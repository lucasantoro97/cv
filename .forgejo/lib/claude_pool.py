#!/usr/bin/env python3
"""Run generic implementers through closed, transactional provider lanes.

Codex Sol runs first and every pass gets a fresh local clone.  Claude fallback
accounts also run in fresh clones, additionally inside a bubblewrap filesystem
namespace.  Failed and no-change attempts are discarded; only a successful
bounded git patch plus bounded new regular files is promoted.  Provider output,
credentials, and pool/account identifiers are never emitted.

For the Codex provider, exit 0 means a diff was promoted and exit 3 means both
passes produced no diff.  The local Gemma provider makes one bounded attempt:
exit 3 means no diff and exit 69 means its CLI or local endpoint was unavailable.
For Claude, exit 0 means the provider completed (possibly without a promotable
diff), while exit 75 means every configured account returned a structured
capacity/auth failure.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import NamedTuple


EXHAUSTED = 75
TIMEOUT = 124
OUTPUT_LIMIT = 125
NO_CHANGE = 3
UNAVAILABLE = 69
KEY = re.compile(r"^[0-9a-f]{12}$")
MODEL = "sonnet"
CODEX_MODEL = "gpt-5.6-sol"
CODEX_MODEL_ID = re.compile(r"^gpt-[a-z0-9]+(?:[.-][a-z0-9]+)*$")
LOCAL_GEMMA_PROVIDER_CLI = "local-gemma"
LOCAL_GEMMA_PROVIDER_ID = "local_gemma"
LOCAL_GEMMA_ENDPOINT = "http://10.200.180.56:8011/v1"
LOCAL_GEMMA_MODEL = "gemma-4-26b-a4b"
LOCAL_GEMMA_IDENTITY = f"{LOCAL_GEMMA_PROVIDER_ID}:{LOCAL_GEMMA_MODEL}"
TOOLS = ("Read", "Edit", "Write", "Glob", "Grep")
DISALLOWED_TOOLS = (
    "Bash",
    "WebFetch",
    "WebSearch",
    "Skill",
    "Agent",
    "Task",
    "NotebookEdit",
)
SAFE_PARENT_ENV = (
    "LANG",
    "LC_ALL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TERM",
)
RETRYABLE_ERRORS = {"rate_limit", "authentication_failed"}
PROTECTED_SCAFFOLDS = (
    ".issue-assets",
    ".issue-raw.txt",
    ".issue-prompt.txt",
    ".context",
    ".standards",
)
MAX_POOL_BYTES = 1024 * 1024
MAX_POOL_ACCOUNTS = 32
MAX_TOKEN_BYTES = 16 * 1024
MAX_PROMPT_BYTES = 256 * 1024
MAX_EVENT_BYTES = 8 * 1024 * 1024
MAX_BASELINE_BYTES = 256 * 1024 * 1024
MAX_PROMOTION_BYTES = 64 * 1024 * 1024
MAX_PROMOTION_FILE_BYTES = 16 * 1024 * 1024
MAX_PROMOTION_FILES = 4096
MIN_TIMEOUT_SEC = 1
MAX_TIMEOUT_SEC = 1800
MIN_TURNS = 1
MAX_TURNS = 50


class PoolError(ValueError):
    pass


class Baseline(NamedTuple):
    head: str
    status: bytes
    protected: tuple[str, ...]
    protected_digest: str


class CodexAttemptResult(NamedTuple):
    returncode: int
    no_change: bool = False
    unavailable: bool = False


class LocalGemmaAttemptResult(NamedTuple):
    returncode: int
    no_change: bool = False
    unavailable: bool = False


def _retryable_cli_failure(output: bytes) -> bool:
    """Trust only provider-owned API error fields from Claude stream-json events."""
    for line in output.decode("utf-8", "replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = event.get("message") if isinstance(event, dict) else None
        legacy_error = (
            isinstance(event, dict)
            and event.get("isApiErrorMessage") is True
            and event.get("error") in RETRYABLE_ERRORS
        )
        current_error = (
            isinstance(message, dict)
            and message.get("is_api_error_message") is True
            and message.get("error") in RETRYABLE_ERRORS
        )
        if (
            isinstance(event, dict)
            and event.get("type") == "assistant"
            and isinstance(message, dict)
            and message.get("role") == "assistant"
            and (legacy_error or current_error)
        ):
            return True
    return False


def _successful_cli_result(output: bytes) -> bool:
    """Require one unambiguous terminal stream-json success event."""
    events: list[dict[str, object]] = []
    for line in output.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(event, dict):
            return False
        events.append(event)
    results = [event for event in events if event.get("type") == "result"]
    if len(results) != 1 or not events or events[-1] is not results[0]:
        return False
    terminal = results[0]
    if terminal.get("subtype") != "success" or terminal.get("is_error") is True:
        return False
    return not any(
        event.get("isApiErrorMessage") is True or event.get("type") == "error"
        for event in events
    )


def _local_gemma_runtime_unavailable(returncode: int, output: bytes) -> bool:
    """Accept fallback only for a Codex runtime/transport error event.

    Provider/model text is untrusted.  A generic non-zero exit therefore remains
    a provider failure; only Codex's JSON ``error`` events with a narrow transport
    signature, plus exec-loader failures, may be classified as unavailable.
    """
    if returncode in (126, 127):
        return True
    runtime_markers = (
        "connection refused",
        "connection reset",
        "connection timed out",
        "network is unreachable",
        "no route to host",
        "name or service not known",
        "temporary failure in name resolution",
        "dns lookup failed",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "http status 502",
        "http status 503",
        "http status 504",
    )
    for line in output.decode("utf-8", "replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "error":
            continue
        message = event.get("message")
        if isinstance(message, str) and any(marker in message.lower() for marker in runtime_markers):
            return True
    return False


def _codex_runtime_unavailable(returncode: int, output: bytes) -> bool:
    """Classify only Codex-owned JSON error events as auth/capacity/transport loss."""
    if returncode in (126, 127):
        return True
    markers = (
        "rate limit",
        "usage limit",
        "quota",
        "not logged in",
        "authentication",
        "credentials",
        "connection refused",
        "connection reset",
        "connection timed out",
        "network is unreachable",
        "no route to host",
        "name or service not known",
        "temporary failure in name resolution",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )
    for line in output.decode("utf-8", "replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        message = event.get("message") if event.get("type") == "error" else None
        if event.get("type") == "turn.failed" and isinstance(event.get("error"), dict):
            message = event["error"].get("message")
        if isinstance(message, str) and any(marker in message.lower() for marker in markers):
            return True
    return False


def _pool(environ: Mapping[str, str]) -> list[str]:
    encoded = environ.get("CLAUDE_CODE_OAUTH_POOL_B64", "")
    if not encoded or len(encoded) > MAX_POOL_BYTES:
        raise PoolError("missing or invalid Claude OAuth pool")
    try:
        decoded = base64.b64decode(encoded, validate=True)
        rows = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoolError("missing or invalid Claude OAuth pool") from exc
    if not isinstance(rows, list) or not rows or len(rows) > MAX_POOL_ACCOUNTS:
        raise PoolError("missing or invalid Claude OAuth pool")
    keys: set[str] = set()
    tokens: set[str] = set()
    pool: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"key", "token"}:
            raise PoolError("missing or invalid Claude OAuth pool")
        key, token = row["key"], row["token"]
        if (
            not isinstance(key, str)
            or not KEY.fullmatch(key)
            or not isinstance(token, str)
            or not token
            or token != token.strip()
            or len(token.encode("utf-8")) > MAX_TOKEN_BYTES
            or key in keys
            or token in tokens
        ):
            raise PoolError("missing or invalid Claude OAuth pool")
        keys.add(key)
        tokens.add(token)
        pool.append(token)
    return pool


def _bounded_int(value: int, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise PoolError(f"invalid {label}")
    return value


def _model(environ: Mapping[str, str]) -> str:
    model = environ.get("AGENT_MODEL", "")
    if model != MODEL:
        raise PoolError("AGENT_MODEL must be exact sonnet")
    return model


def _regular_prompt(path: str) -> tuple[bytes, str]:
    absolute = os.path.abspath(path)
    try:
        info = os.lstat(absolute)
    except OSError as exc:
        raise PoolError("prompt file unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > MAX_PROMPT_BYTES:
        raise PoolError("prompt file is not a bounded regular file")
    try:
        raw = Path(absolute).read_bytes()
        raw.decode("utf-8", "strict")
    except (OSError, UnicodeError) as exc:
        raise PoolError("prompt file is not valid UTF-8") from exc
    if not raw or len(raw) > MAX_PROMPT_BYTES:
        raise PoolError("prompt file is not a bounded regular file")
    return raw, absolute


def _worktree(path: str) -> str:
    absolute = os.path.abspath(path)
    try:
        info = os.lstat(absolute)
    except OSError as exc:
        raise PoolError("worktree unavailable") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise PoolError("worktree must be a real directory")
    return absolute


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired("provider transaction", 0)
    return remaining


def _git_env(environ: Mapping[str, str], root: Path) -> dict[str, str]:
    home = root / "git-home"
    home.mkdir(mode=0o700, exist_ok=True)
    return {
        "PATH": environ.get("PATH") or "/usr/local/bin:/usr/bin:/bin",
        "LANG": environ.get("LANG") or "C.UTF-8",
        "HOME": str(home),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    }


def _git(
    worktree: str,
    git_env: Mapping[str, str],
    deadline: float,
    *args: str,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-C",
            worktree,
            *args,
        ],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(git_env),
        timeout=_remaining(deadline),
        check=False,
    )
    if check and result.returncode != 0:
        raise PoolError("safe git transaction failed")
    return result


def _safe_relative(path: str) -> str:
    if not path or "\x00" in path or path.startswith("/"):
        raise PoolError("unsafe candidate path")
    pure = PurePosixPath(path)
    if any(part in ("", ".", "..", ".git") for part in pure.parts):
        raise PoolError("unsafe candidate path")
    return pure.as_posix()


def _status_roots(raw: bytes) -> tuple[str, ...]:
    roots: list[str] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        if not entry.startswith(b"?? "):
            raise PoolError("Claude baseline has tracked changes")
        path = os.fsdecode(entry[3:]).rstrip("/")
        roots.append(_safe_relative(path))
    return tuple(roots)


def _under(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in roots)


def _protected_candidate_path(path: str, roots: tuple[str, ...]) -> bool:
    """Recognize a path inside a protected root without making it promotable.

    Protected scaffolds may legitimately contain a nested Git checkout (for
    example ``.standards/.git``).  Such paths must be ignored by candidate
    discovery because the whole scaffold is copied only for read access and is
    never promoted.  Unsafe traversal/absolute forms deliberately return false
    so the normal strict path validator still rejects them.
    """
    if not path or "\x00" in path or path.startswith("/"):
        return False
    pure = PurePosixPath(path)
    if any(part in ("", ".", "..") for part in pure.parts):
        return False
    return _under(pure.as_posix(), roots)


def _hash_node(
    root: Path,
    relative: str,
    digest: hashlib._Hash,
    budget: list[int],
    deadline: float,
) -> None:
    _remaining(deadline)
    path = root / relative
    info = os.lstat(path)
    mode = stat.S_IMODE(info.st_mode)
    digest.update(relative.encode("utf-8", "surrogateescape") + b"\0")
    digest.update(f"{stat.S_IFMT(info.st_mode):o}:{mode:o}\0".encode())
    if stat.S_ISREG(info.st_mode):
        budget[0] += info.st_size
        if budget[0] > MAX_BASELINE_BYTES:
            raise PoolError("protected baseline is too large")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                _remaining(deadline)
                digest.update(chunk)
    elif stat.S_ISDIR(info.st_mode):
        for child in sorted(path.iterdir(), key=lambda value: os.fsencode(value.name)):
            _hash_node(root, f"{relative}/{child.name}", digest, budget, deadline)
    else:
        raise PoolError("protected baseline contains a special file")


def _protected_digest(root: str, protected: tuple[str, ...], deadline: float) -> str:
    digest = hashlib.sha256()
    budget = [0]
    for relative in protected:
        path = Path(root) / relative
        if path.exists() or path.is_symlink():
            _hash_node(Path(root), relative, digest, budget, deadline)
        else:
            digest.update(f"absent:{relative}\0".encode("utf-8", "surrogateescape"))
    return digest.hexdigest()


def _baseline(
    worktree: str,
    prompt_file: str,
    git_env: Mapping[str, str],
    deadline: float,
) -> Baseline:
    top = _git(worktree, git_env, deadline, "rev-parse", "--show-toplevel").stdout
    try:
        top_path = top.decode("utf-8", "strict").strip()
    except UnicodeDecodeError as exc:
        raise PoolError("worktree path is not UTF-8") from exc
    if not top_path or not os.path.samefile(top_path, worktree):
        raise PoolError("worktree must be the git top level")
    head = _git(worktree, git_env, deadline, "rev-parse", "--verify", "HEAD").stdout.decode().strip()
    tracked = _git(
        worktree,
        git_env,
        deadline,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=normal",
    ).stdout
    roots = list(_status_roots(tracked))
    for scaffold in PROTECTED_SCAFFOLDS:
        if (Path(worktree) / scaffold).exists() and not _under(scaffold, tuple(roots)):
            roots.append(scaffold)
    try:
        common = os.path.commonpath((worktree, prompt_file))
    except ValueError:
        common = ""
    if common == worktree and Path(prompt_file).exists():
        prompt_relative = _safe_relative(os.path.relpath(prompt_file, worktree))
        if not _under(prompt_relative, tuple(roots)):
            roots.append(prompt_relative)
    protected = tuple(dict.fromkeys(roots))
    return Baseline(
        head=head,
        status=tracked,
        protected=protected,
        protected_digest=_protected_digest(worktree, protected, deadline),
    )


def _assert_baseline(
    worktree: str,
    baseline: Baseline,
    git_env: Mapping[str, str],
    deadline: float,
) -> None:
    current = _baseline(worktree, str(Path(worktree) / ".issue-prompt.txt"), git_env, deadline)
    if (
        current.head != baseline.head
        or current.status != baseline.status
        or current.protected != baseline.protected
        or current.protected_digest != baseline.protected_digest
    ):
        raise PoolError("original worktree changed during Claude transaction")


def _copy_regular_tree(
    source: Path,
    destination: Path,
    deadline: float | None = None,
) -> None:
    if deadline is not None:
        _remaining(deadline)
    info = os.lstat(source)
    if stat.S_ISREG(info.st_mode):
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            with source.open("rb") as input_handle, destination.open("xb") as output:
                for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                    if deadline is not None:
                        _remaining(deadline)
                    output.write(chunk)
            shutil.copystat(source, destination, follow_symlinks=False)
        except BaseException:
            _remove_node(destination)
            raise
    elif stat.S_ISDIR(info.st_mode):
        destination.mkdir(mode=stat.S_IMODE(info.st_mode) or 0o700, parents=True, exist_ok=True)
        for child in source.iterdir():
            _copy_regular_tree(child, destination / child.name, deadline)
    else:
        raise PoolError("protected baseline contains a special file")


def _prepare_clone(
    original: str,
    sandbox: Path,
    baseline: Baseline,
    git_env: Mapping[str, str],
    deadline: float,
) -> None:
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "clone",
            "--no-local",
            "--no-hardlinks",
            "--quiet",
            "--no-checkout",
            "--",
            original,
            str(sandbox),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(git_env),
        timeout=_remaining(deadline),
        check=False,
    )
    if result.returncode != 0:
        raise PoolError("safe local clone failed")
    _git(str(sandbox), git_env, deadline, "checkout", "--quiet", "--detach", baseline.head)
    for relative in baseline.protected:
        source = Path(original) / relative
        if source.exists():
            _copy_regular_tree(source, sandbox / relative, deadline)


def _write_settings(path: Path) -> None:
    allow = [f"{tool}(//workspace/**)" for tool in TOOLS]
    settings = {
        "permissions": {
            "defaultMode": "dontAsk",
            "allow": allow,
            "deny": list(DISALLOWED_TOOLS),
        },
        "enabledPlugins": {},
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def _command(executable: str, model: str, max_turns: int, settings: str) -> list[str]:
    allowed = ",".join(f"{tool}(//workspace/**)" for tool in TOOLS)
    denied = ",".join(DISALLOWED_TOOLS)
    return [
        executable,
        "-p",
        "--safe-mode",
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--tools",
        ",".join(TOOLS),
        "--allowedTools",
        allowed,
        "--disallowedTools",
        denied,
        "--permission-mode",
        "dontAsk",
        "--settings",
        settings,
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-chrome",
        "--no-session-persistence",
        "--output-format",
        "stream-json",
        "--verbose",
    ]


def _child_env(parent: Mapping[str, str], token: str) -> dict[str, str]:
    child = {name: parent[name] for name in SAFE_PARENT_ENV if parent.get(name)}
    child.update(
        {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/home/agent",
            "CLAUDE_CONFIG_DIR": "/home/agent/claude-config",
            "XDG_CONFIG_HOME": "/home/agent/xdg-config",
            "XDG_CACHE_HOME": "/home/agent/xdg-cache",
            "XDG_DATA_HOME": "/home/agent/xdg-data",
            "TMPDIR": "/tmp",
            "CLAUDE_CODE_OAUTH_TOKEN": token,
        }
    )
    for name in ("SSL_CERT_DIR", "SSL_CERT_FILE"):
        if name in child and not child[name].startswith(("/etc/", "/usr/")):
            child.pop(name)
    return child


def _codex_configuration(environ: Mapping[str, str]) -> tuple[str, tuple[str, ...], str]:
    model = environ.get("CODEX_PRIMARY_MODEL") or CODEX_MODEL
    supported = tuple((environ.get("SUPPORTED_CODEX_MODELS") or
                       "gpt-5.6-sol,gpt-5.6-terra").split(","))
    codex_home = environ.get("CODEX_HOME", "")
    if (
        model != CODEX_MODEL
        or not supported
        or any(not CODEX_MODEL_ID.fullmatch(value) for value in supported)
        or len(set(supported)) != len(supported)
        or model not in supported
    ):
        raise PoolError("Codex primary must be exact configured gpt-5.6-sol")
    try:
        home_info = os.lstat(codex_home)
        auth_info = os.lstat(Path(codex_home) / "auth.json")
    except OSError as exc:
        raise PoolError("Codex auth home unavailable") from exc
    if (
        not stat.S_ISDIR(home_info.st_mode)
        or stat.S_IMODE(home_info.st_mode) & 0o077
        or not stat.S_ISREG(auth_info.st_mode)
        or stat.S_IMODE(auth_info.st_mode) & 0o077
        or auth_info.st_nlink != 1
        or auth_info.st_size <= 0
        or auth_info.st_size > MAX_POOL_BYTES
    ):
        raise PoolError("Codex auth home unavailable")
    return model, supported, codex_home


def _codex_child_env(
    parent: Mapping[str, str],
    codex_home: str,
    private_home: str,
) -> dict[str, str]:
    child = {name: parent[name] for name in SAFE_PARENT_ENV if parent.get(name)}
    child.update(
        {
            "PATH": parent.get("PATH") or "/usr/local/bin:/usr/bin:/bin",
            "HOME": private_home,
            "CODEX_HOME": codex_home,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return child


def _sandbox_command(
    environ: Mapping[str, str],
    sandbox: Path,
    private_home: Path,
    private_tmp: Path,
    model: str,
    max_turns: int,
) -> list[str]:
    search_path = environ.get("PATH") or "/usr/local/bin:/usr/bin:/bin"
    bwrap = shutil.which("bwrap", path=search_path)
    claude = shutil.which("claude", path=search_path)
    if not bwrap or not claude:
        raise PoolError("required Claude sandbox runtime unavailable")
    claude_path = Path(claude)
    inner_executable = str(claude_path)
    extra_mount: list[str] = []
    if not str(claude_path).startswith(("/usr/", "/bin/")):
        extra_mount = ["--dir", "/opt", "--ro-bind", str(claude_path.parent), "/opt/claude-bin"]
        inner_executable = f"/opt/claude-bin/{claude_path.name}"
    command = [
        bwrap,
        "--unshare-all",
        "--share-net",
        "--die-with-parent",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind-try",
        "/lib64",
        "/lib64",
        "--dir",
        "/etc",
        "--ro-bind-try",
        "/etc/ssl",
        "/etc/ssl",
        "--ro-bind-try",
        "/etc/ca-certificates",
        "/etc/ca-certificates",
        "--ro-bind-try",
        "/etc/ca-certificates.conf",
        "/etc/ca-certificates.conf",
        "--ro-bind-try",
        "/etc/resolv.conf",
        "/etc/resolv.conf",
        "--ro-bind-try",
        "/etc/hosts",
        "/etc/hosts",
        "--ro-bind-try",
        "/etc/nsswitch.conf",
        "/etc/nsswitch.conf",
        "--ro-bind-try",
        "/etc/passwd",
        "/etc/passwd",
        "--ro-bind-try",
        "/etc/group",
        "/etc/group",
        "--ro-bind-try",
        "/etc/ld.so.cache",
        "/etc/ld.so.cache",
        "--dev",
        "/dev",
        "--dir",
        "/proc",
        "--dir",
        "/home",
        "--bind",
        str(private_home),
        "/home/agent",
        "--bind",
        str(private_tmp),
        "/tmp",
        "--bind",
        str(sandbox),
        "/workspace",
        "--ro-bind",
        str(sandbox / ".git"),
        "/workspace/.git",
        *extra_mount,
        "--chdir",
        "/workspace",
        "--",
        *_command(
            inner_executable,
            model,
            max_turns,
            "/home/agent/settings.json",
        ),
    ]
    return command


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    """Reap the leader and destroy its session even if the leader already exited."""
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    end = time.monotonic() + 0.75
    while _group_exists(pgid) and time.monotonic() < end:
        time.sleep(0.025)
    if _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)


def _capture_process(
    process: subprocess.Popen[bytes],
    deadline: float,
) -> tuple[int, bytes, bool, bool]:
    if process.stdout is None:
        raise PoolError("Claude output pipe unavailable")
    os.set_blocking(process.stdout.fileno(), False)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    output = bytearray()
    timed_out = False
    overflow = False
    eof = False
    returncode: int | None = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            for key, _mask in selector.select(min(0.1, remaining)):
                chunk = os.read(key.fd, 64 * 1024)
                if not chunk:
                    eof = True
                    selector.unregister(process.stdout)
                    break
                if len(output) + len(chunk) > MAX_EVENT_BYTES:
                    overflow = True
                    break
                output.extend(chunk)
            if overflow:
                break
            polled = process.poll()
            if polled is not None:
                returncode = polled
                _terminate_group(process)
                if eof:
                    break
                # Teardown closes stdout inherited by any descendant; drain once.
                while True:
                    try:
                        chunk = os.read(process.stdout.fileno(), 64 * 1024)
                    except BlockingIOError:
                        break
                    if not chunk:
                        eof = True
                        break
                    if len(output) + len(chunk) > MAX_EVENT_BYTES:
                        overflow = True
                        break
                    output.extend(chunk)
                break
    finally:
        selector.close()
        _terminate_group(process)
    if returncode is None:
        returncode = process.returncode if process.returncode is not None else 1
    return returncode, bytes(output), timed_out, overflow


def _candidate_paths(
    original: str,
    sandbox: str,
    baseline: Baseline,
    git_env: Mapping[str, str],
    deadline: float,
) -> tuple[bytes, tuple[str, ...], tuple[str, ...]]:
    tracked_raw = _git(
        sandbox, git_env, deadline, "ls-files", "-z",
    ).stdout
    tracked_paths = {
        _safe_relative(os.fsdecode(value)) for value in tracked_raw.split(b"\0") if value
    }

    def reject_untracked_specials(directory: Path, prefix: str = "") -> None:
        for entry in os.scandir(directory):
            if not prefix and entry.name == ".git":
                continue
            raw_relative = f"{prefix}/{entry.name}".lstrip("/")
            if _protected_candidate_path(raw_relative, baseline.protected):
                continue
            relative = _safe_relative(raw_relative)
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                reject_untracked_specials(Path(entry.path), relative)
            elif not stat.S_ISREG(info.st_mode) and relative not in tracked_paths:
                raise PoolError("Claude candidate contains an untracked special file")

    reject_untracked_specials(Path(sandbox))
    patch = _git(
        sandbox,
        git_env,
        deadline,
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-renames",
        baseline.head,
        "--",
    ).stdout
    changed_raw = _git(
        sandbox,
        git_env,
        deadline,
        "diff",
        "--name-only",
        "-z",
        "--no-ext-diff",
        "--no-renames",
        baseline.head,
        "--",
    ).stdout
    changed = tuple(_safe_relative(os.fsdecode(value)) for value in changed_raw.split(b"\0") if value)
    new_raw = _git(
        sandbox,
        git_env,
        deadline,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).stdout
    new_paths: list[str] = []
    for value in new_raw.split(b"\0"):
        if not value:
            continue
        raw_path = os.fsdecode(value)
        if _protected_candidate_path(raw_path, baseline.protected):
            continue
        new_paths.append(_safe_relative(raw_path))
    new = tuple(new_paths)
    if any(_under(path, baseline.protected) for path in changed):
        raise PoolError("Claude attempted to modify protected scaffold")
    if len(set(changed + new)) != len(changed + new):
        raise PoolError("candidate paths overlap")
    if len(changed) + len(new) > MAX_PROMOTION_FILES or len(patch) > MAX_PROMOTION_BYTES:
        raise PoolError("Claude candidate exceeds promotion bounds")
    total = len(patch)
    for relative in changed + new:
        path = Path(sandbox) / relative
        if path.exists() or path.is_symlink():
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_PROMOTION_FILE_BYTES:
                raise PoolError("Claude candidate contains a special or oversized file")
            total += info.st_size
    for relative in changed:
        path = Path(original) / relative
        if path.exists() or path.is_symlink():
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_PROMOTION_FILE_BYTES:
                raise PoolError("original candidate contains a special or oversized file")
            total += info.st_size
    if total > MAX_PROMOTION_BYTES:
        raise PoolError("Claude candidate exceeds promotion bounds")
    return patch, changed, new


def _remove_node(path: Path) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(info.st_mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def _copy_new_file(source: Path, destination: Path, deadline: float) -> None:
    _remaining(deadline)
    info = os.lstat(source)
    if not stat.S_ISREG(info.st_mode):
        raise PoolError("new candidate is not a regular file")
    current = destination.parent
    missing: list[Path] = []
    while not current.exists():
        missing.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise PoolError("new candidate parent is unsafe")
    for parent in reversed(missing):
        parent.mkdir(mode=0o755)
    fd = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        stat.S_IMODE(info.st_mode) or 0o600,
    )
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as input_handle:
            for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                _remaining(deadline)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        _remove_node(destination)
        raise


def _promote(
    original: str,
    sandbox: str,
    baseline: Baseline,
    git_env: Mapping[str, str],
    deadline: float,
    root: Path,
) -> bool:
    patch, changed, new = _candidate_paths(original, sandbox, baseline, git_env, deadline)
    _assert_baseline(original, baseline, git_env, deadline)
    if not patch and not new:
        return False
    if patch:
        _git(original, git_env, deadline, "apply", "--check", "--binary", "--whitespace=nowarn", "-", input_bytes=patch)
    for relative in new:
        destination = Path(original) / relative
        if destination.exists() or destination.is_symlink():
            raise PoolError("new candidate collides with original worktree")
    backup = root / "promotion-backup"
    backup.mkdir(mode=0o700)
    existing: set[str] = set()
    created_directories: set[Path] = set()
    for relative in changed:
        source = Path(original) / relative
        if source.exists() or source.is_symlink():
            info = os.lstat(source)
            if not stat.S_ISREG(info.st_mode):
                raise PoolError("original candidate path is not a regular file")
            existing.add(relative)
            _copy_regular_tree(source, backup / relative, deadline)
    try:
        if patch:
            _git(original, git_env, deadline, "apply", "--binary", "--whitespace=nowarn", "-", input_bytes=patch)
        for relative in new:
            destination = Path(original) / relative
            parent = destination.parent
            while not parent.exists():
                created_directories.add(parent)
                parent = parent.parent
            _copy_new_file(Path(sandbox) / relative, destination, deadline)
    except BaseException:
        for relative in changed + new:
            _remove_node(Path(original) / relative)
        for relative in existing:
            # Rollback is safety cleanup: once promotion started it must finish
            # even if the execution budget expired.
            _copy_regular_tree(backup / relative, Path(original) / relative)
        for directory in sorted(created_directories, key=lambda value: len(value.parts), reverse=True):
            try:
                directory.rmdir()
            except (FileNotFoundError, OSError):
                pass
        raise
    return True


def _run_attempt(
    prompt: bytes,
    worktree: str,
    prompt_file: str,
    model: str,
    max_turns: int,
    deadline: float,
    token: str,
    environ: Mapping[str, str],
) -> tuple[int, bytes, bool]:
    temp_parent = environ.get("RUNNER_TEMP") or None
    if temp_parent is not None and not os.path.isdir(temp_parent):
        raise PoolError("RUNNER_TEMP is not a directory")
    try:
        with tempfile.TemporaryDirectory(prefix=".claude-implementer.", dir=temp_parent) as raw_root:
            root = Path(raw_root)
            root.chmod(0o700)
            git_env = _git_env(environ, root)
            baseline = _baseline(worktree, prompt_file, git_env, deadline)
            sandbox = root / "candidate"
            _prepare_clone(worktree, sandbox, baseline, git_env, deadline)
            private_home = root / "home"
            private_tmp = root / "tmp"
            private_home.mkdir(mode=0o700)
            private_tmp.mkdir(mode=0o700)
            for name in ("claude-config", "xdg-config", "xdg-cache", "xdg-data"):
                (private_home / name).mkdir(mode=0o700)
            settings = private_home / "settings.json"
            prompt_path = root / "prompt.txt"
            _write_settings(settings)
            fd = os.open(prompt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(prompt)
            child_env = _child_env(environ, token)
            with prompt_path.open("rb") as prompt_handle:
                process = subprocess.Popen(
                    _sandbox_command(environ, sandbox, private_home, private_tmp, model, max_turns),
                    cwd=str(root),
                    env=child_env,
                    stdin=prompt_handle,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                returncode, output, timed_out, overflow = _capture_process(process, deadline)
            if timed_out:
                return TIMEOUT, b"", True
            if overflow:
                return OUTPUT_LIMIT, b"", False
            if returncode == 0 and _successful_cli_result(output):
                try:
                    _promote(worktree, str(sandbox), baseline, git_env, deadline, root)
                except PoolError:
                    return 1, b"", False
            elif returncode == 0:
                return 1, output, False
            return returncode, output, False
    except subprocess.TimeoutExpired:
        return TIMEOUT, b"", True


def _codex_command(
    environ: Mapping[str, str],
    sandbox: Path,
    model: str,
) -> list[str]:
    search_path = environ.get("PATH") or "/usr/local/bin:/usr/bin:/bin"
    codex = shutil.which("codex", path=search_path)
    if not codex:
        raise PoolError("required Codex runtime unavailable")
    return [
        codex,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "workspace-write",
        "--color",
        "never",
        "--json",
        "-C",
        str(sandbox),
        "-c",
        'model_reasoning_effort="ultra"',
        "-m",
        model,
        "-",
    ]


def _run_codex_attempt(
    prompt: bytes,
    worktree: str,
    prompt_file: str,
    model: str,
    codex_home: str,
    deadline: float,
    environ: Mapping[str, str],
) -> CodexAttemptResult:
    temp_parent = environ.get("RUNNER_TEMP") or None
    if temp_parent is not None and not os.path.isdir(temp_parent):
        raise PoolError("RUNNER_TEMP is not a directory")
    try:
        with tempfile.TemporaryDirectory(prefix=".codex-implementer.", dir=temp_parent) as raw_root:
            root = Path(raw_root)
            root.chmod(0o700)
            git_env = _git_env(environ, root)
            baseline = _baseline(worktree, prompt_file, git_env, deadline)
            sandbox = root / "candidate"
            _prepare_clone(worktree, sandbox, baseline, git_env, deadline)
            private_home = root / "home"
            private_home.mkdir(mode=0o700)
            prompt_path = root / "prompt.txt"
            fd = os.open(prompt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(prompt)
            child_env = _codex_child_env(environ, codex_home, str(private_home))
            with prompt_path.open("rb") as prompt_handle:
                process = subprocess.Popen(
                    _codex_command(environ, sandbox, model),
                    cwd=str(root),
                    env=child_env,
                    stdin=prompt_handle,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                returncode, _output, timed_out, overflow = _capture_process(process, deadline)
            if timed_out:
                return CodexAttemptResult(TIMEOUT)
            if overflow:
                return CodexAttemptResult(OUTPUT_LIMIT)
            if returncode != 0:
                return CodexAttemptResult(
                    returncode or 1,
                    unavailable=_codex_runtime_unavailable(returncode or 1, _output),
                )
            try:
                promoted = _promote(
                    worktree,
                    str(sandbox),
                    baseline,
                    git_env,
                    deadline,
                    root,
                )
            except PoolError:
                return CodexAttemptResult(1)
            return CodexAttemptResult(0, no_change=not promoted)
    except subprocess.TimeoutExpired:
        return CodexAttemptResult(TIMEOUT)


def _local_gemma_command(
    environ: Mapping[str, str],
    sandbox: Path,
) -> list[str]:
    """Build the closed, literal-only local provider invocation."""
    search_path = environ.get("PATH") or "/usr/local/bin:/usr/bin:/bin"
    codex = shutil.which("codex", path=search_path)
    if not codex:
        raise FileNotFoundError("codex")
    provider = f"model_providers.{LOCAL_GEMMA_PROVIDER_ID}"
    return [
        codex,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "workspace-write",
        "--color",
        "never",
        "--json",
        "-c",
        f'model_provider="{LOCAL_GEMMA_PROVIDER_ID}"',
        "-c",
        f'{provider}.name="{LOCAL_GEMMA_PROVIDER_CLI}"',
        "-c",
        f'{provider}.base_url="{LOCAL_GEMMA_ENDPOINT}"',
        "-c",
        f'{provider}.wire_api="responses"',
        "-c",
        f"{provider}.requires_openai_auth=false",
        "-c",
        f"{provider}.request_max_retries=0",
        "-c",
        f"{provider}.stream_max_retries=0",
        "-C",
        str(sandbox),
        "-m",
        LOCAL_GEMMA_MODEL,
        "-",
    ]


def _run_local_gemma_attempt(
    prompt: bytes,
    worktree: str,
    prompt_file: str,
    deadline: float,
    environ: Mapping[str, str],
) -> LocalGemmaAttemptResult:
    temp_parent = environ.get("RUNNER_TEMP") or None
    if temp_parent is not None and not os.path.isdir(temp_parent):
        raise PoolError("RUNNER_TEMP is not a directory")
    try:
        with tempfile.TemporaryDirectory(prefix=".local-gemma-implementer.", dir=temp_parent) as raw_root:
            root = Path(raw_root)
            root.chmod(0o700)
            git_env = _git_env(environ, root)
            baseline = _baseline(worktree, prompt_file, git_env, deadline)
            sandbox = root / "candidate"
            _prepare_clone(worktree, sandbox, baseline, git_env, deadline)
            private_home = root / "home"
            private_codex_home = root / "codex-home"
            private_home.mkdir(mode=0o700)
            private_codex_home.mkdir(mode=0o700)
            prompt_path = root / "prompt.txt"
            fd = os.open(prompt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(prompt)
            child_env = _codex_child_env(
                environ,
                str(private_codex_home),
                str(private_home),
            )
            with prompt_path.open("rb") as prompt_handle:
                try:
                    process = subprocess.Popen(
                        _local_gemma_command(environ, sandbox),
                        cwd=str(root),
                        env=child_env,
                        stdin=prompt_handle,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except OSError:
                    return LocalGemmaAttemptResult(UNAVAILABLE, unavailable=True)
                returncode, output, timed_out, overflow = _capture_process(process, deadline)
            if timed_out:
                return LocalGemmaAttemptResult(UNAVAILABLE, unavailable=True)
            if overflow:
                return LocalGemmaAttemptResult(OUTPUT_LIMIT)
            if returncode != 0:
                return LocalGemmaAttemptResult(
                    returncode or 1,
                    unavailable=_local_gemma_runtime_unavailable(returncode or 1, output),
                )
            try:
                promoted = _promote(
                    worktree,
                    str(sandbox),
                    baseline,
                    git_env,
                    deadline,
                    root,
                )
            except PoolError:
                return LocalGemmaAttemptResult(1)
            return LocalGemmaAttemptResult(0, no_change=not promoted)
    except subprocess.TimeoutExpired:
        return LocalGemmaAttemptResult(UNAVAILABLE, unavailable=True)


def run_codex(
    worktree: str,
    prompt_file: str,
    *,
    timeout_sec: int,
    environ: Mapping[str, str] | None = None,
) -> int:
    env = dict(os.environ if environ is None else environ)
    model, _supported, codex_home = _codex_configuration(env)
    timeout = _bounded_int(timeout_sec, MIN_TIMEOUT_SEC, MAX_TIMEOUT_SEC, "timeout")
    prompt, prompt_path = _regular_prompt(prompt_file)
    directory = _worktree(worktree)
    deadline = time.monotonic() + timeout
    retry_suffix = (
        b"\n\nIl primo passaggio non ha prodotto modifiche. Riesamina il repository "
        b"e completa ora il risultato minimo verificabile. Non fare domande e non "
        b"terminare senza una modifica pertinente."
    )
    retry_prompt = prompt + retry_suffix
    if len(retry_prompt) > MAX_PROMPT_BYTES:
        retry_prompt = prompt
    for attempt, attempt_prompt in enumerate((prompt, retry_prompt), start=1):
        outcome = _run_codex_attempt(
            attempt_prompt,
            directory,
            prompt_path,
            model,
            codex_home,
            deadline,
            env,
        )
        if outcome.returncode == 0 and outcome.no_change:
            if attempt == 1:
                print("claude_pool: Codex Sol produced no diff; retrying", file=sys.stderr)
                continue
            print("claude_pool: Codex Sol produced no diff after two passes", file=sys.stderr)
            return NO_CHANGE
        returncode = outcome.returncode
        if returncode == 0:
            print("claude_pool: Codex Sol transaction completed", file=sys.stderr)
            return 0
        if outcome.unavailable:
            print("claude_pool: Codex Sol runtime unavailable", file=sys.stderr)
            return UNAVAILABLE
        if returncode == TIMEOUT:
            print("claude_pool: Codex Sol transaction timed out safely", file=sys.stderr)
        elif returncode == OUTPUT_LIMIT:
            print("claude_pool: Codex Sol output exceeded bounded limit", file=sys.stderr)
        else:
            print("claude_pool: Codex Sol failed safely", file=sys.stderr)
        return returncode or 1
    return NO_CHANGE


def run_local_gemma(
    worktree: str,
    prompt_file: str,
    *,
    timeout_sec: int,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Run one local, unauthenticated Gemma transaction with no configuration inputs."""
    env = dict(os.environ if environ is None else environ)
    timeout = _bounded_int(timeout_sec, MIN_TIMEOUT_SEC, MAX_TIMEOUT_SEC, "timeout")
    prompt, prompt_path = _regular_prompt(prompt_file)
    directory = _worktree(worktree)
    outcome = _run_local_gemma_attempt(
        prompt,
        directory,
        prompt_path,
        time.monotonic() + timeout,
        env,
    )
    if outcome.returncode == 0:
        if outcome.no_change:
            print("claude_pool: local Gemma produced no diff", file=sys.stderr)
            return NO_CHANGE
        print("claude_pool: local Gemma transaction completed", file=sys.stderr)
        return 0
    if outcome.unavailable:
        print("claude_pool: local Gemma runtime unavailable", file=sys.stderr)
        return UNAVAILABLE
    if outcome.returncode == OUTPUT_LIMIT:
        print("claude_pool: local Gemma output exceeded bounded limit", file=sys.stderr)
    else:
        print("claude_pool: local Gemma failed safely", file=sys.stderr)
    return outcome.returncode or 1


def run(
    worktree: str,
    prompt_file: str,
    *,
    max_turns: int,
    timeout_sec: int,
    environ: Mapping[str, str] | None = None,
) -> int:
    env = dict(os.environ if environ is None else environ)
    model = _model(env)
    turns = _bounded_int(max_turns, MIN_TURNS, MAX_TURNS, "max turns")
    timeout = _bounded_int(timeout_sec, MIN_TIMEOUT_SEC, MAX_TIMEOUT_SEC, "timeout")
    prompt, prompt_path = _regular_prompt(prompt_file)
    directory = _worktree(worktree)
    deadline = time.monotonic() + timeout
    for token in _pool(env):
        returncode, output, timed_out = _run_attempt(
            prompt,
            directory,
            prompt_path,
            model,
            turns,
            deadline,
            token,
            env,
        )
        if timed_out:
            print("claude_pool: Claude transaction timed out safely", file=sys.stderr)
            return TIMEOUT
        if returncode == 0:
            print("claude_pool: Claude transaction completed", file=sys.stderr)
            return 0
        if returncode == OUTPUT_LIMIT:
            print("claude_pool: Claude output exceeded bounded limit", file=sys.stderr)
            return OUTPUT_LIMIT
        if not _retryable_cli_failure(output):
            print("claude_pool: non-retryable Claude failure", file=sys.stderr)
            return returncode or 1
        print("claude_pool: Claude capacity/auth unavailable; rotating", file=sys.stderr)
    print("claude_pool: all configured accounts exhausted", file=sys.stderr)
    return EXHAUSTED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=("codex", "claude", LOCAL_GEMMA_PROVIDER_CLI),
        required=True,
    )
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--timeout-sec", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        if args.provider == "codex":
            return run_codex(
                args.worktree,
                args.prompt_file,
                timeout_sec=args.timeout_sec,
            )
        if args.provider == LOCAL_GEMMA_PROVIDER_CLI:
            return run_local_gemma(
                args.worktree,
                args.prompt_file,
                timeout_sec=args.timeout_sec,
            )
        return run(
            args.worktree,
            args.prompt_file,
            max_turns=args.max_turns,
            timeout_sec=args.timeout_sec,
        )
    except PoolError as exc:
        print(f"claude_pool: {exc}", file=sys.stderr)
        return 64
    except OSError as exc:
        print(f"claude_pool: CLI unavailable ({type(exc).__name__})", file=sys.stderr)
        return 127
    except subprocess.SubprocessError as exc:
        print(f"claude_pool: CLI failed safely ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
