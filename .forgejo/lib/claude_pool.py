#!/usr/bin/env python3
"""Run one Claude CLI command against a closed OAuth pool without logging tokens.

Exit 75 means every configured account failed with an explicitly transient
capacity/authentication response.  Callers must surface it as a recoverable
job failure, never mistake it for a completed ticket.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence


EXHAUSTED = 75
KEY = re.compile(r"^[0-9a-f]{12}$")
# These are *protocol records*, not a grep across model output.  A ticket can
# ask the model to emit words such as "rate limit" or "unauthorized"; those
# words must never rotate accounts.  The structured CLI event is authoritative;
# the second form is the exact one-line orchestra limit record.
ORCHESTRA_LIMIT = re.compile(
    r'^error="(?:rate_limit|authentication_failed)", isApiErrorMessage=true$',
)


class PoolError(ValueError):
    pass


def _retryable_cli_failure(output: bytes) -> bool:
    for line in output.decode("utf-8", "replace").splitlines():
        if ORCHESTRA_LIMIT.fullmatch(line):
            return True
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (isinstance(event, dict) and event.get("isApiErrorMessage") is True
                and event.get("error") in {"rate_limit", "authentication_failed"}):
            return True
    return False


def _legacy_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def _pool(environ: dict[str, str]) -> list[tuple[str, str]]:
    encoded = environ.get("CLAUDE_CODE_OAUTH_POOL_B64", "")
    if not encoded:
        legacy = environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
        if not legacy:
            raise PoolError("no Claude OAuth credential configured")
        return [(_legacy_key(legacy), legacy)]
    try:
        decoded = base64.b64decode(encoded, validate=True)
        rows = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoolError("invalid Claude OAuth pool") from exc
    if not isinstance(rows, list) or not rows:
        raise PoolError("invalid Claude OAuth pool")
    pool: list[tuple[str, str]] = []
    keys: set[str] = set()
    tokens: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"key", "token"}:
            raise PoolError("invalid Claude OAuth pool")
        key, token = row["key"], row["token"]
        if (not isinstance(key, str) or not KEY.fullmatch(key)
                or not isinstance(token, str) or not token or token != token.strip()
                or key in keys or token in tokens):
            raise PoolError("invalid Claude OAuth pool")
        keys.add(key); tokens.add(token); pool.append((key, token))
    return pool


def run(argv: Sequence[str], environ: dict[str, str] | None = None) -> int:
    if not argv:
        raise PoolError("missing Claude command")
    env = dict(os.environ if environ is None else environ)
    for key, token in _pool(env):
        child_env = dict(env)
        child_env["CLAUDE_CODE_OAUTH_TOKEN"] = token
        # Each subprocess is a fresh Claude process; OAuth session state from an
        # exhausted account cannot bleed into the next account.
        result = subprocess.run(argv, env=child_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        output = result.stdout or b""
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
        if result.returncode == 0:
            return 0
        if not _retryable_cli_failure(output):
            print(f"claude_pool: non-retryable Claude exit for account={key}", file=sys.stderr)
            return result.returncode or 1
        print(f"claude_pool: capacity/auth unavailable for account={key}", file=sys.stderr)
    print("claude_pool: all configured accounts exhausted", file=sys.stderr)
    return EXHAUSTED


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args.pop(0) != "--":
        print("usage: claude_pool.py -- CLAUDE_COMMAND ...", file=sys.stderr)
        return 64
    try:
        return run(args)
    except PoolError as exc:
        print(f"claude_pool: {exc}", file=sys.stderr)
        return 64
    except OSError as exc:
        print(f"claude_pool: CLI unavailable ({type(exc).__name__})", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
