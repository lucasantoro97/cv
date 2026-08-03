#!/usr/bin/env python3
"""Fail-closed validation for the disposable PIANO planner workspace.

The planner is untrusted.  This module accepts one newly-created plan or one
HOME-Q question, never a pre-existing plan or any other workspace mutation.
Only the sanitized plan is copied outside the disposable clone.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from egress_gate import SECRET
from question import QUESTION_FILE, QuestionError, validate as validate_question
from redact import redact

PLAN_FILE = ".agent-plan.md"
MAX_PLAN_CHARS = 8_000
MAX_PLAN_LINES = 240

_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)")
_ITALIAN_FISCAL_CODE = re.compile(r"(?i)\b[A-Z]{6}\d{2}[A-EHLMPRST]\d{2}[A-Z]\d{3}[A-Z]\b")
_PRIVATE_KEY = re.compile(r"-----BEGIN[^\r\n]*PRIVATE KEY", re.IGNORECASE)
_KNOWN_CREDENTIAL = re.compile(
    r"(?:\bsk-ant-[A-Za-z0-9_-]{8,}\b|\bgh[po]_[A-Za-z0-9]{8,}\b|"
    r"\bgithub_pat_[A-Za-z0-9_]{8,}\b|\bAuthorization\s*:\s*Bearer\s+\S+|"
    r"\bAKIA[0-9A-Z]{16}\b)",
    re.IGNORECASE,
)
_GENERIC_CREDENTIAL_ASSIGNMENT = re.compile(
    r"\b[A-Z][A-Z0-9_]*(?:_TOKEN|_KEY|_SECRET)\s*(?:=|:)\s*['\"]?"
    r"(?P<value>[A-Za-z0-9_-]{20,})\b",
    re.IGNORECASE,
)


class PlanError(ValueError):
    """Planner output is not safe execution input."""


@dataclass(frozen=True)
class Inspection:
    outcome: str
    plan: str | None = None


def _git(worktree: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        cwd=worktree,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _status_paths(worktree: Path) -> list[bytes]:
    """Parse porcelain v1 -z without decoding attacker-controlled pathnames."""
    raw = _git(worktree, "status", "--porcelain=v1", "-z")
    fields = raw.split(b"\0")
    paths: list[bytes] = []
    index = 0
    while index < len(fields):
        record = fields[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise PlanError("malformed git status output")
        code, path = record[:2], record[3:]
        paths.append(path)
        # In -z porcelain rename/copy records carry the original pathname as the
        # following NUL-delimited field.  Both are mutations, so retain both.
        if code[0:1] in {b"R", b"C"} or code[1:2] in {b"R", b"C"}:
            if index >= len(fields) or not fields[index]:
                raise PlanError("malformed rename/copy git status output")
            paths.append(fields[index])
            index += 1
    return paths


def _tracked(worktree: Path) -> set[str]:
    return {
        item.decode("utf-8", "surrogateescape")
        for item in _git(worktree, "ls-files", "-z").split(b"\0")
        if item
    }


def _unexpected_untracked(worktree: Path, tracked: set[str]) -> list[str]:
    """Catch ignored files too; `git status` deliberately does not list them."""
    unexpected: list[str] = []
    for root, dirs, files in os.walk(worktree, topdown=True, followlinks=False):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if name != ".git"]
        for name in [*dirs, *files]:
            path = root_path / name
            relative = path.relative_to(worktree).as_posix()
            if relative in tracked or any(item.startswith(relative + "/") for item in tracked):
                continue
            if relative not in {PLAN_FILE, QUESTION_FILE}:
                unexpected.append(relative)
    return unexpected


def _has_non_sample_hook(worktree: Path) -> bool:
    git_dir = _git(worktree, "rev-parse", "--git-dir").decode("utf-8", "strict").strip()
    hooks = (worktree / git_dir / "hooks").resolve()
    return hooks.is_dir() and any(path.is_file() and not path.name.endswith(".sample") for path in hooks.iterdir())


def _sanitize_pii(value: str) -> str:
    value = redact(value)
    value = _EMAIL.sub("[email redatta]", value)
    value = _PHONE.sub("[telefono redatto]", value)
    return _ITALIAN_FISCAL_CODE.sub("[codice fiscale redatto]", value)


def _high_entropy_credential_assignment(value: str) -> bool:
    """Reject token-shaped keyed assignments, not ordinary descriptive config values."""
    for match in _GENERIC_CREDENTIAL_ASSIGNMENT.finditer(value):
        token = match.group("value")
        # A minimum alphabet prevents ``FEATURE_TOKEN=aaaaaaaa...`` false positives while
        # still rejecting common opaque base64url/hex-style credentials before model ingress.
        if len(set(token)) >= 8:
            return True
    return False


def validate_plan(text: str) -> str:
    """Return a prompt/audit-safe plan or reject secret-bearing planner output."""
    if not text or "\r" in text:
        raise PlanError("plan is empty or contains CR")
    if len(text) > MAX_PLAN_CHARS or text.count("\n") + 1 > MAX_PLAN_LINES:
        raise PlanError("plan exceeds bounded size")
    if any(ord(char) < 32 and char not in "\n\t" for char in text):
        raise PlanError("plan contains control characters")
    if (SECRET.search(text) or _PRIVATE_KEY.search(text) or _KNOWN_CREDENTIAL.search(text)
            or _high_entropy_credential_assignment(text)):
        raise PlanError("plan contains secret-like content")
    cleaned = _sanitize_pii(text).strip()
    if not cleaned:
        raise PlanError("plan is empty after redaction")
    return cleaned + "\n"


def reviewer_for(implementer: str) -> str:
    """Choose an independent reviewer class from recorded execution truth."""
    if implementer in {"codex", "fable", "opus"}:
        return "sonnet"
    if implementer == "sonnet":
        return "fable"
    raise PlanError("missing or unknown implementer class")


def inspect(worktree: Path) -> Inspection:
    plan_path = worktree / PLAN_FILE
    question_path = worktree / QUESTION_FILE
    if plan_path.exists() and question_path.exists():
        return Inspection("mixed")
    if _has_non_sample_hook(worktree):
        return Inspection("mixed")
    try:
        tracked = _tracked(worktree)
        changed = _status_paths(worktree)
        extra = _unexpected_untracked(worktree, tracked)
    except (OSError, subprocess.CalledProcessError, PlanError):
        return Inspection("invalid")
    wanted = PLAN_FILE if plan_path.exists() else QUESTION_FILE if question_path.exists() else None
    if wanted is None:
        return Inspection("absent" if not changed and not extra else "mixed")
    output_path = worktree / wanted
    if output_path.is_symlink() or not output_path.is_file():
        return Inspection("invalid")
    # A committed/checked-out planner output would be implicit plan reuse.
    if wanted in tracked or changed != [wanted.encode("utf-8")] or extra:
        return Inspection("mixed")
    try:
        if wanted == QUESTION_FILE:
            validate_question(output_path.read_text(encoding="utf-8"))
            return Inspection("question")
        return Inspection("valid", validate_plan(output_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, QuestionError, PlanError):
        return Inspection("invalid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--copy-plan", action="store_true")
    parser.add_argument("--reviewer-for")
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    modes = int(args.inspect) + int(args.copy_plan) + int(args.reviewer_for is not None)
    if modes != 1:
        parser.error("choose exactly one operation")
    if args.reviewer_for is not None:
        try:
            print(reviewer_for(args.reviewer_for))
        except PlanError as exc:
            raise SystemExit(f"reviewer invalid: {exc}") from exc
        return
    if args.worktree is None:
        parser.error("--worktree is required")
    result = inspect(args.worktree)
    if args.inspect:
        payload = f"plan={result.outcome}\nplan_valid={'true' if result.outcome == 'valid' else 'false'}\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return
    if result.outcome != "valid" or result.plan is None:
        raise SystemExit(f"plan invalid: {result.outcome}")
    if args.output is None:
        parser.error("--output is required with --copy-plan")
    args.output.write_text(result.plan, encoding="utf-8")
    args.output.chmod(0o600)


if __name__ == "__main__":
    main()
