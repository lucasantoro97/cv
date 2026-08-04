#!/usr/bin/env python3
"""Trusted shared wall-clock budget for all model invocations in one runner job."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path


VERSION = 1
MAX_BUDGET_SEC = 110 * 60
STATE_NAME = "model-budget.json"
KEY_NAME = ".model-budget.key"
OUTCOME_NAME = "model-outcome.json"
CAPACITY = "capacity-exhausted"


class BudgetError(ValueError):
    pass


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _signed(payload: dict[str, object], key: bytes) -> dict[str, object]:
    return {"payload": payload, "signature": hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()}


def _regular_private(path: Path) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise BudgetError(f"unsafe trusted budget file: {path.name}")
    return path.read_bytes()


def _key(root: Path) -> bytes:
    raw = _regular_private(root / KEY_NAME)
    if len(raw) != 32:
        raise BudgetError("trusted budget key is invalid")
    return raw


def _atomic_private(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise BudgetError(f"trusted budget file already exists: {path.name}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def initialize(root: Path, budget_sec: int, *, now: int | None = None) -> None:
    if not 1 <= budget_sec <= MAX_BUDGET_SEC:
        raise BudgetError("model budget is outside the safe bound")
    root = root.resolve(strict=True)
    created = int(time.time() if now is None else now)
    key = os.urandom(32)
    payload = {
        "budget_sec": budget_sec,
        "created_at": created,
        "deadline": created + budget_sec,
        "version": VERSION,
    }
    _atomic_private(root / KEY_NAME, key)
    try:
        _atomic_private(root / STATE_NAME, _canonical(_signed(payload, key)) + b"\n")
    except BaseException:
        (root / KEY_NAME).unlink(missing_ok=True)
        raise


def _verified_payload(path: Path, key: bytes, expected_keys: set[str]) -> dict[str, object]:
    raw = _regular_private(path)
    if len(raw) > 4096:
        raise BudgetError("trusted budget file is oversized")
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise BudgetError("trusted budget file is invalid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"payload", "signature"}:
        raise BudgetError("trusted budget envelope is invalid")
    payload = value["payload"]
    signature = value["signature"]
    if not isinstance(payload, dict) or set(payload) != expected_keys or not isinstance(signature, str):
        raise BudgetError("trusted budget payload is invalid")
    wanted = hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, wanted):
        raise BudgetError("trusted budget signature mismatch")
    return payload


def load(root: Path) -> dict[str, int]:
    payload = _verified_payload(
        root / STATE_NAME,
        _key(root),
        {"budget_sec", "created_at", "deadline", "version"},
    )
    if any(isinstance(payload[name], bool) or not isinstance(payload[name], int) for name in payload):
        raise BudgetError("trusted budget values are invalid")
    budget = int(payload["budget_sec"])
    created = int(payload["created_at"])
    deadline = int(payload["deadline"])
    if payload["version"] != VERSION or not 1 <= budget <= MAX_BUDGET_SEC or deadline - created != budget:
        raise BudgetError("trusted budget arithmetic is invalid")
    return {"budget_sec": budget, "created_at": created, "deadline": deadline}


def allowance(root: Path, cap_sec: int, *, now: float | None = None) -> int:
    if not 1 <= cap_sec <= 30 * 60:
        raise BudgetError("per-call cap is outside the safe bound")
    deadline = load(root)["deadline"]
    remaining = deadline - (time.time() if now is None else now)
    safe_whole_seconds = int(remaining)
    if safe_whole_seconds < 1:
        record(root, CAPACITY)
        raise BudgetError("shared model deadline exhausted")
    return min(cap_sec, safe_whole_seconds)


def record(root: Path, status: str) -> None:
    if status != CAPACITY:
        raise BudgetError("unsupported trusted model outcome")
    key = _key(root)
    path = root / OUTCOME_NAME
    if path.exists() or path.is_symlink():
        existing = outcome(root)
        if existing != status:
            raise BudgetError("trusted model outcome conflicts")
        return
    payload = {"status": status, "version": VERSION}
    _atomic_private(path, _canonical(_signed(payload, key)) + b"\n")


def outcome(root: Path) -> str:
    path = root / OUTCOME_NAME
    if not path.exists() and not path.is_symlink():
        return "none"
    payload = _verified_payload(path, _key(root), {"status", "version"})
    if payload != {"status": CAPACITY, "version": VERSION}:
        raise BudgetError("trusted model outcome is invalid")
    return CAPACITY


def _root(value: str | None) -> Path:
    raw = value or os.environ.get("RUNNER_TEMP")
    if not raw:
        raise BudgetError("RUNNER_TEMP is required")
    root = Path(raw)
    if not root.is_dir() or root.is_symlink():
        raise BudgetError("RUNNER_TEMP is unsafe")
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--budget-sec", type=int, default=MAX_BUDGET_SEC)
    allowed = sub.add_parser("allowance")
    allowed.add_argument("--cap-sec", type=int, required=True)
    recorded = sub.add_parser("record")
    recorded.add_argument("--status", required=True)
    sub.add_parser("outcome")
    args = parser.parse_args(argv)
    try:
        root = _root(args.root)
        if args.command == "init":
            initialize(root, args.budget_sec)
        elif args.command == "allowance":
            print(allowance(root, args.cap_sec))
        elif args.command == "record":
            record(root, args.status)
        else:
            print(outcome(root))
        return 0
    except (BudgetError, OSError) as exc:
        print(f"model_budget: {exc}", file=sys.stderr)
        return 75 if "exhausted" in str(exc) else 2


if __name__ == "__main__":
    raise SystemExit(main())
