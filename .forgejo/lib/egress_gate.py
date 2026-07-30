#!/usr/bin/env python3
"""Fail closed before pushing an agent branch or opening a PR."""
from __future__ import annotations
import argparse, os, re, subprocess, sys
from pathlib import Path

# redact.py is shipped beside this file inside the repo's .forgejo/lib bundle;
# a ticket-bridge checkout is never present in a runner job.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact import contains_nas_path

SECRET = re.compile(r"(?i)(?:api[_-]?key|secret|password|token)\s*(?:=|:|\")\s*['\"]?[A-Za-z0-9_\-]{12,}")
MAX_FILE = 5 * 1024 * 1024

class GateError(ValueError): pass

def reject(value: str, where: str) -> None:
    if contains_nas_path(value): raise GateError(f"NAS path in {where}")
    if SECRET.search(value): raise GateError(f"secret-like content in {where}")

def check(repo: str | Path, base: str, head: str, title: str, body: str) -> None:
    repo = str(repo); reject(title, "PR title"); reject(body, "PR body")
    changed = subprocess.run(["git", "-C", repo, "diff", "--name-status", "-z", f"{base}..{head}"], capture_output=True, check=True).stdout.split(b"\0")
    for item in changed:
        if not item: continue
        name = item.split(b"\t")[-1].decode("utf-8", "surrogateescape"); reject(name, "filename")
        path = Path(repo, name)
        if path.exists() and path.is_file() and path.stat().st_size > MAX_FILE: raise GateError("oversized output")
        if path.exists() and path.is_file() and b"\0" in path.read_bytes()[:8192]: raise GateError("binary output")
    diff = subprocess.run(["git", "-C", repo, "diff", "--no-ext-diff", "--binary", f"{base}..{head}"], text=True, errors="replace", capture_output=True, check=True).stdout
    reject(diff, "commit range")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", default="."); parser.add_argument("--base", required=True); parser.add_argument("--head", required=True); parser.add_argument("--title", required=True); parser.add_argument("--body", required=True)
    args = parser.parse_args()
    try: check(args.repo, args.base, args.head, args.title, args.body)
    except (GateError, subprocess.CalledProcessError) as exc: raise SystemExit(f"EGRESS REFUSE: {exc}")
