#!/usr/bin/env python3
"""Question-channel grammar, canonical envelope, and unprivileged inspection.

Model output is data.  Only this module decides whether it is a question and
only the first line of a rendered comment is an envelope anchor.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

QUESTION_FILE = ".agent-question.md"
MAX_CHARS = 500
MAX_CONTEXT_LINES = 3
ENVELOPE_PREFIX = "<!-- therness-question:v1"
# Paths created by the trusted workflow before its model step.  They are not
# model changes and therefore cannot turn a valid question into a mixed diff.
# Each template passes its own exact subset via ``--scaffold-path``.
KNOWN_SCAFFOLD_PATHS = frozenset({".issue-assets", ".issue-raw.txt", ".issue-prompt.txt", ".context"})


class QuestionError(ValueError):
    """The model output is not a question-channel document."""


@dataclass(frozen=True)
class Question:
    body: str
    mode: str


def _reject_controls(value: str) -> None:
    if any(ord(char) < 32 and char not in "\n\t" for char in value) or "\t" in value:
        raise QuestionError("control characters are not allowed")


def validate(text: str) -> Question:
    """Accept exactly one DOMANDA, or one to three CONTESTO entries."""
    if not text:
        raise QuestionError("question is empty")
    if "\r" in text:
        raise QuestionError("CR is not allowed")
    _reject_controls(text)
    if len(text) > MAX_CHARS:
        raise QuestionError(f"question exceeds {MAX_CHARS} characters")
    # A single final LF is transport, not an extra grammar line.
    lines = text[:-1].split("\n") if text.endswith("\n") else text.split("\n")
    if len(lines) == 1 and lines[0].startswith("DOMANDA: "):
        if not lines[0][len("DOMANDA: "):].strip():
            raise QuestionError("DOMANDA must not be empty")
        mode = "domanda"
    elif 1 <= len(lines) <= MAX_CONTEXT_LINES and all(
            line.startswith("CONTESTO: ") and line[len("CONTESTO: "):].strip()
            for line in lines):
        mode = "contesto"
    else:
        raise QuestionError("use exactly one DOMANDA or one to three CONTESTO lines")
    canonical = "\n".join(lines)
    return Question(canonical, mode)


def canonical_envelope(question: Question | str) -> str:
    body = question.body if isinstance(question, Question) else validate(question).body
    payload = base64.urlsafe_b64encode(body.encode("utf-8")).decode("ascii").rstrip("=")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"{ENVELOPE_PREFIX} sha256={digest} payload={payload} -->"


def parse_envelope(comment: str) -> Question:
    """Parse only first line.  Fences/quotes/markers below it are display data."""
    first = comment.split("\n", 1)[0]
    if not first.startswith(ENVELOPE_PREFIX) or not first.endswith(" -->"):
        raise QuestionError("canonical envelope must be first line")
    fields = first[len(ENVELOPE_PREFIX): -len(" -->")].strip().split(" ")
    if len(fields) != 2 or not fields[0].startswith("sha256=") or not fields[1].startswith("payload="):
        raise QuestionError("malformed envelope")
    digest = fields[0].removeprefix("sha256=")
    encoded = fields[1].removeprefix("payload=")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise QuestionError("malformed digest")
    if not encoded or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for char in encoded):
        raise QuestionError("malformed base64url payload")
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        body = raw.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise QuestionError("invalid payload encoding") from exc
    if hashlib.sha256(raw).hexdigest() != digest:
        raise QuestionError("payload digest mismatch")
    return validate(body)


def _display_fence(body: str) -> str:
    # Tilde avoids making user-supplied backticks structural Markdown.
    longest_tilde = max((len(run.group(0)) for run in re.finditer(r"~+", body)), default=0)
    return "~" * max(3, longest_tilde + 1)


def render_comment(question: Question | str) -> str:
    body = question.body if isinstance(question, Question) else validate(question).body
    fence = _display_fence(body)
    return f"{canonical_envelope(body)}\n\n{fence}text\n{body}\n{fence}\n"


def _normalise_scaffolds(scaffold_paths: tuple[str, ...]) -> tuple[bytes, ...]:
    normalized: list[bytes] = []
    for path in scaffold_paths:
        if path not in KNOWN_SCAFFOLD_PATHS:
            raise QuestionError("unknown workflow scaffold path")
        normalized.append(path.encode("utf-8"))
    return tuple(normalized)


def _changed_paths(worktree: Path, question_file: str, scaffold_paths: tuple[str, ...] = ()) -> bool:
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "status", "--porcelain=v1", "-z"],
        cwd=worktree, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    fields = result.stdout.split(b"\0")
    paths: list[bytes] = []
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4:
            raise QuestionError("malformed git status output")
        path = entry[3:]
        # Rename records carry a second pathname under -z.  Any rename is a diff.
        if entry[:2] in {b"R ", b"C "} and index < len(fields):
            paths.append(fields[index])
            index += 1
        paths.append(path)
    question_bytes = question_file.encode("utf-8")
    scaffold_bytes = _normalise_scaffolds(scaffold_paths)
    def allowed(path: bytes) -> bool:
        return path == question_bytes or any(path == root or path.startswith(root + b"/") for root in scaffold_bytes)
    return any(not allowed(path) for path in paths)


def _is_untracked_new_question(worktree: Path, question_file: str) -> bool:
    """Accept a question only when this run created an untracked path.

    Repository content is untrusted model input.  A committed (or otherwise
    tracked) ``.agent-question.md`` must never become a publishable question.
    """
    tracked = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "ls-files", "--error-unmatch", "--", question_file],
        cwd=worktree, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if tracked.returncode == 0:
        return False
    if tracked.returncode != 1:
        raise QuestionError("git ls-files failed")
    status = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "status", "--porcelain=v1", "-z", "--", question_file],
        cwd=worktree, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return status.stdout == b"?? " + question_file.encode("utf-8") + b"\0"


def inspect(worktree: Path, question_file: str = QUESTION_FILE, scaffold_paths: tuple[str, ...] = ()) -> tuple[str, bool]:
    path = worktree / question_file
    if not path.is_file():
        return "absent", False
    try:
        text = path.read_text(encoding="utf-8")
        validate(text)
        valid = True
    except (OSError, UnicodeError, QuestionError):
        valid = False
    if not _is_untracked_new_question(worktree, question_file):
        return "invalid", False
    changed = _changed_paths(worktree, question_file, scaffold_paths)
    if changed:
        return "mixed", valid
    return ("true" if valid else "invalid"), valid


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--render", action="store_true")
    group.add_argument("--inspect", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--worktree", type=Path, default=Path("."))
    parser.add_argument("--question-file", default=QUESTION_FILE)
    parser.add_argument("--scaffold-path", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.validate or args.render:
        if args.input is None:
            parser.error("--input is required with --validate/--render")
        try:
            question = validate(args.input.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, QuestionError) as exc:
            print(f"question invalid: {exc}", file=sys.stderr)
            raise SystemExit(2)
        if args.render:
            print(render_comment(question), end="")
        return
    status, valid = inspect(args.worktree, args.question_file, tuple(args.scaffold_path))
    if args.output:
        args.output.write_text(f"question={status}\nquestion_valid={'true' if valid else 'false'}\n", encoding="utf-8")
    else:
        print(json.dumps({"question": status, "question_valid": valid}, sort_keys=True))


if __name__ == "__main__":
    main()
