#!/usr/bin/env python3
"""Trusted terminal label/comment transitions for agent workflows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from question import QuestionError, parse_envelope, render_comment, validate

TERMINAL_LABELS = {"ai:run", "ai:running", "ai:done", "ai:failed", "ai:capacity-exhausted",
                   "ai:question", "needs-human"}
ANSWER_LABELS = {"ai:answered-1", "ai:answered-2"}
QUESTION_PUBLISH_MARKER = re.compile(
    r"<!-- therness-question-publish:v1 question-comment-id=([1-9][0-9]*) "
    r"question-sha256=([0-9a-f]{64}) -->"
)


class TransitionError(RuntimeError):
    pass


def _json(request: urllib.request.Request) -> object:
    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


def _request(root: str, path: str, headers: dict[str, str], method: str = "GET", payload: object | None = None) -> urllib.request.Request:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return urllib.request.Request(root + path, data=data, headers=headers, method=method)


def _label_names(value: object) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(row, dict) and isinstance(row.get("name"), str) for row in value):
        raise TransitionError("label API returned unexpected body")
    return {row["name"] for row in value}


def _verify_identity(root: str, headers: dict[str, str], login: str, numeric_id: str) -> None:
    if not login or not numeric_id:
        raise TransitionError("exact bot login and numeric ID are required")
    identity = _json(_request(root, "/user", headers))
    if not isinstance(identity, dict) or identity.get("login") != login or str(identity.get("id", "")) != numeric_id:
        raise TransitionError("issue token identity does not match configured bot")


def _put_labels(root: str, repo: str, issue: str, headers: dict[str, str], desired: set[str]) -> None:
    body = _json(_request(root, f"/repos/{repo}/issues/{issue}/labels", headers, "PUT", {"labels": sorted(desired)}))
    actual = _label_names(body)
    if actual != desired:
        raise TransitionError(f"label PUT mismatch: expected={sorted(desired)!r} actual={sorted(actual)!r}")


def _post_comment(root: str, repo: str, issue: str, headers: dict[str, str], body: str) -> int:
    result = _json(_request(root, f"/repos/{repo}/issues/{issue}/comments", headers, "POST", {"body": body}))
    if (not isinstance(result, dict) or not isinstance(result.get("id"), int)
            or isinstance(result["id"], bool) or result["id"] < 1):
        raise TransitionError("comment POST returned unexpected body")
    return result["id"]


def _comments(root: str, repo: str, issue: str, headers: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    for page in range(1, 11):
        result = _json(_request(root, f"/repos/{repo}/issues/{issue}/comments?limit=100&page={page}", headers))
        if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
            raise TransitionError("comments API returned unexpected body")
        rows.extend(result)
        if len(result) < 100:
            return rows
    raise TransitionError("comments truncated at bounded page limit")


def _marker_posted_by_exact_bot(rows: list[dict], marker: str, login: str, numeric_id: str) -> bool:
    return any(
        isinstance(row.get("body"), str) and QUESTION_PUBLISH_MARKER.fullmatch(row["body"])
        and row["body"] == marker
        and isinstance(row.get("user"), dict) and row["user"].get("login") == login
        and str(row["user"].get("id", "")) == numeric_id
        for row in rows
    )


def _publish_generation(names: set[str]) -> int:
    """The answered counter is the durable boundary between question publishes."""
    answered = {name for name in names if name.startswith("ai:answered-")}
    if not answered:
        return 1
    if answered == {"ai:answered-1"}:
        return 2
    raise TransitionError("question publish generation is invalid")


def _published_question_id(rows: list[dict], envelope: str, digest: str, login: str,
                           numeric_id: str, generation: int) -> int | None:
    """Return only this durable publish generation's exact-bot envelope ID.

    The immutable marker history, not duplicate envelope text, binds retries.  Thus
    the same Q text in round two cannot reuse the round-one comment ID.
    """
    matches = {row["id"] for row in rows
               if isinstance(row.get("id"), int) and not isinstance(row["id"], bool) and row["id"] > 0
               and row.get("body") == envelope
               and isinstance(row.get("user"), dict) and row["user"].get("login") == login
               and str(row["user"].get("id", "")) == numeric_id}
    # Count every prior generation from its own immutable exact-bot question
    # comment.  Filtering history by the current envelope digest breaks a
    # legitimate changed Q2: Q1 is still a prior publish even when Q2 differs.
    question_digests: dict[int, str] = {}
    for row in rows:
        comment_id, user, body = row.get("id"), row.get("user"), row.get("body")
        if not (isinstance(comment_id, int) and not isinstance(comment_id, bool) and comment_id > 0
                and isinstance(user, dict) and user.get("login") == login
                and str(user.get("id", "")) == numeric_id and isinstance(body, str)):
            continue
        try:
            question_digests[comment_id] = hashlib.sha256(parse_envelope(body).body.encode("utf-8")).hexdigest()
        except QuestionError:
            continue
    published: set[int] = set()
    for row in rows:
        user, body = row.get("user"), row.get("body")
        if not (isinstance(user, dict) and user.get("login") == login
                and str(user.get("id", "")) == numeric_id and isinstance(body, str)):
            continue
        marker = QUESTION_PUBLISH_MARKER.fullmatch(body)
        if marker is not None and question_digests.get(int(marker.group(1))) == marker.group(2):
            published.add(int(marker.group(1)))
    if len(published) == generation:
        current = matches & published
        if current != {max(published)}:
            raise TransitionError("question publish generation does not match current envelope")
        return max(published)
    if len(published) != generation - 1:
        raise TransitionError("question publish generation history is ambiguous")
    previous = max(published, default=0)
    pending = matches - published
    pending = {comment_id for comment_id in pending if comment_id > previous}
    if len(pending) > 1:
        raise TransitionError("question publish generation has multiple unmarked envelopes")
    return next(iter(pending), None)


def _counter_requires_human(names: set[str]) -> bool:
    answered = {name for name in names if name.startswith("ai:answered-")}
    return "ai:answered-2" in answered or bool(answered - ANSWER_LABELS) or len(answered) > 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", choices=("done", "failed", "capacity-exhausted", "needs-human"))
    parser.add_argument("--question-file", type=Path)
    parser.add_argument("--verify-exact-bot", action="store_true")
    args = parser.parse_args()
    if (args.status is None) == (args.question_file is None):
        parser.error("use exactly one of --status or --question-file")

    root = os.environ["API_URL"]
    repo = os.environ["REPOSITORY"]
    issue = os.environ["ISSUE"]
    token = os.environ.get("ISSUE_TOKEN") or os.environ.get("GIT_TOKEN")
    if not token:
        raise TransitionError("issue token is required")
    headers = {"Authorization": "token " + token, "Content-Type": "application/json"}
    if args.verify_exact_bot:
        _verify_identity(root, headers, os.environ.get("AGENT_BOT_LOGIN", ""), os.environ.get("AGENT_BOT_ID", ""))

    current = _json(_request(root, f"/repos/{repo}/issues/{issue}", headers))
    if not isinstance(current, dict):
        raise TransitionError("issue GET returned unexpected body")
    labels = current.get("labels")
    if not isinstance(labels, list):
        raise TransitionError("issue labels missing")
    current_names = _label_names(labels)
    keep = current_names - TERMINAL_LABELS

    if args.question_file is not None:
        try:
            question = validate(args.question_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, QuestionError) as exc:
            raise TransitionError(f"question unexpectedly invalid: {exc}") from exc
        if _counter_requires_human(current_names):
            _put_labels(root, repo, issue, headers, keep | {"needs-human"})
            _post_comment(root, repo, issue, headers, "Agent stopped safely: question round counter requires human review.")
            return
        # First make the label transition durable.  The answer-agent sweep owns
        # recovery if a crash leaves this label without the exact marker below.
        _put_labels(root, repo, issue, headers, keep | {"ai:question"})
        envelope = render_comment(question)
        login, numeric_id = os.environ.get("AGENT_BOT_LOGIN", ""), os.environ.get("AGENT_BOT_ID", "")
        digest = hashlib.sha256(question.body.encode("utf-8")).hexdigest()
        generation = _publish_generation(current_names)
        rows = _comments(root, repo, issue, headers)
        question_id = _published_question_id(rows, envelope, digest, login, numeric_id, generation)
        if question_id is None:
            question_id = _post_comment(root, repo, issue, headers, envelope)
        marker = (f"<!-- therness-question-publish:v1 question-comment-id={question_id} "
                  f"question-sha256={digest} -->")
        if not _marker_posted_by_exact_bot(_comments(root, repo, issue, headers), marker, login, numeric_id):
            _post_comment(root, repo, issue, headers, marker)
        return

    status = args.status
    assert status is not None
    desired = keep | ({"needs-human"} if status == "needs-human" else
                      {"ai:failed", "ai:capacity-exhausted"} if status == "capacity-exhausted" else
                      {"ai:" + status})
    _put_labels(root, repo, issue, headers, desired)
    comment = {
        "done": "Agent completed.",
        "failed": "Agent failed safely; no branch was published.",
        "capacity-exhausted": "Agent paused safely: Claude OAuth capacity is exhausted; supervisor retry may resume it.",
        "needs-human": "Agent stopped safely: question protocol violation needs human review.",
    }[status]
    _post_comment(root, repo, issue, headers, comment)


if __name__ == "__main__":
    try:
        main()
    except (TransitionError, urllib.error.URLError, OSError, ValueError) as exc:
        print(f"terminal transition failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
