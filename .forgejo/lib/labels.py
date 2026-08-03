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
LIST_PAGE_SIZE = 50
MAX_LIST_PAGES = 10
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


def _comment_page(request: urllib.request.Request) -> tuple[object, int | None]:
    """Read one comments page plus a bounded, canonical total when supplied."""
    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read()
        total_text = response.headers.get("X-Total-Count")
    if total_text is None:
        total = None
    else:
        if not re.fullmatch(r"0|[1-9][0-9]*", total_text):
            raise TransitionError("comments total count invalid")
        total = int(total_text)
        if total > LIST_PAGE_SIZE * MAX_LIST_PAGES:
            raise TransitionError("comments total count exceeds bound")
    return (json.loads(raw) if raw else None), total


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


def _issue(root: str, repo: str, issue: str, headers: dict[str, str]) -> dict:
    current = _json(_request(root, f"/repos/{repo}/issues/{issue}", headers))
    if not isinstance(current, dict):
        raise TransitionError("issue GET returned unexpected body")
    return current


def _issue_state(root: str, repo: str, issue: str, headers: dict[str, str]) -> tuple[set[str], str | None]:
    """Read labels plus the observed server revision for a transition boundary."""
    current = _issue(root, repo, issue, headers)
    labels = current.get("labels")
    if not isinstance(labels, list):
        raise TransitionError("issue labels missing")
    revision = current.get("updated_at")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise TransitionError("issue revision invalid")
    return _label_names(labels), revision


def _label_records(root: str, repo: str, issue: str, headers: dict[str, str]) -> dict[str, int]:
    rows = _json(_request(root, f"/repos/{repo}/issues/{issue}/labels", headers))
    if not isinstance(rows, list):
        raise TransitionError("issue labels API returned unexpected body")
    records: dict[str, int] = {}
    for row in rows:
        name = row.get("name") if isinstance(row, dict) else None
        numeric_id = row.get("id") if isinstance(row, dict) else None
        if (not isinstance(name, str) or not name or isinstance(numeric_id, bool)
                or not isinstance(numeric_id, int) or numeric_id <= 0 or name in records):
            raise TransitionError("issue labels API returned invalid label record")
        records[name] = numeric_id
    return records


def _repository_label_id(root: str, repo: str, headers: dict[str, str], name: str) -> int:
    """Resolve a label name to Forgejo's immutable numeric ID, fail closed."""
    matches: list[int] = []
    page_fingerprints: set[str] = set()
    row_fingerprints: set[str] = set()
    for page in range(1, MAX_LIST_PAGES + 1):
        rows = _json(_request(root, f"/repos/{repo}/labels?limit={LIST_PAGE_SIZE}&page={page}", headers))
        if not isinstance(rows, list):
            raise TransitionError("repository labels API returned unexpected body")
        if len(rows) > LIST_PAGE_SIZE:
            raise TransitionError("repository labels page exceeds bound")
        if not rows:
            break
        try:
            page_fingerprint = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            raise TransitionError("repository labels row is not canonical JSON") from exc
        if page_fingerprint in page_fingerprints:
            raise TransitionError("repository labels repeated page")
        page_fingerprints.add(page_fingerprint)
        for row in rows:
            if not isinstance(row, dict):
                raise TransitionError("repository labels API returned invalid label record")
            row_fingerprint = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            if row_fingerprint in row_fingerprints:
                raise TransitionError("repository labels duplicate row")
            row_fingerprints.add(row_fingerprint)
            if row.get("name") == name:
                numeric_id = row.get("id")
                if isinstance(numeric_id, bool) or not isinstance(numeric_id, int) or numeric_id <= 0:
                    raise TransitionError("repository label ID invalid")
                matches.append(numeric_id)
    else:
        raise TransitionError("repository labels truncated at bounded page limit")
    if len(matches) != 1:
        raise TransitionError(f"required label is not defined exactly once: {name}")
    return matches[0]


def _is_control_label(name: str) -> bool:
    return name in TERMINAL_LABELS or name.startswith("ai:answered-")


def _add_label(root: str, repo: str, issue: str, headers: dict[str, str], name: str) -> None:
    """Attach one known repository label and confirm it after response loss."""
    if name in _label_records(root, repo, issue, headers):
        return
    numeric_id = _repository_label_id(root, repo, headers, name)
    try:
        _json(_request(root, f"/repos/{repo}/issues/{issue}/labels", headers, "POST", {"labels": [numeric_id]}))
    except (urllib.error.URLError, OSError, ValueError):
        # The server may have accepted the request before its response vanished.
        pass
    if name not in _label_records(root, repo, issue, headers):
        raise TransitionError(f"label add was not durably applied: {name}")


def _remove_exact_label(root: str, repo: str, issue: str, headers: dict[str, str],
                        name: str, numeric_id: int) -> None:
    """Remove only the label ID observed in our predecessor, never replace all labels."""
    records = _label_records(root, repo, issue, headers)
    if name not in records:
        return
    if records[name] != numeric_id:
        raise TransitionError(f"label ID changed during transition: {name}")
    try:
        _json(_request(root, f"/repos/{repo}/issues/{issue}/labels/{numeric_id}", headers, "DELETE"))
    except (urllib.error.URLError, OSError, ValueError):
        # A DELETE with a lost response is safe only when the fresh read proves it.
        pass
    if name in _label_records(root, repo, issue, headers):
        raise TransitionError(f"label remove was not durably applied: {name}")


def _mutation_transition(root: str, repo: str, issue: str, headers: dict[str, str],
                         expected: tuple[set[str], str | None], desired: set[str]) -> set[str]:
    """Transition exact control labels with add/remove endpoints, preserving all others.

    Forgejo exposes no compare-and-swap for issue labels.  We therefore bind to a
    fresh predecessor, add the successor, then remove only predecessor label IDs.
    A concurrent non-control label survives; a concurrent terminal/control label
    causes refusal instead of being silently erased.
    """
    current = _issue_state(root, repo, issue, headers)
    current_names, _revision = current
    if current_names != expected[0]:
        # Exact durable successor makes an interrupted retry idempotent.
        stale_controls = {name for name in current_names - desired if _is_control_label(name)}
        if desired <= current_names and not stale_controls:
            return current_names
        raise TransitionError("label/timeline generation changed before terminal transition")
    records = _label_records(root, repo, issue, headers)
    if set(records) != current_names:
        raise TransitionError("issue label reads disagree before transition")
    for name in sorted(desired - current_names):
        _add_label(root, repo, issue, headers, name)
    # Delete only IDs present in the committed predecessor.  This cannot erase a
    # concurrently attached unrelated label, even in the unavoidable API race.
    for name in sorted(current_names - desired):
        _remove_exact_label(root, repo, issue, headers, name, records[name])
    actual = set(_label_records(root, repo, issue, headers))
    if not desired <= actual or (current_names - desired) & actual:
        raise TransitionError(f"label transition mismatch: desired={sorted(desired)!r} actual={sorted(actual)!r}")
    unexpected_controls = {name for name in actual - desired if _is_control_label(name)}
    if unexpected_controls:
        raise TransitionError(f"conflicting terminal/control labels appeared: {sorted(unexpected_controls)!r}")
    return actual


def _close_done(root: str, repo: str, issue: str, headers: dict[str, str]) -> None:
    """Close a verified done issue; a lost PATCH response is reconciled by GET."""
    state = _issue(root, repo, issue, headers).get("state")
    if state == "closed":
        return
    if state != "open":
        raise TransitionError("issue state is neither open nor closed")
    try:
        _json(_request(root, f"/repos/{repo}/issues/{issue}", headers, "PATCH", {"state": "closed"}))
    except (urllib.error.URLError, OSError, ValueError):
        pass
    if _issue(root, repo, issue, headers).get("state") != "closed":
        raise TransitionError("done issue close was not durably applied")


def _post_comment(root: str, repo: str, issue: str, headers: dict[str, str], body: str) -> int:
    result = _json(_request(root, f"/repos/{repo}/issues/{issue}/comments", headers, "POST", {"body": body}))
    if (not isinstance(result, dict) or not isinstance(result.get("id"), int)
            or isinstance(result["id"], bool) or result["id"] < 1):
        raise TransitionError("comment POST returned unexpected body")
    return result["id"]


def _comments(root: str, repo: str, issue: str, headers: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    page_fingerprints: set[str] = set()
    row_fingerprints: set[str] = set()
    row_ids: set[int] = set()
    total_mode: bool | None = None
    declared_total: int | None = None
    for page in range(1, MAX_LIST_PAGES + 1):
        result, page_total = _comment_page(
            _request(root, f"/repos/{repo}/issues/{issue}/comments?limit={LIST_PAGE_SIZE}&page={page}", headers)
        )
        has_total = page_total is not None
        if total_mode is None:
            total_mode = has_total
            declared_total = page_total
        elif has_total != total_mode or (has_total and page_total != declared_total):
            raise TransitionError("comments total count changed")
        if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
            raise TransitionError("comments API returned unexpected body")
        if len(result) > LIST_PAGE_SIZE:
            raise TransitionError("comments page exceeds bound")
        if not result:
            if declared_total is not None and len(rows) != declared_total:
                raise TransitionError("comments ended before declared total")
            return rows
        page_fingerprint = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if page_fingerprint in page_fingerprints:
            raise TransitionError("comments repeated page")
        page_fingerprints.add(page_fingerprint)
        for row in result:
            row_id = row.get("id")
            if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id <= 0:
                raise TransitionError("comments row id invalid")
            if row_id in row_ids:
                raise TransitionError("comments duplicate row id")
            row_ids.add(row_id)
            row_fingerprint = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            if row_fingerprint in row_fingerprints:
                raise TransitionError("comments duplicate row")
            row_fingerprints.add(row_fingerprint)
        if declared_total is not None and len(rows) + len(result) > declared_total:
            raise TransitionError("comments exceed declared total")
        rows.extend(result)
        if declared_total is not None:
            if len(rows) == declared_total:
                return rows
            if len(result) < LIST_PAGE_SIZE:
                raise TransitionError("comments short page before declared total")
    raise TransitionError("comments truncated at bounded page limit")


def _marker_posted_by_exact_bot(rows: list[dict], marker: str, login: str, numeric_id: str) -> bool:
    return any(
        isinstance(row.get("body"), str) and QUESTION_PUBLISH_MARKER.fullmatch(row["body"])
        and row["body"] == marker
        and isinstance(row.get("user"), dict) and row["user"].get("login") == login
        and str(row["user"].get("id", "")) == numeric_id
        for row in rows
    )


def _terminal_comment(status: str) -> str:
    message = {
        "done": "Agent completed.",
        # The terminal publisher does not create or inspect branches/PRs.  Do not
        # turn a workflow failure into a false publication assertion.
        "failed": "Agent failed safely. Publication state was not established by this terminal transition.",
        "capacity-exhausted": "Agent paused safely: Claude OAuth capacity is exhausted; supervisor retry may resume it.",
        "needs-human": "Agent stopped safely: question protocol violation needs human review.",
    }[status]
    return f"{message}\n\n<!-- therness-terminal:v1 status={status} -->"


def _terminal_comment_posted_by_exact_bot(rows: list[dict], status: str, login: str, numeric_id: str) -> bool:
    expected = _terminal_comment(status)
    return any(
        row.get("body") == expected
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

    initial = _issue_state(root, repo, issue, headers)
    current_names = initial[0]
    keep = current_names - TERMINAL_LABELS

    if args.question_file is not None:
        try:
            question = validate(args.question_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, QuestionError) as exc:
            raise TransitionError(f"question unexpectedly invalid: {exc}") from exc
        if _counter_requires_human(current_names):
            _mutation_transition(root, repo, issue, headers, initial, keep | {"needs-human"})
            login, numeric_id = os.environ.get("AGENT_BOT_LOGIN", ""), os.environ.get("AGENT_BOT_ID", "")
            if not _terminal_comment_posted_by_exact_bot(
                    _comments(root, repo, issue, headers), "needs-human", login, numeric_id):
                _post_comment(root, repo, issue, headers, _terminal_comment("needs-human"))
            return
        # First make the label transition durable.  The answer-agent sweep owns
        # recovery if a crash leaves this label without the exact marker below.
        _mutation_transition(root, repo, issue, headers, initial, keep | {"ai:question"})
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
    _mutation_transition(root, repo, issue, headers, initial, desired)
    # Completion is a durable issue-state transition, not merely a label.  Failed
    # tickets deliberately remain open for the autonomous supervisor.
    if status == "done":
        _close_done(root, repo, issue, headers)
    login, numeric_id = os.environ.get("AGENT_BOT_LOGIN", ""), os.environ.get("AGENT_BOT_ID", "")
    if not _terminal_comment_posted_by_exact_bot(_comments(root, repo, issue, headers), status, login, numeric_id):
        _post_comment(root, repo, issue, headers, _terminal_comment(status))


if __name__ == "__main__":
    try:
        main()
    except (TransitionError, urllib.error.URLError, OSError, ValueError) as exc:
        print(f"terminal transition failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
