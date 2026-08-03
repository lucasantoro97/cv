#!/usr/bin/env python3
"""Download issue assets under stable, anti-flood selection rules.

Trusted answer manifests are selected before original issue attachments.  Asset
names/content are untrusted; stable Forgejo IDs are used for order and paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import urllib.parse
import urllib.request
from typing import Any

MAX_ORIGINAL_FILES = 10
MAX_ORIGINAL_BYTES = 25 * 1024 * 1024
MAX_MANIFEST_FILES = 1
MAX_MANIFEST_BYTES = 1 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
ASSET_PAGE_SIZE = 100
MAX_ASSET_PAGES = 10
MAX_LISTING_PAGE_BYTES = 1024 * 1024
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
PLAN_ARTIFACTS = {"piano_esecuzione.md", ".agent-plan.md", "agent-plan.md", "piano_v1", "piano_v1.md"}
ANSWER_MANIFEST = "answer-manifest.json"
MANIFEST_DIGEST_FIELD = "manifest_sha256"


def validate_answer_manifest(payload: bytes) -> dict[str, Any]:
    """Verify canonical manifest self-digest before exposing it to a model job."""
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("answer manifest JSON invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("answer manifest shape invalid")
    digest = value.get(MANIFEST_DIGEST_FIELD)
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("answer manifest digest invalid")
    unsigned = dict(value)
    del unsigned[MANIFEST_DIGEST_FIELD]
    canonical = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != digest:
        raise ValueError("answer manifest digest mismatch")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Reject every redirect before urllib can replay Authorization elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _api(url: str, token: str, limit: int) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("Authorization", "token " + token)
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(req, timeout=30) as resp:
        return resp.read(limit + 1)


def _trusted_attachment_url(api_root: str, candidate: object) -> bool:
    """Accept only same-origin Forgejo attachment URLs.

    ``download_url`` comes from an untrusted issue-asset listing.  Do not let a
    lookalike hostname, alternate port, or API path receive the issue token.
    """
    if not isinstance(candidate, str):
        return False
    try:
        root = urllib.parse.urlsplit(api_root)
        asset = urllib.parse.urlsplit(candidate)
        root_port = root.port
        asset_port = asset.port
    except ValueError:
        return False
    if not root.scheme or not root.hostname:
        return False
    # API_URL may be installed below a reverse-proxy path.  Attachments live at
    # that installation root, never under an arbitrary API endpoint.
    api_marker = "/api/"
    if api_marker not in root.path:
        return False
    installation_path = root.path.split(api_marker, 1)[0].rstrip("/")
    attachment_prefix = installation_path + "/attachments/"
    return (asset.scheme == root.scheme and asset.hostname == root.hostname
            and asset_port == root_port and asset.path.startswith(attachment_prefix))


def _stable_id(asset: dict[str, Any]) -> int:
    value = asset.get("id")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("asset has no non-negative stable ID")
    return value


def _asset_name(asset: dict[str, Any]) -> str:
    value = asset.get("name")
    return value if isinstance(value, str) else ""


ANSWER_MARKER = re.compile(
    r"\A<!-- therness-answer:v1 question-comment-id=[1-9][0-9]* "
    r"manifest-asset-id=([1-9][0-9]*) manifest-sha256=([0-9a-f]{64}) -->\n"
)


def _manifest_ref(comments: list[dict[str, Any]], assets: list[dict[str, Any]],
                  login: str, numeric_id: str) -> tuple[int, str] | None:
    """Return one exact bot-authored answer comment's asset commitment.

    Forgejo Attachment responses do not include uploader provenance.  The comment
    endpoint does, so it is the only supported place that binds the asset ID/digest.
    """
    # A fixed manifest filename is replaced for every answer round.  Historical
    # exact-bot comments remain immutable, so only a marker whose committed asset
    # is still present can describe the next runner input.
    present = {
        _stable_id(asset) for asset in assets
        if isinstance(asset, dict) and _asset_name(asset) == ANSWER_MANIFEST
        and isinstance(asset.get("id"), int) and not isinstance(asset.get("id"), bool)
        and asset["id"] > 0
    }
    matches: list[tuple[int, str]] = []
    for comment in comments:
        user, body = comment.get("user"), comment.get("body")
        if not (isinstance(user, dict) and user.get("login") == login
                and str(user.get("id")) == numeric_id and isinstance(body, str)):
            continue
        marker = ANSWER_MARKER.match(body)
        if marker and int(marker.group(1)) in present:
            matches.append((int(marker.group(1)), marker.group(2)))
    if len(matches) > 1:
        raise SystemExit("multiple trusted answer manifest comments")
    return matches[0] if matches else None


def select_assets(listing: list[dict[str, Any]], *, manifest_id: int | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Return manifest first, then original assets by ascending stable ID.

    A malformed ID is ignored rather than becoming an order-dependent fallback.
    Plan artifacts are never execution input.
    """
    candidates: list[dict[str, Any]] = []
    for asset in listing:
        if not isinstance(asset, dict):
            continue
        try:
            _stable_id(asset)
        except ValueError:
            continue
        if _asset_name(asset).casefold() in PLAN_ARTIFACTS:
            continue
        candidates.append(asset)
    candidates.sort(key=_stable_id)
    manifest = [asset for asset in candidates if manifest_id is not None and _stable_id(asset) == manifest_id]
    if manifest and _asset_name(manifest[0]) != ANSWER_MANIFEST:
        raise SystemExit("trusted answer manifest asset name invalid")
    if manifest_id is not None and not manifest:
        raise SystemExit("trusted answer manifest asset missing")
    manifest_ids = {_stable_id(asset) for asset in manifest}
    originals = [asset for asset in candidates if _stable_id(asset) not in manifest_ids and _asset_name(asset) != ANSWER_MANIFEST]
    return [("manifest", asset) for asset in manifest] + [("original", asset) for asset in originals[:MAX_ORIGINAL_FILES]]


def _target(dest: pathlib.Path, kind: str, asset: dict[str, Any]) -> pathlib.Path:
    asset_id = _stable_id(asset)
    name = SAFE_NAME.sub("_", pathlib.PurePosixPath(_asset_name(asset) or "asset").name)[:120] or "asset"
    return dest / f"{kind}-{asset_id:020d}-{name}"


def _list_assets(root: str, repo: str, issue: str, token: str) -> list[dict[str, Any]]:
    """Return complete bounded listing; a full final page is unsafe to trust."""
    listing: list[dict[str, Any]] = []
    for page in range(1, MAX_ASSET_PAGES + 1):
        raw = _api(
            f"{root}/repos/{repo}/issues/{issue}/assets?limit={ASSET_PAGE_SIZE}&page={page}",
            token,
            MAX_LISTING_PAGE_BYTES,
        )
        try:
            rows = json.loads(raw or b"[]")
        except (TypeError, ValueError) as exc:
            raise SystemExit("asset listing JSON invalid") from exc
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise SystemExit("asset listing is not a list")
        if len(rows) > ASSET_PAGE_SIZE:
            raise SystemExit("asset listing page exceeds bound")
        listing.extend(rows)
        if len(rows) < ASSET_PAGE_SIZE:
            return listing
    raise SystemExit("asset listing truncated at bounded page limit")


def _list_comments(root: str, repo: str, issue: str, token: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in range(1, MAX_ASSET_PAGES + 1):
        raw = _api(f"{root}/repos/{repo}/issues/{issue}/comments?limit={ASSET_PAGE_SIZE}&page={page}", token, MAX_LISTING_PAGE_BYTES)
        try:
            page_rows = json.loads(raw or b"[]")
        except (TypeError, ValueError) as exc:
            raise SystemExit("comment listing JSON invalid") from exc
        if not isinstance(page_rows, list) or any(not isinstance(row, dict) for row in page_rows):
            raise SystemExit("comment listing is not a list")
        rows.extend(page_rows)
        if len(page_rows) < ASSET_PAGE_SIZE:
            return rows
    raise SystemExit("comment listing truncated at bounded page limit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    root = os.environ["API_URL"]
    repo = os.environ["REPOSITORY"]
    issue = os.environ["ISSUE"]
    token = os.environ["GIT_TOKEN"]
    dest = pathlib.Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    listing = _list_assets(root, repo, issue, token)
    login, numeric_id = os.environ.get("ANSWER_BOT_LOGIN", ""), os.environ.get("ANSWER_BOT_ID", "")
    ref = _manifest_ref(_list_comments(root, repo, issue, token), listing, login, numeric_id) if login and numeric_id else None
    selected = select_assets(listing, manifest_id=ref[0] if ref else None)
    total = 0
    counts = {"manifest": 0, "original": 0}
    for kind, asset in selected:
        cap = MAX_MANIFEST_BYTES if kind == "manifest" else MAX_ORIGINAL_BYTES
        url = asset.get("download_url")
        if not _trusted_attachment_url(root, url):
            raise SystemExit(f"{kind} asset {_stable_id(asset)} download URL untrusted")
        if kind == "original" and ref is not None and counts["manifest"] != 1:
            raise SystemExit("original assets require one verified answer manifest")
        payload = _api(url, token, min(cap, MAX_TOTAL_BYTES - total))
        if len(payload) > cap:
            raise SystemExit(f"{kind} asset {_stable_id(asset)} exceeds quota")
        if total + len(payload) > MAX_TOTAL_BYTES:
            raise SystemExit("assets exceed total quota")
        if kind == "manifest":
            try:
                value = validate_answer_manifest(payload)
                if ref is None or value.get(MANIFEST_DIGEST_FIELD) != ref[1]:
                    raise ValueError("answer manifest comment digest mismatch")
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        target = _target(dest, kind, asset)
        target.write_bytes(payload)
        target.chmod(0o600)
        total += len(payload)
        counts[kind] += 1
    print(f"assets downloaded: manifest={counts['manifest']} original={counts['original']} bytes={total}")


if __name__ == "__main__":
    main()
