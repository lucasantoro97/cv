#!/usr/bin/env python3
"""Download issue attachments through the API only, under contract caps.

Untrusted input: filenames and content. Names are normalised, never used as paths
from the API string directly, and URLs from issue *text* are never fetched.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, urllib.request

MAX_FILES = 10
MAX_BYTES = 25 * 1024 * 1024
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _api(url: str, token: str) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("Authorization", "token " + token)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read(MAX_BYTES + 1)


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
    listing = json.loads(_api(f"{root}/repos/{repo}/issues/{issue}/assets", token) or b"[]")
    for index, asset in enumerate(listing[:MAX_FILES]):
        url = asset.get("browser_download_url")
        if not url or not url.startswith(root.split("/api/")[0]):
            continue
        payload = _api(url, token)
        if len(payload) > MAX_BYTES:
            raise SystemExit(f"asset {index} exceeds size cap")
        name = SAFE_NAME.sub("_", pathlib.PurePosixPath(asset.get("name", f"asset{index}")).name)[:120]
        target = dest / f"{index:02d}_{name or 'asset'}"
        target.write_bytes(payload)
        target.chmod(0o600)
    print(f"assets downloaded: {min(len(listing), MAX_FILES)}")


if __name__ == "__main__":
    main()
