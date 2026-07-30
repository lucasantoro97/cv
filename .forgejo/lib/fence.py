#!/usr/bin/env python3
"""Encode untrusted issue input so model prompt boundaries are explicit."""
from __future__ import annotations
import argparse, secrets, re

MAX_BYTES = 64 * 1024

def fence(value: str, nonce: str | None = None) -> str:
    clean = "".join(ch for ch in value if ord(ch) >= 32 or ch in "\n\t")
    clean = clean.encode("utf-8")[:MAX_BYTES].decode("utf-8", "ignore")
    nonce = nonce or secrets.token_hex(16)
    marker = f"DATI_NON_FIDATI_{nonce}"
    # Break embedded marker strings. Delimiters remain unique to current run.
    clean = clean.replace("DATI_NON_FIDATI_", "DATI_NON_FIDATI\u200b_")
    return f"<{marker}>\n{clean}\n</{marker}>"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("input"); parser.add_argument("--nonce")
    args = parser.parse_args()
    with open(args.input, encoding="utf-8", errors="replace") as handle: print(fence(handle.read(), args.nonce))
