#!/usr/bin/env python3
"""Independent review gate: decide whether an agent PR may merge without a person.

The committente removed the human from the merge decision and asked for Fable to review "as if it
were a human". That only means anything if the reviewer is genuinely independent and if some
changes remain out of its reach. Both are enforced here, not left to the reviewer's judgement:

  INDEPENDENCE — the reviewer is a different model from the implementer, gets a fresh context, and
  sees only the DIFF. It never reads the implementer's reasoning, so it cannot be talked into
  agreeing by an explanation it did not verify.

  PROTECTED PATHS — a change that touches the trust boundary is never auto-merged, no matter how
  good the review. That includes the intake authentication, the ACL, the egress gate, the workflow
  gate, and THIS FILE. Without that carve-out a single PR could disable the reviewer and every
  later PR would sail through: the gate must not be able to open itself.

Exit 0 = may merge. Exit 2 = needs a person. Any other exit = the gate failed and, because a
failed gate must not approve, that also means no merge.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

# Anything matching these is a boundary change: the reviewer may comment, never merge.
PROTECTED = (
    re.compile(r"forgejo-runner/workflows/"),          # the gate itself, including this file
    re.compile(r"(^|/)(imap_bridge|notifier|common)\.py$"),   # who may create and be told things
    re.compile(r"safety/"),                            # NAS boundaries, doveadm, mail identity
    re.compile(r"(^|/)corpus\.py$"),                   # access_scope lives here
    re.compile(r"(^|/)retrievers/base\.py$"),          # the visibility filter every retriever uses
    re.compile(r"contracts/"),                         # frozen contracts
    re.compile(r"(^|/)egress_gate\.py$"),
    re.compile(r"\.env|credentials|token|secret", re.IGNORECASE),
)

VERDICT = re.compile(r"^\s*VERDETTO:\s*(APPROVA|RIFIUTA|UMANO)\s*$", re.MULTILINE)

PROMPT = """Sei il revisore indipendente di questa modifica. Non l'hai scritta tu e non sai come e
perche' e' stata scritta: vedi solo il diff, ed e' voluto.

Il tuo verdetto sostituisce quello di una persona, quindi giudica come giudicherebbe un collega
che deve conviverci: la modifica fa cio' che il ticket chiede, senza rompere altro e senza
lasciare trappole.

Rifiuta se trovi anche una sola di queste:
- scarta dati in silenzio (un except che ingoia, un timeout trattato come successo, un troncamento
  non dichiarato). E' la famiglia di guasti che questa infrastruttura produce piu' spesso;
- allarga un permesso o rende leggibile a piu' persone qualcosa che prima non lo era;
- un default che in caso di dubbio CONCEDE invece di negare;
- toglie o indebolisce un test esistente per far passare il codice nuovo;
- introduce una lista scritta a mano dove serviva una regola, o viceversa;
- fa qualcosa che il ticket non chiedeva.

Chiedi un umano (VERDETTO: UMANO) quando il diff e' corretto ma la DECISIONE non e' tua: cambia
una politica aziendale, tocca soldi o contratti, o e' irreversibile.

Rispondi in italiano, massimo 12 righe, e chiudi con una riga esatta:
VERDETTO: APPROVA
oppure
VERDETTO: RIFIUTA
oppure
VERDETTO: UMANO

Diff da giudicare:
"""


def changed_files(repo: str, base: str, head: str) -> list[str]:
    out = subprocess.run(["git", "-C", repo, "diff", "--name-only", f"{base}...{head}"],
                         capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def protected_hits(paths: list[str]) -> list[str]:
    return [p for p in paths if any(rx.search(p) for rx in PROTECTED)]


def read_verdict(text: str) -> str | None:
    """Last explicit verdict line wins; anything else is not a verdict.

    Parsing loosely here would be the whole hole: 'non approvo' contains 'APPROVA'. The line must
    be exactly the contract, so a reviewer that rambles cannot accidentally merge.
    """
    found = VERDICT.findall(text or "")
    return found[-1] if found else None


def main() -> int:
    parse = argparse.ArgumentParser()
    parse.add_argument("--repo", default=".")
    parse.add_argument("--base", required=True)
    parse.add_argument("--head", required=True)
    parse.add_argument("--model", default=os.environ.get("REVIEW_MODEL", "fable"))
    parse.add_argument("--max-diff-bytes", type=int, default=200_000)
    args = parse.parse_args()

    paths = changed_files(args.repo, args.base, args.head)
    if not paths:
        print("REVIEW: nessun file modificato")
        return 2
    blocked = protected_hits(paths)
    if blocked:
        print("REVIEW: confine di fiducia toccato, serve una persona:")
        for path in blocked:
            print(f"  {path}")
        return 2

    diff = subprocess.run(["git", "-C", args.repo, "diff", f"{args.base}...{args.head}"],
                          capture_output=True, text=True, check=True).stdout
    if len(diff.encode()) > args.max_diff_bytes:
        # A diff nobody can read in one pass is not reviewable by a model either. Say so instead
        # of approving something that was never actually examined.
        print(f"REVIEW: diff troppo grande ({len(diff.encode())} byte), serve una persona")
        return 2

    proc = subprocess.run(
        ["claude", "-p", PROMPT + diff, "--model", args.model, "--max-turns", "1"],
        capture_output=True, text=True, timeout=900)
    review = (proc.stdout or "") + (proc.stderr or "")
    print(review[-4000:])
    if proc.returncode != 0:
        print("REVIEW: il revisore non ha risposto; nessuna fusione")
        return 3

    verdict = read_verdict(review)
    payload = {"model": args.model, "files": len(paths), "verdict": verdict}
    print("REVIEW_RESULT " + json.dumps(payload, ensure_ascii=False))
    if verdict == "APPROVA":
        return 0
    if verdict == "RIFIUTA":
        return 4
    # No verdict at all is NOT an approval: an unparseable review means nobody reviewed it.
    return 2


if __name__ == "__main__":
    sys.exit(main())
