#!/usr/bin/env python3
"""Independent model review gate for automatic agent PR merges.

The release decision is fully automatic. Independence and protected-path handling are enforced
here rather than left to either reviewer's judgement:

  INDEPENDENCE — the reviewer is a different model from the implementer, gets a fresh context, and
  sees only the DIFF. It never reads the implementer's reasoning, so it cannot be talked into
  agreeing by an explanation it did not verify.

  PROTECTED PATHS — default is fail closed. The trusted workflow may explicitly request automatic
  handling, but then two distinct Codex model classes must independently approve the same diff.
  That includes intake authentication, ACLs, egress, workflow gates, and this file.

Exit 0 = may merge. Exit 2 = policy boundary or unparseable output. Any other exit = failure and, because a
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
    re.compile(r"(^|/)\.forgejo/"),
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

Rifiuta anche quando il diff richiede autorita' operativa non dimostrata: cambia una politica
aziendale, tocca soldi o contratti, oppure e' irreversibile.

Rispondi in italiano, massimo 12 righe, e chiudi con una riga esatta:
VERDETTO: APPROVA
oppure
VERDETTO: RIFIUTA
Non chiedere intervento umano: APPROVA soltanto con prove sufficienti, altrimenti RIFIUTA.

Diff da giudicare:
"""


def changed_files(repo: str, base: str, head: str) -> list[str]:
    out = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-C", repo, "diff", "--name-only", f"{base}...{head}"],
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


def run_reviewer(repo: str, model: str, prompt: str) -> subprocess.CompletedProcess[str]:
    """Run one fresh reviewer. Codex models are read-only; Claude remains compatible."""
    child_env = dict(os.environ)
    child_env.pop("CODEX_AUTH_B64", None)
    if model.startswith("gpt-"):
        command = ["codex", "exec", "--sandbox", "read-only", "-C", repo,
                   "-m", model, prompt]
        child_env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    else:
        command = ["claude", "-p", prompt, "--model", model, "--max-turns", "1"]
    return subprocess.run(command, capture_output=True, text=True, timeout=900, env=child_env)


def main() -> int:
    parse = argparse.ArgumentParser()
    parse.add_argument("--repo", default=".")
    parse.add_argument("--base", required=True)
    parse.add_argument("--head", required=True)
    parse.add_argument("--model", default=os.environ.get("REVIEW_MODEL", "fable"))
    parse.add_argument("--second-model")
    parse.add_argument("--allow-protected-auto", action="store_true")
    parse.add_argument("--max-diff-bytes", type=int, default=200_000)
    args = parse.parse_args()

    paths = changed_files(args.repo, args.base, args.head)
    if not paths:
        print("REVIEW: nessun file modificato")
        return 2
    blocked = protected_hits(paths)
    if blocked and not args.allow_protected_auto:
        print("REVIEW: confine di fiducia toccato; fusione automatica non autorizzata:")
        for path in blocked:
            print(f"  {path}")
        return 2

    diff = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-C", args.repo, "diff", f"{args.base}...{args.head}"],
                          capture_output=True, text=True, check=True).stdout
    if len(diff.encode()) > args.max_diff_bytes:
        # A diff nobody can read in one pass is not reviewable by a model either. Say so instead
        # of approving something that was never actually examined.
        print(f"REVIEW: diff troppo grande ({len(diff.encode())} byte); nessuna fusione")
        return 2

    models = [args.model]
    if blocked:
        if not args.second_model or args.second_model == args.model:
            print("REVIEW: secondo revisore indipendente mancante")
            return 2
        models.append(args.second_model)
        print("REVIEW: confine di fiducia, servono due approvazioni modello indipendenti")
    for model in models:
        proc = run_reviewer(args.repo, model, PROMPT + diff)
        review = (proc.stdout or "") + (proc.stderr or "")
        print(review[-4000:])
        if proc.returncode != 0:
            print("REVIEW: il revisore non ha risposto; nessuna fusione")
            return 3
        verdict = read_verdict(review)
        payload = {"model": model, "files": len(paths), "verdict": verdict}
        print("REVIEW_RESULT " + json.dumps(payload, ensure_ascii=False))
        if verdict != "APPROVA":
            return 4 if verdict in {"RIFIUTA", "UMANO"} else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
