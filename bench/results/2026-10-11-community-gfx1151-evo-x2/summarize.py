#!/usr/bin/env python3
"""Résume les runs de bin/banc-communaute.py (un fichier .jsonl par configuration) en un tableau Markdown, au format
de docs/COMMUNITY_BENCHMARKS.md de Strata : médiane et étendue par configuration et taille de prompt, d'après les
lignes du moteur (débits) et du client (TTFT). Les requêtes de chauffe sont exclues.

    python3 bin/resumer-banc.py <dossier des .jsonl>
"""
import json
import statistics
import sys
from pathlib import Path


def med_etendue(v, fmt="{:,.1f}"):
    v = [x for x in v if x is not None]
    if not v:
        return "not measured"
    m = statistics.median(v)
    return f"{fmt.format(m)} ({fmt.format(min(v))}-{fmt.format(max(v))})"


def main():
    d = Path(sys.argv[1])
    print("| Configuration | Prompt tokens | Reused | Generated | Runs | Prompt tok/s median (range) | "
          "Decode tok/s median (range) | TTFT s median (range) | Drafts accepted |")
    print("| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
    for f in sorted(d.glob("*.jsonl")):
        runs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        runs = [r for r in runs if not r.get("chauffe")]
        for mots in sorted({r["mots"] for r in runs}):
            g = [r for r in runs if r["mots"] == mots]
            acc = sum(r.get("brouillons_acceptes", 0) for r in g)
            prop = sum(r.get("brouillons_proposes", 0) for r in g)
            toks = sorted({r.get("moteur_prompt") for r in g})
            print(f"| {f.stem} | {toks[0]:,}-{toks[-1]:,} | {max(r.get('moteur_reutilises', 0) for r in g)} | "
                  f"{statistics.median(r.get('moteur_generes') for r in g):.0f} | {len(g)} | "
                  f"{med_etendue([r.get('moteur_lecture_tok_s') for r in g], '{:,.0f}')} | "
                  f"{med_etendue([r.get('moteur_generation_tok_s') for r in g])} | "
                  f"{med_etendue([r.get('ttft_s') for r in g])} | "
                  f"{acc} / {prop} ({100 * acc / prop:.0f}%) |" if prop else "| |")


if __name__ == "__main__":
    main()
