#!/usr/bin/env python3
"""Banc de vitesse pour le rapport communautaire de Strata (docs/COMMUNITY_BENCHMARKS.md).

Lancé sur evo-X2 (bibliothèque standard seulement) contre un Strata déjà démarré :
    STRATA_URL=http://100.101.45.102:8090 STRATA_API_KEY=… python3 -I banc-communaute.py <étiquette> <journal du moteur> \
        [--mots 330,2700,8000,30000] [--runs 3] [--max-tokens 512] > runs.jsonl

Chaque prompt commence par un nonce propre au run (« run <étiquette>-<n>-<taille> ») et décale ses mots, pour que
rien ne soit réutilisé d'un run à l'autre (le moteur réutilise les préfixes communs). Une requête de chauffe de
~1 200 tokens, exclue, passe d'abord. Pour chaque requête : mesure côté client (TTFT = premier delta de contenu non
vide, durée totale) et la ligne du moteur « strata serve: prompt N tokens = R reused + F read in … » lue dans son
journal (tokens réutilisés et lus, débits du moteur, brouillons acceptés). Une ligne JSON par requête sur stdout.
"""
import json
import os
import re
import sys
import time
import urllib.request

URL, CLE = os.environ["STRATA_URL"], os.environ["STRATA_API_KEY"]
LIGNE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated "
                   r"in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def options(argv):
    o = {"mots": [330, 2700, 8000, 30000], "runs": 3, "max_tokens": 512}
    i = 0
    while i < len(argv):
        if argv[i] == "--mots":
            o["mots"] = [int(x) for x in argv[i + 1].split(",")]
        elif argv[i] == "--runs":
            o["runs"] = int(argv[i + 1])
        elif argv[i] == "--max-tokens":
            o["max_tokens"] = int(argv[i + 1])
        i += 2
    return o


def ligne_moteur(journal, depuis):
    """La dernière ligne « strata serve: prompt … » écrite après l'octet `depuis` du journal du moteur."""
    for _ in range(50):
        with open(journal, "rb") as f:
            f.seek(depuis)
            texte = f.read().decode("utf-8", "replace")
        m = [x for x in LIGNE.finditer(texte)]
        if m:
            g = m[-1].groups()
            return {"moteur_prompt": int(g[0]), "moteur_reutilises": int(g[1]), "moteur_lus": int(g[2]),
                    "moteur_lecture_ms": int(g[3]), "moteur_lecture_tok_s": float(g[4]),
                    "moteur_generes": int(g[5]), "moteur_generation_ms": int(g[6]),
                    "moteur_generation_tok_s": float(g[7]), "brouillons_acceptes": int(g[8]),
                    "brouillons_proposes": int(g[9])}
        time.sleep(0.2)
    return {}


def requete(etiquette, n_mots, max_tokens, journal):
    decalage = sum(map(ord, etiquette)) % 997
    prompt = f"run {etiquette}\n" + " ".join(f"mot{(i + decalage) % 997}" for i in range(n_mots)) + \
        "\nÉcris une longue histoire en français sur un phare breton."
    corps = json.dumps({"model": "strata", "max_tokens": max_tokens, "stream": True, "reasoning_effort": "none",
                        "stream_options": {"include_usage": True},
                        "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", corps,
                                 {"content-type": "application/json", "authorization": f"Bearer {CLE}"})
    depuis = os.path.getsize(journal)
    t0, t1, usage = time.time(), None, {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for l in r:
            if not l.startswith(b"data: ") or l.strip() == b"data: [DONE]":
                continue
            d = json.loads(l[6:])
            if t1 is None and d.get("choices") and d["choices"][0]["delta"].get("content"):
                t1 = time.time()
            usage = d.get("usage") or usage
    t2 = time.time()
    res = {"etiquette": etiquette, "mots": n_mots, "prompt_tokens": usage.get("prompt_tokens"),
           "reutilises_api": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "generes": usage.get("completion_tokens"), "ttft_s": round(t1 - t0, 3) if t1 else None,
           "total_s": round(t2 - t0, 3)}
    res.update(ligne_moteur(journal, depuis))
    return res


def main():
    etiquette, journal = sys.argv[1], sys.argv[2]
    o = options(sys.argv[3:])
    chauffe = requete(f"{etiquette}-chauffe", 330, 64, journal)
    print(json.dumps({**chauffe, "chauffe": True}), flush=True)
    for run in range(1, o["runs"] + 1):
        for n in o["mots"]:
            r = requete(f"{etiquette}-{run}-{n}", n, o["max_tokens"], journal)
            r["run"] = run
            print(json.dumps(r), flush=True)
            print(f"{etiquette} run {run} {r['prompt_tokens']:>6} tk : lecture {r.get('moteur_lecture_tok_s')} tok/s, "
                  f"génération {r.get('moteur_generation_tok_s')} tok/s, réutilisés {r.get('moteur_reutilises')}",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
