#!/usr/bin/env python3
"""llama.cpp-Gegenstück zu Stratas `tools/quant_quality.py run` — 17.10.2026.

Liest denselben Textsatz (set.ids.json, Token-IDs aus Stratas Tokenizer) Position für Position durch einen laufenden
llama-server: je Position eine Anfrage mit dem Präfix bis dorthin, `n_predict` 1 und `n_probs` 4000, mit
`cache_prompt`, damit nur das neue Token gerechnet wird. Geschrieben wird Stratas STRATA_LOGPOS-Format (TSV), damit
`quant_quality.py compare` beide Läufe direkt vergleicht:
  pos, Ziel-ID, logp(Ziel), argmax-ID, logp(argmax), argmax==Ziel, logp(<|im_end|>), logp(Ziel), 64 × id:logp
Die Positionsnummer `pos` übernimmt die Zählweise des Strata-Laufs (--pos-offset, aus dessen erster Zeile ablesen).
Fehlt das Ziel unter den 4000 wahrscheinlichsten Token, wird der 4000. Wert eingesetzt und gezählt (Spalte 8 dann leer markiert: "nan").

  python3 qq_llamacpp.py --url http://127.0.0.1:18300 --ids set.ids.json --out llamacpp.tsv --im-end 248046
"""
import argparse, json, time, urllib.request

TOPK, NPROBS = 64, 4000


def frage(url, ids):
    body = {"prompt": ids, "n_predict": 1, "n_probs": NPROBS, "temperature": 0, "cache_prompt": True,
            "post_sampling_probs": False}
    req = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    cp = d["completion_probabilities"][0]
    return [(t["id"], t["logprob"]) for t in cp["top_logprobs"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True); p.add_argument("--ids", required=True); p.add_argument("--out", required=True)
    p.add_argument("--im-end", type=int, default=-1); p.add_argument("--pos-offset", type=int, default=0)
    p.add_argument("--max-pos", type=int, default=0, help="nur zum Testen: so viele Positionen je Abschnitt")
    a = p.parse_args()
    st = json.load(open(a.ids))
    segs, zeile, fehlend = [], 0, 0
    with open(a.out, "w") as f:
        for s in st["segments"]:
            ids, t0, start = s["ids"], time.time(), zeile
            n = len(ids) - 1 if not a.max_pos else min(a.max_pos, len(ids) - 1)
            for i in range(n):
                top = frage(a.url, ids[:i + 1])
                d = dict(top)
                ziel = ids[i + 1]
                lz = d.get(ziel)
                exakt = lz is not None
                if not exakt:
                    lz = top[-1][1]; fehlend += 1
                am, lam = top[0]
                lim = d.get(a.im_end, top[-1][1])
                felder = [str(i + a.pos_offset), str(ziel), f"{lz:.6f}", str(am), f"{lam:.6f}", str(int(am == ziel)),
                          f"{lim:.6f}", f"{lz:.6f}" if exakt else "nan"] + [f"{t}:{l:.6f}" for t, l in top[:TOPK]]
                f.write("\t".join(felder) + "\n"); zeile += 1
            f.flush()
            segs.append({"name": s["name"], "first_line": start, "end_line": zeile})
            print(f"{s['name']}: {zeile - start} Positionen, {time.time() - t0:.0f} s, Ziel außerhalb Top-{NPROBS}: {fehlend}", flush=True)
    json.dump({"segments": segs, "engine": "llama.cpp", "url": "local", "n_probs": NPROBS, "target_outside_top": fehlend},
              open(a.out + ".seg.json", "w"), indent=1)


if __name__ == "__main__":
    main()
