#!/usr/bin/env python3
"""Schreib- und Einlesetempo über die Kontexttiefe — 17.10.2026, für die Kurven im Repo ugreen-idx6011-egpu-llm.

Je Tiefe ein frischer Prompt (eindeutiges Kennzeichen vorn, damit kein Gesprächs-Cache hilft), danach 256 neue
Token ohne Denken. Gelesen werden die Zeiten der Engine aus `timings` (prompt_per_second, predicted_per_second,
draft_n / draft_n_accepted). Ergebnisse als JSON-Zeilen.

  python3 tiefe.py --url http://127.0.0.1:8090 --model arbeit --label mit-mtp --out ergebnisse/tiefe.jsonl
"""
import argparse, json, random, time, urllib.request

TIEFEN = [(1024, 3), (4096, 3), (16384, 3), (65536, 3), (131072, 2), (250000, 2)]
WOERTER = ("Haus Garten Fluss Rechner Netz Strom Licht Wolke Brücke Straße Fenster Lampe Buch Seite Karte Zug Hafen "
           "Markt Feld Wald Berg Tal Stadt Dorf Kabel Platte Speicher Schalter Turm Mauer Brunnen Weg Stein Glas").split()
AUFGABE = "\n\nIgnoriere die Wortliste oben. Erzähle stattdessen eine ruhige Geschichte über einen Leuchtturmwärter, etwa 300 Wörter."


def frage(url, modell, text, max_tokens):
    body = {"model": modell, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": 0, "reasoning_effort": "none"}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    return d, time.time() - t0


def wortliste(n, rng):
    return " ".join(rng.choice(WOERTER) for _ in range(n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True); p.add_argument("--model", required=True)
    p.add_argument("--label", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    rng = random.Random(17102026)
    d, _ = frage(a.url, a.model, wortliste(2000, rng), 1)      # Kalibrierung: Token je Wort
    je_wort = d["usage"]["prompt_tokens"] / 2000
    print(f"{a.label}: {je_wort:.3f} Token je Wort", flush=True)
    for ziel, laeufe in TIEFEN:
        for i in range(laeufe):
            n = int((ziel - 80) / je_wort)
            text = f"Lauf {time.time_ns()} {a.label}.\n" + wortliste(n, rng) + AUFGABE
            d, dauer = frage(a.url, a.model, text, 256)
            t = d.get("timings", {})
            z = {"label": a.label, "modell": a.model, "ziel": ziel, "lauf": i + 1,
                 "prompt_token": d["usage"]["prompt_tokens"], "neu_token": d["usage"]["completion_tokens"],
                 "prompt_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
                 "draft_n": t.get("draft_n"), "draft_ok": t.get("draft_n_accepted"), "sekunden": round(dauer, 1)}
            print(json.dumps(z, ensure_ascii=False), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(z, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
