#!/usr/bin/env python3
"""Two (or N) clients at once: fresh ~4K-token prompts, 256-token cap, temperature 0, reasoning off.
Reports per-request engine decode rate, wall time per round and total generated tokens per second of wall time.

    python parallel.py --url http://127.0.0.1:18096 --clients 2 --rounds 3 --out parallel.json
"""
import argparse
import json
import threading
import time
import urllib.request

WORDS = ("server cache token engine kernel memory prompt decode layer expert route batch stream "
         "thread queue buffer tensor graph model weight").split()


def prompt(tag, n_words=3600):
    body = " ".join(WORDS[(i * 7 + len(tag)) % len(WORDS)] for i in range(n_words))
    return f"[{tag}] Read the word list below, then describe in plain English what kind of program it could belong to.\n\n{body}"


def one(url, tag, cap, out):
    body = {"model": "x", "max_tokens": cap, "temperature": 0, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": prompt(tag)}]}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        t = d.get("timings") or {}
        out.append({"tag": tag, "ok": True, "wall_s": round(time.time() - t0, 3), "prompt_tokens": t.get("prompt_n"),
                    "generated_tokens": t.get("predicted_n"), "decode_tps": t.get("predicted_per_second"),
                    "prompt_tps": t.get("prompt_per_second"), "finish": d["choices"][0].get("finish_reason"),
                    "visible_chars": len(d["choices"][0]["message"].get("content") or "")})
    except Exception as e:  # noqa: BLE001
        out.append({"tag": tag, "ok": False, "wall_s": round(time.time() - t0, 3), "error": repr(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18096")
    ap.add_argument("--clients", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--out", default="parallel.json")
    a = ap.parse_args()
    nonce = str(int(time.time()))
    rounds = []
    for r in range(a.rounds):
        res = []
        t0 = time.time()
        th = [threading.Thread(target=one, args=(a.url, f"r{r}c{c}-{nonce}", a.cap, res)) for c in range(a.clients)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.time() - t0
        gen = sum(x.get("generated_tokens") or 0 for x in res if x["ok"])
        rounds.append({"round": r, "wall_s": round(wall, 3), "generated_tokens": gen,
                       "aggregate_tps": round(gen / wall, 1) if wall else None, "requests": res})
        print(f"round {r}: {len(res)} requests, {gen} tokens in {wall:.2f} s -> {gen / wall:.1f} tok/s aggregate; "
              + ", ".join(f"{x.get('decode_tps')}" for x in res), flush=True)
    json.dump({"clients": a.clients, "cap": a.cap, "rounds": rounds}, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
