#!/usr/bin/env python3
"""Needle-in-a-haystack against a running Strata server (strata-qwen36 engine).  Only the standard library.

  python tests/long_context_qwen36.py [--url http://127.0.0.1:8080] [--model qwen3.6-35b-a3b]
                                      [--lengths 2000,4000,8000,16000,28000] [--depths 0.1,0.5,0.9]
                                      [--max-tokens 3000] [--seed 1]

For every (length, depth) a secret code is hidden inside filler text at that relative depth and the model is asked for it.
Prompt sizes come from the server's own `usage.prompt_tokens` (a first small request calibrates chars-per-token), so no
tokenizer is needed here.  Also reports prefill / decode speed and runs a multi-turn cache-reuse check.
Thinking is on by default in Qwen3.6: --max-tokens must leave room for it; a run that ends in "length" with no answer is
reported as INCONCLUSIVE, not as a failure.
"""
import argparse, json, random, sys, time, urllib.request

TOPICS = ["harbour", "orchard", "railway", "library", "glacier", "market", "lighthouse", "observatory", "vineyard", "canal"]
VERBS = ["was surveyed", "was repainted", "was inspected", "was documented", "was reopened", "was measured"]


def sentence(rng, i):
    return (f"Report {i}: the {rng.choice(TOPICS)} number {rng.randint(100, 999)} {rng.choice(VERBS)} on day "
            f"{rng.randint(1, 365)} by team {rng.choice('ABCDEFGH')}{rng.randint(1, 99)}, with a recorded value of "
            f"{rng.randint(1000, 99999)} units.")


def build(rng, n_sent, depth, needle_sentence):
    s = [sentence(rng, i) for i in range(n_sent)]
    s.insert(min(n_sent, max(0, int(depth * n_sent))), needle_sentence)
    return " ".join(s)


def chat(url, model, messages, max_tokens, extra=None, timeout=1800):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    body.update(extra or {})
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["_wall"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--lengths", default="2000,4000,8000,16000,28000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # calibrate: tokens per filler sentence
    probe = [sentence(rng, i) for i in range(300)]
    r = chat(a.url, a.model, [{"role": "user", "content": " ".join(probe)}], 1)
    tps = r["usage"]["prompt_tokens"] / 300.0
    print(f"calibration: ~{tps:.1f} tokens per sentence\n")

    rows, fails, inconcl = [], 0, 0
    print(f"{'target':>7} {'prompt':>7} {'depth':>5}  {'result':<12} {'prefill t/s':>11} {'decode t/s':>10} {'wall s':>7}")
    for L in [int(x) for x in a.lengths.split(",")]:
        for d in [float(x) for x in a.depths.split(",")]:
            code = f"{rng.choice(['ZEBRA', 'COBALT', 'MAPLE', 'QUARTZ'])}-{rng.randint(1000, 9999)}"
            needle = f"IMPORTANT: the secret access code of the vault is {code}. Remember it exactly."
            n_sent = max(10, int((L - 120) / tps))
            text = build(rng, n_sent, d, needle)
            msg = [{"role": "user", "content": text + "\n\nWhat is the secret access code of the vault? "
                                                      "Answer with the code only."}]
            try:
                o = chat(a.url, a.model, msg, a.max_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"{L:>7} {'-':>7} {d:>5.1f}  ERROR {e}")
                fails += 1
                continue
            ch = o["choices"][0]
            content = ch["message"].get("content") or ""
            reasoning = ch["message"].get("reasoning_content") or ""
            if code in content:
                res = "PASS"
            elif ch["finish_reason"] == "length" and not content:
                res = "INCONCLUSIVE"
                inconcl += 1
            else:
                res = "FAIL"
                fails += 1
            t = o.get("timings", {})
            print(f"{L:>7} {o['usage']['prompt_tokens']:>7} {d:>5.1f}  {res:<12} {t.get('prompt_per_second', 0):>11.0f} "
                  f"{t.get('predicted_per_second', 0):>10.1f} {o['_wall']:>7.1f}")
            if res == "FAIL":
                print(f"        expected {code!r}; got content={content[:120]!r}; code in reasoning: {code in reasoning}")

    # multi-turn cache reuse: turn 2 extends turn 1's conversation (the template may rewrite the old assistant turn)
    print("\nmulti-turn cache reuse:")
    big = " ".join(sentence(rng, i) for i in range(int(4000 / tps)))
    m1 = [{"role": "user", "content": big + "\n\nSay only: OK"}]
    o1 = chat(a.url, a.model, m1, 400)
    m2 = m1 + [{"role": "assistant", "content": o1["choices"][0]["message"].get("content") or "OK"},
               {"role": "user", "content": "Now say only: DONE"}]
    o2 = chat(a.url, a.model, m2, 400)
    pt = o2["usage"]["prompt_tokens"]
    cached = o2["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"  turn 2: prompt {pt} tokens, cached {cached} ({100 * cached / max(1, pt):.0f}%), "
          f"prefill {o2.get('timings', {}).get('prompt_ms', 0):.0f} ms")
    print("  (a low % means the template rewrote the previous turn: GDN state snapshots would be needed)")

    print(f"\n{'ALL PASSED' if not fails and not inconcl else 'DONE'}: {fails} failed, {inconcl} inconclusive")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
