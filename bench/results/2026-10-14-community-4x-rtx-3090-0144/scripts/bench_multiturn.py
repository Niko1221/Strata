#!/usr/bin/env python3
"""Agent-style multi-turn benchmark against an OpenAI-compatible endpoint (Strata, vLLM).

N users each hold one conversation of T turns.  Turn 1 sends a large unique document (~DOC tokens) plus a task; every
later turn sends the full history (previous answers included) plus ~ADD new tokens and a new task - the way an agent
loop or a chat client resends its context.  Users run concurrently, so conversations interleave like a main agent and
its sub-agents.  What matters: time to first token of the follow-up turns (does the engine reuse the conversation or
read it again?), decode rate, aggregate output.  The --out row also carries every turn's reply sha256 (uid, turn,
tokens, sha256): with --temperature 0 two arms of an exact change must match turn by turn (the later turns resend the
earlier replies, so the first difference cascades into all following turns of that user).

  python3 bench_multiturn.py --url http://127.0.0.1:8090 --users 8 --turns 5 --label strata-agent
"""
import argparse, concurrent.futures as cf, datetime, hashlib, json, statistics, threading, time, urllib.request

WORDS = ("system design latency throughput cache memory kernel scheduler packet buffer index query storage replica "
         "shard consensus leader follower snapshot checkpoint journal compaction bloom filter hash tree page block "
         "extent inode socket thread process lock mutex barrier fence atomic vector matrix tensor gradient").split()


def doc(uid, turn, n_tok):
    # deterministic, unique per user and turn; ~1.3 words per token
    out, k = [], uid * 7919 + turn * 104729
    for i in range(int(n_tok / 1.3)):
        k = (k * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out.append(WORDS[(k >> 33) % len(WORDS)])
        if i % 17 == 16:
            out.append(f"(note {uid}.{turn}.{i})\n")
    return " ".join(out)


def call(a, messages):
    body = {"model": a.model, "messages": messages, "max_tokens": a.max_tok, "temperature": a.temperature, "stream": True,
            "stream_options": {"include_usage": True}}
    if a.vllm:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["reasoning_effort"] = "none"
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first, text, usage = None, [], {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            c = json.loads(line[5:])
            if c.get("usage"):
                usage = c["usage"]
            for ch in c.get("choices") or []:
                d = (ch.get("delta") or {}).get("content")
                if d:
                    if t_first is None:
                        t_first = time.perf_counter()
                    text.append(d)
    t1 = time.perf_counter()
    gen = usage.get("completion_tokens", len(text))
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    return {"ttft": (t_first or t1) - t0, "e2e": t1 - t0, "gen": gen, "prompt": usage.get("prompt_tokens", 0),
            "cached": cached, "dec": (gen - 1) / (t1 - t_first) if t_first and gen > 1 and t1 > t_first else 0.0,
            "text": "".join(text)}


def user(a, uid, rows, lock):
    msgs = [{"role": "system", "content": f"You are agent #{uid}. Answer concisely and precisely."}]
    for t in range(a.turns):
        n = a.doc if t == 0 else a.add
        msgs.append({"role": "user", "content": f"[agent {uid}, step {t + 1}] New material:\n{doc(uid, t, n)}\n\n"
                                                f"Task: summarise what is new in this material and how it relates "
                                                f"to the earlier steps."})
        r = call(a, msgs)
        msgs.append({"role": "assistant", "content": r["text"]})
        r["sha256"] = hashlib.sha256(r.pop("text").encode("utf-8")).hexdigest()
        r.update(uid=uid, turn=t + 1)
        with lock:
            rows.append(r)
        time.sleep(a.think)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="x")
    ap.add_argument("--label", required=True)
    ap.add_argument("--vllm", action="store_true")
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--doc", type=int, default=8000, help="tokens of the first turn's document")
    ap.add_argument("--add", type=int, default=3000, help="new tokens per later turn")
    ap.add_argument("--max-tok", type=int, default=400)
    ap.add_argument("--think", type=float, default=1.0, help="seconds a user waits between turns")
    ap.add_argument("--temperature", type=float, default=0.7, help="0 = greedy (reply hashes comparable across arms)")
    ap.add_argument("--out", default="")
    ap.add_argument("--uid-offset", type=int, default=0,
                    help="first user id: another offset = other documents = new conversations (soak cycles)")
    a = ap.parse_args()
    rows, lock = [], threading.Lock()
    ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(a.users) as ex:
        list(ex.map(lambda u: user(a, u, rows, lock), range(a.uid_offset, a.uid_offset + a.users)))
    wall = time.perf_counter() - t0
    print(f"# {a.label}: {a.users} users x {a.turns} turns, doc {a.doc} + {a.add}/turn, {a.max_tok} out")
    print(f"{'turn':>4} {'prompt':>7} {'cached':>7} {'TTFT p50':>9} {'TTFT p95':>9} {'dec p50':>8}")
    summary = {"label": a.label, "ts": ts, "users": a.users, "turns": a.turns, "temperature": a.temperature,
               "wall_s": round(wall, 1)}
    for t in range(1, a.turns + 1):
        rs = [r for r in rows if r["turn"] == t]
        tt = sorted(r["ttft"] for r in rs)
        p95 = tt[min(len(tt) - 1, int(0.95 * len(tt)))]
        row = {"prompt": int(statistics.mean(r["prompt"] for r in rs)), "cached": int(statistics.mean(r["cached"] for r in rs)),
               "ttft_p50": round(statistics.median(tt), 2), "ttft_p95": round(p95, 2),
               "dec_p50": round(statistics.median(r["dec"] for r in rs), 1)}
        summary[f"turn{t}"] = row
        print(f"{t:>4} {row['prompt']:>7} {row['cached']:>7} {row['ttft_p50']:>9.2f} {row['ttft_p95']:>9.2f} {row['dec_p50']:>8.1f}")
    out_tok = sum(r["gen"] for r in rows)
    in_tok = sum(r["prompt"] for r in rows)
    summary.update(out_tok=out_tok, agg_out_tps=round(out_tok / wall, 1), prompt_tok=in_tok,
                   follow_ttft_p50=round(statistics.median(r["ttft"] for r in rows if r["turn"] > 1), 2),
                   reply_fields=["uid", "turn", "gen", "sha256"],
                   replies=[[r["uid"], r["turn"], r["gen"], r["sha256"]]
                            for r in sorted(rows, key=lambda r: (r["uid"], r["turn"]))])
    print(f"wall {wall:.0f}s, output {out_tok} tok = {out_tok / wall:.1f} tok/s, prompt tokens sent {in_tok}, "
          f"follow-up TTFT p50 {summary['follow_ttft_p50']} s")
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
