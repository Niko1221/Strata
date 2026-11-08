#!/usr/bin/env python3
"""Concurrency sweep against an OpenAI-compatible endpoint (Strata, llama.cpp, vLLM).

N closed-loop users each send ROUNDS streamed chat requests one after another; every request carries a unique
header first (so no prompt/conversation cache hit), a ~PROMPT_TOK-token document and asks for a long answer capped at
MAX_TOK.  Per level it reports the aggregate output rate (completion tokens / wall), the same rate over the steady
window only (agg_out_tps_steady, see run_level), TTFT, per-request decode rate (tokens after the first / time after
the first) and end-to-end latency, and appends a JSON line to --out.

The prompt depends only on the request id (uid) and the prompt kind (--prompt-tok), never on the label or the clock,
so two arms of an A/B send byte-identical requests.  With --temperature 0 an exact engine change must then return
the same text: every request's reply sha256 is written to a sidecar JSONL (<out>.requests.jsonl, one line per level,
with uid, max_tokens, tokens, timestamps) and compare_replies.py compares two labels uid by uid.

  python3 bench_concurrency.py --url http://127.0.0.1:8080 --levels 1,2,4,8,16 --label batch8
  python3 compare_replies.py results/sweep.requests.jsonl ab-x-A-short ab-x-B-short
"""
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import statistics
import threading
import time
import urllib.request

TOPICS = [
    "how NVMe SSD wear leveling works", "the CAP theorem in distributed systems",
    "TCP congestion control (CUBIC and BBR)", "why B-trees are used in database indexes",
    "virtual memory and page tables", "event-driven vs synchronous server architectures",
    "how GPU tensor cores execute matrix multiplication", "cache coherency protocols in multi-core CPUs",
    "the Raft consensus algorithm", "garbage collection in the JVM", "how a compiler performs register allocation",
    "the design of the Linux CFS/EEVDF scheduler", "public-key infrastructure and TLS handshakes",
    "how LSM-trees work in RocksDB", "mixture-of-experts language models", "speculative decoding for LLMs",
]
FILLER = (
    "The system under discussion consists of several cooperating components. Each component exposes a narrow "
    "interface, keeps its own state and communicates through explicit messages. Failures are expected and handled "
    "locally where possible; otherwise they are escalated with enough context for an operator to act. Performance "
    "is measured end to end, and every optimisation is justified by a measurement rather than by intuition. "
)


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def prompt_for(a, uid):
    """Deterministic: uid + prompt kind only.  The unique header comes first, so two requests (also a short and a long
    run on the same server: same uids, different --prompt-tok) share no cacheable prefix beyond the chat template."""
    topic = TOPICS[uid % len(TOPICS)]
    doc = FILLER * max(1, a.prompt_tok // 60)
    rid = uid + a.uid_offset
    tag = hashlib.sha256(f"{a.prompt_tok}/{rid}".encode()).hexdigest()[:8]   # differs from the first character on
    return (f"[{tag} | {a.prompt_tok}-token request {rid}]\nBackground notes:\n{doc}\n\n"
            f"Task: write a long, thorough and well-structured technical explanation of {topic}. "
            f"Use several sections and concrete examples. Do not stop early.")


def mix_lengths(a):
    """--max-tok-min: the reply-length multiset every user gets (rotated per user, see run_level), ROUNDS lengths
    evenly spaced over [max-tok-min, max-tok]; one round -> the midpoint for everyone."""
    lo, hi, n = a.max_tok_min, a.max_tok, a.rounds
    if n == 1:
        return [(lo + hi) // 2]
    return [lo + round(k * (hi - lo) / (n - 1)) for k in range(n)]


def one_request(a, uid, max_tok=None):
    max_tok = max_tok or a.max_tok
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt_for(a, uid)}], "max_tokens": max_tok,
            "stream": True, "temperature": a.temperature, "seed": uid, "reasoning_effort": "none",
            "stream_options": {"include_usage": True}}
    if a.vllm:
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body.pop("reasoning_effort")
    hdr = {"Content-Type": "application/json"}
    if a.key:
        hdr["Authorization"] = "Bearer " + a.key
    urls = a.url.split(",")          # several instances: round-robin by request id (data-parallel setups)
    req = urllib.request.Request(urls[uid % len(urls)].rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=hdr)
    t0 = time.perf_counter()
    t_first = None
    chunks = 0
    usage = None
    err = None
    text, think = [], []
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    c = json.loads(data)
                except ValueError:
                    continue
                if c.get("error"):
                    err = str(c["error"])[:200]
                if c.get("usage"):
                    usage = c["usage"]
                for ch in c.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        if t_first is None:
                            t_first = time.perf_counter()
                        chunks += 1
                        text.append(d.get("content") or "")
                        think.append(d.get("reasoning_content") or "")
    except Exception as e:  # noqa: BLE001 - a failed request is a result too
        err = repr(e)[:200]
    t1 = time.perf_counter()
    reply = "".join(text)
    if any(think):                   # reasoning (if any) ahead of the answer, NUL-separated
        reply = "".join(think) + "\x00" + reply
    gen = (usage or {}).get("completion_tokens", chunks)
    ptok = (usage or {}).get("prompt_tokens", 0)
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    ttft = (t_first - t0) if t_first else float("nan")
    dec = (gen - 1) / (t1 - t_first) if t_first and t1 > t_first and gen > 1 else float("nan")
    return {"uid": uid, "max_tok": max_tok, "t0": t0, "t_first": t_first, "t1": t1, "ttft": ttft, "e2e": t1 - t0,
            "gen": gen, "prompt": ptok, "cached": cached, "dec_tps": dec, "err": err,
            "sha256": hashlib.sha256(reply.encode("utf-8")).hexdigest()}


def steady_window(results, n):
    """Steady window of a closed-loop level: from the moment every user has its first token (all slots busy, the
    admission ramp is over) to the moment the first user finishes its last request (from there on slots drain).
    Returns (start, end, tokens emitted in it); tokens of a request are spread linearly over t_first..t1."""
    by_user = {}
    for r in results:
        by_user.setdefault(r["user"], []).append(r)
    if len(by_user) < n or any(not r["t_first"] for r in results):
        return None
    start = max(min(r["t_first"] for r in rs) for rs in by_user.values())
    end = min(max(r["t1"] for r in rs) for rs in by_user.values())
    if end - start <= 0:
        return None
    tok = 0.0
    for r in results:
        if start <= r["t_first"] <= end:
            tok += 1                                   # the first token
        span = r["t1"] - r["t_first"]
        if r["gen"] > 1 and span > 0:
            ov = min(end, r["t1"]) - max(start, r["t_first"])
            if ov > 0:
                tok += (r["gen"] - 1) * ov / span
    return start, end, tok


def run_level(a, n, base_uid):
    results, lock = [], threading.Lock()
    lengths = mix_lengths(a) if a.max_tok_min and a.max_tok_min < a.max_tok else None

    def user(i):
        for r in range(a.rounds):
            # mixed reply lengths: user i gets the same multiset as every other user, rotated by i, so all users
            # carry equal work (no closed-loop tail from one user drawing only long replies)
            mt = lengths[(r + i) % len(lengths)] if lengths else a.max_tok
            res = one_request(a, base_uid + i * a.rounds + r, mt)
            res["user"] = i
            with lock:
                results.append(res)

    ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(user, range(n)))
    wall = time.perf_counter() - t0
    ok = [r for r in results if not r["err"] and r["gen"] > 0]
    errs = [r["err"] for r in results if r["err"]]
    tot_gen = sum(r["gen"] for r in ok)
    tot_prompt = sum(r["prompt"] for r in ok)
    ttft = [r["ttft"] for r in ok if r["ttft"] == r["ttft"]]
    dec = [r["dec_tps"] for r in ok if r["dec_tps"] == r["dec_tps"]]
    e2e = [r["e2e"] for r in ok]
    sw = steady_window(results, n) if len(ok) == len(results) else None
    results.sort(key=lambda r: r["uid"])
    row = {
        "label": a.label, "ts": ts, "users": n, "rounds": a.rounds, "requests": len(results), "ok": len(ok),
        "errors": len(errs), "err_sample": errs[:3], "wall_s": round(wall, 2), "max_tok": a.max_tok,
        "max_tok_min": a.max_tok_min, "temperature": a.temperature, "prompt_kind": a.prompt_tok,
        "prompt_tok_avg": round(tot_prompt / max(1, len(ok))),
        "out_tok_total": tot_gen, "agg_out_tps": round(tot_gen / wall, 1),
        "agg_out_tps_steady": round(sw[2] / (sw[1] - sw[0]), 1) if sw else None,
        "steady_s": round(sw[1] - sw[0], 2) if sw else None,
        "agg_total_tps": round((tot_gen + tot_prompt) / wall, 1),
        "ttft_p50": round(pct(ttft, 50), 2), "ttft_p95": round(pct(ttft, 95), 2), "ttft_max": round(max(ttft) if ttft else float("nan"), 2),
        "dec_tps_p50": round(pct(dec, 50), 1), "dec_tps_p5": round(pct(dec, 5), 1), "dec_tps_p95": round(pct(dec, 95), 1),
        "e2e_p50": round(pct(e2e, 50), 1), "e2e_p95": round(pct(e2e, 95), 1),
        "cached_tok": sum(r["cached"] for r in ok),
        # one digest over all (uid, reply sha256): equal digests = every reply identical (greedy exactness check)
        "replies_sha256": hashlib.sha256("".join(f"{r['uid']}:{r['sha256']}\n" for r in results).encode()).hexdigest(),
    }
    rel = (lambda x: round(x - t0, 3) if x else None)
    reqs = {"label": a.label, "ts": ts, "users": n, "fields": REQ_FIELDS,
            "reqs": [[r["uid"], r["user"], r["max_tok"], r["gen"], r["prompt"], rel(r["t0"]), rel(r["t_first"]),
                      rel(r["t1"]), r["sha256"], r["err"]] for r in results]}
    return row, reqs


REQ_FIELDS = ["uid", "user", "max_tok", "gen", "prompt", "t0", "t_first", "t_end", "sha256", "err"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default="")
    ap.add_argument("--label", default="run")
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--rounds", type=int, default=2, help="requests per user, sent back to back")
    ap.add_argument("--max-tok", type=int, default=512)
    ap.add_argument("--max-tok-min", type=int, default=0,
                    help="> 0: mixed reply lengths: ROUNDS lengths evenly spaced over [max-tok-min, max-tok], every user "
                         "gets all of them in a rotated order (equal work per user), so the slots do not finish in "
                         "lockstep but no user is left running alone at the end")
    ap.add_argument("--prompt-tok", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--req-out", default="auto",
                    help="per-request JSONL (uid, max_tokens, tokens, t0/t_first/t_end relative to the level start, "
                         "reply sha256); auto = <out without .jsonl>.requests.jsonl, '' = off")
    ap.add_argument("--uid-offset", type=int, default=0,
                    help="added to the request id in the prompt header only: set it when the same prompt kind runs "
                         "twice on one live server with a prompt cache (default 0 keeps prompts identical across arms)")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--vllm", action="store_true", help="thinking off via chat_template_kwargs (vLLM)")
    a = ap.parse_args()
    if a.req_out == "auto":
        a.req_out = (a.out[:-6] if a.out.endswith(".jsonl") else a.out) + ".requests.jsonl" if a.out else ""
    for w in range(a.warmup):
        one_request(a, 10_000_000 + w)
    print(f"# {a.label}: prompt~{a.prompt_tok} tok, max_tokens {a.max_tok}, {a.rounds} request(s) per user, "
          f"temperature {a.temperature}"
          + (f", reply lengths {mix_lengths(a)} rotated per user" if a.max_tok_min and a.max_tok_min < a.max_tok else ""),
          flush=True)
    print(f"{'users':>5} {'ok/req':>7} {'wall':>7} {'agg out':>8} {'dec p50':>8} {'dec p5':>7} {'TTFT p50':>9} "
          f"{'TTFT p95':>9} {'TTFT max':>9} {'e2e p50':>8} {'e2e p95':>8} {'steady':>7}", flush=True)
    for i, n in enumerate(int(x) for x in a.levels.split(",")):
        row, reqs = run_level(a, n, 1_000_000 * (i + 1))
        print(f"{n:>5} {row['ok']:>3}/{row['requests']:<3} {row['wall_s']:>7.1f} {row['agg_out_tps']:>8.1f} "
              f"{row['dec_tps_p50']:>8.1f} {row['dec_tps_p5']:>7.1f} {row['ttft_p50']:>9.2f} {row['ttft_p95']:>9.2f} "
              f"{row['ttft_max']:>9.2f} {row['e2e_p50']:>8.1f} {row['e2e_p95']:>8.1f} "
              f"{row['agg_out_tps_steady'] if row['agg_out_tps_steady'] is not None else '-':>7}"
              + (f"  ERR {row['err_sample']}" if row["errors"] else ""), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        if a.req_out:
            with open(a.req_out, "a") as f:
                f.write(json.dumps(reqs, separators=(",", ":")) + "\n")
        time.sleep(2)


if __name__ == "__main__":
    main()
