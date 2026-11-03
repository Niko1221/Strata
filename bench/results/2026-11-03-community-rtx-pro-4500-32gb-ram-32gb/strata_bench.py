#!/usr/bin/env python3
"""Strata A/B benchmark with natural text, read from the engine's own request records.

usage: strata_bench.py LABEL OUT.jsonl --docs-dir DIR [--prefill-runs N] [--decode-reps N] [--port 8080]

Prefill: ~7K-token slices of real documentation, each starting with a unique salt line so nothing is reused from the
conversation cache. Decode: three fixed prompt types (prose, code, JSON), greedy. A run is kept only if the engine's
request list shows our request and nothing else overlapping it (a foreign request would contaminate the numbers).
"""
import json, os, random, sys, time, urllib.request

PORT = 8080
DOCS = None   # set with --docs-dir: a folder with the repository's docs/*.md files
MODEL = "qwen3.8-flash-next-iq3_xxs"


def http(path, body=None, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def metrics():
    return http("/metrics", timeout=5)


def wait_idle(quiet_s=60, limit_s=1800):
    """Idle = nothing running and the newest request finished at least quiet_s ago."""
    t0 = time.time()
    while time.time() - t0 < limit_s:
        m = metrics()
        newest_end = max([(r["time"] + (r.get("duration_s") or 0)) for r in m["requests"][:3]] or [0])
        if (m["live"].get("running") or 0) == 0 and m["time"] - newest_end >= quiet_s:
            return True
        time.sleep(5)
    return False


def pool():
    text = ""
    for f in ("DETAILS.md", "MODELS.md", "HOW_IT_WORKS.md", "TERMINAL.md", "TERMINAL_LAUNCH.md", "README.md", "TROUBLESHOOTING.md"):
        p = os.path.join(DOCS, f)
        if os.path.exists(p):
            text += open(p, errors="replace").read() + "\n"
    return text.splitlines(keepends=True)


def prefill_prompt(rnd, lines, chars=26000):
    start = rnd.randrange(0, max(1, len(lines) - 400))
    out, n, i = [], 0, start
    while n < chars and i < len(lines):
        out.append(lines[i]); n += len(lines[i]); i += 1
    salt = "%016x" % rnd.getrandbits(64)
    return f"Reference id {salt}.\n\n" + "".join(out) + "\n\nIn one short sentence, what is the main topic of the text above?"


DECODE_PROMPTS = {
    "prose": "Write a detailed, well-structured 900-word explanation of how the internet's DNS resolution works, from a browser typing a URL to the page loading.",
    "code": "Write a complete Python module implementing an LRU cache with per-entry TTL, thread safety, type hints, docstrings, and a pytest test file. Output only code.",
    "json": "Generate a JSON array of 40 fictional customer records with the fields id, name, email, address (street, city, country), signup_date and tags (array of strings). Output only JSON.",
}


def run_one(prompt, max_tokens, temperature=0.0):
    before = metrics()
    seen = {r["id"] for r in before["requests"]}
    t_start = time.time()
    http("/v1/chat/completions", {"model": MODEL, "max_tokens": max_tokens, "temperature": temperature,
                                  "messages": [{"role": "user", "content": prompt}]})
    t_end = time.time()
    time.sleep(1.0)
    after = metrics()
    new = [r for r in after["requests"] if r["id"] not in seen]
    if len(new) != 1:
        return None, f"{len(new)} new requests in the engine list (foreign traffic?)"
    r = new[0]
    if (after["live"].get("running") or 0) != 0:
        return None, "engine still busy afterwards"
    return r, None


def main():
    global PORT, DOCS
    label, out = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    DOCS = args[args.index("--docs-dir") + 1]
    pf_runs = int(args[args.index("--prefill-runs") + 1]) if "--prefill-runs" in args else 4
    dc_reps = int(args[args.index("--decode-reps") + 1]) if "--decode-reps" in args else 2
    if "--port" in args:
        PORT = int(args[args.index("--port") + 1])
    rnd = random.Random(os.urandom(8))
    lines = pool()
    eng = metrics()["engine"]
    meta = {k: eng.get(k) for k in ("pcie_frac", "spec_min_p", "batch_slots", "expert_slots", "drafting", "vram_free_mib", "spec", "mtp_max")}
    print(f"[{label}] engine {meta}", flush=True)
    results = []
    with open(out, "a") as f:
        def record(kind, name, r):
            row = {"label": label, "kind": kind, "name": name, "t": time.time(), "meta": meta,
                   "prompt_read": r.get("prompt_read"), "prompt_tokens": r["prompt_tokens"], "reused": r.get("reused"),
                   "output_tokens": r["output_tokens"], "ttft_s": r.get("first_engine_token_s"), "decode_tok_s": r.get("decode_tok_s"),
                   "prompt_ms": r.get("prompt_ms"), "drafts_offered": r.get("drafts_offered"), "drafts_accepted": r.get("drafts_accepted"),
                   "file_mb": r.get("file_mb"), "hit_rate": r.get("hit_rate")}
            if row["prompt_ms"]:
                row["prefill_tok_s"] = round(row["prompt_read"] / (row["prompt_ms"] / 1000), 1)
            f.write(json.dumps(row) + "\n"); f.flush(); results.append(row)
            return row

        # warm-up (not recorded): one short generation so the first measured request is not a cold start
        if not wait_idle(60):
            print("engine never became idle", flush=True); sys.exit(3)
        run_one("Say hello.", 16)
        for i in range(pf_runs):
            for attempt in range(4):
                if not wait_idle(30): print("not idle", flush=True); sys.exit(3)
                r, err = run_one(prefill_prompt(rnd, lines), 8)
                if r: break
                print(f"  prefill run {i+1} discarded: {err}", flush=True)
            else:
                continue
            row = record("prefill", f"p{i+1}", r)
            print(f"  prefill {i+1}: read {row['prompt_read']} tok  {row.get('prefill_tok_s')} tok/s  first token {row['ttft_s']}s", flush=True)
        for rep in range(dc_reps):
            for name, prompt in DECODE_PROMPTS.items():
                for attempt in range(4):
                    if not wait_idle(30): print("not idle", flush=True); sys.exit(3)
                    r, err = run_one(prompt, 1200)
                    if r: break
                    print(f"  decode {name} discarded: {err}", flush=True)
                else:
                    continue
                row = record("decode", name, r)
                acc = (row["drafts_accepted"] / row["drafts_offered"]) if row.get("drafts_offered") else None
                print(f"  decode {name}: {row['output_tokens']} tok  {row['decode_tok_s']} tok/s  drafts accepted {None if acc is None else round(acc, 2)}", flush=True)
    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return xs[len(xs) // 2] if xs else None
    pf = [r["prefill_tok_s"] for r in results if r["kind"] == "prefill"]
    print(f"[{label}] SUMMARY prefill median {med(pf)} tok/s (n={len(pf)}, min {min(pf) if pf else None}, max {max(pf) if pf else None})", flush=True)
    for name in DECODE_PROMPTS:
        d = [r["decode_tok_s"] for r in results if r["kind"] == "decode" and r["name"] == name]
        print(f"[{label}] SUMMARY decode {name}: median {med(d)} tok/s (n={len(d)})", flush=True)
    alld = [r["decode_tok_s"] for r in results if r["kind"] == "decode"]
    print(f"[{label}] SUMMARY decode all: median {med(alld)} tok/s", flush=True)


if __name__ == "__main__":
    main()
