#!/usr/bin/env python3
"""RES-20261029-01 G1.3 (TASK-20261029-04), reused by G2 (TASK-20261029-06): decode microbench for Strata on v100host.
Standard library only. G2 adds the batch path's window breakdown (the serve shape decodes through it).

Two fixed tasks (code, prose) at each context size, temperature 0, thinking off, 512 output tokens. One pass per engine
start: repeats are restarts, so no request reuses the prompt cache and every arm starts from the same expert profile.
The prompt text depends only on (context, task), never on the arm, so output hashes compare across arms.

Per request: client TTFT and decode tok/s ((tokens - 1) / (total - TTFT)), the output's sha256, and the engine's own
per-request lines (prompt and generation tok/s, drafts accepted, decode expert-cache hit rate, PCIe share) taken from
the bytes the engine log gained during that request; those bytes are also saved raw next to the output.

usage: decode-bench.py --base http://127.0.0.1:8081/v1 --model ID --log ENGINE_LOG --ctx 4096,16384
                       --arm NAME --rep N --out DIR
"""
import argparse, hashlib, json, os, random, re, time, urllib.request

WORDS = ("amber basin cedar delta ember fjord garnet harbor iris juniper kestrel lumen meadow nimbus orchid pylon "
         "quartz raven sierra tundra umber vessel willow xenon yarrow zephyr").split()
TOKENS_PER_WORD = 4.6        # these "word123" fillers: v41bench's 2,000 target at 0.55 words/token gave 5,075 tokens
TASKS = {
    "code": "Ignore the notes above; they are filler. Write a complete, well-commented Python module that implements an "
            "LRU cache class with get, put, delete, resize and stats methods, followed by unittest tests for every "
            "method. Write all of the code.",
    "prose": "Ignore the notes above; they are filler. Write a long essay, at least 700 words with headed sections, on "
             "how lighthouses were built, lit and kept running in the nineteenth century.",
}
PROMPT_RE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated "
                       r"in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
HIT_RE = re.compile(r"decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\); (\d+) more read by the GPU "
                    r"over PCIe or from another GPU \(([\d.]+)% of all (\d+) routed\)")


BATCH_RE = re.compile(r"strata batch: (\d+) windows, avg ([\d.]+) rows, ([\d.]+) ms/window = run ([\d.]+) \(CUDA0 GPU-reach "
                      r"wait ([\d.]+) \+ CPU experts ([\d.]+)\).*?adapt wait ([\d.]+);.*?routed entries per window: VRAM ([\d.]+) "
                      r"\(([\d.]+)%\), PCIe ([\d.]+) \(([\d.]+)%\), CPU ([\d.]+) \(([\d.]+)%\); adaptive tier: (\d+) rounds, (\d+) "
                      r"experts swapped in")


def prompt_for(ctx, task):
    rnd = random.Random(ctx * 1000 + len(task))
    n = max(0, int((ctx - 120) / TOKENS_PER_WORD))
    filler = " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" for _ in range(n))
    return f"[g13 v1 ctx{ctx} {task}] Notes: {filler}\n\n{TASKS[task]}"


def chat(base, model, prompt, max_tokens, timeout=600):     # a 128K prompt takes ~100 s; a stuck window fails fast
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning") or d.get("reasoning_content") or "")
                if piece:
                    ttft = ttft if ttft is not None else time.time() - t0
                    text.append(piece)
    total = time.time() - t0
    ct = (usage or {}).get("completion_tokens", 0)
    dec = (ct - 1) / (total - ttft) if ttft is not None and ct > 1 and total > ttft else None
    return "".join(text), {"ttft_s": ttft, "total_s": round(total, 3), "prompt_tokens": (usage or {}).get("prompt_tokens"),
                           "completion_tokens": ct, "decode_tok_s": round(dec, 2) if dec else None}


def log_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def log_since(path, off):
    time.sleep(0.5)                                   # the engine prints its summary just after the last token
    with open(path, "rb") as f:
        f.seek(off)
        return f.read().decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8081/v1"); ap.add_argument("--model", required=True)
    ap.add_argument("--log", required=True); ap.add_argument("--ctx", default="4096,16384")
    ap.add_argument("--tasks", default="code,prose"); ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--arm", required=True); ap.add_argument("--rep", type=int, default=1); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    chat(a.base, a.model, "[g13 warmup] Name three colours.", 32)            # same warm-up in every arm
    rows = []
    for ctx in [int(x) for x in a.ctx.split(",")]:
        for task in a.tasks.split(","):
            off = log_size(a.log)
            started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            text, m = chat(a.base, a.model, prompt_for(ctx, task), a.max_tokens)
            chunk = log_since(a.log, off)
            tag = f"{a.arm}-r{a.rep}-ctx{ctx}-{task}"
            open(os.path.join(a.out, f"out-{tag}.txt"), "w", encoding="utf-8").write(text)
            open(os.path.join(a.out, f"enginelog-{tag}.txt"), "w", encoding="utf-8").write(chunk)
            row = {"arm": a.arm, "rep": a.rep, "ctx": ctx, "task": task, "started": started, **m,
                   "sha256": hashlib.sha256(text.encode()).hexdigest()}
            p = PROMPT_RE.findall(chunk); h = HIT_RE.findall(chunk)
            if p:
                g = p[-1]
                row.update({"eng_prompt_tokens": int(g[0]), "eng_reused": int(g[1]), "eng_prompt_tok_s": float(g[4]),
                            "eng_generated": int(g[5]), "eng_gen_ms": int(g[6]), "eng_gen_tok_s": float(g[7]),
                            "drafts_accepted": int(g[8]), "drafts_offered": int(g[9])})
            if h:
                g = h[-1]
                row.update({"hit_pct": float(g[0]), "hits": int(g[1]), "lookups": int(g[2]), "pcie_reads": int(g[3]),
                            "pcie_pct": float(g[4]), "routed": int(g[5]),
                            "vram_share_of_routed": round(int(g[1]) / int(g[5]), 4) if int(g[5]) else None})
            b = BATCH_RE.findall(chunk)
            if b:
                g = b[-1]                                 # the request's last summary
                row.update({"batch_lines": len(b), "win_n": int(g[0]), "win_rows": float(g[1]), "win_ms": float(g[2]),
                            "win_gpu_wait_ms": float(g[4]), "win_cpu_experts_ms": float(g[5]), "win_adapt_wait_ms": float(g[6]),
                            "routed_vram_pct": float(g[8]), "routed_pcie_pct": float(g[10]), "routed_cpu_pct": float(g[12]),
                            "adapt_rounds": int(g[13]), "adapt_swapped": int(g[14])})
            rows.append(row)
            print(json.dumps(row), flush=True)
    with open(os.path.join(a.out, "results.jsonl"), "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
