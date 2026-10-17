"""Request driver for this report: sends one plan's requests to a running Strata server, one at a time, and reads
each request's numbers from the engine's own log lines.

It does not start or stop the engine. Start the arm (the launch commands are in README.md), then run e.g.

    python bench.py --plan pw --arm next8-pw-on --url http://<host>:<port> --engine-log <engine log file> \
        --text long300k.txt --out runs.jsonl

Plans (the request bodies are the ones this report's numbers came from; see README.md, "Method"):
    pw  the pipelined-windows A/B: 9 requests, three of them one conversation that grows to 301,147 tokens
    ab  the prompt-path A/Bs (MMQ patch, STRATA_PREFILL_PIPE): 5 requests
    pf  the STRATA_PREFILL_TIMING profile: 3 requests
    mp  the draft-floor sweep: 3 prompts x 4 spec_min_p values x 2 rounds, one-shot requests
Every plan except mp starts with the same two warm-up requests, which are not recorded.

Numbers are parsed from the engine log written while each request ran (the per-request "strata serve: prompt"
line, and the "strata prefill", "strata decode timing" and "strata pipeline" lines when the engine prints them).
Decode speed is the engine's, never generated tokens / request time. client_latency_s is the client-side time of
the whole non-streaming request. TTFT is not measured (requests are not streamed).
--dry-run prints the request bodies' sizes and hashes instead of sending them.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_prompts import slices  # noqa: E402

Q = "\n\nIn two sentences: what is the code above about?"
CODE = ("Write a Python function that parses a CSV file with a header row into a list of dicts, converting numeric fields "
        "to int or float, and include three pytest tests for it.")
LRU = ("Write a small Python module with a class LRUCache (get, put, capacity eviction), full type hints and docstrings, "
       "then pytest tests for it.")
PENS = ("A shop sells pens at 3 for $4 and notebooks at $2.50 each. Mia buys 7 pens and 4 notebooks and pays with a $20 "
        "bill. How much change does she get? Explain step by step, then check the answer another way.")
TRAIN = ("A train leaves at 9:40 and travels 210 km at 84 km/h, then waits 12 minutes, then travels 95 km at 76 km/h. "
         "When does it arrive? Explain step by step.")
FOLLOW = "Now list five functions defined in that code and say in one line what each does."
LONGEST = "Which of those five functions is the longest, and why?"
ADD = "More of the same file:\n{more}\n\nWhat does this part add?"
MINP_VALUES = [0.85, 0.5, 0.7, 0.95]

LINE = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                  r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)(?:, drafts accepted (\d+) of (\d+))?")
CHUNKS = re.compile(r"strata prefill: (\d+) tokens in (\d+)-token chunks")
DT = re.compile(r"decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = verify ([\d.]+) "
                r".*?stage ([\d.]+)\) \+ commit/emit ([\d.]+) \+ draft ([\d.]+)")
PL = re.compile(r"strata pipeline: (\d+) windows in (\d+) ms \(([\d.]+) ms/window\): (\d+) speculative, (\d+) on the path, "
                r"(\d+) rolled back, (\d+) below the gate \(theta ([\d.]+)\)")
PLC = re.compile(r"strata pipeline classes: fresh (\d+) windows ([\d.]+) ms ([\d.]+) tok \| speculative (\d+) windows "
                 r"([\d.]+) ms ([\d.]+) tok \| forced-chain disagreements (\d+), chain late (\d+)")
PLK = re.compile(r"strata pipeline calibration \(p_on decile: on/scored\): (.*)")


def parse(fresh):
    """The numbers of one request from the engine log text it produced (the last match of each line kind)."""
    row = {}
    m = LINE.findall(fresh)
    if m:
        m = m[-1]
        row.update({"prompt_tokens": int(m[0]), "reused_tokens": int(m[1]), "read_tokens": int(m[2]),
                    "read_ms": int(m[3]), "prompt_tok_s": float(m[4]), "generated_tokens": int(m[5]),
                    "decode_ms": int(m[6]), "decode_tok_s": float(m[7]),
                    "drafts_accepted": int(m[8]) if m[8] else None, "drafts_offered": int(m[9]) if m[9] else None})
    row["batched_chunks"] = [[int(a), int(b)] for a, b in CHUNKS.findall(fresh)]
    d = DT.findall(fresh)
    if d:
        d = d[-1]
        row["decode_timing"] = {"windows": int(d[0]), "avg_T": float(d[1]), "tokens_per_window": float(d[2]),
                                "ms_per_window": float(d[3]), "verify_ms": float(d[4]), "stage_ms": float(d[5]),
                                "commit_emit_ms": float(d[6]), "draft_ms": float(d[7])}
    p, c, k = PL.findall(fresh), PLC.findall(fresh), PLK.findall(fresh)
    if p and c:
        p, c = p[-1], c[-1]
        row["pipeline"] = {
            "windows": int(p[0]), "decode_ms": int(p[1]), "ms_per_window": float(p[2]), "launched_guesses": int(p[3]),
            "held": int(p[4]), "rolled_back": int(p[5]), "below_gate": int(p[6]), "theta": float(p[7]),
            "fresh_windows": int(c[0]), "fresh_mean_ms": float(c[1]), "fresh_tokens_per_window": float(c[2]),
            "held_windows": int(c[3]), "held_mean_ms": float(c[4]), "held_tokens_per_window": float(c[5]),
            "forced_chain_disagreements": int(c[6]), "chain_late": int(c[7]),
            "calibration_held_of_scored": {a: b for a, b in (x.split(":") for x in k[-1].split())} if k else None}
    return row


def sha1_pipe(content, reasoning):
    return hashlib.sha1((content + "|" + reasoning).encode()).hexdigest()[:12]


def sha256_nul(content, reasoning):
    return hashlib.sha256((reasoning + "\x00" + content).encode()).hexdigest()[:12]


def plan(name, text):
    """(label, kind, payload, max_tokens, extra body fields). kind 'new' starts a one-message request, 'conv'
    appends a user turn to the running conversation (the previous reply's content is appended before it)."""
    s = slices(text)
    S1K, S30K, MORE = s["S1K"], s["S30K"], s["MORE"]
    if name == "pw":
        return [("R_code", "new", "[pc] " + CODE, 256, {}),
                ("R_reason", "new", "[pc] " + PENS, 256, {}),
                ("R_code512", "new", "[pc] " + LRU, 512, {}),
                ("A0_1k_fresh", "new", "[pc] A0\n" + S1K + Q, 64, {}),
                ("A_30k_fresh", "new", "[pc] A\n" + S30K + Q, 64, {}),
                ("B_300k_fresh", "conv", "[pc] B\n" + text + Q, 64, {}),
                ("C_decode_at300k", "conv", FOLLOW, 256, {}),
                ("E_other", "new", "[pc] E\n" + PENS, 64, {}),
                ("F_back300k", "conv", LONGEST, 128, {})]
    if name in ("ab", "pf"):
        t = "[" + name + "]"
        rows = [("code256", "new", t + " " + CODE, 256, {}),
                ("read1k", "new", t + " A0\n" + S1K + Q, 64, {}),
                ("read30k", "conv", t + " A\n" + S30K + Q, 64, {}),
                ("inc1k_at30k", "conv", ADD.format(more=MORE), 64, {}),
                ("reason256", "new", t + " " + PENS, 256, {})]
        return rows if name == "ab" else rows[1:4]
    if name == "mp":
        out = []
        prompts = [("code256", "[mp] " + CODE, 256), ("code512", "[mp] " + LRU, 512), ("reason", "[mp] " + TRAIN, 256)]
        for rnd in range(2):
            for label, content, n in prompts:
                for v in (MINP_VALUES if rnd == 0 else list(reversed(MINP_VALUES))):
                    out.append((label, "new", content, n,
                                {"strata_checkpoint": False, "strata_tune": {"spec_min_p": v}, "_round": rnd}))
        return out
    raise SystemExit("unknown plan " + name)


def warmups(name, text):
    if name == "mp":
        return [("[mp warm-up] Say hello.", 8, {"strata_checkpoint": False})]
    return [("[warm-up] Say hello.", 16, {}), ("[warm-up 2]\n" + slices(text)["S1K"] + Q, 16, {})]


def body(messages, n, extra):
    b = {"model": "x", "messages": messages, "max_tokens": n, "temperature": 0}
    b.update({k: v for k, v in extra.items() if not k.startswith("_")})
    return b


def post(url, b):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=["pw", "ab", "pf", "mp"])
    ap.add_argument("--arm", required=True, help="a label written into every row")
    ap.add_argument("--url", help="the server's base URL")
    ap.add_argument("--engine-log", help="the engine's log file (its per-request lines are parsed)")
    ap.add_argument("--text", required=True, help="the file make_prompts.py --out wrote")
    ap.add_argument("--out", default="runs.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    with open(a.text, encoding="utf-8", newline="") as f:
        text = f.read()
    if a.dry_run:
        conv = []
        for label, kind, payload, n, extra in plan(a.plan, text):
            msgs = [{"role": "user", "content": payload}] if kind == "new" else conv + [{"role": "user", "content": payload}]
            if kind == "conv":
                conv = msgs + [{"role": "assistant", "content": "<reply>"}]
            b = json.dumps(body(msgs, n, extra))
            print(label, kind, n, len(payload), hashlib.sha256(payload.encode()).hexdigest()[:12], extra or "", len(b))
        return
    if not (a.url and a.engine_log):
        sys.exit("--url and --engine-log are needed unless --dry-run")
    for content, n, extra in warmups(a.plan, text):
        post(a.url, body([{"role": "user", "content": content}], n, extra))
    conv = []
    with open(a.out, "a", encoding="utf-8") as out:
        for i, (label, kind, payload, n, extra) in enumerate(plan(a.plan, text), 1):
            msgs = [{"role": "user", "content": payload}] if kind == "new" else conv + [{"role": "user", "content": payload}]
            pos = os.path.getsize(a.engine_log)
            t0 = time.time()
            resp = post(a.url, body(msgs, n, extra))
            t1 = time.time()
            time.sleep(1.0)  # let the engine finish its log lines for this request
            with open(a.engine_log, "rb") as fh:
                fh.seek(pos)
                fresh = fh.read().decode("utf-8", "replace")
            msg = resp["choices"][0]["message"]
            content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
            if kind == "conv":
                conv = msgs + [{"role": "assistant", "content": content}]
            row = {"arm": a.arm, "plan": a.plan, "order": i, "request": label, "max_tokens": n,
                   "client_latency_s": round(t1 - t0, 1)}
            if a.plan == "mp":
                row.update({"round": extra["_round"], "spec_min_p": extra["strata_tune"]["spec_min_p"],
                            "reply_hash": sha256_nul(content, reasoning), "hash_rule": "sha256(reasoning NUL content)[:12]"})
            else:
                row.update({"reply_hash": sha1_pipe(content, reasoning), "hash_rule": "sha1(content | reasoning)[:12]"})
            row.update(parse(fresh))
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps({k: row.get(k) for k in ("request", "prompt_tokens", "reused_tokens", "prompt_tok_s",
                                                      "decode_tok_s", "reply_hash")}), flush=True)


if __name__ == "__main__":
    main()
