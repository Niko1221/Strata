"""tools/kv_stream_sweep.py - KV streaming counters and speed across context lengths.

Drives a running Strata server with conversations of growing length and reads the engine log's
per-request lines - the prompt/generation speeds and the cumulative KV streaming counters
("KV streaming: H% of B block reads hit VRAM, M MiB read from RAM") - differencing snapshots of
the cumulative counters taken before and after each request (the engine reprints its summary
several times per request, so lines are diffed per segment, not paired). One row per request:

    length  round  kind   prompt tok/s  gen tok/s  seg hit%  seg blk reads  MiB from RAM  KiB/token

    python tools/kv_stream_sweep.py --log engine.log --lengths 16k,32k,64k,128k,262k
    python tools/kv_stream_sweep.py --log engine.log --lengths 64k --rounds 3 --gen-tokens 512

Each round reads a fresh haystack (filler rotated per length+round, so rounds do not reuse each
other's prefix) with a tiny answer, then continues the same conversation with a long-generation
request - so the generation speed is measured with the full context resident and the counter
deltas belong to decode, not to reading the prompt.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("needle_bench", ROOT / "tools" / "needle_bench.py")
_nb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nb)          # for haystack() and CHARS_PER_TOKEN only; no side effects

PROMPT_RE = ("strata serve: prompt ", " tokens = ", " reused + ", " read in ", " ms (", " tok/s), ",
             " generated in ", " ms (", " tok/s")


def parse_prompt(line: str):
    """(prompt_tok, reused, read, read_tok_s, gen, gen_ms, gen_tok_s, drafts_ok, drafts_all) or None."""
    p, s = PROMPT_RE[0], line
    fields = {}
    try:
        i = s.index(p) + len(p)
        nxt = [PROMPT_RE[1], PROMPT_RE[2], PROMPT_RE[3], PROMPT_RE[4], PROMPT_RE[5], PROMPT_RE[6], PROMPT_RE[7]]
        vals = []
        for sep in nxt:
            j = s.index(sep, i)
            vals.append(s[i:j])
            i = j + len(sep)
        j = s.index(" tok/s", i)
        vals.append(s[i:j])
        (prompt_tok, reused, read, read_ms, read_tok_s, gen, gen_ms, gen_tok_s) = vals
        d = s.find("drafts accepted", i)
        if d >= 0:
            tail = s[d + len("drafts accepted"):].split()
            drafts_ok, drafts_all = int(tail[0]), int(tail[2].rstrip(","))
        else:
            drafts_ok = drafts_all = None
        return {"prompt_tok": int(prompt_tok), "reused": int(reused), "read": int(read),
                "read_tok_s": float(read_tok_s), "gen": int(gen), "gen_ms": int(gen_ms),
                "gen_tok_s": float(gen_tok_s), "drafts_ok": drafts_ok, "drafts_all": drafts_all}
    except (ValueError, IndexError):
        return None


def parse_kv(line: str):
    """(hit_pct, blocks, mib) or None."""
    k = "strata serve: KV streaming: "
    if k not in line:
        return None
    try:
        body = line[line.index(k) + len(k):]
        pct, rest = body.split("% of ", 1)
        blocks, rest = rest.split(" block reads hit VRAM, ", 1)
        mib = rest.split(" MiB", 1)[0]
        return {"hit_pct": float(pct), "blocks": int(blocks), "mib": float(mib)}
    except (ValueError, IndexError):
        return None


class LogFollower:
    """Incremental parser for the engine log: bundles each KV streaming line with the last prompt line."""

    def __init__(self, path: Path):
        self.path = path
        self.file = open(path, "r", encoding="utf-8", errors="replace")
        self.pending_prompt = None
        self.rows = []

    def pump(self) -> list:
        out = []
        for line in self.file:
            kv = parse_kv(line)
            if kv:
                if self.pending_prompt:
                    out.append(dict(self.pending_prompt, **kv))
                    self.pending_prompt = None
                continue
            p = parse_prompt(line)
            if p:
                self.pending_prompt = p
        return out


def chat(url: str, key: str, messages: list, max_tokens: int, timeout: float) -> dict:
    body = {"model": "strata", "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": messages}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def sweep_rows(all_rows: list) -> list:
    """Collapse a whole-log row list into one row per request, with per-request counter deltas.

    The engine reprints its (cumulative) summary several times per request; consecutive rows that
    share (prompt_tok, reused, read, gen) belong to one request. Each request's segment delta is
    its last cumulative counters minus the previous request's last cumulative counters.
    """
    requests, prev = [], None
    for r in all_rows:
        key = (r["prompt_tok"], r["reused"], r["read"], r["gen"])
        if requests and requests[-1]["key"] == key:
            requests[-1]["last"] = r
        else:
            requests.append({"key": key, "last": r})
    out = []
    for req in requests:
        r = req["last"]
        if prev is None:
            prev = r
            continue
        db = r["blocks"] - prev["blocks"]
        dm = r["mib"] - prev["mib"]
        dmiss = (r["blocks"] * (1 - r["hit_pct"] / 100)) - (prev["blocks"] * (1 - prev["hit_pct"] / 100))
        if db < 0:                              # a new conversation reset the counters: this request owns its whole sum
            db, dm = r["blocks"], r["mib"]
            dmiss = r["blocks"] * (1 - r["hit_pct"] / 100)
        prev = r
        tok = r["gen"] if r["gen"] > 32 else (r["read"] or r["gen"])
        out.append({"prompt_tok": r["prompt_tok"], "reused": r["reused"], "read": r["read"], "gen": r["gen"],
                    "read_tok_s": r["read_tok_s"], "gen_tok_s": r["gen_tok_s"],
                    "drafts_ok": r["drafts_ok"], "drafts_all": r["drafts_all"],
                    "seg_blocks": db, "seg_mib": round(dm, 2), "seg_misses": int(dmiss),
                    "seg_hit_pct": round(100 * (1 - dmiss / db), 3) if db else None,
                    "kib_per_token": round(dm * 1024 / tok, 2) if tok else None})
    return out


def print_rows(rows: list) -> None:
    print(f"\n{'ctx tok':>8} {'read':>6} {'gen':>5} {'prompt t/s':>10} {'gen t/s':>8} {'drafts':>9} "
          f"{'seg hit%':>9} {'seg blk reads':>14} {'MiB RAM':>9} {'KiB/tok':>8}")
    for r in rows:
        drafts = f"{r['drafts_ok']}/{r['drafts_all']}" if r["drafts_ok"] is not None else "-"
        hit = f"{r['seg_hit_pct']:.2f}" if r["seg_hit_pct"] is not None else "-"
        kit = f"{r['kib_per_token']:.2f}" if r["kib_per_token"] is not None else "-"
        print(f"{r['prompt_tok']:>8,} {r['read']:>6} {r['gen']:>5} {r['read_tok_s']:>10.1f} {r['gen_tok_s']:>8.1f} "
              f"{drafts:>9} {hit:>9} {r['seg_blocks']:>14,} {r['seg_mib']:>9.1f} {kit:>8}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--log", required=True, help="the engine log the server writes (counters are read from it)")
    ap.add_argument("--lengths", default="16k,32k,64k,128k,262k")
    ap.add_argument("--rounds", type=int, default=2, help="fresh-haystack rounds per length")
    ap.add_argument("--gen-tokens", type=int, default=384, help="generation length of the decode request")
    ap.add_argument("--timeout", type=float, default=2400)
    ap.add_argument("--parse-only", action="store_true",
                    help="do not drive requests; parse --log (whole file) and print the table")
    ap.add_argument("--out", help="also write the rows as JSON")
    a = ap.parse_args()

    if a.parse_only:
        follower = LogFollower(Path(a.log))
        rows = sweep_rows(follower.pump())
        print_rows(rows)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rows, indent=1))
        return 0

    try:
        with urllib.request.urlopen(a.url.rstrip("/") + "/metrics", timeout=10) as r:
            ctx = int((json.loads(r.read()).get("engine") or {}).get("max_context") or 0)
    except (OSError, ValueError):
        ctx = 0
        print("warning: server not reachable on " + a.url, file=sys.stderr)

    follower = LogFollower(Path(a.log))
    rows = []
    cum = {"blocks": 0, "mib": 0.0, "misses": 0.0}
    started = False

    def snapshot(kind: str, length: str, rnd: int):
        nonlocal cum, started
        got = follower.pump()
        if not got:
            return
        last = got[-1]
        if not started:                       # first snapshot only anchors the baseline
            cum = {"blocks": last["blocks"], "mib": last["mib"],
                   "misses": last["blocks"] * (1 - last["hit_pct"] / 100)}
            started = True
            return
        blocks, mib = last["blocks"], last["mib"]
        misses = blocks * (1 - last["hit_pct"] / 100)
        db, dm, dmiss = blocks - cum["blocks"], mib - cum["mib"], misses - cum["misses"]
        cum = {"blocks": blocks, "mib": mib, "misses": misses}
        tok = last["read"] if kind == "read" else last["gen"]
        rows.append({"length": length, "round": rnd, "kind": kind,
                     "prompt_tok": last["prompt_tok"], "read": last["read"], "gen": last["gen"],
                     "read_tok_s": last["read_tok_s"], "gen_tok_s": last["gen_tok_s"],
                     "drafts_ok": last["drafts_ok"], "drafts_all": last["drafts_all"],
                     "seg_blocks": db, "seg_mib": round(dm, 2), "seg_misses": int(dmiss),
                     "seg_hit_pct": round(100 * (1 - dmiss / db), 3) if db else None,
                     "kib_per_token": round(dm * 1024 / tok, 2) if tok else None})

    for L in a.lengths.split(","):
        tokens = int(float(L.lower().rstrip("k")) * 1024) if L.lower().endswith("k") else int(L)
        tokens = int(tokens * 0.98)
        if ctx and tokens + a.gen_tokens + 600 > ctx:
            print(f"{L:>5}: skipped (the server's context is {ctx})")
            continue
        for rnd in range(a.rounds):
            text = _nb.haystack(int(tokens * _nb.CHARS_PER_TOKEN))
            cut = (tokens * 7919 + rnd * 104729) % max(len(text) - 4096, 1)
            text = text[cut:] + text[:cut]     # rotate: this round shares no long prefix with earlier ones
            question = ("\n\nSummarize the section about '{q}' in one short sentence, then reply READY."
                        ).format(q=["the engine", "the server", "the kernels", "the setup", "the bench"][rnd % 5])
            messages = [{"role": "user", "content": text + question}]

            chat(a.url, a.api_key, messages, 24, a.timeout)
            snapshot("read", L, rnd)

            follow = ("Continue this conversation. In detail (about 300 words), explain how the streaming "
                      "of a large cache works; do not ask questions.")
            messages += [{"role": "assistant", "content": "READY."},
                         {"role": "user", "content": follow}]
            chat(a.url, a.api_key, messages, a.gen_tokens, a.timeout)
            snapshot("decode", L, rnd)

    for _ in range(20):                        # let the log catch up with the last request
        if not follower.pump():
            break
        time.sleep(0.5)

    print_rows(rows)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))
        print("\nrows -> " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
