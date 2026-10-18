#!/usr/bin/env python3
"""Long-context matrix for one already-running Strata server.

Measures 64K, 128K, 256K and the largest prompt that still leaves room for a
256-token answer. Each length has fresh single requests, a cached resend, two
fresh requests at once, and one cached pair when two snapshots actually parked.
Speeds are the engine's timings. Thinking is off.

A resend counts as cached only when the engine log says the conversation cache
restored it, and at least 90% of the prompt tokens were reused. Before that
resend, one short request is sent to every slot so the prompt is no longer
sitting in a slot. A skipped park is reported as cache unavailable. It is not
averaged into the cached row.

This file is meant to be copied next to a community report. It does not start
or stop the server and it does not read a model config.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
# When this file lives in bench/results/<report>/, the repo tools package is
# four levels up from the file and is passed with --tools. The orchestrator
# passes it. A checkout that keeps this file beside tools/ also works.
for candidate in (TOOLS, TOOLS.parents[3] / "tools" if len(TOOLS.parents) > 3 else None):
    if candidate and (candidate / "strata_bench.py").is_file():
        sys.path.insert(0, str(candidate))
        break

import strata_bench  # noqa: E402
import re

GEN = 256
SLACK = 8
MARGIN = 48
FRESH_REUSED_MAX = 1024
CACHE_FRACTION = 0.90
SHORT_GEN = 16
TARGETS = (65536, 131072, 262144)
VOCAB = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
         "ledger quartz meadow anchor violet thunder").split()
CACHE_EVENT = re.compile(
    r"conversation cache: (parked|skipped|restored) (\d+) tokens.*?parked=(\d+) bytes=(\d+)"
    r"(?: evictions=(\d+))?"
)


def stat(values):
    v = [x for x in values if isinstance(x, (int, float))]
    if not v:
        return None
    return {"median": round(statistics.median(v), 3), "min": round(min(v), 3),
            "max": round(max(v), 3), "n": len(v)}


def log_end(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    return path.stat().st_size


def read_log(path: Path | None, pos: int) -> tuple[str, int]:
    if path is None or not path.is_file():
        return "", pos
    with path.open("rb") as f:
        f.seek(pos)
        data = f.read()
    end = data.rfind(b"\n") + 1
    if end <= 0:
        return "", pos
    return data[:end].decode("utf-8", "replace"), pos + end


def cache_events(text: str) -> list[dict]:
    out = []
    for m in CACHE_EVENT.finditer(text):
        out.append({"kind": m.group(1), "tokens": int(m.group(2)),
                    "parked": int(m.group(3)), "bytes": int(m.group(4)),
                    "evictions": int(m.group(5) or 0)})
    return out


def avail_gib() -> float | None:
    if sys.platform != "win32":
        return None
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                    ("tpage", ctypes.c_ulonglong), ("apage", ctypes.c_ulonglong),
                    ("tvirt", ctypes.c_ulonglong), ("avirt", ctypes.c_ulonglong),
                    ("ext", ctypes.c_ulonglong)]

    ms = MS()
    ms.l = ctypes.sizeof(MS)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
        return None
    return round(ms.avail / 2 ** 30, 2)


def words_for(suite, tokens: int, seed: int) -> str:
    n = max(50, int((tokens - suite.overhead - 40) / max(suite.tokens_per_word, 0.5)))
    rng = random.Random(seed)
    body = " ".join(rng.choice(VOCAB) for _ in range(n))
    return (f"Here is a list of words:\n{body}\n\n"
            "Count from 1, one integer per line, and keep going until you are stopped. "
            "Do not summarize the list.")


def messages(suite, tokens: int, run: int, label: str, seed: int) -> list:
    tag = suite.tag(label, run, seed)
    return [{"role": "user", "content": f"[bench {tag}]\n" + words_for(suite, tokens, seed)}]


def classify_fresh(d: dict) -> str:
    if not d.get("ok"):
        return "failed"
    reused = d.get("reused_tokens") or 0
    if reused > FRESH_REUSED_MAX:
        return "contaminated"
    return "fresh"


def classify_cached(d: dict, events: list[dict], parked_kind: str | None) -> str:
    if parked_kind == "skipped":
        return "cache_skipped"
    if not d or not d.get("ok"):
        return "failed"
    prompt = d.get("prompt_tokens") or 0
    reused = d.get("reused_tokens") or 0
    restored = any(e["kind"] == "restored" for e in events)
    high = bool(prompt and reused >= CACHE_FRACTION * prompt)
    if restored and high:
        return "cached"
    if restored:
        return "restored_partial"
    if high:
        return "reused_without_restore"
    return "reread"


def classify_pair_cached(rows: list[dict], events: list[dict]) -> None:
    """A shared log window. High reuse counts as cached only when there is a restore line per such request."""
    highs = 0
    for r in rows:
        prompt = r.get("prompt_tokens") or 0
        reused = r.get("reused_tokens") or 0
        r["_high"] = bool(r.get("ok") and prompt and reused >= CACHE_FRACTION * prompt)
        highs += bool(r["_high"])
    nrest = sum(1 for e in events if e["kind"] == "restored")
    for r in rows:
        high = bool(r.pop("_high", False))
        if not r.get("ok"):
            r["kind"] = "failed"
        elif high and nrest >= highs and nrest >= 1:
            r["kind"] = "cached"
        elif high and nrest >= 1:
            r["kind"] = "restored_ambiguous"
        elif high:
            r["kind"] = "reused_without_restore"
        elif nrest:
            r["kind"] = "restored_partial"
        else:
            r["kind"] = "reread"


def send(suite, msgs, gen: int, timeout: int = 3600) -> dict:
    """One request. If the built prompt is slightly over the window, rebuild is the caller's job."""
    return strata_bench.digest(strata_bench.chat(suite.base, suite.key, msgs, gen, timeout=timeout))


def occupy(suite, n: int, label: str) -> list[dict]:
    """One short new prompt per slot, sent together, so none of them still holds the long prompt."""
    res: dict[int, dict] = {}

    def go(i: int):
        msg = [{"role": "user", "content": f"[occupy {suite.tag(label, i)}] Reply with the single word ok."}]
        res[i] = strata_bench.digest(strata_bench.chat(suite.base, suite.key, msg, SHORT_GEN, timeout=600))

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [res[i] for i in range(n)]


def pair(suite, msgs: list[list], gen: int) -> list[dict]:
    """Send len(msgs) requests together. The barrier is before the HTTP call."""
    res: dict[int, dict] = {}
    barrier = threading.Barrier(len(msgs))

    def go(i: int):
        barrier.wait()
        res[i] = strata_bench.digest(
            strata_bench.chat(suite.base, suite.key, msgs[i], gen, timeout=3600))

    t0 = time.perf_counter()
    threads = [threading.Thread(target=go, args=(i,)) for i in range(len(msgs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    rows = [res[i] for i in range(len(msgs))]
    for row in rows:
        row["pair_wall_s"] = round(wall, 3)
    return rows


def overlap(rows: list[dict]) -> str:
    """Whether the slower request was still waiting while the faster one had already finished."""
    good = [r for r in rows if r.get("ok") and r.get("ttft_s") is not None]
    if len(good) < 2:
        return "unknown"
    ordered = sorted(good, key=lambda r: r["ttft_s"])
    fast, slow = ordered[0], ordered[-1]
    fast_done = fast["ttft_s"] + ((fast.get("decode_ms") or 0) / 1000)
    prompt_sum = ((fast.get("prompt_ms") or 0) + (slow.get("prompt_ms") or 0)) / 1000
    decode = "decode overlapped" if slow["ttft_s"] < fast_done else "decode did not overlap"
    if prompt_sum and slow["ttft_s"] >= 0.75 * prompt_sum:
        prefill = "prefill looks serial"
    else:
        prefill = "prefill may have overlapped"
    return f"{prefill}; {decode}"


def low_ram(label: str) -> bool:
    return label == "full-window" and (avail_gib() or 99) < 4


def decode_collapsed(row: dict) -> bool:
    dec = row.get("decode_tps")
    return bool(row.get("ok") and (row.get("generated_tokens") or 0) >= 64
                and isinstance(dec, (int, float)) and dec < 15)


def one_length(suite, log: Path | None, label: str, tokens: int, runs: int, slots: int,
               logfn, stop) -> dict:
    fresh_rows, cached_rows, pairs, cached_pairs = [], [], [], []
    saved: list[list] = []
    confirmed_skip = False
    did_cached_pair = False
    cache_note = ""

    def cached_pair_once():
        nonlocal did_cached_pair, cache_note
        if did_cached_pair or stop["flag"] or len(saved) < 2:
            return
        did_cached_pair = True
        if low_ram(label):
            cache_note = "cached concurrent skipped: available RAM under 4 GiB"
            stop["flag"] = "low RAM"
            logfn(f"  {label}: {cache_note}")
            return
        occupy(suite, slots, f"{label}-flush-cached-pair")
        time.sleep(0.3)
        pos = log_end(log)
        t0 = time.perf_counter()
        rows = pair(suite, [saved[0], saved[1]], GEN)
        wall = time.perf_counter() - t0
        time.sleep(0.5)
        text, _ = read_log(log, pos)
        ev = cache_events(text)
        classify_pair_cached(rows, ev)
        note = overlap(rows)
        cached_pairs.append({"wall_s": round(wall, 3), "requests": rows, "cache_events": ev, "overlap": note})
        logfn(f"  {label} cached pair: wall {wall:.1f}s {note} kinds {[r.get('kind') for r in rows]} "
              f"ttft {[r.get('ttft_s') for r in rows]} pp {[r.get('prompt_tps') for r in rows]}")
        for row in rows:
            if decode_collapsed(row):
                stop["flag"] = f"decode {row.get('decode_tps')} tok/s on {label} cached pair"
                logfn(f"  STOP: {stop['flag']}")
                break

    for run in range(runs):
        if stop["flag"]:
            break
        if low_ram(label):
            logfn(f"  {label}: skipped a fresh run, available RAM is under 4 GiB")
            fresh_rows.append({"ok": False, "err": "skipped: available RAM under 4 GiB", "run": run})
            stop["flag"] = "low RAM"
            break
        aimed = tokens
        msgs = messages(suite, aimed, run, label + "-single", 1000 + run)
        pos = log_end(log)
        d = send(suite, msgs, GEN)
        if d.get("err") and "exceeds the context" in str(d.get("err")) and aimed > 8192:
            aimed -= 8192
            logfn(f"  {label}: prompt did not fit, retrying at {aimed}")
            msgs = messages(suite, aimed, run, label + "-single", 1000 + run)
            pos = log_end(log)
            d = send(suite, msgs, GEN)
        tokens = aimed
        time.sleep(0.5)
        text, _ = read_log(log, pos)
        events = cache_events(text)
        park = next((e["kind"] for e in events if e["kind"] in ("parked", "skipped")), None)
        d.update(run=run, park=park, cache_events=events, kind=classify_fresh(d))
        fresh_rows.append(d)
        if park == "parked" and d.get("kind") == "fresh":
            saved.append(msgs)
        logfn(f"  {label} fresh {run + 1}: prompt {d.get('prompt_tokens')} reused {d.get('reused_tokens')} "
              f"pp {d.get('prompt_tps')} tg {d.get('decode_tps')} gen {d.get('generated_tokens')} park {park}")
        if decode_collapsed(d):
            stop["flag"] = f"decode {d.get('decode_tps')} tok/s on {label}"
            logfn(f"  STOP: {stop['flag']}")
            break
        if confirmed_skip:
            cached_pair_once()
            continue
        occupy(suite, slots, f"{label}-flush-{run}")
        time.sleep(0.3)
        pos = log_end(log)
        d2 = send(suite, msgs, GEN)
        time.sleep(0.5)
        text, _ = read_log(log, pos)
        ev = cache_events(text)
        kind = classify_cached(d2, ev, park)
        d2.update(run=run, kind=kind, cache_events=ev, park=park)
        cached_rows.append(d2)
        logfn(f"  {label} resend {run + 1}: {kind} reused {d2.get('reused_tokens')} "
              f"pp {d2.get('prompt_tps')} tg {d2.get('decode_tps')} ttft {d2.get('ttft_s')}")
        if park == "skipped":
            confirmed_skip = True
        if decode_collapsed(d2):
            stop["flag"] = f"decode {d2.get('decode_tps')} tok/s on {label} resend"
            logfn(f"  STOP: {stop['flag']}")
            break
        cached_pair_once()

    if not did_cached_pair and not cache_note:
        if confirmed_skip:
            cache_note = "cached concurrent not measured: the conversation cache skipped the park"
        elif stop["flag"]:
            cache_note = "cached concurrent not measured: stopped before two snapshots had parked"
        else:
            cache_note = "cached concurrent not measured: fewer than two parked snapshots"
        logfn(f"  {label}: {cache_note}")

    for run in range(runs):
        if stop["flag"]:
            break
        if low_ram(label):
            pairs.append({"run": run, "error": "skipped: available RAM under 4 GiB"})
            stop["flag"] = "low RAM"
            break
        msgs = [messages(suite, tokens, run, label + "-pairA", 2000 + run),
                messages(suite, tokens, run, label + "-pairB", 3000 + run)]
        pos = log_end(log)
        t0 = time.perf_counter()
        rows = pair(suite, msgs, GEN)
        wall = time.perf_counter() - t0
        time.sleep(0.5)
        text, _ = read_log(log, pos)
        ev = cache_events(text)
        note = overlap(rows)
        for row in rows:
            row["kind"] = classify_fresh(row)
        pairs.append({"run": run, "wall_s": round(wall, 3), "requests": rows,
                      "cache_events": ev, "overlap": note})
        logfn(f"  {label} pair {run + 1}: wall {wall:.1f}s {note} "
              f"ttft {[r.get('ttft_s') for r in rows]} pp {[r.get('prompt_tps') for r in rows]}")
        for row in rows:
            if decode_collapsed(row):
                stop["flag"] = f"decode {row.get('decode_tps')} tok/s on {label} pair"
                logfn(f"  STOP: {stop['flag']}")
                break
    return {"target_tokens": tokens, "avail_gib": avail_gib(), "fresh": fresh_rows, "resend": cached_rows,
            "pairs": pairs, "cached_pairs": cached_pairs, "cached_pair_note": cache_note}


def summarize(block: dict) -> dict:
    fresh = [r for r in block["fresh"] if r.get("kind") == "fresh" and r.get("ok")]
    cached = [r for r in block["resend"] if r.get("kind") == "cached" and r.get("ok")]
    pair_reqs = [r for p in block["pairs"] for r in p.get("requests") or [] if r.get("kind") == "fresh" and r.get("ok")]
    cached_pair_reqs = [r for p in block.get("cached_pairs") or [] for r in p.get("requests") or []
                        if r.get("kind") == "cached" and r.get("ok")]
    return {
        "fresh_prompt_tps": stat(r.get("prompt_tps") for r in fresh),
        "fresh_decode_tps": stat(r.get("decode_tps") for r in fresh if (r.get("generated_tokens") or 0) >= 32),
        "fresh_ttft_s": stat(r.get("ttft_s") for r in fresh),
        "fresh_prompt_tokens": stat(r.get("prompt_tokens") for r in fresh),
        "fresh_reused": stat(r.get("reused_tokens") or 0 for r in fresh),
        "fresh_generated": stat(r.get("generated_tokens") for r in fresh),
        "fresh_drafts_accepted": stat(r.get("drafts_accepted") for r in fresh),
        "cached_prompt_tps": stat(r.get("prompt_tps") for r in cached),
        "cached_decode_tps": stat(r.get("decode_tps") for r in cached if (r.get("generated_tokens") or 0) >= 32),
        "cached_ttft_s": stat(r.get("ttft_s") for r in cached),
        "cached_prompt_tokens": stat(r.get("prompt_tokens") for r in cached),
        "cached_reused": stat(r.get("reused_tokens") for r in cached),
        "cached_n": len(cached),
        "resend_kinds": [r.get("kind") for r in block["resend"]],
        "pair_prompt_tps": stat(r.get("prompt_tps") for r in pair_reqs),
        "pair_decode_tps": stat(r.get("decode_tps") for r in pair_reqs if (r.get("generated_tokens") or 0) >= 32),
        "pair_ttft_s": stat(r.get("ttft_s") for r in pair_reqs),
        "pair_wall_s": stat(p.get("wall_s") for p in block["pairs"]),
        "pair_overlap": [p.get("overlap") for p in block["pairs"]],
        "pair_prompt_tokens": stat(r.get("prompt_tokens") for r in pair_reqs),
        "pair_reused": stat(r.get("reused_tokens") or 0 for r in pair_reqs),
        "pair_generated": stat(r.get("generated_tokens") for r in pair_reqs),
        "cached_pair_prompt_tps": stat(r.get("prompt_tps") for r in cached_pair_reqs),
        "cached_pair_decode_tps": stat(r.get("decode_tps") for r in cached_pair_reqs
                                       if (r.get("generated_tokens") or 0) >= 32),
        "cached_pair_ttft_s": stat(r.get("ttft_s") for r in cached_pair_reqs),
        "cached_pair_prompt_tokens": stat(r.get("prompt_tokens") for r in cached_pair_reqs),
        "cached_pair_reused": stat(r.get("reused_tokens") for r in cached_pair_reqs),
        "cached_pair_generated": stat(r.get("generated_tokens") for r in cached_pair_reqs),
        "cached_pair_n": len(cached_pair_reqs),
        "cached_pair_kinds": [r.get("kind") for p in block.get("cached_pairs") or [] for r in p.get("requests") or []],
        "cached_pair_overlap": [p.get("overlap") for p in block.get("cached_pairs") or []],
        "cached_pair_note": block.get("cached_pair_note") or "",
    }


def run(url: str, engine_log: str | None, out: Path, runs: int = 3) -> dict:
    base = url.rstrip("/")
    key = ""
    pre = strata_bench.collect_machine(base, key, None)
    ctx = int(pre.get("context") or 0)
    slots = int(pre.get("parallel_slots") or 2)
    suite = strata_bench.Suite(base, key, ctx, runs, False, log=print)
    suite.calibrate()
    full = max(4096, ctx - SLACK - GEN - MARGIN) if ctx else 0
    plan = [(str(n), n) for n in TARGETS if not ctx or n + GEN + SLACK <= ctx]
    if full and full > TARGETS[-1]:
        plan.append(("full-window", full))
    stop = {"flag": ""}
    lengths = {}
    log = Path(engine_log) if engine_log else None
    for label, tokens in plan:
        print(f"length {label} target {tokens} avail_gib {avail_gib()}", flush=True)
        try:
            block = one_length(suite, log, label, tokens, runs, max(2, slots), print, stop)
        except Exception as exc:  # noqa: BLE001 - keep the lengths already measured
            stop["flag"] = f"error {type(exc).__name__}: {exc}"
            print(f"  STOP: {stop['flag']}", flush=True)
            block = {"target_tokens": tokens, "fresh": [], "resend": [], "pairs": [], "cached_pairs": [],
                     "cached_pair_note": stop["flag"], "summary": {}}
        else:
            block["summary"] = summarize(block)
        lengths[label] = block
        result = _payload(ctx, slots, stop["flag"], lengths)
        _write(out, result)
        print(f"wrote {out} through {label}", flush=True)
        if stop["flag"]:
            break
    result = _payload(ctx, slots, stop["flag"], lengths)
    _write(out, result)
    print(f"wrote {out}", flush=True)
    return result


def _payload(ctx, slots, stop, lengths) -> dict:
    packed = {}
    for k, v in lengths.items():
        packed[k] = {"target_tokens": v.get("target_tokens"), "avail_gib": v.get("avail_gib"),
                     "summary": v.get("summary") or {}, "fresh": v.get("fresh") or [],
                     "resend": v.get("resend") or [], "pairs": v.get("pairs") or [],
                     "cached_pairs": v.get("cached_pairs") or [],
                     "cached_pair_note": v.get("cached_pair_note") or ""}
    return {"schema": "long-matrix-1", "context": ctx, "slots": slots, "gen": GEN,
            "avail_gib_end": avail_gib(), "stop": stop, "lengths": packed}


def _write(out: Path, result: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    tmp.replace(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--engine-log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args(argv)
    run(a.url, a.engine_log, Path(a.out), a.runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
