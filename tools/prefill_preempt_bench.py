"""tools/prefill_preempt_bench.py - does the feature actually solve the head-of-line blocking?

Sends a long prompt A (the ~50K-token stand-in for the community's 200K one), queues a small request B once A
is clearly inside its prefill, and measures B's time-to-first-token and A's total wall time - with
--prefill-preempt off, and with it on.  Greedy, --adapt-swaps 0, --pcie-frac 0, --suffix-draft 0, same warm-up
as the parity harness.  The point is the RATIO, not the absolute numbers.

    python3 tools/prefill_preempt_bench.py [--tokens 50000] [--max-new 32]
"""
from __future__ import annotations

import argparse
import faulthandler
import signal

faulthandler.register(signal.SIGUSR1)   # DEBUG: kill -USR1 <pid> dumps every thread's stack
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from prefill_preempt_test import Engine, deterministic_tokens, engine_args  # noqa: E402

CONFIG = ROOT / "strata-iq3_xxs.json"


def one_pass(exe: str, cfg: dict, workdir: Path, name: str, a_ids, b_ids, warm_ids, max_new: int,
             preempt: bool, chunk: int, args_max_context: int) -> dict:
    ctx = args_max_context
    e = Engine(exe, engine_args(cfg, prefill=chunk, preempt=preempt, max_context=ctx), workdir / f"bench-{name}.log")
    try:
        e.gen(None, warm_ids, 8)
        e.collect(None)
        out = {"preempt": preempt}
        e.gen(1, a_ids, max_new)
        t_a0 = time.time()
        b_queued = {"sent": False}

        def on_line(line: str) -> bool:
            if line.startswith("PP ") and not b_queued["sent"]:
                print(f"[bench]   A at {line.split()[1]} tokens ({time.time() - t_a0:.0f} s)", flush=True)
                if int(line.split()[1]) >= 3 * chunk:      # A is three chunks into its prefill: queue B
                    out["a_at_b"] = time.time() - t_a0
                    e.gen(2, b_ids, max_new)
                    if preempt:
                        e.send("YIELD")
                    b_queued["sent"] = True
                    out["b_queued_at"] = time.time()
            return line.startswith("SUSPENDED")

        # A's first leg (returns early on SUSPENDED when preempting)
        a_first = e.collect(1, stop_on=on_line)
        if not b_queued["sent"]:
            raise RuntimeError("A finished before B could be queued")
        if preempt:
            if not a_first.get("stopped"):
                raise RuntimeError("A never parked for B")
        else:
            # no preemption: the first collect ran A to its DONE; B's lines follow
            out["a_total_s"] = time.time() - t_a0
        print("[bench]   collecting B", flush=True)
        b = e.collect(2)
        print("[bench]   B collected", flush=True)
        # B's latency: from the moment it was queued (mid-A's prefill) to its DONE - the wait is the point
        out["b_total_s"] = time.time() - out["b_queued_at"]
        out["b_tokens"] = len(b["tokens"])
        if preempt:
            out["a_total_s"] = None
            # A's remaining: resume and run to the end
            e.send("RESUME id=1")
            t_r0 = time.time()
            a_rest = e.collect(1)
            out["a_total_s"] = time.time() - t_a0
            out["a_resume_s"] = time.time() - t_r0
            out["a_tokens"] = len(a_rest["tokens"])
        return out
    finally:
        e.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default=str(ROOT / "build" / "strata"))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--workdir", default=str(ROOT / "bench" / "results" / "prefill-preempt"))
    ap.add_argument("--tokens", type=int, default=50000)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    a_ids = deterministic_tokens(args.tokens, seed=7)
    b_ids = deterministic_tokens(220, seed=99)
    warm_ids = deterministic_tokens(120, seed=5)

    results = []
    for preempt in (False, True):
        name = "on" if preempt else "off"
        print(f"[bench] pass {name}: A={args.tokens} tokens, B=220 tokens, max_new={args.max_new}", flush=True)
        t0 = time.time()
        r = one_pass(args.engine, cfg, workdir, name, a_ids, b_ids, warm_ids, args.max_new, preempt, args.chunk,
                     args.tokens + 2 * args.max_new + 4096)
        r["pass_s"] = time.time() - t0
        results.append(r)
        print(f"[bench]   {r}", flush=True)
    off, on = results
    if off["b_total_s"] > 0:
        cut = 100.0 * (1.0 - on["b_total_s"] / off["b_total_s"])
        print(f"[bench] B's wait cut by {cut:.0f}% ({off['b_total_s']:.1f} s -> {on['b_total_s']:.1f} s); "
              f"A's total {off['a_total_s']:.1f} s -> {on['a_total_s']:.1f} s "
              f"(+{100.0 * (on['a_total_s'] / off['a_total_s'] - 1.0):.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
