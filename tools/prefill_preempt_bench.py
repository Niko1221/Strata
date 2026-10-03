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

if hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1)   # DEBUG: kill -USR1 <pid> dumps every thread's stack
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from prefill_preempt_test import Engine, deterministic_tokens, engine_args, output_differences  # noqa: E402

CONFIG = ROOT / "strata-iq3_xxs.json"


def latency_metrics(queued_at: float, observed: dict) -> dict:
    """Pump receipt timestamps, so delayed collect() never becomes a queue/TTFT measurement."""
    started, done = observed['started'], observed['done']
    first = observed.get('first_token')
    if started < queued_at or done < started or (first is not None and not started <= first <= done):
        raise ValueError('invalid request timing order')
    return {'b_queue_wait_s': started - queued_at,
            'b_ttft_s': first - queued_at if first is not None else None,
            'b_total_s': done - queued_at}


def one_pass(exe: str, cfg: dict, workdir: Path, name: str, a_ids, b_ids, warm_ids, max_new: int,
             preempt: bool, chunk: int, args_max_context: int, expert_slots=None, kv_resident=None) -> dict:
    ctx = args_max_context
    e = Engine(exe, engine_args(cfg, prefill=chunk, preempt=preempt, max_context=ctx,
                               expert_slots=expert_slots, kv_resident=kv_resident), workdir / f"bench-{name}.log")
    try:
        e.gen(None, warm_ids, 8)
        e.collect(None)
        out = {"preempt": preempt}
        t_a0 = time.monotonic()
        e.gen(1, a_ids, max_new)
        b_queued = {"sent": False}

        def on_line(line: str) -> bool:
            if line.startswith("PP ") and not b_queued["sent"]:
                print(f"[bench]   A at {line.split()[1]} tokens ({time.monotonic() - t_a0:.0f} s)", flush=True)
                if int(line.split()[1]) >= 3 * chunk:      # A is three chunks into its prefill: queue B
                    out["a_at_b"] = time.monotonic() - t_a0
                    out["b_queued_at"] = time.monotonic()
                    e.gen(2, b_ids, max_new)
                    if preempt:
                        e.send("YIELD")
                    b_queued["sent"] = True
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
            out["a_total_s"] = a_first["observed"]["done"] - t_a0
        print("[bench]   collecting B", flush=True)
        b = e.collect(2)
        print("[bench]   B collected", flush=True)
        out.update(latency_metrics(out["b_queued_at"], b["observed"]))
        out["b_tokens"] = len(b["tokens"])
        out["B"] = b
        if preempt:
            out["a_total_s"] = None
            # A's remaining: resume and run to the end
            e.send("RESUME id=1")
            t_r0 = time.monotonic()
            a_rest = e.collect(1)
            out["a_total_s"] = a_rest["observed"]["done"] - t_a0
            out["a_resume_s"] = a_rest["observed"]["done"] - t_r0
            out["a_tokens"] = len(a_rest["tokens"])
            out["A"] = {**a_rest, "tokens": a_first["tokens"] + a_rest["tokens"]}
        else:
            out["a_tokens"] = len(a_first["tokens"])
            out["A"] = a_first
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
    ap.add_argument("--max-context", type=int, default=None)
    ap.add_argument("--kv-resident", type=int, default=None)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    a_ids = deterministic_tokens(args.tokens, seed=7)
    b_ids = deterministic_tokens(220, seed=99)
    warm_ids = deterministic_tokens(120, seed=5)
    ctx = args.max_context or args.tokens + 2 * args.max_new + 4096
    if args.chunk < 1 or args.max_new < 1 or args.tokens <= 4 * args.chunk + 1 or ctx < args.tokens + args.max_new + 8:
        ap.error('A must leave a chunk after B arrives; context must fit prompt + max_new + 8')
    probe = Engine(args.engine, engine_args(cfg, prefill=args.chunk, preempt=False, max_context=ctx,
                   kv_resident=args.kv_resident), workdir / 'bench-probe.log')
    try:
        expert_slots = int(probe.info.get('expert_slots', 0) or 0)
        if not expert_slots:
            raise RuntimeError('probe did not advertise expert_slots; cannot pin benchmark geometry')
    finally: probe.close()

    results = []
    for preempt in (False, True):
        name = "on" if preempt else "off"
        print(f"[bench] pass {name}: A={args.tokens} tokens, B=220 tokens, max_new={args.max_new}", flush=True)
        t0 = time.time()
        r = one_pass(args.engine, cfg, workdir, name, a_ids, b_ids, warm_ids, args.max_new, preempt, args.chunk,
                     ctx, expert_slots, args.kv_resident)
        r["pass_s"] = time.time() - t0
        results.append(r)
        print(f"[bench]   { {k: v for k, v in r.items() if k not in ('A', 'B')} }", flush=True)
    off, on = results
    (workdir / 'benchmark.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    differences = output_differences(off['A'], on['A'], 'A') + output_differences(off['B'], on['B'], 'B')
    if differences:
        for failure in differences: print(f'[bench] FAIL {failure}', flush=True)
        return 1
    if off["b_queue_wait_s"] > 0:
        cut = 100.0 * (1.0 - on["b_queue_wait_s"] / off["b_queue_wait_s"])
        print(f"[bench] B's queue wait cut by {cut:.0f}% ({off['b_queue_wait_s']:.1f} s -> {on['b_queue_wait_s']:.1f} s); "
              f"A's total {off['a_total_s']:.1f} s -> {on['a_total_s']:.1f} s "
              f"(+{100.0 * (on['a_total_s'] / off['a_total_s'] - 1.0):.1f}%)", flush=True)
        print(f"[bench] B TTFT {off['b_ttft_s']} -> {on['b_ttft_s']}; "
              f"total {off['b_total_s']:.3f} -> {on['b_total_s']:.3f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
