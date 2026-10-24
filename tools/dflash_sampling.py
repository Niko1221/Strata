#!/usr/bin/env python3
"""GPU integration gate for DFlash's sampled target-only fallback.

No downloads. Supply an existing DFlash run configuration and a directory outside
the checkout for logs. Uses the native serve protocol and CLI, fixed expert
residency, and the same target/sampler seed on each side. This checks real draft
execution on greedy requests, request reset, penalties, filters and greedy regression.

    python tools/dflash_sampling.py --config strata-iq3_xxs.json \
        --engine build/strata --baseline /path/to/previous/strata --output /tmp/df-sampling

Exact token equality is a gate on the supplied fixture, not a promise of bitwise
equality across all CPU/GPU arithmetic paths or adaptive cache configurations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading


def replace(args: list[str], key: str, value: str | None) -> None:
    while key in args:
        i = args.index(key)
        del args[i:i + 2]
    if value is not None:
        args.extend((key, value))


def validate_residency(info, requested, reference=None):
    """A requested budget is not evidence of identical CPU/GPU expert placement."""
    for key in ("expert_slots", "expert_slots_primary"):
        if int(info.get(key, -1)) != requested:
            raise AssertionError(f"{key}={info.get(key)}; requested {requested}. "
                                 "Reduce --expert-cache so every arm fits without automatic shrinking.")
    if reference is not None:
        for key in ("context", "kv", "kv_resident", "expert_slots", "expert_slots_primary",
                    "spec", "pcie_frac", "pool_workers"):
            if key not in reference or key not in info or reference[key] != info[key]:
                raise AssertionError(f"incomparable engine setting {key}: "
                                     f"{reference.get(key)} versus {info.get(key)}")


class Engine:
    def __init__(self, exe: Path, args: list[str], cwd: Path, env: dict[str, str], log: Path):
        self.info = {}
        self.log = log.open("w", encoding="utf-8")
        self.p = subprocess.Popen([str(exe), *args, "--serve"], cwd=cwd, env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, text=True, bufsize=1)
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            while not self.line().startswith("READY"):
                pass
        except BaseException:
            self.close()
            raise

    def _read(self):
        assert self.p.stdout is not None
        for line in self.p.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def line(self) -> str:
        try:
            line = self.lines.get(timeout=240)
        except queue.Empty:
            raise RuntimeError("engine response timed out; inspect the engine log") from None
        if line is None:
            raise RuntimeError(f"engine exited ({self.p.poll()}); inspect the engine log")
        if line.startswith("INFO "):
            self.info = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
        return line

    def request(self, ids: list[int], sampling: str, count: int) -> dict:
        assert self.p.stdin is not None
        self.p.stdin.write(f"GEN {count} {sampling} " + ",".join(map(str, ids)) + "\n")
        self.p.stdin.flush()
        tokens = []
        while True:
            line = self.line()
            if line.startswith("T "):
                tokens.append(int(line[2:]))
            elif line.startswith("ERR"):
                return {"error": line, "tokens": tokens}
            elif line.startswith("DONE"):
                fields = line.split()
                assert len(tokens) == int(fields[1]), line
                return {"tokens": tokens, "decode_ms": float(fields[4]),
                        "accepted": int(fields[6]), "offered": int(fields[7]), "done": line}

    def close(self):
        try:
            if self.p.poll() is None:
                assert self.p.stdin is not None
                self.p.stdin.write("QUIT\n")
                self.p.stdin.flush()
                self.p.wait(timeout=30)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.p.kill()
            self.p.wait()
        finally:
            self.reader.join(timeout=5)
            for pipe in (self.p.stdin, self.p.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except BrokenPipeError:
                        pass   # a buffered QUIT may fail again when a dead child's stdin closes
            self.log.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--baseline", type=Path, help="previous engine, for DFlash-off greedy byte equality")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--tokens", default="50,3815,2923,539,2923,13,6558,281,2007,8,283,220,18,87,61,17,471,220,22,87,478,220,17,13")
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--expert-cache", type=int, default=2000)
    a = ap.parse_args()
    if a.expert_cache < 1:
        ap.error("--expert-cache must be positive")
    if a.count < 2 or a.count > 128:
        ap.error("--count must be in 2..128")
    out = a.output.resolve()
    root = Path(__file__).resolve().parents[1]
    if out == root or root in out.parents:
        ap.error("--output must be outside the checkout")
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    args = list(cfg["args"])
    if "--dflash" not in args or "--mtp" in args:
        ap.error("configuration must select DFlash without MTP")
    ids = [int(t) for t in a.tokens.replace(",", " ").split()]
    if not ids or len(ids) + a.count + 8 > 192:
        ap.error("prompt + output + 8 must fit the test's 192-cell draft capacity")
    for key, value in (("--expert-cache", str(a.expert_cache)), ("--adapt-every", "100000"),
                       ("--adapt-swaps", "0"), ("--pcie-frac", "0"), ("--prefill", "256"),
                       ("--max-context", "512"), ("--kv-resident", "512"),
                       ("--spec", "8"), ("--dflash-block", "4"), ("--spec-min-p", "0"),
                       ("--dflash-window", "192"), ("--suffix-draft", "0"),
                       ("--lookup-chain", "0"), ("--prompt-cache", "0")):
        replace(args, key, value)
    target_args = args.copy()
    for key in ("--dflash", "--dflash-block", "--dflash-window", "--dflash-vocab", "--dflash-head", "--dflash-mask-token"):
        replace(target_args, key, None)
    cwd = Path(cfg.get("cwd", root)).resolve()
    env = dict(os.environ, STRATA_IQ_MT_MIN="1", STRATA_PREFILL_CPU_SHARE="0",
               STRATA_SPEC_PROB="0", STRATA_SPEC_COUPLED="0", STRATA_SPEC_GUMBEL="0")
    cases = [
        ("greedy", "temperature=0 seed=1234"),
        ("cold", "temperature=0.3 seed=1234 top_k=20 top_p=0.95"),
        ("filtered", "temperature=0.7 seed=3456 top_k=40 top_p=0.9 min_p=0.05"),
        ("seed7", "temperature=0.7 seed=7 top_k=40 top_p=0.9 min_p=0.05"),
        ("penalties", "temperature=1.2 seed=5678 top_k=64 top_p=1 min_p=0.02 penalty_last_n=64 penalty_repeat=1.1 penalty_freq=0.2 penalty_present=0.3"),
        ("greedy_again", "temperature=0 seed=1234"),
        ("filtered_repeat", "temperature=0.7 seed=3456 top_k=40 top_p=0.9 min_p=0.05"),
    ]
    results = {}
    infos = {}
    for name, binary, arm_args in [("target", a.engine, target_args), ("dflash", a.engine, args)] + (
            [("previous", a.baseline, target_args)] if a.baseline else []):
        engine = Engine(binary.resolve(), arm_args, cwd, env, out / f"{name}-engine.txt")
        try:
            infos[name] = engine.info
            (out / "engine-info.json").write_text(json.dumps(infos, indent=2))
            validate_residency(engine.info, a.expert_cache, infos.get("target"))
            records = {}
            if name == "dflash":
                refused = engine.request(ids, "temperature=0.7 seed=1", 192)
                assert "DFlash capacity" in refused.get("error", "") and not refused["tokens"], refused
                records["capacity_refusal"] = refused
            for label, sampling in cases:
                if name == "previous" and not label.startswith("greedy"):
                    continue
                record = engine.request(ids, sampling, a.count)
                assert "error" not in record, record
                assert len(record["tokens"]) == a.count, (label, record)
                if name == "dflash":
                    if label.startswith("greedy"):
                        assert record["offered"] > 0, "DFlash was silently bypassed"
                    else:
                        assert record["offered"] == 0, "sampled request bypassed the target-only fallback"
                records[label] = record
                print(name, label, "accepted/offered", record["accepted"], record["offered"], flush=True)
            results[name] = records
            (out / "results.json").write_text(json.dumps(results, indent=2))
        finally:
            engine.close()
    for label, _ in cases:
        assert results["dflash"][label]["tokens"] == results["target"][label]["tokens"], f"sampled target parity: {label}"
    for name in ("target", "dflash"):
        assert results[name]["greedy"]["tokens"] == results[name]["greedy_again"]["tokens"], name
        assert results[name]["filtered"]["tokens"] == results[name]["filtered_repeat"]["tokens"], name
    if a.baseline:
        for label in ("greedy", "greedy_again"):
            assert results["previous"][label]["tokens"] == results["target"][label]["tokens"], f"DFlash-off regression: {label}"
    # The CLI must select the same target-only fallback before loading the drafter.
    cli_sampling = ["--tokens", ",".join(map(str, ids)), "--max-new", str(a.count),
                    "--seed", "3456", "--temperature", "0.7", "--top-k", "40", "--top-p", "0.9"]
    trace = out / "cli-cycles.jsonl"
    trace.unlink(missing_ok=True)
    with (out / "cli.txt").open("w") as log:
        subprocess.run([str(a.engine.resolve()), *args, *cli_sampling], cwd=cwd,
                       env=dict(env, STRATA_DF_CYCLES=str(trace)), stdout=log, stderr=log,
                       check=True, timeout=240)
    with (out / "cli-target.txt").open("w") as log:
        subprocess.run([str(a.engine.resolve()), *target_args, *cli_sampling], cwd=cwd,
                       env=env, stdout=log, stderr=log, check=True, timeout=240)
    def cli_output(path: Path) -> list[int]:
        line = next(line for line in path.read_text().splitlines() if line.startswith("output  :"))
        return [int(t) for t in line.split(":", 1)[1].split()]
    cli_tokens = cli_output(out / "cli.txt")
    assert len(cli_tokens) == a.count and cli_tokens == cli_output(out / "cli-target.txt"), "CLI sampled target parity"
    assert "sampled CLI requests use target-only decoding" in (out / "cli.txt").read_text()
    assert not trace.exists() or not trace.read_text().strip(), "sampled CLI ran the drafter"
    print("DFlash sampled fallback, target token parity, request reset and greedy regression: PASS")


if __name__ == "__main__":
    main()
