"""T6: a long soak through a running server (docs/TEST_REQUESTS.md: "an hour or more of back-to-back requests").
The pattern follows the RTX 3060 community report's soak: every 4th request is a known-answer check (17 x 19 = 323,
16 tokens), the others are fresh prompts of about 512 / 4,096 / 32,000 tokens in turn (a unique tag each, so the
prompt cache never helps), 512 generated tokens. A sampler records the GPU (nvidia-smi) and the engine's RAM every
10 s.

    python3 soak.py --out DIR [--minutes 65]
Writes DIR/requests.jsonl, DIR/samples.csv and DIR/summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_client import body, chat, health  # noqa: E402

VOCAB = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember ledger "
         "quartz meadow anchor violet thunder").split()
GPU_FIELDS = ["memory.used", "utilization.gpu", "temperature.gpu", "power.draw", "clocks.current.graphics",
              "clocks.current.memory", "pcie.link.gen.current", "pcie.link.width.current"]
TARGETS = (512, 4096, 32000)


def gpu_sample() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().split("\n")[0]
        return dict(zip(GPU_FIELDS, [x.strip() for x in out.split(",")]))
    except (OSError, subprocess.SubprocessError):
        return {}


def engine_rss_mib() -> float | None:
    try:
        pids = subprocess.run(["pgrep", "-x", "strata"], capture_output=True, text=True).stdout.split()
        total = 0
        for pid in pids:
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
        return round(total / 1024, 1) if pids else None
    except OSError:
        return None


def mem_available_gib() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return round(int(line.split()[1]) / 2**20, 2)
    return float("nan")


def sampler(path: Path, stop: threading.Event) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "engine_rss_mib", "mem_available_gib", *GPU_FIELDS])
        while not stop.is_set():
            g = gpu_sample()
            w.writerow([round(time.time(), 1), engine_rss_mib(), mem_available_gib(), *[g.get(k) for k in GPU_FIELDS]])
            f.flush()
            stop.wait(10)


def prompt(i: int, words: int) -> str:
    rng = random.Random(1000 + i)
    text = " ".join(rng.choice(VOCAB) for _ in range(words))
    return (f"Soak request {i} (unique tag {rng.getrandbits(48):012x}).\nHere is a list of words:\n{text}\n\n"
            "Summarize what kinds of words appear in the list, in five sentences.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--minutes", type=float, default=65)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    th = threading.Thread(target=sampler, args=(a.out / "samples.csv", stop), daemon=True)
    th.start()
    ratio = {t: 1.0 for t in TARGETS}          # tokens per word, learned from the server's own counts
    rows, began, i = [], time.time(), 0
    log = (a.out / "requests.jsonl").open("w", encoding="utf-8")
    try:
        while time.time() - began < a.minutes * 60:
            if i % 4 == 0:
                kind, target = "check", None
                req = body("Calculate 17 multiplied by 19. Reply only with the integer result.", 16)
            else:
                target = TARGETS[i % 4 - 1]
                kind = f"p{target}"
                words = max(50, int((target - 60) / ratio[target]))
                req = body(prompt(i, words), 512)
            row = chat(req, timeout=1200)
            row.update(index=i, kind=kind, target=target, elapsed_min=round((time.time() - began) / 60, 2))
            if kind == "check":
                row["correct"] = row["ok"] and "323" in (row.get("text") or "")
            elif row["ok"] and row.get("prompt_tokens"):
                ratio[target] = row["prompt_tokens"] / max(1, words)
            row["text"] = (row.get("text") or "")[:200]
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            rows.append(row)
            print(f"[{row['elapsed_min']:6.1f} min] #{i:3d} {kind:7s} ok={row['ok']} prompt={row.get('prompt_tokens')} "
                  f"pp={row.get('prompt_tps')} decode={row.get('decode_tps')} {row.get('error') or ''}"
                  + (f" correct={row['correct']}" if kind == "check" else ""), flush=True)
            if not row["ok"] and health() is None:
                print("the server does not answer /health any more: stopping the soak", flush=True)
                break
            i += 1
    finally:
        stop.set()
        th.join(15)
    observed = (time.time() - began) / 60

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 2) if xs else None

    summary = {"observed_minutes": round(observed, 1), "requests": len(rows),
               "failed": sum(not r["ok"] for r in rows),
               "checks": sum(r["kind"] == "check" for r in rows),
               "checks_correct": sum(bool(r.get("correct")) for r in rows if r["kind"] == "check"),
               "by_kind": {}}
    half = observed / 2
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind and r["ok"]]
        summary["by_kind"][kind] = {
            "n": len(rs), "prompt_tokens_median": med([r.get("prompt_tokens") for r in rs]),
            "decode_tps_first_half": med([r.get("decode_tps") for r in rs if r["elapsed_min"] <= half]),
            "decode_tps_second_half": med([r.get("decode_tps") for r in rs if r["elapsed_min"] > half]),
            "prompt_tps_first_half": med([r.get("prompt_tps") for r in rs if r["elapsed_min"] <= half]),
            "prompt_tps_second_half": med([r.get("prompt_tps") for r in rs if r["elapsed_min"] > half])}
    try:
        with (a.out / "samples.csv").open() as f:
            s = list(csv.DictReader(f))

        def col(k):
            return [float(x[k]) for x in s if x.get(k) not in (None, "", "None", "[N/A]")]
        rss = col("engine_rss_mib")
        summary["samples"] = len(s)
        summary["gpu_temp_max_c"] = max(col("temperature.gpu"), default=None)
        summary["gpu_power_max_w"] = max(col("power.draw"), default=None)
        summary["vram_used_max_mib"] = max(col("memory.used"), default=None)
        summary["engine_rss_first_last_max_mib"] = [rss[0], rss[-1], max(rss)] if rss else None
        summary["mem_available_min_gib"] = min(col("mem_available_gib"), default=None)
    except (OSError, ValueError) as e:
        summary["samples_error"] = str(e)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
