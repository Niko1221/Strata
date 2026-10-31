# P0 of the phase-2 plan (2026-10-24): plain vs plain+MTP single-stream tok/s on the 4090 rig.
# Same protocol as tools/grammar_acceptance.py's perf phase (its prompt, max_tokens, seed, the
# tok/s from usage), one server at a time: first the shipping args (--spec 2, no MTP), then the
# same args with --mtp <rt>.  Writes bench/results/<stamp>/mtp-ab.json.
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from grammar_acceptance import PERF_PROMPT, run_once  # noqa: E402

PORT = 8199
BASE = f"http://127.0.0.1:{PORT}"
CFG = ROOT / ".zcode-obj" / "mtp-ab.cfg.json"
LOG = ROOT / ".zcode-obj" / "mtp-ab.serve.log"
OUT = ROOT / "bench" / "results" / sys.argv[1] if len(sys.argv) > 1 else ROOT / "bench" / "results" / "mtp-ab"

ENGINE = ["--pack", "F:/Strata-data/packs/swift-iq3_s",
          "--native", "F:/Strata-data/models/swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
          "--resident-budget-gib", "40", "--expert-profile", "F:/Strata/data/expert-profile.bin",
          "--expert-cache", "auto", "--prefill", "auto", "--spec", "2",
          "--max-context", "16384", "--kv", "int8"]
MTP = ["--mtp", "F:/Strata-data/mtp/rt"]
RUNS = 3


def start(args):
    cfg = {"exe": str(ROOT / "build" / "strata.exe"), "args": args, "cwd": str(ROOT),
           "tokenizer": "F:/Strata-data/packs/swift-iq3_s/tokenizer",
           "model_name": "Qwen3.8-Flash-Next", "log": str(LOG),
           "port": PORT, "host": "127.0.0.1", "gpu": 0,
           "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}}
    CFG.write_text(json.dumps(cfg), encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                             "--config", str(CFG), "--port", str(PORT)],
                            stdout=open(LOG, "ab"), stderr=subprocess.STDOUT)
    for _ in range(120):
        time.sleep(5)
        try:
            with urllib.request.urlopen(BASE + "/v1/status", timeout=5) as r:
                if r.status == 200:
                    return proc
        except OSError:
            pass
    proc.kill()
    raise RuntimeError("the server did not come up")


def stop(proc):
    proc.terminate()
    try:
        proc.wait(20)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(5)


def measure(label):
    speeds, toks = [], []
    for i in range(RUNS):
        tp, n = run_once(BASE, False)
        speeds.append(round(tp, 2))
        toks.append(n)
        print(f"  {label} {i + 1}: {tp:.1f} tok/s ({n} tokens)", flush=True)
    return {"runs": RUNS, "all_tok_s": speeds, "median_tok_s": statistics.median(speeds), "tokens": toks}


def main():
    stamp = sys.argv[1] if len(sys.argv) > 1 else "mtp-ab"
    out_dir = ROOT / "bench" / "results" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"plain_args": ENGINE, "mtp_args": MTP, "runs": RUNS}
    p = start(ENGINE)
    try:
        results["plain"] = measure("plain")
    finally:
        stop(p)
    time.sleep(5)
    p = start(ENGINE + MTP)
    try:
        with urllib.request.urlopen(BASE + "/v1/status", timeout=5) as r:
            results["mtp_info"] = json.loads(r.read()).get("engine")
        results["mtp"] = measure("plain+mtp")
    finally:
        stop(p)
    results["mtp_gain_pct"] = round((results["mtp"]["median_tok_s"] / results["plain"]["median_tok_s"] - 1) * 100.0, 1)
    (out_dir / "mtp-ab.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("plain", "mtp", "mtp_gain_pct")}, indent=1))


if __name__ == "__main__":
    main()
