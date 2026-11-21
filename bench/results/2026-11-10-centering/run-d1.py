#!/usr/bin/env python
"""D1: the 128K out-of-sample run (registered in bench/results/2026-11-10-centering/README.md).

The question: the offsets were calibrated on cwe-4k.  C3 measured them on that same text (in
sample).  D1 is the only test of whether the calibrated offset transfers to texts it was not
calibrated on.  The arms are the product codec (q4lm32w, 16 levels on both sides) with and without
the calibration, plus int8 as the reference, on the two 128K texts.

Gates (fixed before the run):
  G7  the arm with the calibration must be better than the same arm without it on BOTH texts
      (paired per-segment difference < 0 per text).  If the advantage turns negative on one text,
      the offset is an artifact of the calibration text.
  G8  whole-segment KL <= int8 on both texts and p99 <= int8 on both texts (K2/K3 of the ladder).

Cost: a 128K run is the most expensive thing in this project, so the runs are staged: text 1
first (int8, calibrated, uncalibrated), and text 2 only if text 1 does not already answer G7.
The staging does not change the gate: both texts are needed for G7 and G8.

Geometry: the production window at 128K (S=256, R=64, chunk 6,144 -- the byte model of the
paper), and the verify-window launch switch that every campaign arm used.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable
DATA = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval")
OUT = DATA / "d1"
CENTER = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-centering\center-table.bin")
CFG = REPO / "bench" / "configs" / "cwe4k-swift-iq3_xxs-merged.json"
IDS = REPO / "data" / "quant_quality" / "cwe-128k"

BASE_ENV = ["STRATA_KV_WINDOW_S=256", "STRATA_KV_WINDOW_R=64", "STRATA_KV_WINDOW_BATCH=6144",
            "STRATA_VERIFY_ASYNC_LAUNCH=1"]

# text, arm, --kv, calibrated, extra env
# The fp16 arm is the reference the absolute KL is measured against (the same reference the
# predecessor's 128K numbers use), so it is run again here: that also checks on the merged build
# what the predecessor measured on 128K (int8 whole segment 0.10715 on T1, 0.06435 on T2).
# The q4lm8v+window+calibration arm is the *shipped* configuration with the calibration added:
# it reproduces the predecessor's window arm on the merged build and answers whether the
# calibration pays in the long context, where the 4K measurement could not decide it (its own
# paired effect at 4K was +0.00218, not significant).
ARMS = [
    ("t1", "fp16", "fp16", False, {}),
    ("t1", "int8", "int8", False, {}),
    ("t1", "q4lm32w", "q4lm32w", False, {}),
    ("t1", "q4lm32w-center", "q4lm32w", True, {}),
    ("t1", "q4lm8v-win-center", "q4lm8v", True, {"STRATA_KV_WINDOW": "1"}),
    ("t1", "q4lm8v-win", "q4lm8v", False, {"STRATA_KV_WINDOW": "1"}),
]

# a subset of arms, so that a single arm can be measured without re-running the others
ONLY = [a for a in os.environ.get("RUN_D1_ONLY", "").split(",") if a]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strata_running():
    raw = subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True).stdout
    return any(line.lower().lstrip().startswith("strata.exe")
               for line in raw.decode("utf-8", errors="replace").splitlines())


def run_arm(text, tag, kv, centered, extra=None):
    extra = extra or {}
    ids = IDS / f"{text}.ids.json"
    out = OUT / f"{text}-{tag}.tsv"
    env = list(BASE_ENV)
    if centered:
        env.append(f"STRATA_KV_CENTER={CENTER}")
    for k, v in extra.items():
        env.append(f"{k}={v}")
    cmd = [PY, str(REPO / "tools" / "quant_quality.py"), "run",
           "--config", str(CFG), "--ids", str(ids), "--out", str(out), "--kv", kv,
           "--prefill", "6144"]
    for e in env:
        cmd += ["--env", e]
    print(f"=== arm {text}/{tag} ===", flush=True)
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0:
        print(f"arm {text}/{tag}: quant_quality.py exited with {rc}", flush=True)
        return rc
    lines = sum(1 for _ in open(out, encoding="utf-8", errors="replace"))
    seg = Path(str(out) + ".seg.json")
    segments = len(json.loads(seg.read_text(encoding="utf-8"))["segments"]) if seg.exists() else 0
    print(f"arm {text}/{tag}: {lines} lines, {segments} segments, sha256 {sha256(out)}", flush=True)
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if strata_running():
        print("a strata.exe is running; refusing to start (one GPU job at a time)", flush=True)
        return 1
    rc = 0
    for text, tag, kv, centered, extra in ARMS:
        if ONLY and tag not in ONLY:
            continue
        rc = run_arm(text, tag, kv, centered, extra) or rc
        time.sleep(30)   # the campaign rule: 30 s between engine starts
    print("=== compare against the fp16 reference (the absolute numbers AND the paired differences) ===", flush=True)
    # the reference is always the fp16 arm, also when only a subset was run: the paired difference
    # between two arms is the difference of their KLs against fp16, not their KL against int8 (the
    # C4 run showed how easily the two get mixed up).  Before 2026-11-11 the reference was
    # runs[0], which in the RUN_D1_ONLY case was the *new* arm itself -- the compare then had a
    # single file and aborted (exit 2), and the tail tool measured against the wrong reference.
    ref = str(OUT / "t1-fp16.tsv")
    runs = [str(OUT / f"{t}-{tag}.tsv") for t, tag, _kv, _c, _e in ARMS
            if tag != "fp16" and (not ONLY or tag in ONLY) and Path(OUT / f"{t}-{tag}.tsv").exists()]
    if not runs:
        print("no arm to compare against the fp16 reference", flush=True)
        return 1
    cmd = [PY, str(REPO / "tools" / "quant_quality.py"), "compare", ref] + runs + \
          ["--paired", "--ids", str(IDS / "t1.ids.json"), "--segments",
           "--json", str(OUT / "compare-fp16.json"), "--paired-json", str(OUT / "paired-fp16.json")]
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode or rc
    print("=== the tail and the top-1, paired (the criteria the calibration was nominated for) ===", flush=True)
    # the tail tool's reference is the fp16 arm, never the arm under test; the pairs are the two
    # codecs that the calibration was measured on (16/16 and the shipped 16/8)
    for a_tag, b_tag in (("q4lm32w", "q4lm32w-center"), ("q4lm8v-win", "q4lm8v-win-center")):
        a = str(OUT / f"t1-{a_tag}.tsv")
        b = str(OUT / f"t1-{b_tag}.tsv")
        if Path(a).exists() and Path(b).exists():
            print(f"--- pair {a_tag} (a) against {b_tag} (b) ---", flush=True)
            cmd = [PY, str(REPO / "tools" / "kv_tail_top1.py"), "--ref", ref,
                   "--int8", str(OUT / "t1-int8.tsv"), "--a", a, "--b", b,
                   "--ids", str(IDS / "t1.ids.json")]
            rc = subprocess.run(cmd, cwd=str(REPO)).returncode or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
