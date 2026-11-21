#!/usr/bin/env python
"""C4: the product codec (16 levels on both sides) with and without the centering.

Registered in bench/results/2026-11-10-centering/README.md (stage C4) before the run.

Why this arm: the q4lm8v arm carries 8 levels on the value side -- a 3-bit codebook inside 4-bit
bytes.  It saves no byte and costs 1.10 dB of block error (the parity harness, with the
centering: lloyd16K+lloyd8V+LS 6.527e-03 against lloyd16+LS 5.072e-03).  The product codec is
`--kv q4lm32w` (16/16, with the exact-cell window).  Two arms on cwe-4k, same ids, same build:

  a) q4lm32w            the 16/16 window arm, no centering
  b) q4lm32w + center   the same, with the calibrated per-channel offset

b against a isolates the centering; a against the q4lm8v window arm (b1/q4lm8v-win.tsv) isolates
the level split.  Both against int8 (b1/int8.tsv) is the K1 gate of the ladder (section 7).

Two things this driver learned the hard way:

  * STRATA_VERIFY_ASYNC_LAUNCH=1 is not optional on this card.  Without it the first attempt of
    both arms died in the verify window ("an earlier window never finished on the GPU (#267)",
    the lost wakeup on gfx1200/Windows; the upstream sets the switch by default only for
    gfx1151).  Every campaign arm (B1, B2, C3) ran with it.
  * the effectiveness gate must not be a hash alone.  The failed arms differ from each other in
    their hashes because they are two different broken transcripts; a gate that only compares
    hashes calls that a pass.  A run counts only with 32,746 lines, eight segments and no ERR
    line - and then the two arms must differ.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable
DATA = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval")
B1 = DATA / "b1"
OUT = DATA / "b4"
CENTER = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-centering\center-table.bin")
CFG = REPO / "bench" / "configs" / "cwe4k-swift-iq3_xxs-merged.json"
IDS = REPO / "data" / "quant_quality" / "cwe-4k" / "set.ids.json"
LINES = 32746

# the window geometry of the production arm (b1/q4lm8v-win.tsv): S=256, R=64, chunk 512,
# and the verify-window launch switch every campaign arm used (see the module docstring)
BASE_ENV = ["STRATA_KV_WINDOW_S=256", "STRATA_KV_WINDOW_R=64", "STRATA_KV_WINDOW_BATCH=512",
            "STRATA_VERIFY_ASYNC_LAUNCH=1"]

ARMS = [
    ("q4lm32w", "q4lm32w", False),
    ("q4lm32w-center", "q4lm32w", True),
]


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


def check_run(out: Path) -> bool:
    """A run counts only with the full transcript, eight segments and no ERR line."""
    if not out.exists():
        return False
    lines = sum(1 for _ in open(out, encoding="utf-8", errors="replace"))
    seg = Path(str(out) + ".seg.json")
    segments = 0
    if seg.exists():
        try:
            segments = len(json.loads(seg.read_text(encoding="utf-8"))["segments"])
        except Exception:
            segments = 0
    err = False
    with open(out, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("^ERR "):
                err = True
                break
    ok = lines == LINES and segments == 8 and not err
    print(f"  {out.name}: {lines} lines, {segments} segments, ERR line: {err} -> "
          f"{'valid' if ok else 'INVALID'}", flush=True)
    return ok


def run_arm(tag, kv, centered):
    out = OUT / f"cwe4k-{tag}.tsv"
    env = list(BASE_ENV)
    if centered:
        env.append(f"STRATA_KV_CENTER={CENTER}")
    cmd = [PY, str(REPO / "tools" / "quant_quality.py"), "run",
           "--config", str(CFG), "--ids", str(IDS), "--out", str(out), "--kv", kv]
    for e in env:
        cmd += ["--env", e]
    print(f"=== arm {tag}: {' '.join(cmd[3:])} ===", flush=True)
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0:
        print(f"arm {tag}: quant_quality.py exited with {rc}", flush=True)
        return rc
    print(f"arm {tag}: sha256 {sha256(out)}", flush=True)
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if strata_running():
        print("a strata.exe is running; refusing to start (one GPU job at a time)", flush=True)
        return 1
    rc = 0
    for tag, kv, centered in ARMS:
        rc = run_arm(tag, kv, centered) or rc
        time.sleep(30)   # the campaign rule: 30 s between engine starts
    a, b = OUT / "cwe4k-q4lm32w.tsv", OUT / "cwe4k-q4lm32w-center.tsv"
    va, vb = check_run(a), check_run(b)
    if not (va and vb):
        print("effectiveness gate: FAILED (a run is not a complete transcript)", flush=True)
        return 1
    sa, sb = sha256(a), sha256(b)
    print(f"effectiveness gate: the two arms must differ: {sa[:16]} / {sb[:16]} -> "
          f"{'OK' if sa != sb else 'FAILED (identical: the centering did nothing)'}", flush=True)
    if sa == sb:
        rc = 1
    print("=== compare against the fp16 reference (the absolute numbers) ===", flush=True)
    cmd = [PY, str(REPO / "tools" / "quant_quality.py"), "compare",
           str(B1 / "fp16.tsv"), str(B1 / "q4lm8v-win.tsv"), str(a), str(b),
           "--segments", "--json", str(OUT / "compare-fp16.json")]
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode or rc
    print("=== compare against int8 (the K1 gate of the ladder) ===", flush=True)
    cmd = [PY, str(REPO / "tools" / "quant_quality.py"), "compare",
           str(B1 / "int8.tsv"), str(B1 / "q4lm8v-win.tsv"), str(a), str(b),
           "--paired", "--ids", str(IDS), "--segments",
           "--json", str(OUT / "compare-int8.json"), "--paired-json", str(OUT / "paired.json")]
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
