"""Compare native fused-prefill output fingerprints in fresh CUDA test processes.

Pass --flag with the opt-in named in the corresponding benchmark report.
The executable also checks the FP64/MMQ numerical references. No model is needed.
"""
import argparse
import os
from pathlib import Path
import subprocess


def run(binary, flag, value, tile):
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env.update({flag: value, "STRATA_PF_FUSED_TILE": str(tile)})
    result = subprocess.run([str(binary), "--only=IQ", "--chunks=2048"], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    print(result.stdout, end="", flush=True)
    if result.returncode == 77:
        raise SystemExit(77)
    result.check_returncode()
    hashes = [line.strip() for line in result.stdout.splitlines() if "fused output bits hash " in line]
    if len(hashes) != 6:
        raise RuntimeError("expected fingerprints for all six format pairs")
    if value == "1" and flag + "=1" not in result.stdout:
        if "inactive: requires SM86" in result.stdout:
            raise SystemExit(77)
        raise RuntimeError("the requested CUDA variant did not report activation; enable its SM86 CMake option")
    return hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("binary", type=Path)
    ap.add_argument("--flag", required=True, choices=["STRATA_PF_PREFETCH_ONE", "STRATA_PF_IQ3_STAGE2"])
    ap.add_argument("--reference-binary", type=Path, help="optional prior build with the same fingerprint output")
    a = ap.parse_args()
    for tile in (64, 128):
        off = run(a.binary.resolve(), a.flag, "0", tile)
        if a.reference_binary and run(a.reference_binary.resolve(), a.flag, "0", tile) != off:
            raise RuntimeError(f"disabled path differs from the reference build (tile {tile})")
        if run(a.binary.resolve(), a.flag, "1", tile) != off:
            raise RuntimeError(f"variant output differs from its disabled control (tile {tile})")
    print("PASS: six format pairs, both tile sizes, identical output fingerprints")


if __name__ == "__main__":
    main()
