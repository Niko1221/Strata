#!/usr/bin/env python3
"""ggml-cpu's IQ1_S / IQ1_M dots with and without BMI2 pdep, side by side (STRATA_GGML_PDEP in CMakeLists.txt).

    python tools/bench_ggml_pdep.py [--ggml-dir DIR] [--build-root DIR] [--passes 2] [-- ggml_dot_bench args]

Builds ggml_dot_bench twice, CPU only (no CUDA, a few minutes the first time): once with pdep
(-DSTRATA_GGML_PDEP=ON -DGGML_BMI2=ON) and once without (-DSTRATA_GGML_PDEP=OFF), runs both --passes times in
alternating order, keeps each format's best time, and prints a Markdown table: ns per 256 weights with and without
pdep, the speed-up, and whether the two builds computed the same results bit for bit (the bench's output checksum).
The formats other than IQ1_S / IQ1_M are the control: their code does not use pdep, so they should not move.

On AMD Zen 1 / Zen+ / Zen 2 (and Hygon) pdep runs in microcode and IQ1_S / IQ1_M are expected to be several times
faster without it; on Intel and Zen 3+ both builds should be close.  --out FILE also writes the table and the raw
runs (FILE.json) for a report.  --ggml-dir is the llama.cpp checkout to build ggml from (STRATA_GGML_DIR; empty:
CMake fetches the pinned commit).  Standard library only.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (("pdep", ["-DSTRATA_GGML_PDEP=ON", "-DGGML_BMI2=ON"]), ("no-pdep", ["-DSTRATA_GGML_PDEP=OFF"]))
ROW = re.compile(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([0-9a-f]{16})$")
HEAD = re.compile(r"^ggml_dot_bench: (.*); ggml-cpu BMI2 \(pdep in the IQ1_S / IQ1_M dots\): (yes|no);")


def tool(name: str) -> str | None:
    here = Path(sys.executable).parent / (name + (".exe" if platform.system() == "Windows" else ""))
    return str(here) if here.exists() else shutil.which(name)


def build(variant: str, defs: list[str], a) -> Path:
    cmake = tool("cmake")
    if cmake is None:
        raise SystemExit("cmake not found (setup's .venv has one: run this with .venv's python)")
    bdir = Path(a.build_root) / variant
    cfg = [cmake, "-S", str(ROOT), "-B", str(bdir), "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=OFF", *defs]
    ninja = tool("ninja")
    if ninja is not None and not (bdir / "CMakeCache.txt").exists():
        cfg += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    if a.ggml_dir:
        cfg.append(f"-DSTRATA_GGML_DIR={a.ggml_dir}")
    print(f"[{variant}] configure", flush=True)
    subprocess.run(cfg, check=True, stdout=subprocess.DEVNULL)
    print(f"[{variant}] build ggml_dot_bench", flush=True)
    subprocess.run([cmake, "--build", str(bdir), "--target", "ggml_dot_bench", "--config", "Release", "-j", str(a.jobs)],
                   check=True, stdout=subprocess.DEVNULL)
    exe = next((p for p in (bdir / "ggml_dot_bench", bdir / "ggml_dot_bench.exe", bdir / "Release" / "ggml_dot_bench.exe")
                if p.exists()), None)
    if exe is None:
        raise SystemExit(f"ggml_dot_bench not found under {bdir}")
    return exe


def run(exe: Path, extra: list[str]) -> dict:
    out = subprocess.run([str(exe), *extra], check=True, capture_output=True, text=True).stdout
    res = {"cpu": "?", "bmi2": "?", "rows": {}, "text": out}
    for line in out.splitlines():
        if m := HEAD.match(line):
            res["cpu"], res["bmi2"] = m.group(1), m.group(2)
        elif m := ROW.match(line.strip()):
            res["rows"][m.group(1)] = {"bytes": float(m.group(2)), "ns": float(m.group(3)), "gbs": float(m.group(4)),
                                      "sum": m.group(5)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ggml-dir", default="", help="llama.cpp checkout for STRATA_GGML_DIR (default: the pinned commit)")
    ap.add_argument("--build-root", default=str(ROOT / "build-pdep-bench"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--passes", type=int, default=2, help="runs per build, alternating; each format's best time counts")
    ap.add_argument("--no-build", action="store_true", help="reuse the two builds under --build-root")
    ap.add_argument("--out", help="also write the table here (Markdown) and the raw runs to OUT.json")
    ap.add_argument("bench_args", nargs=argparse.REMAINDER, help="after --: arguments for ggml_dot_bench")
    a = ap.parse_args()
    extra = [x for x in a.bench_args if x != "--"]
    exes = {}
    for v, defs in VARIANTS:
        if a.no_build:
            bdir = Path(a.build_root) / v
            exes[v] = next(p for p in (bdir / "ggml_dot_bench", bdir / "ggml_dot_bench.exe",
                                       bdir / "Release" / "ggml_dot_bench.exe") if p.exists())
        else:
            exes[v] = build(v, defs, a)
    runs = {v: [] for v, _ in VARIANTS}
    for p in range(max(1, a.passes)):
        order = VARIANTS if p % 2 == 0 else VARIANTS[::-1]
        for v, _ in order:
            print(f"[{v}] pass {p + 1}", flush=True)
            runs[v].append(run(exes[v], extra))
    with_p, without = runs["pdep"][0], runs["no-pdep"][0]
    lines = [f"CPU: {with_p['cpu']}", "",
             f"ggml-cpu with pdep: BMI2 {with_p['bmi2']}; without: BMI2 {without['bmi2']} "
             f"(best of {len(runs['pdep'])} runs per build; ns per 256 weights, one row at a time)", "",
             "| type | bytes / 256 | ns, with pdep | ns, without | speed-up | same results |",
             "|---|---|---|---|---|---|"]
    for t in with_p["rows"]:
        if t not in without["rows"]:
            continue
        a_ns = min(r["rows"][t]["ns"] for r in runs["pdep"] if t in r["rows"])
        b_ns = min(r["rows"][t]["ns"] for r in runs["no-pdep"] if t in r["rows"])
        same = len({r["rows"][t]["sum"] for r in runs["pdep"] + runs["no-pdep"] if t in r["rows"]}) == 1
        lines.append(f"| {t} | {with_p['rows'][t]['bytes']:.1f} | {a_ns:.1f} | {b_ns:.1f} | {a_ns / b_ns:.2f}x | "
                     f"{'yes' if same else 'NO'} |")
    if with_p["bmi2"] != "yes":
        lines += ["", "Note: the pdep build reports no BMI2 - this compiler or CPU never took the pdep branch, so the "
                      "two builds run the same code."]
    table = "\n".join(lines)
    print("\n" + table)
    if a.out:
        Path(a.out).write_text(table + "\n")
        Path(a.out + ".json").write_text(json.dumps(runs, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
