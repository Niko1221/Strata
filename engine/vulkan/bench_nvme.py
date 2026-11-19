#!/usr/bin/env python3
"""Random-read throughput of the GGUF's drive at the queue depths the expert streamer uses.

The native engine reads 589,824 B (Q4_K) or 860,160 B (Q6_K) expert tensors with
one io_uring reader and a bounded queue. This measures that access pattern with
fio, read-only, against the model file itself:

    python3 engine/vulkan/bench_nvme.py --model /path/to/model.gguf

It reports the drive model, the block sizes and queue depths, IOPS and MB/s, and
writes JSON with enough provenance to publish the number (drive, file, fio
version, direct I/O, runtime). Do not run it at the same time as the engine: the
two would share the drive and neither number would mean anything.

The default block sizes match the two expert tensor sizes in the test model;
`--bs 4096` measures the small random reads the SYCL expert path does instead.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL = Path("/home/g/Code/llama.cpp/models/Qwen3.8-35B-A3B-Q4_K_M.gguf")
BLOCK_SIZES = (589824, 860160)          # Q4_K gate/up, Q6_K down
QUEUE_DEPTHS = (1, 8, 16, 32)


def drive_model(path: Path) -> str:
    """The physical drive behind `path`, from sysfs (best effort)."""
    if not sys.platform.startswith("linux"):
        return "unknown"
    try:
        resolved = str(path.resolve())
        # st_dev is not the drive; find the longest mount point that contains the file.
        mounts = Path("/proc/self/mountinfo").read_text(encoding="utf-8", errors="replace").splitlines()
        candidates = [fields[4] for fields in (line.split() for line in mounts)
                      if len(fields) >= 10 and resolved.startswith(fields[4])]
        if not candidates:
            return "unknown"
        mount = max(candidates, key=len)
        source = ""
        for line in Path("/proc/self/mounts").read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[1] == mount:
                source = fields[0]
                break
        if not source.startswith("/dev/"):
            return source or "unknown"
        return _physical_model(Path(source).resolve().name)
    except (OSError, ValueError):
        return "unknown"


def _physical_model(name: str, seen: frozenset[str] = frozenset()) -> str:
    """Walk device-mapper/LVM slaves to the drive that has a real model string."""
    if not name or name in seen:
        return name or "unknown"
    # Try the block device as named, then a partition's parent (nvme0n1p2 ->
    # nvme0n1, sda1 -> sda). Do not strip digits from dm-0: that is the mapper.
    variants = [name]
    import re
    match = re.match(r"^(.*\d)p\d+$", name)
    if match:
        variants.append(match.group(1))
    plain = re.sub(r"\d+$", "", name)
    if plain and plain not in variants:
        variants.append(plain)
    for variant in variants:
        for candidate in (Path("/sys/block") / variant, Path("/sys/block") / variant / "device"):
            model = candidate / "model"
            if model.is_file():
                text = model.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    return text
    for slave in sorted((Path("/sys/block") / name / "slaves").glob("*")):
        found = _physical_model(slave.name, seen | {name})
        if found and found != slave.name and found != name:
            return found
    return name


def _fio_version(fio: str) -> str:
    try:
        return subprocess.run([fio, "--version"], capture_output=True, text=True,
                              timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_one(fio: str, model: Path, bs: int, depth: int, seconds: float,
            direct: bool, offset: int, ioengine: str = "io_uring") -> dict:
    cmd = [fio, "--name=expert-read", f"--filename={model}", "--rw=randread",
           f"--bs={bs}", f"--iodepth={depth}", f"--ioengine={ioengine}", "--numjobs=1",
           f"--direct={'1' if direct else '0'}", "--readonly",
           f"--offset={offset}", f"--runtime={seconds:g}", "--time_based",
           "--percentile_list=1:50:99", "--output-format=json", "--eta=never"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=max(60, seconds * 4),
                            check=False)
    if result.returncode != 0:
        return {"bs": bs, "iodepth": depth, "error": (result.stderr or result.stdout)[-400:]}
    try:
        # fio prints a note before the JSON when an I/O engine caps the queue depth.
        text = result.stdout[result.stdout.index("{"):]
        read = json.loads(text)["jobs"][0]["read"]
        # Completion latency is what an expert read waits on. fio keys percentiles
        # as "50.000000" / "99.000000".
        clat = read.get("clat_ns", {})
        pct = clat.get("percentile", {})
        return {"bs": bs, "iodepth": depth,
                "iops": round(read["iops"], 1),
                "bw_mib_s": round(read["bw"] / 1024.0, 1),
                "mean_us": round(clat.get("mean", 0) / 1000.0, 1),
                "p50_us": round(pct.get("50.000000", 0) / 1000.0, 1),
                "p99_us": round(pct.get("99.000000", 0) / 1000.0, 1),
                "io_bytes": read.get("io_bytes", 0)}
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return {"bs": bs, "iodepth": depth, "error": f"unrecognized fio output: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--seconds", type=float, default=10, help="seconds per block-size x queue-depth point")
    ap.add_argument("--bs", type=int, nargs="*", default=list(BLOCK_SIZES))
    ap.add_argument("--depths", type=int, nargs="*", default=list(QUEUE_DEPTHS))
    ap.add_argument("--offset", type=int, default=0, help="start of the expert tensor region in the GGUF")
    ap.add_argument("--no-direct", action="store_true", help="use buffered I/O (direct may fail on compressed fs)")
    ap.add_argument("--ioengine", default="io_uring", help="fio I/O engine (default io_uring, which honours --iodepth)")
    ap.add_argument("--out", help="write the full JSON report here")
    a = ap.parse_args()
    fio = shutil.which("fio")
    if not fio:
        print("fio is not installed; install it (pacman -S fio / apt install fio) and retry", file=sys.stderr)
        return 2
    model = Path(a.model).expanduser().resolve()
    if not model.is_file():
        print(f"no such model: {model}", file=sys.stderr)
        return 2
    report = {
        "schema_version": 1,
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "drive": drive_model(model), "model_file": str(model), "model_bytes": model.stat().st_size,
        "fio": _fio_version(fio), "direct_io": not a.no_direct, "ioengine": a.ioengine,
        "seconds_per_point": a.seconds,
        "offset": a.offset, "kernel": platform.release(), "results": [],
    }
    print(f"drive: {report['drive']}  file: {model.name}  fio: {report['fio']}")
    print(f"{'block':>9} {'queue':>6} {'IOPS':>9} {'MB/s':>9} {'p99 us':>9}")
    for bs in a.bs:
        for depth in a.depths:
            r = run_one(fio, model, bs, depth, a.seconds, not a.no_direct, a.offset, a.ioengine)
            report["results"].append(r)
            if "error" in r:
                print(f"{bs:>9} {depth:>6}  ERROR: {r['error'][:90]}")
            else:
                print(f"{bs:>9} {depth:>6} {r['iops']:>9} {r['bw_mib_s']:>9} {r['p99_us']:>9}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {a.out}")
    return 0 if all("error" not in r for r in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
