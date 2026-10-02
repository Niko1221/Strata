#!/usr/bin/env python3
"""Verify the VRAM budget of a running strata-hip container - the third layer of the 10 GiB contract.

Sampling device memory from sysfs (the same KFD walk as hipinfo.py) means this works from inside or
outside the container and needs no ROCm python bindings.

The contract is about Strata's own share. Inside the container, use --pid with
its engine PID to audit AMD DRM allocation totals, deduplicated by client and
filtered to the selected device. Missing attribution fails the audit. Without
--pid the guard judges raw card memory, including the desktop. --others-mib is
only a fixed subtraction estimate; prefer process attribution.

    vram-guard.py --pid $(pgrep -x strata) --budget-mib 10240 --for 240 --output /logs/vram-audit.json
    vram-guard.py --budget-mib 10240 --for 60 --engine-log /logs/strata-hip.log

Exit code: 0 within budget, 1 breach (peak share > budget), 2 no device / nothing
sampled.  The guard only reports; it cannot free VRAM.  It does not raise the budget, ever - the
plan's fallback for a breach is a pinned --expert-cache (docs/DOCKER_GFX1101_PLAN.md).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hipinfo  # noqa: E402  (same directory)


def sample(device_index: int) -> tuple[int, int] | None:
    """Exact (used_bytes, free_bytes), or None if the device cannot be read."""
    gpus = hipinfo.kfd_gpus()
    if not gpus:
        return None
    try:
        g = hipinfo.pick(gpus, device_index)
    except SystemExit:
        return None
    if g["vram_total_b"] <= 0:
        return None
    return g["vram_used_b"], g["vram_total_b"] - g["vram_used_b"]


def process_vram(pid: int, pdev: str, proc_root: Path = Path("/proc")) -> int | None:
    """Count AMD DRM clients once, on the selected PCI device; never infer zero.

    drm-total-vram counts requested allocation footprint (including evicted
    resources), rather than only currently resident pages. Multiple fds for
    one DRM client must not be summed twice.
    """
    clients = {}
    try:
        for fd in (proc_root / str(pid) / "fdinfo").iterdir():
            try:
                fields = dict(line.split(":", 1) for line in fd.read_text().splitlines() if ":" in line)
            except FileNotFoundError:
                continue
            fields = {k: v.strip() for k, v in fields.items()}
            if fields.get("drm-driver") != "amdgpu" or fields.get("drm-pdev") != pdev:
                continue
            client = fields.get("drm-client-id")
            memory = fields.get("drm-total-vram") or fields.get("drm-memory-vram")
            if not client or not memory:
                continue
            match = re.fullmatch(r"(\d+) KiB", memory)
            if not match:
                continue
            clients[client] = max(clients.get(client, 0), int(match[1]) * 1024)
    except (PermissionError, FileNotFoundError):
        return None
    return sum(clients.values()) if clients else None


def engine_note(log_path: str) -> str | None:
    """The engine's own auto-sizing decision, to sit next to the measured peak in the audit."""
    try:
        lines = [l.strip() for l in Path(log_path).read_text(errors="replace").splitlines()
                 if "expert cache auto" in l or "expert cache" in l.lower() and "slots" in l.lower()]
    except OSError:
        return None
    return lines[-1] if lines else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-mib", type=int, default=int(os.environ.get("STRATA_VRAM_BUDGET_MIB", 10240)))
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="must be zero; no over-budget tolerance is permitted")
    ap.add_argument("--for", dest="seconds", type=float, default=0.0,
                    help="sample for N seconds (0 = until --exec finishes or Ctrl-C)")
    ap.add_argument("--exec", dest="cmd", default="", help="run this shell command and sample while it runs")
    ap.add_argument("--interval", type=float, default=0.25)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--pid", type=int, default=0, help="engine PID in this process namespace; audit AMD DRM client VRAM")
    ap.add_argument("--others-mib", type=int, default=0,
                    help="MiB the desktop/GUI holds: judge Strata's own share (peak - others) against "
                         "the budget (default 0: judge the raw card total, stricter than the contract)")
    ap.add_argument("--engine-log", default=os.environ.get("STRATA_LOG", ""))
    ap.add_argument("--output", default="", help="write the audit JSON here")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if not 0 < a.budget_mib <= 10240 or a.tolerance != 0 or not math.isfinite(a.interval) or a.interval <= 0 or a.seconds < 0 or not math.isfinite(a.seconds) or a.others_mib < 0:
        ap.error("budget must be 1..10240 MiB, tolerance must be zero, interval positive, others nonnegative")

    if a.pid < 0 or (a.pid and a.others_mib):
        ap.error("PID must be positive and cannot be combined with desktop subtraction")
    gpus = hipinfo.kfd_gpus()
    if not gpus:
        print("vram-guard: no AMD GPU readable", file=sys.stderr)
        return 2
    gpu = hipinfo.pick(gpus, a.device)
    arch = gpu["arch"]
    pdev = Path("/sys/class/drm", Path(gpu["render_node"]).name, "device").resolve().name if gpu["render_node"] else "unknown"
    if a.pid and process_vram(a.pid, pdev) is None:
        print("vram-guard: cannot attribute AMD DRM VRAM to this PID/device", file=sys.stderr)
        return 2
    limit = a.budget_mib * 1024 ** 2
    peak = 0
    peak_share = 0
    device_total = 0
    total_used: list[int] = []
    breaches: list[dict] = []
    stop = {"now": False}

    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))

    child = subprocess.Popen(a.cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) \
        if a.cmd else None
    started = time.time()
    deadline = started + a.seconds if a.seconds else None

    while True:
        s = sample(a.device)
        if s:
            used_bytes, free_bytes = s
            used, free = used_bytes / 1024 ** 2, free_bytes / 1024 ** 2
            device_total = used + free
            share_bytes = process_vram(a.pid, pdev) if a.pid else max(0, used_bytes - a.others_mib * 1024 ** 2)
            if share_bytes is None:
                # A disappearing process ends its audit. Losing access while it
                # is alive is an attribution failure, not a successful zero.
                if not Path("/proc", str(a.pid)).exists() and total_used:
                    break
                print("vram-guard: lost process attribution", file=sys.stderr)
                return 2
            share = share_bytes / 1024 ** 2
            total_used.append(used)
            peak = max(peak, used)
            peak_share = max(peak_share, share)
            if share_bytes > limit:
                breaches.append({"t": round(time.time() - started, 2), "used_mib": used,
                                 "strata_share_mib": share})
            if not a.quiet and len(total_used) % max(1, int(2.0 / a.interval)) == 0:
                print(f"  vram {used:9.3f} MiB total, Strata's share {share:9.3f} MiB "
                      f"(peak {peak_share} / budget {a.budget_mib})", flush=True)
        if stop["now"] or (child and child.poll() is not None) or (deadline and time.time() >= deadline):
            break
        time.sleep(a.interval)
    if child and child.poll() is None:
        child.terminate()

    if not total_used:
        print("vram-guard: no samples - no AMD GPU readable from here (run it on the host, or pass "
              "--device /dev/dri/renderD<N> into the container)", file=sys.stderr)
        return 2

    audit = {
        "budget_mib": a.budget_mib, "tolerance": a.tolerance, "limit_mib": a.budget_mib,
        "limit_bytes": limit,
        "others_mib": a.others_mib, "pid": a.pid, "pci_device": pdev,
        "judged": "process_drm_total" if a.pid else ("strata_share" if a.others_mib else "raw_card_total"),
        "peak_mib": peak, "peak_strata_share_mib": peak_share,
        "peak_bytes": int(peak * 1024 ** 2), "peak_share_bytes": int(peak_share * 1024 ** 2),
        "mean_mib": round(sum(total_used) / len(total_used), 1),
        "min_mib": min(total_used), "samples": len(total_used), "device_total_mib": device_total,
        "window_seconds": round(time.time() - started, 1), "arch": arch,
        "verdict": "PASS" if not breaches else "FAIL", "breaches": breaches[:20],
        "attribution": "DRM allocation totals per unique client" if a.pid else
                       "fixed subtraction is an estimate; raw card total is conservative",
        "engine_note": engine_note(a.engine_log) if a.engine_log else None,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    print(f"vram-guard: Strata's share peaked at {peak_share} MiB, card peaked at {peak} MiB "
          f"({a.others_mib} MiB of it the desktop's), over {audit['window_seconds']}s "
          f"({len(total_used)} samples) against a {a.budget_mib} MiB budget -> {audit['verdict']}")
    if device_total and peak >= device_total - 256:
        print(f"            NOTE: the card itself came within {device_total - peak} MiB of full "
              f"({device_total} MiB) - the desktop grew into its share; Strata may have behaved")
    if audit["engine_note"]:
        print(f"            engine: {audit['engine_note']}")
    if a.output:
        Path(a.output).write_text(json.dumps(audit, indent=1) + "\n")
        print(f"            audit written to {a.output}")
    return 0 if not breaches else 1


if __name__ == "__main__":
    raise SystemExit(main())
