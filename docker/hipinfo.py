#!/usr/bin/env python3
"""Device discovery for Strata's HIP container - no ROCm python bindings, no third-party imports.

The AMD GPU list is read from the kernel's KFD topology exactly the way setup.py's amd_gpus()
reads it (src: setup.py:713-748), so the container, ./setup.sh and this script agree about which
card is HIP device 0.  Integrated GPUs are reported but flagged, never selected implicitly.

Usage (all output on stdout, non-zero exit + message on stderr when there is no usable device):

    hipinfo.py                       human-readable summary of every GPU
    hipinfo.py --arch                gfx target of HIP device N (--device, default 0), e.g. gfx1101
    hipinfo.py --vram                "<total_mib> <used_mib> <free_mib>" of HIP device N
    hipinfo.py --render-node         /dev/dri/renderD<minor> of HIP device N
    hipinfo.py --reserve-mib         vram-reserve-mib that honours $STRATA_VRAM_BUDGET_MIB
    hipinfo.py --json                everything, as one JSON object
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Arches Strata's HIP backend accepts after the gfx1101 widening (docs/DOCKER_GFX1101_PLAN.md C1/C2).
# Keep in sync with cmake/hip_backend.cmake and src/core/device.cu.
SUPPORTED_ARCHS = ("gfx1100", "gfx1101")

# What the engine binds after --expert-cache auto has already spent the free memory: draft head +
# logits, the spec-verify window, the prompt/MMQ workspace.  Measured on gfx1101 with IQ3_XXS,
# --prefill 512 --spec 4 --mtp: 1425 MiB over a 10 GiB budget, hence this floor of 1536 MiB.
# Override with $STRATA_VRAM_LATER_MIB or --later-mib (0 reproduces the engine's own arithmetic).
LATER_MIB = int(os.environ.get("STRATA_VRAM_LATER_MIB", "1536"))


def kfd_gpus() -> list[dict]:
    """AMD GPUs in HIP order: the GPU nodes of the KFD topology in node-number order, CPUs skipped."""
    base = Path("/sys/class/kfd/kfd/topology/nodes")
    out = []
    if not base.is_dir():
        return out
    for node in sorted((p for p in base.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
        try:
            props = {}
            for line in (node / "properties").read_text().splitlines():
                key, _, value = line.partition(" ")
                props[key] = value.strip()
            ver = int(props.get("gfx_target_version") or 0)
            if ver == 0 or int(props.get("simd_count") or 0) == 0:
                continue                       # a CPU node, or a GPU without compute
        except (OSError, ValueError):
            continue
        arch = f"gfx{ver // 10000}{(ver // 100) % 100:x}{ver % 100:x}"
        minor = props.get("drm_render_minor", "")
        dev = Path(f"/sys/class/drm/renderD{minor}/device")
        def read(name: str) -> int:
            try:
                return int((dev / name).read_text().strip())
            except (OSError, ValueError):
                return 0
        try:
            name = (dev / "product_name").read_text().strip()
        except OSError:
            name = f"AMD Radeon ({arch})"
        out.append({
            "index": len(out), "arch": arch, "name": name, "kfd_node": node.name,
            "render_node": f"/dev/dri/renderD{minor}" if minor else "",
            "vram_total_b": read("mem_info_vram_total"), "vram_used_b": read("mem_info_vram_used"),
            "simd_count": int(props.get("simd_count") or 0),
            "supported": arch in SUPPORTED_ARCHS,
            # An APU's GPU shares system RAM: tiny mem_info_vram_total, and never the intended target.
            "integrated": read("mem_info_vram_total") < 2 * 1024 ** 3,
        })
    return out


def visible_devices() -> list[int]:
    """The HIP/ROCR visible-device list, if the environment sets one."""
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        value = os.environ.get(var, "").strip()
        if value:
            try:
                return [int(x) for x in value.split(",") if x.strip() != ""]
            except ValueError:
                return []
    return []


def pick(gpus: list[dict], device: int) -> dict:
    """The GPU that HIP index `device` maps to, honouring HIP_VISIBLE_DEVICES the way the runtime does."""
    visible = visible_devices()
    if visible:
        if device >= len(visible):
            sys.exit(f"hipinfo: HIP device {device} is not in HIP_VISIBLE_DEVICES="
                     f"{','.join(map(str, visible))}")
        device = visible[device]
    if device >= len(gpus):
        sys.exit(f"hipinfo: HIP device {device} does not exist "
                 f"(the KFD topology has {len(gpus)} GPU node(s); is the amdgpu driver loaded, "
                 f"and is /dev/kfd plus the render node passed to the container?)")
    return gpus[device]


def reserve_mib(total_mib: int, budget_mib: int, slack_mib: int, later_mib: int = LATER_MIB,
                others_mib: int = 0) -> int:
    """--vram-reserve-mib that keeps **Strata's own share** of the card at or under budget_mib.

    The budget is what Strata may take, not what the card may reach: what is left over
    (device_total - budget, ~2 GiB on a 12 GiB card) belongs to the desktop and the GUI, and
    whatever they already hold (`others_mib`) is not charged against Strata a second time.

    '--expert-cache auto' sizes from FREE memory (src/program/generate.cpp:2020-2040), which makes
    the arithmetic come out independent of the desktop:

        total_occupancy ~= device_total - reserve + later_mib
        strata_share    ~= total_occupancy - others_mib

    so the reserve that lands strata_share on (budget - slack) is the one below.  `later_mib` is
    what the engine still binds AFTER that decision: the draft head and its logits, the
    speculative-verify window and the prompt/MMQ workspace.  Not theoretical - measured at 1425 MiB
    on a 12 GiB card with --expert-profile, which makes the sizing skip its own prefill allowance
    because the prompt path 'borrows' cache slots.  The engine's multi-GPU sizing adds exactly such
    an allowance (generate.cpp:1883-1889, +1024 MiB); the single-GPU path has none, so the container
    carries it here.

    The floor matters as much as the formula.  A reserve smaller than `later` would let the engine's
    post-sizing allocations push the card past 100 %, so the floor is max(700, later + slack): if a
    desktop has grown past its share, Strata takes less rather than overflowing the card.
    """
    floor = max(700, later_mib + slack_mib)
    return max(floor, total_mib - budget_mib + slack_mib + later_mib - others_mib)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=0, help="HIP device index (default 0)")
    ap.add_argument("--arch", action="store_true")
    ap.add_argument("--vram", action="store_true")
    ap.add_argument("--render-node", action="store_true")
    ap.add_argument("--reserve-mib", action="store_true")
    ap.add_argument("--allocation-guard", action="store_true",
                    help="free memory is already clamped by the HIP allocation ledger")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--budget-mib", type=int, default=int(os.environ.get("STRATA_VRAM_BUDGET_MIB", 10240)))
    ap.add_argument("--slack-mib", type=int, default=int(os.environ.get("STRATA_VRAM_SLACK_MIB", 256)))
    ap.add_argument("--later-mib", type=int, default=LATER_MIB,
                    help=f"VRAM the engine binds after --expert-cache auto (default {LATER_MIB})")
    ap.add_argument("--others-mib", type=int, default=-1,
                    help="MiB already held by the desktop/other processes; default: what sysfs says now")
    a = ap.parse_args()

    gpus = kfd_gpus()
    if not gpus:
        sys.exit("hipinfo: no AMD GPU in /sys/class/kfd - pass --device /dev/kfd and the render node "
                 "to the container, and check the amdgpu driver")
    g = pick(gpus, a.device)

    if a.arch:
        print(g["arch"])
        return 0
    if a.render_node:
        if not g["render_node"]:
            sys.exit("hipinfo: this device has no drm render node")
        print(g["render_node"])
        return 0
    if a.vram or a.reserve_mib:
        if g["vram_total_b"] <= 0:
            sys.exit("hipinfo: mem_info_vram_total is unreadable for "
                     f"{g['render_node'] or 'this device'}; cannot honour a VRAM budget")
        total, used = g["vram_total_b"] // 1024 ** 2, g["vram_used_b"] // 1024 ** 2
        if a.vram:
            print(f"{total} {used} {total - used}")
            return 0
        # 'used' here is everyone but Strata, because this runs before the engine is loaded: exactly
        # the desktop's share that must not be charged against Strata's budget again.
        others = used if a.others_mib < 0 else a.others_mib
        print(max(700, a.later_mib) if a.allocation_guard else
              reserve_mib(total, a.budget_mib, a.slack_mib, a.later_mib, others))
        return 0

    if a.json:
        print(json.dumps({"device": g, "all": gpus,
                          "budget_mib": a.budget_mib, "slack_mib": a.slack_mib, "later_mib": a.later_mib,
                          "others_mib": (g["vram_used_b"] // 1024 ** 2) if a.others_mib < 0 else a.others_mib,
                          "reserve_mib": reserve_mib(g["vram_total_b"] // 1024 ** 2, a.budget_mib,
                                                     a.slack_mib, a.later_mib,
                                                     (g["vram_used_b"] // 1024 ** 2) if a.others_mib < 0
                                                     else a.others_mib)
                          if g["vram_total_b"] else None}, indent=1))
        return 0

    for x in gpus:
        note = "" if x["supported"] else "  <- not supported by Strata's HIP backend"
        if x["integrated"]:
            note += "  (integrated: shared system RAM)"
        print(f"GPU {x['index']}: {x['name']}  {x['arch']}  "
              f"{x['vram_total_b'] / 1024**3:.1f} GiB  {x['render_node'] or 'no render node'}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
