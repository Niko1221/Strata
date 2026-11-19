"""strata_control/model_memory.py - Strict hardware memory budgeting and preflight launch rejection for RAM32."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
import shutil
import sys
from typing import Any

from terminal.strata_control.bundle_identity import VerifiedBundle
from terminal.strata_control.backend_contracts import BackendCapabilities


@dataclass(frozen=True)
class HardwareSnapshot:
    total_physical_ram_bytes: int
    available_physical_ram_bytes: int
    commit_limit_bytes: int
    available_commit_bytes: int
    total_vram_bytes: int
    free_vram_bytes: int


@dataclass(frozen=True)
class ResourcePolicy:
    max_host_budget_bytes: int = 24 * 1024 * 1024 * 1024  # 24 GiB cap
    min_os_ram_headroom_bytes: int = 4 * 1024 * 1024 * 1024  # 4 GiB OS reserve
    min_commit_headroom_bytes: int = 4 * 1024 * 1024 * 1024  # 4 GiB commit reserve
    vram_reserve_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GiB VRAM reserve


@dataclass(frozen=True)
class MemoryPlan:
    is_launchable: bool
    is_qualifiable: bool
    planned_host_bytes: int
    planned_vram_bytes: int
    headroom_bytes: int
    rejection_reason: str | None = None


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def capture_live_hardware_snapshot() -> HardwareSnapshot:
    """Read live physical RAM, commit limit, and GPU VRAM on Windows (with POSIX fallbacks)."""
    total_ram = 0
    avail_ram = 0
    total_commit = 0
    avail_commit = 0

    if sys.platform == "win32":
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_ram = int(stat.ullTotalPhys)
            avail_ram = int(stat.ullAvailPhys)
            total_commit = int(stat.ullTotalPageFile)
            avail_commit = int(stat.ullAvailPageFile)
    else:
        try:
            total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            avail_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
            total_commit = total_ram
            avail_commit = avail_ram
        except (AttributeError, ValueError):
            total_ram = 32 * 1024**3
            avail_ram = 16 * 1024**3
            total_commit = total_ram
            avail_commit = avail_ram

    # GPU VRAM detection (NVML or torch if available, fallback default)
    total_vram = 0
    free_vram = 0
    try:
        import torch
        if torch.cuda.is_available():
            free_vram, total_vram = torch.cuda.mem_get_info(0)
    except Exception:
        pass

    if total_vram == 0:
        # Fallback query via nvidia-smi if available
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
                text=True,
                timeout=2,
            )
            parts = [int(p.strip()) for p in out.strip().split(",")]
            total_vram = parts[0] * 1024 * 1024
            free_vram = parts[1] * 1024 * 1024
        except Exception:
            total_vram = 16 * 1024 * 1024 * 1024
            free_vram = 12 * 1024 * 1024 * 1024

    return HardwareSnapshot(
        total_physical_ram_bytes=total_ram,
        available_physical_ram_bytes=avail_ram,
        commit_limit_bytes=total_commit,
        available_commit_bytes=avail_commit,
        total_vram_bytes=total_vram,
        free_vram_bytes=free_vram,
    )


def plan_memory(
    bundle: VerifiedBundle,
    caps: BackendCapabilities | None,
    hardware: HardwareSnapshot,
    policy: ResourcePolicy | None = None,
    dense_bytes: int = 6 * 1024 * 1024 * 1024,
    expert_resident_bytes: int = 4 * 1024 * 1024 * 1024,
    scratch_bytes: int = 1 * 1024 * 1024 * 1024,
    planned_vram_bytes: int | None = None,
) -> MemoryPlan:
    """Compute strict separate host RAM and GPU VRAM budget; reject launch if headroom violated."""
    policy = policy or ResourcePolicy()

    # Rule 1: Host and VRAM are NEVER unified into a single pool
    planned_host = dense_bytes + expert_resident_bytes + scratch_bytes
    planned_vram = planned_vram_bytes if planned_vram_bytes is not None else 2 * 1024 * 1024 * 1024

    # Static qualification check (capacity of hardware architecture)
    can_qualify = (
        hardware.total_physical_ram_bytes >= (planned_host + policy.min_os_ram_headroom_bytes)
        and hardware.total_vram_bytes >= (planned_vram + policy.vram_reserve_bytes)
    )

    # Dynamic launch check 1: host budget cap
    if planned_host > policy.max_host_budget_bytes:
        return MemoryPlan(
            is_launchable=False,
            is_qualifiable=False,
            planned_host_bytes=planned_host,
            planned_vram_bytes=planned_vram,
            headroom_bytes=0,
            rejection_reason=f"Planned host RAM {planned_host / 1e9:.2f}GB exceeds budget cap {policy.max_host_budget_bytes / 1e9:.2f}GB",
        )

    # Dynamic launch check 2: available physical RAM headroom
    if hardware.available_physical_ram_bytes < (planned_host + policy.min_os_ram_headroom_bytes):
        return MemoryPlan(
            is_launchable=False,
            is_qualifiable=can_qualify,
            planned_host_bytes=planned_host,
            planned_vram_bytes=planned_vram,
            headroom_bytes=0,
            rejection_reason=(
                f"Insufficient available RAM: requires {planned_host / 1e9:.2f}GB + "
                f"{policy.min_os_ram_headroom_bytes / 1e9:.2f}GB OS reserve, but only "
                f"{hardware.available_physical_ram_bytes / 1e9:.2f}GB available"
            ),
        )

    # Dynamic launch check 3: VRAM headroom
    if hardware.free_vram_bytes < (planned_vram + policy.vram_reserve_bytes):
        return MemoryPlan(
            is_launchable=False,
            is_qualifiable=can_qualify,
            planned_host_bytes=planned_host,
            planned_vram_bytes=planned_vram,
            headroom_bytes=0,
            rejection_reason=(
                f"Insufficient free VRAM: requires {planned_vram / 1e9:.2f}GB + "
                f"{policy.vram_reserve_bytes / 1e9:.2f}GB reserve, but only "
                f"{hardware.free_vram_bytes / 1e9:.2f}GB free"
            ),
        )

    headroom = hardware.available_physical_ram_bytes - planned_host
    return MemoryPlan(
        is_launchable=True,
        is_qualifiable=can_qualify,
        planned_host_bytes=planned_host,
        planned_vram_bytes=planned_vram,
        headroom_bytes=headroom,
    )
