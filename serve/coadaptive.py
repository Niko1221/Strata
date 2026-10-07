"""Independent desktop resource signals; native MEMORY remains the only allocator.

Experimental single-GPU policy. Shadow mode records recommendations by default.
Explicit live mode adjusts cache reserves and the existing per-request CPU/GPU
expert split. The heuristic is not a measured cost optimizer. It never changes
model precision or context.
GPU utilization while Strata is generating is NOT evidence of another workload.
"""
from __future__ import annotations

import math

GIB, MIB = 2**30, 2**20


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class CoAdaptive:
    def __init__(self, config=None, *, ram_headroom_gib=4, vram_reserve_mib=256, pcie_frac=.5):
        config = {} if config is None else config
        if (not isinstance(config, dict) or set(config) - {"enabled", "mode"}
                or not isinstance(config.get("enabled", False), bool)
                or config.get("mode", "shadow") not in ("shadow", "live")):
            raise ValueError("coadaptive accepts boolean enabled and mode shadow or live")
        self.enabled = config.get("enabled", False)
        self.policy_mode = config.get("mode", "shadow")
        if not number(pcie_frac) or not 0 <= pcie_frac <= 1:
            raise ValueError("coadaptive requires explicit pcie_frac in [0, 1]")
        self.ram_floor = max(2, ram_headroom_gib)
        self.gpu_floor = max(256, vram_reserve_mib)
        self.base_fraction = pcie_frac
        self.mode = "balanced"
        self.reason = "awaiting_telemetry" if self.enabled else "disabled"
        self.last_stamp = self.candidate = self.since = None
        self.last_valid = None
        self.gpu_idle_sample = None

    @property
    def active(self):
        return self.enabled and self.policy_mode == "live"

    def limits(self):
        if not self.enabled:
            return None
        # CPU pressure must not evict GPU experts. GPU pressure must not evict RAM.
        return self.ram_floor, self.gpu_floor + (512 if self.mode == "yield_gpu" else 0)

    def fraction(self, now):
        if not self.enabled or self.last_valid is None or not 0 <= now - self.last_valid <= 5:
            return None
        if self.mode == "yield_cpu":
            return min(.9, self.base_fraction + .25)
        if self.mode == "yield_gpu":
            return max(.05, self.base_fraction - .25)
        return self.base_fraction

    def observe(self, snapshot, now, *, engine_idle=False):
        if not self.enabled:
            return False
        work = snapshot.get("workload", {}) if isinstance(snapshot, dict) else {}
        stamp = snapshot.get("sampled_at") if isinstance(snapshot, dict) else None
        keys = ("ram_total", "ram_used", "gpu_mem_total", "gpu_mem_used")
        valid = (number(stamp) and number(now) and 0 <= now - stamp <= 5
                 and all(number(snapshot.get(k)) for k in keys)
                 and 0 <= snapshot["ram_used"] <= snapshot["ram_total"] and snapshot["ram_total"] > 0
                 and 0 <= snapshot["gpu_mem_used"] <= snapshot["gpu_mem_total"] and snapshot["gpu_mem_total"] > 0
                 and work.get("complete") is True and number(work.get("cpu_percent"))
                 and 0 <= work["cpu_percent"] <= 100)
        if not valid or self.last_stamp is not None and stamp <= self.last_stamp:
            self.candidate = self.since = self.last_valid = None
            self.reason = "telemetry_unavailable"
            return False
        if self.last_stamp is not None and stamp - self.last_stamp > 5:
            self.candidate = self.since = None
        self.last_stamp = self.last_valid = stamp
        gpu_util = snapshot.get("gpu_util")
        if engine_idle and number(gpu_util) and 0 <= gpu_util <= 100:
            self.gpu_idle_sample = (stamp, gpu_util)
        # An attributable provider may supply this; NVML's device total during
        # generation cannot. Idle observations expire instead of becoming facts.
        external_gpu = work.get("gpu_percent")
        if not number(external_gpu) or not 0 <= external_gpu <= 100:
            external_gpu = (self.gpu_idle_sample[1] if self.gpu_idle_sample is not None
                            and 0 <= stamp - self.gpu_idle_sample[0] <= 5 else None)
        free_ram = snapshot["ram_total"] - snapshot["ram_used"]
        commit = snapshot.get("ram_commit_available")
        if snapshot.get("ram_commit_required") or "ram_commit_available" in snapshot:
            if not number(commit) or commit < 0:
                self.candidate = self.since = self.last_valid = None
                self.reason = "commit_unavailable"
                return False
            free_ram = min(free_ram, commit)
        free_gpu = (snapshot["gpu_mem_total"] - snapshot["gpu_mem_used"]) / MIB
        cpu_busy = work["cpu_percent"] >= 20
        gpu_busy = free_gpu < self.gpu_floor or external_gpu is not None and external_gpu >= 60
        ram_busy = free_ram < self.ram_floor * GIB
        # Safety takes precedence; both busy does not invent a free compute tier.
        if gpu_busy:
            candidate = "balanced" if cpu_busy or ram_busy else "yield_gpu"
        elif cpu_busy or ram_busy:
            candidate = "yield_cpu" if free_gpu >= self.gpu_floor + 256 else "balanced"
        else:
            candidate = "balanced"
        # A newly busy destination cancels an existing promotion immediately.
        # The quiet recovery dwell must not keep routing into a congested tier.
        if candidate == "balanced" and (gpu_busy or cpu_busy or ram_busy) and self.mode != "balanced":
            self.mode, self.reason = "balanced", "destination_pressure"
            self.candidate = self.since = None
            return True
        if candidate == self.mode:
            self.candidate = self.since = None
            self.reason = candidate
            return False
        if candidate != self.candidate:
            self.candidate, self.since = candidate, stamp
        # Resource pressure reacts promptly; return to baseline only after quiet.
        if stamp - self.since < (30 if candidate == "balanced" else 8):
            self.reason = "pending_" + candidate
            return False
        self.mode, self.reason = candidate, candidate
        self.candidate = self.since = None
        return True

    def status(self, now):
        return {"enabled": self.enabled, "policy_mode": self.policy_mode, "active": self.active,
                "mode": self.mode, "reason": self.reason,
                "targets": self.limits(), "next_request_pcie_frac": self.fraction(now),
                "routing_boundary": "next request", "multi_gpu": False,
                "storage_aware_routing": False, "dynamic_worker_count": False}
