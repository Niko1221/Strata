"""Independent desktop resource signals; native MEMORY remains the only allocator.

Experimental single-GPU policy. Shadow mode records recommendations by default.
Explicit live mode adjusts cache reserves. The existing per-request CPU/GPU
expert split changes only with matched calibration showing a clear cost benefit.
Capacity uses conservative thresholds; this is not a tensor-placement optimizer.
It never changes model precision or context.
GPU utilization while Strata is generating is NOT evidence of another workload.
"""
from __future__ import annotations

import math
from serve.routing_costs import RoutingCosts

GIB, MIB = 2**30, 2**20


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class CoAdaptive:
    def __init__(self, config=None, *, ram_headroom_gib=4, vram_reserve_mib=256, pcie_frac=.5):
        config = {} if config is None else config
        if (not isinstance(config, dict) or set(config) - {"enabled", "mode", "routing_profile", "min_free_vram_mib"}
                or not isinstance(config.get("enabled", False), bool)
                or config.get("mode", "shadow") not in ("shadow", "live")):
            raise ValueError("coadaptive accepts boolean enabled and mode shadow or live")
        self.enabled = config.get("enabled", False)
        self.policy_mode = config.get("mode", "shadow")
        self.routing = RoutingCosts.load(config.get("routing_profile"))
        self.runtime_key = None
        self.selected_fraction = pcie_frac
        if not number(pcie_frac) or not 0 <= pcie_frac <= 1:
            raise ValueError("coadaptive requires explicit pcie_frac in [0, 1]")
        self.ram_floor = max(2, ram_headroom_gib)
        floor = config.get("min_free_vram_mib", max(256, vram_reserve_mib))
        if not number(floor) or not 256 <= floor <= 262144 or int(floor) != floor:
            raise ValueError("min_free_vram_mib must be an integer of at least 256")
        # The startup fitting reserve and the running allocator floor serve
        # different purposes: lazy graph allocations consume part of startup room.
        self.gpu_floor = floor
        self.base_fraction = pcie_frac
        self.mode = "balanced"
        self.reason = "awaiting_telemetry" if self.enabled else "disabled"
        self.last_stamp = self.candidate = self.since = None
        self.last_valid = None
        self.gpu_idle_sample = None
        self.idle_since = None

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
        return self.selected_fraction

    def fairness_decision(self, snapshot, now):
        """Bounded cooperative delay, not an unmeasured CPU/GPU route change.

        The service applies this via a short native lease at safe boundaries.
        Unknown or stale external work cannot be inferred from Strata's own
        device utilization. A delay offers time to other work; it does not claim
        a scheduling guarantee or free resources that the engine must retain.
        """
        result = {"delay_ms": 0, "reason": "disabled"}
        if not self.active:
            return result
        work = snapshot.get("workload", {}) if isinstance(snapshot, dict) else {}
        work = work if isinstance(work, dict) else {}
        stamp = snapshot.get("sampled_at") if isinstance(snapshot, dict) else None
        if (not number(now) or not number(stamp) or not 0 <= now - stamp <= 5
                or work.get("complete") is not True):
            result["reason"] = "workload_unavailable"
            return result
        cpu, gpu = work.get("cpu_percent"), work.get("gpu_percent")
        cpu = cpu if number(cpu) and 0 <= cpu <= 100 else None
        gpu = gpu if number(gpu) and 0 <= gpu <= 100 else None
        if cpu is None and gpu is None:
            result["reason"] = "workload_unavailable"
            return result
        cpu_delay = max(0, (cpu - 20) / 4) if cpu is not None else 0
        gpu_delay = max(0, (gpu - 60) / 2) if gpu is not None else 0
        result["delay_ms"] = min(20, math.ceil(max(cpu_delay, gpu_delay)))
        result["reason"] = ("external_cpu_and_gpu" if cpu_delay and gpu_delay else
                            "external_cpu" if cpu_delay else "external_gpu" if gpu_delay else "quiet")
        return result

    def choose_route(self, snapshot, native, context, output_tokens, now, sampling=None):
        snapshot = {**snapshot, "workload": dict(snapshot.get("workload", {}))}
        if snapshot["workload"].get("gpu_percent") is None and self.gpu_idle_sample is not None:
            stamp, value = self.gpu_idle_sample
            if 0 <= now - stamp <= 5:
                snapshot["workload"]["gpu_percent"] = value
        # RAM pressure limits CPU promotion independently of CPU utilization.
        total_ram, used_ram = snapshot.get("ram_total"), snapshot.get("ram_used")
        free_ram = total_ram - used_ram if (number(total_ram) and number(used_ram)
                                             and 0 <= used_ram <= total_ram) else None
        if snapshot.get("ram_commit_required") or "ram_commit_available" in snapshot:
            commit = snapshot.get("ram_commit_available")
            free_ram = min(free_ram, commit) if (free_ram is not None and number(commit) and commit >= 0) else None
        free_gpu = native.get("free_mib") if isinstance(native, dict) else None
        free_gpu = free_gpu if number(free_gpu) and free_gpu >= 0 else None
        if isinstance(native, dict) and "budget_free_mib" in native:
            budget = native["budget_free_mib"]
            # CUDA free bytes may exceed the current WDDM resident budget. A
            # numeric zero is the strongest pressure signal, not missing data.
            free_gpu = min(free_gpu, max(0, budget)) if (free_gpu is not None and number(budget)) else None
        self.selected_fraction = self.routing.choose(
            key=self.runtime_key, baseline=self.base_fraction, current=self.selected_fraction,
            snapshot=snapshot, native=native, context=context, prompt_read=context,
            output_tokens=output_tokens, now=now, sampling=sampling,
            allow_more_gpu=free_gpu is not None and free_gpu >= self.gpu_floor,
            allow_more_cpu=free_ram is not None and free_ram >= self.ram_floor * GIB)
        return self.selected_fraction

    def observe(self, snapshot, now, *, engine_idle=False):
        if not self.enabled:
            return False
        work = snapshot.get("workload", {}) if isinstance(snapshot, dict) else {}
        work = work if isinstance(work, dict) else {}
        stamp = snapshot.get("sampled_at") if isinstance(snapshot, dict) else None
        keys = ("ram_total", "ram_used", "gpu_mem_total", "gpu_mem_used")
        valid = (number(stamp) and number(now) and 0 <= now - stamp <= 5
                 and (not snapshot.get("native_capacity_required") or number(snapshot.get("native_free_mib")))
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
        self.idle_since = (stamp if self.idle_since is None else self.idle_since) if engine_idle else None
        # Device utilization is averaged over a recent interval. Immediately
        # after generation it may still describe Strata's own work.
        if (engine_idle and stamp - self.idle_since >= 2
                and number(gpu_util) and 0 <= gpu_util <= 100):
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
        gpu_compute_busy = external_gpu is not None and external_gpu >= 60
        gpu_capacity_busy = free_gpu < self.gpu_floor
        gpu_busy = gpu_capacity_busy or gpu_compute_busy
        ram_busy = free_ram < self.ram_floor * GIB
        # Safety takes precedence; both busy does not invent a free compute tier.
        if gpu_busy:
            # Low free VRAM proves capacity pressure, not external GPU compute.
            # The memory policy already reclaims toward the running floor. Own
            # prompt workspace must not manufacture a +512 MiB foreground target.
            candidate = "yield_gpu" if gpu_compute_busy and not (cpu_busy or ram_busy) else "balanced"
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
                "routing_costs": self.routing.status(),
                "routing_boundary": "next request", "multi_gpu": False,
                "storage_aware_routing": False, "dynamic_worker_count": False}
