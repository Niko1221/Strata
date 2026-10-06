"""Bounded cache budgets; the caller owns sampling and applying native allocations.

All telemetry values are bytes, with a caller-supplied ``sampled_at`` wall-clock
timestamp. A proposal is not an allocation: call ``applied`` only after the engine
has successfully loaded it, or ``live_actual`` for native live acknowledgements.
The policy never changes model precision or context.
"""
from __future__ import annotations

import math

GIB = 2**30
MIB = 2**20


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class MemoryPolicy:
    def __init__(self, config=None, resident_cap_gib=42.0, vram_reserve_mib=1536):
        config = config or {}
        self.enabled = config.get("enabled", False) is True
        self.mode = config.get("mode", "reload")
        if self.mode not in ("reload", "live"):
            raise ValueError("memory policy mode must be reload or live")
        self.cap = float(resident_cap_gib)
        self.reserve_floor = int(vram_reserve_mib)
        self.ram_target = self._setting(config, "ram_target_percent", 95, 50, 95) / 100
        self.vram_target = self._setting(config, "vram_target_percent", 99, 50, 99) / 100
        self.headroom = self._setting(config, "min_ram_headroom_gib", 5.5, 2, 128)
        self.configured_headroom = self.headroom
        self.configured_reserve_floor = self.reserve_floor
        self.fixed_resource_limits = False
        self.overhead = self._setting(config, "overhead_ram_gib", 2, 2, 128)
        self.cooldown = self._setting(config, "cooldown_seconds", 600, 600, 86400)
        self.pressure_duration = self._setting(config, "pressure_seconds", 60, 2, 3600)
        self.growth_duration = self._setting(config, "growth_seconds", 120, 120, 3600)
        self.recovery_duration = self._setting(config, "recovery_seconds", 0, 0, 3600)
        if 0 < self.recovery_duration < 30:
            raise ValueError("invalid memory policy setting: recovery_seconds")
        self.max_age = self._setting(config, "max_sample_age_seconds", 5, 1, 60)
        if not _number(self.cap) or self.cap < 1 or not _number(vram_reserve_mib) or self.reserve_floor < 0:
            raise ValueError("memory policy requires a positive resident cap and nonnegative VRAM reserve")
        self.current = None
        self.last_applied = None
        self.last_sample = None
        self.pressure_since = None
        self.growth_since = None
        self.gpu_growth_since = None
        self.gpu_baseline = None
        self.reconcile_after_load = False
        self.pressure_recovery_ceiling_gib = None
        self.recovery_since = None
        self.last_reason = "disabled" if not self.enabled else "awaiting_telemetry"

    @staticmethod
    def _setting(config, name, default, minimum, maximum):
        value = config.get(name, default)
        if not _number(value) or not minimum <= value <= maximum:
            raise ValueError(f"invalid memory policy setting: {name}")
        return float(value)

    def _reading(self, snapshot, now):
        if not isinstance(snapshot, dict) or not _number(now):
            return None
        keys = ("sampled_at", "ram_used", "ram_total", "gpu_mem_used", "gpu_mem_total")
        if not all(_number(snapshot.get(k)) for k in keys):
            return None
        stamp = snapshot["sampled_at"]
        if not 0 <= now - stamp <= self.max_age:
            return None
        for prefix in ("ram", "gpu_mem"):
            used, total = snapshot[prefix + "_used"], snapshot[prefix + "_total"]
            if total <= 0 or not 0 <= used <= total:
                return None
        return snapshot

    def _reset_windows(self):
        self.pressure_since = self.growth_since = None
        self.gpu_growth_since = None
        self.recovery_since = None

    def update_resource_limits(self, headroom_gib=None, vram_reserve_mib=None):
        """Retarget reserves in place; native acknowledgements still own actual sizes.

        Both omitted limits restore the original configured percentage policy.
        Validate the complete pair before changing either limit or its windows.
        """
        fixed = headroom_gib is not None or vram_reserve_mib is not None
        if fixed:
            if (not _number(headroom_gib) or not 2 <= headroom_gib <= 128
                    or not _number(vram_reserve_mib) or vram_reserve_mib < 0
                    or int(vram_reserve_mib) != vram_reserve_mib):
                raise ValueError("invalid fixed resource limits")
            headroom, reserve = float(headroom_gib), int(vram_reserve_mib)
        else:
            headroom, reserve = self.configured_headroom, self.configured_reserve_floor
        if (fixed == self.fixed_resource_limits and headroom == self.headroom
                and reserve == self.reserve_floor):
            return
        self.fixed_resource_limits = fixed
        self.headroom, self.reserve_floor = headroom, reserve
        self.reconcile_after_load = False
        self.pressure_recovery_ceiling_gib = None
        self.gpu_baseline = None
        self._reset_windows()

    def _required_headroom(self, reading):
        if self.fixed_resource_limits:
            return self.headroom
        return max(self.headroom, reading["ram_total"] / GIB * (1 - self.ram_target))

    def _budget(self, reading, arena_gib=0, loaded=False):
        free_ram = (reading["ram_total"] - reading["ram_used"]) / GIB
        headroom = self._required_headroom(reading)
        # Loaded telemetry already includes dense/vision/runtime allocations.
        # Charge the startup allowance only before those allocations exist.
        startup_allowance = 0 if loaded else self.overhead
        resident = max(1.0, min(self.cap, free_ram + arena_gib - headroom - startup_allowance))
        reserve = self.reserve_floor
        if loaded:
            reserve = self.current["vram_reserve_mib"]
            free_vram = (reading["gpu_mem_total"] - reading["gpu_mem_used"]) / MIB
            desired_free = (self.reserve_floor if self.fixed_resource_limits else
                            reading["gpu_mem_total"] / MIB * (1 - self.vram_target))
            # Native auto sizing owns the GPU cache. Reserve changes only reclaim
            # external pressure, or give stable recovered space back to that cache.
            if free_vram < desired_free:
                reserve += math.ceil(desired_free - free_vram)
            elif free_vram - desired_free >= (32 if self.fixed_resource_limits else 512):
                reserve = max(self.reserve_floor, reserve - math.floor(free_vram - desired_free))
            if self.fixed_resource_limits:
                reserve = max(self.reserve_floor, reserve)
        else:
            # At load time the native floor is retained even when the caller asks
            # for a high utilization target; GPU workspace must still fit.
            if not self.fixed_resource_limits:
                reserve = max(reserve, math.ceil(reading["gpu_mem_total"] / MIB * (1 - self.vram_target)))
        return {"resident_budget_gib": round(resident, 3), "vram_reserve_mib": reserve}

    def plan_for_load(self, snapshot, now):
        """Return a fresh bounded budget for a naturally unloaded model."""
        if not self.enabled:
            return None
        reading = self._reading(snapshot, now)
        if reading is None:
            self.last_reason = "telemetry_unavailable"
            return None
        plan = self._budget(reading)
        plan["reason"] = "load_budget"
        return plan

    def applied(self, plan, now):
        """Record only a successfully applied engine allocation."""
        resident, reserve = plan["resident_budget_gib"], plan["vram_reserve_mib"]
        if (not _number(now) or not _number(resident) or not 1 <= resident <= self.cap
                or not _number(reserve) or reserve < self.reserve_floor):
            raise ValueError("invalid applied memory budget")
        self.current = {"resident_budget_gib": resident, "vram_reserve_mib": int(reserve)}
        self.last_applied = now
        self.last_reason = plan.get("reason", "applied")
        self.reconcile_after_load = False
        self.pressure_recovery_ceiling_gib = None
        self.gpu_baseline = None
        self._reset_windows()

    def live_actual(self, resident_mib, reserve_mib, now, reason, completed=False, loaded=False):
        """Native committed sizes can be below the requested minimum after rounding or partial failure."""
        if (not _number(resident_mib) or resident_mib < 0 or resident_mib > self.cap * 1024
                or not _number(reserve_mib) or reserve_mib < 0):
            raise ValueError("invalid live memory allocation")
        self.current = {"resident_budget_gib": resident_mib / 1024,
                        "vram_reserve_mib": int(reserve_mib)}
        self.last_reason = reason
        if completed:
            self.last_applied = now
        self.reconcile_after_load = self.mode == "live" and completed and loaded
        if loaded or reason == "native_error":
            self.pressure_recovery_ceiling_gib = None
        self.gpu_baseline = None
        self._reset_windows()

    def complete_live_plan(self, plan, resident_before_gib, limitation=None):
        """Remember only applied pressure capacity, captured before any progress ACK.

        The service calls this after recording a matching terminal applied ACK.
        Proposals, progress, errors and failed writes never arm recovery.
        """
        if (self.fixed_resource_limits or self.mode != "live" or not self.recovery_duration
                or self.current is None):
            self.pressure_recovery_ceiling_gib = None
            return
        actual = self.current["resident_budget_gib"]
        reason = plan.get("reason")
        if reason == "sustained_pressure":
            if (_number(resident_before_gib) and 0 < resident_before_gib <= self.cap
                    and plan["resident_budget_gib"] < resident_before_gib
                    and actual < resident_before_gib
                    and self.pressure_recovery_ceiling_gib is None):
                self.pressure_recovery_ceiling_gib = resident_before_gib
        elif reason == "pressure_recovery":
            requested = int(plan["resident_budget_gib"] * 1024) / 1024
            if (actual < requested or limitation == "ram_capacity_or_rounding"
                    or self.pressure_recovery_ceiling_gib is None
                    or actual >= self.pressure_recovery_ceiling_gib):
                self.pressure_recovery_ceiling_gib = None
        else:
            self.pressure_recovery_ceiling_gib = None

    def record_loaded(self, snapshot, now):
        """Record the first fresh *idle* GPU reading after successful load.

        The caller must wait for idle: a busy generation's temporary workspace
        is not an external application's allocation. Existing baselines stay
        fixed until the next successful allocation, never follow free-space
        oscillations. A pre-load sample cannot establish a loaded baseline.
        """
        if (not self.enabled or self.fixed_resource_limits or self.current is None
                or self.gpu_baseline is not None):
            return False
        reading = self._reading(snapshot, now)
        if reading is None or reading["sampled_at"] < self.last_applied:
            return False
        self.gpu_baseline = (reading["gpu_mem_total"],
                             (reading["gpu_mem_total"] - reading["gpu_mem_used"]) / MIB)
        return True

    def observe(self, snapshot, loaded, info, now):
        """Return a proposal after sustained pressure/growth; never unload on idle.

        ``info['arena_mib']`` must be the engine's actual resident arena. Unknown
        allocations freeze the decision rather than guessing which RAM is ours.
        Repeated samples, clock regressions and missing samples cannot advance a
        stability window. The caller may inspect status even during generation.
        """
        if not self.enabled:
            return None
        reading = self._reading(snapshot, now)
        if reading is None:
            self._reset_windows()
            self.last_reason = "telemetry_unavailable"
            return None
        stamp = reading["sampled_at"]
        if self.last_sample is not None and stamp <= self.last_sample:
            self._reset_windows()
            self.last_reason = "telemetry_not_advancing"
            return None
        if self.last_sample is not None and stamp - self.last_sample > self.max_age:
            self._reset_windows()
        self.last_sample = stamp
        arena = info.get("arena_mib") if isinstance(info, dict) else None
        if not loaded or self.current is None or not _number(arena) or arena <= 0:
            self._reset_windows()
            if not loaded:
                self.pressure_recovery_ceiling_gib = None
            self.last_reason = "awaiting_allocation" if loaded else "unloaded"
            return None
        if ((self.reconcile_after_load or self.fixed_resource_limits)
                and self.last_applied is not None and stamp <= self.last_applied):
            self._reset_windows()
            self.last_reason = "awaiting_post_load_telemetry"
            return None
        plan = self._budget(reading, arena / 1024, loaded=True)
        if self.fixed_resource_limits:
            return self._observe_fixed(reading, plan, stamp)
        ram_delta = plan["resident_budget_gib"] - self.current["resident_budget_gib"]
        vram_delta = plan["vram_reserve_mib"] - self.current["vram_reserve_mib"]
        free_ram = (reading["ram_total"] - reading["ram_used"]) / GIB
        required_free_ram = self._required_headroom(reading)
        pressure = (reading["ram_used"] / reading["ram_total"] > self.ram_target
                    or free_ram < required_free_ram
                    or reading["gpu_mem_used"] / reading["gpu_mem_total"] > self.vram_target)
        if self.reconcile_after_load:
            # READY includes startup allocations. Reclaim an overestimated RAM
            # allowance once from a fresh reading, without lowering the GPU reserve.
            # Later growth and partial/error ACKs retain the ordinary cooldown.
            self.reconcile_after_load = False
            if not pressure and ram_delta >= 2:
                plan["vram_reserve_mib"] = self.current["vram_reserve_mib"]
                plan["reason"] = self.last_reason = "post_load_headroom"
                self._reset_windows()
                return plan
        # Live CUDA cache mappings commit in 32 MiB blocks. At the 99% target,
        # a 16 GiB card's entire pressure deficit is only 164 MiB; the legacy
        # reload threshold would make GPU-only reclamation unreachable there.
        shrink_vram = 32 if self.mode == "live" else 256
        grow_vram = 32 if self.mode == "live" else 512
        shrink = pressure and (ram_delta <= -1 or vram_delta >= shrink_vram)
        grow = not pressure and (ram_delta >= 2 or vram_delta <= -grow_vram)
        cache = info.get("expert_cache_mib")
        gpu_grow = (not pressure and self.gpu_baseline is not None
                    and reading["gpu_mem_total"] == self.gpu_baseline[0]
                    and _number(cache) and cache > 0
                    and (reading["gpu_mem_total"] - reading["gpu_mem_used"]) / MIB
                    - self.gpu_baseline[1] >= 512)
        self.pressure_since = (stamp if self.pressure_since is None else self.pressure_since) if shrink else None
        self.growth_since = (stamp if self.growth_since is None else self.growth_since) if grow else None
        self.gpu_growth_since = (stamp if self.gpu_growth_since is None else self.gpu_growth_since) if gpu_grow else None
        recover = (self.mode == "live" and self.recovery_duration > 0
                   and self.pressure_recovery_ceiling_gib is not None
                   and not pressure and stamp > self.last_applied and ram_delta >= 2
                   and free_ram - required_free_ram >= 2
                   and self.pressure_recovery_ceiling_gib - self.current["resident_budget_gib"] >= 2)
        self.recovery_since = (stamp if self.recovery_since is None else self.recovery_since) if recover else None
        reason = None
        if self.pressure_since is not None and stamp - self.pressure_since >= self.pressure_duration:
            reason = "sustained_pressure"
            # Pressure must never grow a different cache during the same reload.
            plan["resident_budget_gib"] = min(plan["resident_budget_gib"], self.current["resident_budget_gib"])
            plan["vram_reserve_mib"] = max(plan["vram_reserve_mib"], self.current["vram_reserve_mib"])
        elif self.recovery_since is not None and stamp - self.recovery_since >= self.recovery_duration:
            reason = "pressure_recovery"
            # Only restore previously admitted RAM, in material two-GiB steps.
            # Keep exact MiB arithmetic and leave GPU sizing to ordinary growth.
            plan = {"resident_budget_gib": self.current["resident_budget_gib"] + 2,
                    "vram_reserve_mib": self.current["vram_reserve_mib"]}
        elif self.growth_since is not None and stamp - self.growth_since >= self.growth_duration:
            reason = "stable_headroom"
            # Growth must not shrink a different cache without pressure.
            plan["resident_budget_gib"] = max(plan["resident_budget_gib"], self.current["resident_budget_gib"])
            plan["vram_reserve_mib"] = min(plan["vram_reserve_mib"], self.current["vram_reserve_mib"])
        elif self.gpu_growth_since is not None and stamp - self.gpu_growth_since >= self.growth_duration:
            # Even at the reserve floor, native auto sizing can use newly freed
            # external VRAM after a safe reload with exactly the same arguments.
            reason = "stable_gpu_headroom"
            plan = dict(self.current)
        if reason is None:
            self.last_reason = "pressure_debounce" if shrink else "growth_debounce" if grow or gpu_grow else "stable"
            return None
        # Ordinary growth waits for the long cooldown. Pressure relief and
        # bounded restoration of previously admitted RAM earn separate windows.
        if reason not in ("sustained_pressure", "pressure_recovery") and now - self.last_applied < self.cooldown:
            self.last_reason = "cooldown"
            return None
        plan["reason"] = reason
        self.last_reason = reason
        return plan

    def _observe_fixed(self, reading, plan, stamp):
        """Fixed targets use bounded fresh growth and the existing pressure window."""
        self.reconcile_after_load = False
        ram_delta = plan["resident_budget_gib"] - self.current["resident_budget_gib"]
        vram_delta = plan["vram_reserve_mib"] - self.current["vram_reserve_mib"]
        free_ram = (reading["ram_total"] - reading["ram_used"]) / GIB
        free_vram = (reading["gpu_mem_total"] - reading["gpu_mem_used"]) / MIB
        pressure = (free_ram < self.headroom or free_vram < self.reserve_floor
                    or self.current["vram_reserve_mib"] < self.reserve_floor)
        shrink = pressure and (ram_delta <= -1 or vram_delta >= 32)
        # Use raw headroom as well as the rounded budget: rounding cannot earn
        # a two-GiB expansion from slightly less safe space.
        ram_grow = ram_delta >= 2 and free_ram - self.headroom >= 2
        vram_grow = vram_delta <= -32
        grow = not pressure and (ram_grow or vram_grow)
        self.pressure_since = (stamp if self.pressure_since is None else self.pressure_since) if shrink else None
        self.growth_since = (stamp if self.growth_since is None else self.growth_since) if not pressure and ram_grow else None
        self.gpu_growth_since = (stamp if self.gpu_growth_since is None else self.gpu_growth_since) if not pressure and vram_grow else None
        ram_ready = self.growth_since is not None and stamp - self.growth_since >= 30
        vram_ready = self.gpu_growth_since is not None and stamp - self.gpu_growth_since >= 30
        if self.pressure_since is not None and stamp - self.pressure_since >= self.pressure_duration:
            # Reclamation must not grow the other cache, even after retargeting.
            plan["resident_budget_gib"] = min(plan["resident_budget_gib"], self.current["resident_budget_gib"])
            plan["vram_reserve_mib"] = max(plan["vram_reserve_mib"], self.current["vram_reserve_mib"])
            reason = "sustained_pressure"
        elif ram_ready or vram_ready:
            plan = {"resident_budget_gib": self.current["resident_budget_gib"] + 2 if ram_ready else
                    self.current["resident_budget_gib"],
                    "vram_reserve_mib": max(plan["vram_reserve_mib"],
                                            self.current["vram_reserve_mib"] - 128) if vram_ready else
                    self.current["vram_reserve_mib"]}
            reason = "stable_headroom"
        else:
            self.last_reason = "pressure_debounce" if shrink else "growth_debounce" if grow else "stable"
            return None
        plan["reason"] = self.last_reason = reason
        return plan

    def status(self):
        return {"enabled": self.enabled, "mode": self.mode, "current": dict(self.current) if self.current else None,
                "reason": self.last_reason, "last_applied_at": self.last_applied,
                "resident_cap_gib": self.cap, "vram_reserve_floor_mib": self.reserve_floor,
                "ram_target_percent": self.ram_target * 100, "vram_target_percent": self.vram_target * 100,
                "min_ram_headroom_gib": self.headroom, "cooldown_seconds": self.cooldown,
                "fixed_resource_limits": self.fixed_resource_limits,
                "resource_targets": {"headroom_gib": self.headroom,
                                     "vram_reserve_mib": self.reserve_floor} if self.fixed_resource_limits else None,
                "recovery_seconds": self.recovery_duration,
                "pressure_recovery_ceiling_gib": self.pressure_recovery_ceiling_gib,
                "gpu_baseline_free_mib": self.gpu_baseline[1] if self.gpu_baseline is not None else None}
