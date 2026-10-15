"""Fixed reserve targets and private process aggregates; no allocation or persistence.

The server owns the single telemetry loop and applies changed limits to its memory
policy. Process observations never include command lines, windows or content.
"""
from __future__ import annotations

import math
import os
import time

GIB = 2**30
CATALOG = {
    "full": {"label": "Full", "headroom_gib": 2, "vram_reserve_mib": 256},
    "daily": {"label": "Daily", "headroom_gib": 4, "vram_reserve_mib": 700},
    "busy": {"label": "Busy", "headroom_gib": 8, "vram_reserve_mib": 1536},
}
MAX_SAMPLE_GAP = 3.0  # telemetry is 1s; its existing policy observer runs every 2s


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean_config(config):
    """Validate the whole opt-in block before the caller persists or mutates it."""
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config) - {"enabled", "selection"}:
        raise ValueError("resource_presets accepts only enabled and selection")
    enabled, selection = config.get("enabled", False), config.get("selection", "auto")
    if not isinstance(enabled, bool):
        raise ValueError("resource_presets enabled must be a boolean")
    if not isinstance(selection, str) or selection not in ("auto", "full", "daily", "busy"):
        raise ValueError("resource_presets selection must be auto, full, daily or busy")
    return {"enabled": enabled, "selection": selection}


class ResourcePresets:
    clean_config = staticmethod(clean_config)

    def __init__(self, config=None):
        self.enabled = False
        self.selection = "auto"
        self.effective = None
        self.reason = "disabled"
        self._last_sample = self._candidate = self._since = None
        self.configure(config)

    def configure(self, config):
        config = clean_config(config)
        before = self.limits()
        if self.enabled == config["enabled"] and self.selection == config["selection"]:
            return False
        self.enabled, self.selection = config["enabled"], config["selection"]
        self._last_sample = self._candidate = self._since = None
        self.effective = ("daily" if self.selection == "auto" else self.selection) if self.enabled else None
        self.reason = ("awaiting_workload" if self.selection == "auto" else "manual") if self.enabled else "disabled"
        return self.limits() != before

    def limits(self):
        if not self.enabled:
            return None
        profile = CATALOG[self.effective]
        return profile["headroom_gib"], profile["vram_reserve_mib"]

    def status(self):
        profile = CATALOG.get(self.effective)
        return {"enabled": self.enabled, "selection": self.selection, "effective": self.effective,
                "reason": self.reason,
                "targets": ({k: profile[k] for k in ("headroom_gib", "vram_reserve_mib")} if profile else None),
                "catalog": {key: dict(value) for key, value in CATALOG.items()}}

    def _unavailable(self):
        self._last_sample = self._candidate = self._since = None
        self.reason = "workload_unavailable"
        return False

    def observe(self, snapshot, now):
        """Earn dwell only on fresh, advancing, complete workload aggregates."""
        if not self.enabled or self.selection != "auto":
            return False
        if not isinstance(snapshot, dict) or not _number(now):
            return self._unavailable()
        stamp, work = snapshot.get("sampled_at"), snapshot.get("workload")
        if (not _number(stamp) or not 0 <= now - stamp <= MAX_SAMPLE_GAP or not isinstance(work, dict)
                or work.get("complete") is not True
                or not _number(work.get("cpu_percent")) or not 0 <= work["cpu_percent"] <= 100
                or not _number(work.get("rss_bytes")) or work["rss_bytes"] < 0):
            return self._unavailable()
        if self._last_sample is not None:
            if stamp == self._last_sample:
                return False
            if stamp < self._last_sample:
                return self._unavailable()
            if stamp - self._last_sample > MAX_SAMPLE_GAP:
                self._candidate = self._since = None
        self._last_sample = stamp
        if work["cpu_percent"] >= 20 or work["rss_bytes"] >= 8 * GIB:
            candidate, reason = "busy", "busy_workload"
        elif work["cpu_percent"] >= 5 or work["rss_bytes"] >= 2 * GIB:
            candidate, reason = "daily", "moderate_workload"
        else:
            candidate, reason = "full", "light_workload"
        if candidate == self.effective:
            self._candidate = self._since = None
            self.reason = reason
            return False
        if candidate != self._candidate:
            self._candidate, self._since = candidate, stamp
        ranks = {"full": 0, "daily": 1, "busy": 2}
        duration = 8 if ranks[candidate] > ranks[self.effective] else 60
        if stamp - self._since < duration:
            self.reason = "pending_" + candidate
            return False
        self.effective, self.reason = candidate, reason
        self._candidate = self._since = None
        return True


class WorkloadSampler:
    # Known Windows OS processes can deny their resource counters even to the
    # desktop user. They are not eligible application workloads. Unknown names
    # or missing counters of eligible processes still make the sample incomplete.
    _SYSTEM_NAMES = frozenset(("system idle process", "system", "registry", "memory compression",
                               "secure system", "smss", "csrss", "wininit", "services", "lsass",
                               "winlogon", "svchost", "fontdrvhost", "dwm"))
    _ATTRS = ("pid", "name", "ppid", "create_time", "cpu_times", "memory_info")
    MAX_PROCESSES = 4096
    MAX_SCAN_SECONDS = .75

    def __init__(self, ps):
        self.ps = ps
        self._previous = {}
        self._last_sample = None
        self._last_clock = self._last_source = None
        self._bulk = None
        # Fake psutil providers used by tests keep the portable path. Importing
        # this optional sensor never affects Linux or unsupported Windows ABIs.
        if os.name == "nt" and getattr(ps, "__name__", None) == "psutil":
            try:
                try:
                    from .windows_process_snapshot import WindowsProcessSnapshot
                except ImportError:
                    from windows_process_snapshot import WindowsProcessSnapshot
                self._bulk = WindowsProcessSnapshot()
            except (ImportError, OSError, AttributeError):
                pass

    @staticmethod
    def unavailable():
        return {"complete": False, "cpu_percent": None, "rss_bytes": None}

    def sample(self, now, exclude_pids=()):
        if self.ps is None or not _number(now):
            self._previous, self._last_sample = {}, None
            self._last_clock = self._last_source = None
            return self.unavailable()
        complete, processes = True, {}
        source, clock_now = "psutil", now
        started = time.monotonic()
        try:
            cores = self.ps.cpu_count(logical=True)
            if not _number(cores) or cores < 1:
                raise ValueError("CPU count unavailable")
            if self._bulk is not None:
                try:
                    snapshot = self._bulk.sample(self.MAX_PROCESSES, self.MAX_SCAN_SECONDS)
                    if (not _number(snapshot.wall_time) or not _number(snapshot.monotonic_time)
                            or abs(snapshot.wall_time - now) > MAX_SAMPLE_GAP):
                        raise ValueError("native process snapshot is stale")
                    processes, now, clock_now = snapshot.processes, snapshot.wall_time, snapshot.monotonic_time
                    source = "windows_bulk"
                except Exception:  # noqa: BLE001 - unavailable native counters use conservative fallback
                    pass
            if source == "psutil":
                for index, proc in enumerate(self.ps.process_iter(attrs=list(self._ATTRS), ad_value=None)):
                    if index >= self.MAX_PROCESSES or time.monotonic() - started > self.MAX_SCAN_SECONDS:
                        complete = False
                        break
                    info = proc.info
                    pid = info.get("pid")
                    if isinstance(pid, bool) or not isinstance(pid, int):
                        complete = False
                        continue
                    processes[pid] = info
        except Exception:  # noqa: BLE001 - optional process sensors must never stop telemetry
            self._previous, self._last_sample = {}, None
            self._last_clock = self._last_source = None
            return self.unavailable()
        excluded = set(exclude_pids) | {os.getpid()}
        names = {}
        for pid, info in processes.items():
            name = info.get("name")
            name = name.lower().removesuffix(".exe") if isinstance(name, str) else None
            names[pid] = name
            if name == "strata":
                excluded.add(pid)
        # Resolve descendants without depending on process enumeration order.
        children = {}
        for pid, info in processes.items():
            parent = info.get("ppid")
            parent_info = processes.get(parent)
            created = info.get("create_time_ticks", info.get("create_time"))
            parent_created = parent_info.get("create_time_ticks", parent_info.get("create_time")) if parent_info else None
            # Do not attach a child to an unrelated process reusing its old PPID.
            if _number(created) and _number(parent_created) and parent_created > created:
                continue
            children.setdefault(parent, []).append(pid)
        pending = list(excluded)
        while pending:
            for pid in children.get(pending.pop(), ()):
                if pid not in excluded:
                    excluded.add(pid)
                    pending.append(pid)
        elapsed = clock_now - self._last_clock if self._last_clock is not None and source == self._last_source else None
        advancing = elapsed is not None and 0 < elapsed <= MAX_SAMPLE_GAP
        complete = complete and advancing
        previous, cpu, rss = {}, 0.0, 0
        for pid, info in processes.items():
            if pid in excluded or pid in (0, 4):
                continue
            name = names[pid]
            # Windows exposes Memory Compression with an empty name on some
            # psutil versions. Its direct System parent identifies a kernel
            # process; unknown application names still fail closed below.
            unnamed_kernel = os.name == "nt" and not name and info.get("ppid") == 4
            if name in self._SYSTEM_NAMES or unnamed_kernel:
                continue
            if not name:
                complete = False
            created, times, memory = info.get("create_time"), info.get("cpu_times"), info.get("memory_info")
            user, system = getattr(times, "user", None), getattr(times, "system", None)
            resident = getattr(memory, "rss", None)
            if (not all(_number(value) for value in (created, user, system, resident))
                    or min(created, user, system, resident) < 0 or created > now):
                complete = False
                continue
            rss += resident
            total = user + system
            identity = (pid, info.get("create_time_ticks", created))
            previous[identity] = total
            baseline = self._previous.get(identity)
            if advancing and baseline is None:
                # A process born within this interval has accumulated no CPU
                # before it. Account its own lifetime instead of invalidating
                # every aggregate during healthy CLI creation. An older unseen
                # identity still needs a baseline; never borrow another PID's.
                if created >= self._last_sample and total <= elapsed * cores:
                    baseline = 0.0
            if not advancing or baseline is None or total < baseline:
                complete = False
                continue
            cpu += (total - baseline) / elapsed / cores * 100
        self._previous, self._last_sample = previous, now
        self._last_clock, self._last_source = clock_now, source
        return {"complete": complete, "cpu_percent": min(100.0, max(0.0, cpu)), "rss_bytes": rss}
