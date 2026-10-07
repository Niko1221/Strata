"""Preset decisions and private workload sampling, without processes or a model."""
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from serve.resource_presets import GIB, ResourcePresets, WorkloadSampler, clean_config
    from serve.telemetry import Telemetry
except ModuleNotFoundError:  # support the repository's direct script convention
    from resource_presets import GIB, ResourcePresets, WorkloadSampler, clean_config
    from telemetry import Telemetry


def reading(now, cpu=0, rss=0, complete=True):
    return {"sampled_at": now, "workload": {"complete": complete,
                                          "cpu_percent": cpu, "rss_bytes": rss}}


class PresetTests(unittest.TestCase):
    def observe_until(self, presets, start, stop, **kwargs):
        for now in range(start, stop + 1):
            presets.observe(reading(now, **kwargs), now)

    def test_disabled_is_legacy_and_catalog_cannot_mutate_policy(self):
        presets = ResourcePresets()
        self.assertIsNone(presets.limits())
        self.assertFalse(presets.observe(reading(1, cpu=100), 1))
        status = presets.status()
        status["catalog"]["full"]["headroom_gib"] = 100
        self.assertEqual(presets.status()["catalog"]["full"]["headroom_gib"], 2)

    def test_config_rejects_invalid_complete_block_without_mutating(self):
        presets = ResourcePresets({"enabled": True, "selection": "busy"})
        for invalid in ([], True, {"enabled": 1}, {"selection": "fast"},
                        {"enabled": True, "selection": "full", "unknown": True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                presets.configure(invalid)
            self.assertEqual(presets.limits(), (8, 1536))
        self.assertEqual(clean_config(None), {"enabled": False, "selection": "auto"})

    def test_auto_initial_daily_and_absence_requires_sixty_seconds(self):
        presets = ResourcePresets({"enabled": True})
        self.assertEqual(presets.limits(), (4, 700))
        self.observe_until(presets, 0, 59)
        self.assertEqual(presets.status()["effective"], "daily")
        self.assertTrue(presets.observe(reading(60), 60))
        self.assertEqual(presets.limits(), (2, 256))

    def test_busy_cpu_or_rss_requires_eight_seconds_and_relaxes_slowly(self):
        for pressure in ({"cpu": 20}, {"rss": 8 * GIB}):
            presets = ResourcePresets({"enabled": True})
            self.observe_until(presets, 0, 7, **pressure)
            self.assertEqual(presets.status()["effective"], "daily")
            self.observe_until(presets, 8, 8, **pressure)
            self.assertEqual(presets.limits(), (8, 1536))
            self.observe_until(presets, 9, 68, rss=2 * GIB)
            self.assertEqual(presets.status()["effective"], "busy")
            self.observe_until(presets, 69, 69, rss=2 * GIB)
            self.assertEqual(presets.status()["effective"], "daily")

    def test_moderate_workload_returns_full_to_daily_after_eight_seconds(self):
        presets = ResourcePresets({"enabled": True, "selection": "full"})
        presets.configure({"enabled": True, "selection": "auto"})
        self.assertEqual(presets.status()["effective"], "daily")
        self.observe_until(presets, 0, 60)
        self.assertEqual(presets.status()["effective"], "full")
        self.observe_until(presets, 61, 68, cpu=5)
        self.assertEqual(presets.status()["effective"], "full")
        self.observe_until(presets, 69, 69, cpu=5)
        self.assertEqual(presets.status()["effective"], "daily")

    def test_manual_latches_and_disable_restores_legacy(self):
        presets = ResourcePresets({"enabled": True, "selection": "full"})
        self.observe_until(presets, 0, 100, cpu=90)
        self.assertEqual(presets.limits(), (2, 256))
        presets.configure({"enabled": False, "selection": "busy"})
        self.assertIsNone(presets.limits())

    def test_unchanged_config_preserves_effective_and_dwell_at_observer_cadence(self):
        presets = ResourcePresets({"enabled": True})
        for now in range(0, 60, 2):
            presets.observe(reading(now), now)
        self.assertFalse(presets.configure({"enabled": True, "selection": "auto"}))
        self.assertTrue(presets.observe(reading(60), 60))
        self.assertEqual(presets.status()["effective"], "full")
        self.assertFalse(presets.configure({"enabled": True}))
        self.assertEqual(presets.status()["effective"], "full")

    def test_gaps_invalid_stale_or_duplicate_cannot_earn_absence(self):
        for interrupted in (reading(30, complete=False), reading(30, cpu=float("nan")),
                            reading(20), reading(30, rss=None)):
            presets = ResourcePresets({"enabled": True})
            self.observe_until(presets, 0, 29)
            presets.observe(interrupted, 30)
            self.observe_until(presets, 31, 90)
            self.assertEqual(presets.status()["effective"], "daily")
            self.observe_until(presets, 91, 91)
            self.assertEqual(presets.status()["effective"], "full")
        presets = ResourcePresets({"enabled": True})
        self.observe_until(presets, 0, 59)
        self.assertFalse(presets.observe(reading(59), 60))
        self.assertEqual(presets.status()["effective"], "daily")
        self.observe_until(presets, 120, 179)
        self.assertEqual(presets.status()["effective"], "daily")


def process(pid, name="editor.exe", ppid=0, created=0, cpu=0, rss=1, **changes):
    info = {"pid": pid, "name": name, "ppid": ppid, "create_time": created,
            "cpu_times": SimpleNamespace(user=cpu, system=0),
            "memory_info": SimpleNamespace(rss=rss)}
    info.update(changes)
    return SimpleNamespace(pid=pid, info=info)


class SamplerTests(unittest.TestCase):
    def sampler(self, processes):
        ps = mock.Mock()
        ps.cpu_count.return_value = 4
        ps.process_iter.side_effect = lambda **kwargs: iter(processes)
        return WorkloadSampler(ps), ps

    def test_whole_machine_cpu_pid_reuse_and_private_aggregate(self):
        processes = [process(101, name="Codex.exe", cpu=1, rss=2 * GIB)]
        sampler, ps = self.sampler(processes)
        self.assertFalse(sampler.sample(0)["complete"])
        processes[0] = process(101, name="Codex.exe", cpu=2, rss=2 * GIB)
        sample = sampler.sample(1)
        self.assertEqual(sample, {"complete": True,
                                  "cpu_percent": 25, "rss_bytes": 2 * GIB})
        processes[0] = process(101, name="Codex.exe", created=2, cpu=200)
        self.assertFalse(sampler.sample(2)["complete"])
        self.assertNotIn("cmdline", ps.process_iter.call_args.kwargs["attrs"])

    def test_healthy_cli_birth_every_sample_can_earn_busy_dwell(self):
        processes = [process(101, rss=9 * GIB)]
        sampler, _ = self.sampler(processes)
        presets = ResourcePresets({"enabled": True})
        first = sampler.sample(0)
        self.assertFalse(first["complete"])
        presets.observe({"sampled_at": 0, "workload": first}, 0)
        for now in range(1, 10):
            processes[:] = [process(101, rss=9 * GIB),
                            process(200 + now, name="python.exe", created=now - .5, cpu=.25)]
            workload = sampler.sample(now)
            self.assertTrue(workload["complete"])
            self.assertEqual(workload["cpu_percent"], 6.25)
            changed = presets.observe({"sampled_at": now, "workload": workload}, now)
            self.assertEqual(changed, now == 9)
        self.assertEqual(presets.status()["effective"], "busy")

    def test_empty_first_scan_is_incomplete_until_advancing_sample(self):
        sampler, _ = self.sampler([])
        self.assertFalse(sampler.sample(0)["complete"])
        self.assertTrue(sampler.sample(1)["complete"])
        self.assertFalse(sampler.sample(1)["complete"])

    def test_reappearing_older_or_future_identity_still_incomplete(self):
        for created in (.5, 2.5):
            processes = [process(101)]
            sampler, _ = self.sampler(processes)
            sampler.sample(0)
            self.assertTrue(sampler.sample(1)["complete"])
            processes.append(process(102, created=created, cpu=.25))
            self.assertFalse(sampler.sample(2)["complete"])
        processes = [process(101)]
        sampler, _ = self.sampler(processes)
        sampler.sample(0)
        processes.clear()
        self.assertTrue(sampler.sample(1)["complete"])
        processes.append(process(101, created=0))
        self.assertFalse(sampler.sample(2)["complete"])

    def test_genuinely_reused_pid_counts_new_lifetime_without_old_cpu_delta(self):
        processes = [process(101, cpu=100)]
        sampler, _ = self.sampler(processes)
        sampler.sample(0)
        self.assertTrue(sampler.sample(1)["complete"])
        processes[:] = [process(101, created=1.5, cpu=.5)]
        sample = sampler.sample(2)
        self.assertTrue(sample["complete"])
        self.assertEqual(sample["cpu_percent"], 12.5)

    def test_excludes_native_and_server_descendants_in_any_order(self):
        processes = [process(104, ppid=103, rss=9 * GIB), process(103, ppid=102, cpu=50),
                     process(102, name="strata.exe", cpu=50), process(201, name="ChatGPT.exe"),
                     process(301, ppid=300, rss=9 * GIB), process(300)]
        sampler, _ = self.sampler(processes)
        sampler.sample(0, exclude_pids=(300,))
        sample = sampler.sample(1, exclude_pids=(300,))
        self.assertTrue(sample["complete"])
        self.assertNotIn("codex_present", sample)
        self.assertEqual(sample["rss_bytes"], 1)
        self.assertEqual(sample["cpu_percent"], 0)

    def test_unreadable_system_is_skipped_but_missing_eligible_stats_incomplete(self):
        processes = [process(4, name=None, cpu_times=None, memory_info=None, create_time=None),
                     process(50, name="lsass.exe", cpu_times=None, memory_info=None, create_time=None),
                     process(101)]
        sampler, _ = self.sampler(processes)
        sampler.sample(0)
        self.assertTrue(sampler.sample(1)["complete"])
        processes.append(process(102, name="Codex.exe", cpu_times=None))
        self.assertFalse(sampler.sample(2)["complete"])

    def test_windows_unnamed_kernel_child_does_not_freeze_valid_samples(self):
        # psutil returns an empty name for this workstation's Memory Compression
        # process, a direct child of the Windows System process (PID 4).
        processes = [process(236, name="", ppid=4, rss=300 * 2**20), process(101)]
        sampler, _ = self.sampler(processes)
        with mock.patch.object(__import__(WorkloadSampler.__module__, fromlist=["os"]).os, "name", "nt"):
            sampler.sample(0)
            sample = sampler.sample(1)
        self.assertTrue(sample["complete"])
        self.assertEqual(sample["rss_bytes"], 1)

    def test_unknown_application_name_remains_incomplete(self):
        sampler, _ = self.sampler([process(101, name="")])
        sampler.sample(0)
        self.assertFalse(sampler.sample(1)["complete"])

    def test_scan_count_and_time_bounds_fail_closed(self):
        sampler, _ = self.sampler([process(101), process(102)])
        sampler.MAX_PROCESSES = 1
        self.assertFalse(sampler.sample(0)["complete"])
        self.assertFalse(sampler.sample(1)["complete"])
        sampler.MAX_PROCESSES = 4096
        module = __import__(WorkloadSampler.__module__, fromlist=["time"])
        with mock.patch.object(module.time, "monotonic", side_effect=(0, 1)):
            self.assertFalse(sampler.sample(2)["complete"])

    def test_reused_parent_pid_does_not_exclude_existing_external_process(self):
        processes = [process(101, ppid=300, created=1, rss=2 * GIB),
                     process(300, created=2)]
        sampler, _ = self.sampler(processes)
        sampler.sample(0, exclude_pids=(300,))
        sample = sampler.sample(1, exclude_pids=(300,))
        self.assertTrue(sample["complete"])
        self.assertEqual(sample["rss_bytes"], 2 * GIB)

    def test_optional_dependency_errors_and_gaps_fail_closed(self):
        self.assertFalse(WorkloadSampler(None).sample(1)["complete"])
        sampler, ps = self.sampler([process(101)])
        sampler.sample(0)
        self.assertTrue(sampler.sample(1)["complete"])
        self.assertFalse(sampler.sample(10)["complete"])
        ps.process_iter.side_effect = RuntimeError("sensor error")
        self.assertFalse(sampler.sample(11)["complete"])


class TelemetryWorkloadTests(unittest.TestCase):
    def telemetry(self):
        telemetry = Telemetry.__new__(Telemetry)
        telemetry.gpu = mock.Mock()
        telemetry.gpu.ok.return_value = False
        telemetry.ps = None
        telemetry.fallback = mock.Mock()
        telemetry.fallback.ram.return_value = (1, 2)
        telemetry._disk = lambda: (None, None)
        telemetry.extra = None
        return telemetry

    def test_disabled_samples_no_processes(self):
        telemetry = self.telemetry()
        with mock.patch(Telemetry.__module__ + ".time.time", return_value=123):
            self.assertNotIn("workload", telemetry.sample())

    def test_workload_sample_precedes_extra_and_supports_callable(self):
        telemetry = self.telemetry()
        calls = []
        telemetry.workload_sampler = lambda now: calls.append(("workload", now)) or reading(now)["workload"]
        telemetry.extra = lambda: calls.append(("extra", None)) or {"tok_s": 2}
        with mock.patch(Telemetry.__module__ + ".time.time", return_value=123):
            sample = telemetry.sample()
        self.assertEqual(calls, [("workload", 123), ("extra", None)])
        self.assertTrue(sample["workload"]["complete"])
        self.assertEqual(sample["tok_s"], 2)

    def test_sampler_errors_are_incomplete_and_extra_still_runs(self):
        telemetry = self.telemetry()
        telemetry.workload_sampler = mock.Mock()
        telemetry.workload_sampler.sample.side_effect = RuntimeError("sensor")
        telemetry.extra = lambda: {"tok_s": 2}
        sample = telemetry.sample()
        self.assertFalse(sample["workload"]["complete"])
        self.assertEqual(sample["tok_s"], 2)


if __name__ == "__main__":
    unittest.main()
