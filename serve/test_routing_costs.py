import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from serve.routing_costs import RoutingCosts, runtime_key
from serve.frontend import ChatTemplate
from serve.server import Service, StrataEngine, ByteTokenizer
from serve.test_live_memory import engine
from serve.test_memory_policy import sample
from serve.test_live_memory import service, propose, deliver, ack
from serve.memory_policy import MemoryPolicy
from serve.coadaptive import CoAdaptive


def profile():
    return {"schema": 1, "runtime_key": "runtime", "sampling": {}, "created_unix": 1,
            "max_prompt_read": 8192, "max_output_tokens": 128, "records": [
        dict(fraction=f, prefill_ms_per_token=0, decode_ms_per_token=cost + n,
             fixed_ms=0, external_cpu=40, external_gpu=0, resident_mib=32768, cache_mib=736,
             context_min=0, context_max=8192, pair=str(n), correct=True)
        for n in range(3) for f, cost in ((.15, 70), (.37, 60), (.65, 40))]}


class RoutingCostsTests(unittest.TestCase):
    def choose(self, p=None, **kwargs):
        c = RoutingCosts(p or profile())
        args = dict(key="runtime", baseline=.37, current=.37,
                    snapshot={"sampled_at": 10, "workload": {"complete": True, "cpu_percent": 40, "gpu_percent": 0}},
                    native={"sampled_at": 10, "resident_mib": 32768, "cache_mib": 736},
                    context=4096, prompt_read=20, output_tokens=128, now=11)
        args.update(kwargs)
        return c, c.choose(**args)

    def test_costs_select_gpu_when_matched_measurements_win(self):
        self.assertEqual(self.choose()[1], .65)

    def test_can_select_cpu_when_actual_cost_is_lower(self):
        p = profile()
        for row in p["records"]:
            if row["fraction"] == .15:
                row["decode_ms_per_token"] = 30
        self.assertEqual(self.choose(p)[1], .15)

    def test_transfer_cost_can_erase_gpu_compute_benefit(self):
        p = profile()
        for row in p["records"]:
            if row["fraction"] == .65:
                row["fixed_ms"] = 4000
        self.assertEqual(self.choose(p)[1], .37)

    def test_migration_cost_must_pay_back(self):
        self.assertEqual(self.choose(switch_ms=4000)[1], .37)

    def test_noise_or_small_improvement_keeps_baseline(self):
        p = profile()
        for row in p["records"]:
            if row["fraction"] == .65:
                row["decode_ms_per_token"] = 59
        self.assertEqual(self.choose(p)[1], .37)

    def test_no_unmatched_cherry_picked_samples(self):
        p = profile()
        p["records"][-1]["pair"] = "unrelated"
        self.assertEqual(self.choose(p)[1], .37)

    def test_different_matched_subsets_rank_by_paired_benefit(self):
        p = profile()
        prototype = p['records'][0]
        p['records'] = []
        # The .15 subset is cheap overall, but saves only 10 ms/token.
        # The .65 subset saves 40 ms/token against its own matched baseline.
        for n in range(6):
            base = 100 if n < 3 else 200
            candidate, cost = (.15, 90) if n < 3 else (.65, 160)
            for f, c in ((.37, base), (candidate, cost)):
                p['records'].append(dict(prototype, pair=str(n), fraction=f, decode_ms_per_token=c))
        self.assertEqual(self.choose(p)[1], .65)

    def test_stale_missing_or_changed_resource_observations_hold(self):
        for override in (dict(key="new-engine"), dict(now=20), dict(context=65536),
                         dict(native={"sampled_at": 10, "resident_mib": 30720, "cache_mib": 736}),
                         dict(snapshot={"sampled_at": 10, "workload": {"complete": True, "cpu_percent": 40}})):
            self.assertEqual(self.choose(**override)[1], .37)

    def test_destination_admission_wins_over_cost(self):
        self.assertEqual(self.choose(allow_more_gpu=False)[1], .37)

    def test_warm_decode_cannot_authorize_cold_or_long_or_changed_sampling(self):
        p = profile()
        p['max_prompt_read'] = 5
        for override in (dict(prompt_read=20), dict(prompt_read=5, output_tokens=2048),
                         dict(prompt_read=5, sampling={'temperature': 1})):
            c, fraction = self.choose(p, **override)
            self.assertEqual(fraction, .37)
            self.assertEqual(c.reason, 'outside_request_calibration')

    def test_bad_quality_or_nonfinite_profile_is_rejected(self):
        for k, v in (("correct", False), ("decode_ms_per_token", float("nan")), ("fraction", 2)):
            p = profile()
            p["records"][0][k] = v
            with self.assertRaises(ValueError):
                RoutingCosts(p)

    def test_missing_calibration_does_not_guess(self):
        self.assertEqual(RoutingCosts().choose(key="x", baseline=.37, current=.5,
            snapshot={}, native=None, context=1, prompt_read=1, output_tokens=1, now=1), .37)

    def test_expired_calibration_and_duplicate_pairs_are_rejected(self):
        p = profile()
        p['created_unix'] = 1
        self.assertEqual(self.choose(p, now=100000)[1], .37)
        p['records'].append(dict(p['records'][0]))
        with self.assertRaises(ValueError):
            RoutingCosts(p)


class RuntimeIdentityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.binary = Path(directory.name) / "fixture.bin"
        self.binary.write_bytes(b"runtime identity fixture; never executed")
        nvml = patch("serve.telemetry._Nvml")
        nvml.start().return_value.name.return_value = "fixture GPU"
        self.addCleanup(nvml.stop)
        cpu = patch("serve.telemetry._cpu_name", return_value="fixture CPU")
        cpu.start()
        self.addCleanup(cpu.stop)

    def native(self, cpus=None):
        return StrataEngine(str(self.binary), ["--max-context", "65536"], lazy=True, cpus=cpus)

    def test_real_lazy_native_metadata_supports_parking_identity_without_loading(self):
        with patch("serve.server.subprocess.Popen") as popen:
            native = self.native([0, 2, 4])
            self.assertEqual(len(native.spawn), 7)
            svc = Service(native, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
            identity = svc._parking_identity()
            self.assertEqual(identity, {"runtime": runtime_key(native), "context": 65536, "vision": False})
            self.assertEqual(len(identity["runtime"]), 64)
            popen.assert_not_called()

    def test_cpu_assignment_invalidates_identity_but_unpinned_legacy_is_stable(self):
        unpinned = self.native()
        legacy = SimpleNamespace(spawn=unpinned.spawn[:5])
        baseline = runtime_key(unpinned)
        self.assertEqual(baseline, runtime_key(legacy))
        first = self.native([0, 2])
        second = self.native([1, 3])
        self.assertNotEqual(runtime_key(first), baseline)
        self.assertNotEqual(runtime_key(first), runtime_key(second))
        first.spawn = (*first.spawn[:5], True, first.spawn[6])
        self.assertEqual(runtime_key(first), runtime_key(self.native([0, 2])))

    def test_unknown_restart_metadata_cannot_authorize_identity(self):
        native = self.native()
        for metadata in (native.spawn[:4], native.spawn[:6], (*native.spawn, "future-option")):
            with self.subTest(length=len(metadata)), self.assertRaisesRegex(ValueError, "restart metadata"):
                runtime_key(SimpleNamespace(spawn=metadata))


class NativeCapacityTests(unittest.TestCase):
    def test_recovery_can_finish_the_last_partial_two_gib_step(self):
        from serve.test_coadaptive import reading
        p = MemoryPolicy({"enabled": True, "mode": "live"}, 32, 320)
        p.update_resource_limits(4, 320)
        p.applied({"resident_budget_gib": 30.5, "vram_reserve_mib": 320}, 0)
        for now in range(1, 32):
            plan = p.observe(reading(now, free_ram=10, free_gpu=400), True, {"arena_mib": 30.5 * 1024}, now)
        self.assertEqual(plan['resident_budget_gib'], 32)

    def test_recent_strata_gpu_work_is_not_attributed_to_another_app(self):
        from serve.test_coadaptive import reading
        c = CoAdaptive({"enabled": True})
        c.observe(reading(1, gpu_util=100), 1, engine_idle=True)
        self.assertIsNone(c.gpu_idle_sample)
        c.observe(reading(2, gpu_util=0), 2, engine_idle=True)
        c.observe(reading(3, gpu_util=0), 3, engine_idle=True)
        self.assertEqual(c.gpu_idle_sample, (3, 0))

    def test_native_capacity_ceiling_stops_pointless_growth(self):
        from serve.test_coadaptive import reading
        p = MemoryPolicy({"enabled": True, "mode": "live"}, 32, 256)
        p.update_resource_limits(4, 256)
        p.reclaim_gpu_headroom = True
        p.applied({"resident_budget_gib": 32, "vram_reserve_mib": 256}, 0)
        p.complete_live_plan({"reason": "stable_headroom"}, 32, "gpu_capacity")
        for now in range(1, 40):
            self.assertIsNone(p.observe(reading(now, free_ram=4, free_gpu=2048), True, {"arena_mib": 32768}, now))

    def test_unreachable_gpu_floor_does_not_repeat_but_ram_relief_still_can(self):
        s = service()
        propose(s)
        line = ack(s.memory_live_pending["id"]).replace("error=", "error=")
        # Fixture helper has no error suffix; append the allocator's limitation.
        line = line.strip() + " error=prefill_cache_floor\n"
        deliver(s, line)
        previous = s.memory_request_id
        reading = sample(603, used=61)
        reading["native_free_mib"] = 1024
        s.memory_snapshot.return_value = reading
        s.memory_policy.observe.return_value = dict(resident_budget_gib=40, vram_reserve_mib=1600, reason="sustained_pressure")
        with patch("serve.server.time.time", return_value=603):
            s.observe_memory()
        self.assertEqual(s.memory_request_id, previous)
        s.memory_policy.observe.return_value["resident_budget_gib"] = 38
        with patch("serve.server.time.time", return_value=604):
            s.observe_memory()
        self.assertGreater(s.memory_request_id, previous)

    def test_parser_rejects_partial_negative_and_nonfinite_readings(self):
        for line in ("CAPACITY free_mib=-1 total_mib=8192 resident_mib=32 cache_mib=700",
                     "CAPACITY free_mib=99999 total_mib=8192 resident_mib=32 cache_mib=700",
                     "CAPACITY free_mib=nan total_mib=8192 resident_mib=32 cache_mib=700",
                     "CAPACITY free_mib=400"):
            self.assertIsNone(StrataEngine._capacity_reading(line, object(), 1))

    def test_unreachable_floor_error_allows_recovery_to_lower_target(self):
        s = service()
        propose(s)
        line = ack(s.memory_live_pending['id'], status='error').strip() + ' error=prefill_cache_floor\n'
        deliver(s, line)
        self.assertEqual(s.memory_retry_at, 0)
        self.assertEqual(s.memory_error, 'prefill_cache_floor')
        previous = s.memory_request_id
        reading = sample(603, used=40)
        reading['native_free_mib'] = 1024
        s.memory_snapshot.return_value = reading
        s.memory_policy.observe.return_value = dict(resident_budget_gib=40, vram_reserve_mib=1024, reason='stable_headroom')
        with patch('serve.server.time.time', return_value=603):
            s.observe_memory()
        self.assertGreater(s.memory_request_id, previous)

    def test_partial_shrink_error_invalidates_previous_growth_ceiling(self):
        s = service()
        s.engine.info['expert_cache_mib'] = 9000
        s.memory_policy.gpu_capacity_ceiling = True
        propose(s)
        deliver(s, ack(s.memory_live_pending['id'], status='error').strip() + ' error=prefill_cache_floor\n')
        self.assertFalse(s.memory_policy.gpu_capacity_ceiling)

    def test_allocator_headroom_replaces_optimistic_nvml_and_expires(self):
        e = engine()
        s = Service(e, ByteTokenizer(), None)
        s.telemetry = SimpleNamespace(snapshot=lambda: {"now": sample(10, used=40)})
        e.native_capacity = StrataEngine._capacity_reading(
            "CAPACITY free_mib=300 total_mib=8192 resident_mib=32768 cache_mib=736", e.proc, 10)
        with patch("serve.server.time.time", return_value=11):
            x = s.memory_snapshot()
        self.assertEqual(x["gpu_mem_total"] - x["gpu_mem_used"], 300 * 2**20)
        with patch("serve.server.time.time", return_value=20):
            self.assertIsNone(s.memory_snapshot()["native_free_mib"])
        e.native_capacity["proc"] = object()
        with patch("serve.server.time.time", return_value=11):
            self.assertIsNone(s.memory_snapshot()["native_free_mib"])


if __name__ == "__main__":
    unittest.main()
