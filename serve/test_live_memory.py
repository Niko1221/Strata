"""Live control protocol and service lifecycle checks without a model or GPU."""
import io
import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from serve.server import ByteTokenizer, Service, StrataEngine, engine_args
from serve.test_memory_policy import sample
from serve.memory_policy import MemoryPolicy


ARGS = ["--resident-budget-gib", "42", "--vram-reserve-mib", "1536", "--live-memory"]


def ack(request_id=1, status="applied", resident=40960, reserve=1536):
    return (f"MEMORY {request_id} status={status} resident_mib={resident} expert_cache_mib=8192 "
            f"expert_slots=128 vram_free_mib=1024 vram_reserve_mib={reserve}\n")


def engine():
    e = StrataEngine("unused", ARGS, lazy=True)
    e.proc = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(), poll=lambda: None, wait=lambda **kw: 0)
    e.ended = e.unloaded = False
    e.info.update(live_memory=1, memory_protocol=1, arena_mib=43008, expert_cache_mib=8192)
    e.lines = queue.Queue()
    return e


def service():
    e = engine()
    svc = Service(e, ByteTokenizer(), None)
    with mock.patch("serve.server.time.time", return_value=0):
        svc.configure_memory({"enabled": True, "mode": "live"})
    svc.memory_snapshot = mock.Mock(return_value=sample(601, used=61))
    return svc


def propose(svc, now=601):
    svc.memory_policy.observe = mock.Mock(return_value={"resident_budget_gib": 40,
                                                       "vram_reserve_mib": 1536,
                                                       "reason": "sustained_pressure"})
    with mock.patch("serve.server.time.time", return_value=now):
        svc.observe_memory()


def deliver(svc, line, proc=None, now=602):
    svc.engine.memory_acks.put((proc or svc.engine.proc, StrataEngine._memory_ack(line)))
    svc.memory_policy.observe.return_value = None
    with mock.patch("serve.server.time.time", return_value=now):
        svc.observe_memory()


class LiveMemoryTests(unittest.TestCase):
    def test_16_gib_gpu_only_pressure_and_recovery_are_reachable_at_99_percent(self):
        p = MemoryPolicy({"enabled": True, "mode": "live"})
        p.applied({"resident_budget_gib": 42, "vram_reserve_mib": 1536}, 0)
        info = {"arena_mib": 43008, "expert_cache_mib": 8192}
        for now in range(600, 661):
            plan = p.observe(sample(now, used=50, gpu_used=15.99, gpu_total=16), True, info, now)
            if now < 660:
                self.assertIsNone(plan)
        self.assertEqual(plan["reason"], "sustained_pressure")
        self.assertEqual(plan["resident_budget_gib"], 42)
        self.assertEqual(plan["vram_reserve_mib"], 1690)
        p.applied(plan, 660)
        for now in range(661, 1261):
            recovered = p.observe(sample(now, used=50, gpu_used=14, gpu_total=16), True, info, now)
            if now < 1260:
                self.assertIsNone(recovered)
        self.assertEqual(recovered["reason"], "stable_headroom")
        self.assertEqual(recovered["vram_reserve_mib"], 1536)
        self.assertEqual(recovered["resident_budget_gib"], 42)

    def test_engine_args_opt_in_and_legacy_default(self):
        self.assertEqual(engine_args({"args": ARGS[:-1]}), ARGS[:-1])
        cfg = {"args": ARGS[:-1], "memory_policy": {"enabled": True, "mode": "live"}}
        self.assertEqual(engine_args(cfg), ARGS)
        for extra in ({"backend": "hip"}, {"gpu": "0,1"}, {"args": ARGS + ["--layer-split", "auto"]}):
            with self.assertRaises(ValueError):
                engine_args({**cfg, **extra})

    def test_configuration_requires_real_native_capability(self):
        svc = service()
        svc.engine.info.pop("live_memory")
        svc.engine.unload = mock.Mock()
        with self.assertRaisesRegex(ValueError, "advertising"):
            svc.configure_memory({"enabled": True, "mode": "live"})
        svc.engine.unload.assert_not_called()

    def test_live_admission_rejects_generated_parallel_and_competing_capacity_owner(self):
        cfg = {"args": ARGS[:-1], "memory_policy": {"enabled": True, "mode": "live"}}
        for extra in ({"parallel": 2}, {"vram_elastic": True},
                      {"args": ARGS[:-1] + ["--batch", "2"]},
                      {"args": ARGS[:-1] + ["--slots", "2"]},
                      {"args": ARGS[:-1] + ["--batch-groups", "2"]}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                engine_args({**cfg, **extra})
        for mode in ("live", "reload"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "both own"):
                engine_args({**cfg, "vram_elastic": True,
                             "memory_policy": {"enabled": True, "mode": mode}})
        self.assertEqual(engine_args({**cfg, "parallel": 1}), ARGS)
        legacy = {"args": ARGS[:-1], "parallel": 2, "vram_elastic": True,
                  "memory_policy": {"enabled": False}}
        args = engine_args(legacy)
        self.assertIn("--vram-elastic", args)
        self.assertIn("--batch", args)
        self.assertNotIn("--live-memory", args)

    def test_live_service_rejects_actual_batch_slots_and_direct_competing_flags(self):
        for extra_args, info, batch in (([], {"batch_slots": 2}, 0), ([], {}, 2),
                                        (["--slots", "2"], {}, 0),
                                        (["--vram-elastic"], {}, 0)):
            with self.subTest(args=extra_args, info=info, batch=batch):
                e = engine()
                e.spawn = ("unused", ARGS + extra_args, None, None, None)
                e.info.update(info)
                e.batch = batch
                svc = Service(e, ByteTokenizer(), None)
                with self.assertRaises(ValueError):
                    svc.configure_memory({"enabled": True, "mode": "live"})
                self.assertIsNone(svc.memory_policy)

    def test_pump_keeps_control_slot_and_generation_channels_separate(self):
        e = engine()
        e.slot_q = [queue.Queue()]
        e.proc.stdout = io.StringIO(ack() + "MEMORY malformed\nBT 0 99\nBDONE 0 1 stop 1.0\n"
                                  "T 10\nDONE 1 2 3 4 stop\n")
        proc = e.proc
        e._pump()
        self.assertEqual(list(e.lines.queue), ["T 10\n", "DONE 1 2 3 4 stop\n", None])
        self.assertEqual(list(e.slot_q[0].queue), ["BT 0 99\n", "BDONE 0 1 stop 1.0\n", None])
        self.assertEqual(e.memory_acks.qsize(), 1)
        source_proc, parsed = e.memory_acks.get_nowait()
        self.assertIs(source_proc, proc)
        self.assertEqual(parsed["id"], 1)

    def test_generation_and_memory_writers_cannot_bypass_pipe_owner(self):
        e = engine()
        entered = [threading.Event(), threading.Event()]
        written = threading.Event()
        errors = []
        class Pipe(io.StringIO):
            def write(self, text):
                written.set()
                return super().write(text)
        e.proc.stdin = Pipe()
        def run(index, command):
            entered[index].set()
            try:
                command()
            except Exception as error:
                errors.append(error)
        writers = [threading.Thread(target=run, args=(0, lambda: e._send("GEN 1 2"))),
                   threading.Thread(target=run, args=(1, lambda: e.request_memory(1, 40960, 1536, e.proc)))]
        try:
            with e.pipe_lock:
                for writer in writers:
                    writer.start()
                for started in entered:
                    self.assertTrue(started.wait(2))
                self.assertFalse(written.wait(0.1), "a protocol writer bypassed the pipe owner")
        finally:
            for writer in writers:
                writer.join(2)
        for writer in writers:
            self.assertFalse(writer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(set(e.proc.stdin.getvalue().splitlines()), {"GEN 1 2", "MEMORY 1 40960 1536"})

    def test_restart_does_not_inherit_old_live_capability(self):
        e = engine()
        e.close = mock.Mock()
        def fresh(self, *args):
            self.info = {"engine": "new", "batch_slots": 0}
        with mock.patch.object(StrataEngine, "__init__", fresh):
            e.restart()
        self.assertNotIn("live_memory", e.info)
        self.assertNotIn("memory_protocol", e.info)
        self.assertEqual(e.info["engine"], "new")
        self.assertFalse(e.starting)

    def test_eager_start_rejects_missing_capability_before_serving(self):
        proc = mock.Mock(stdout=io.StringIO("INFO arena_mib=43008\nREADY 4096 stop\n"), poll=lambda: None)
        with mock.patch("serve.server.subprocess.Popen", return_value=proc), mock.patch("serve.server.contain"):
            with self.assertRaisesRegex(ValueError, "advertising"):
                StrataEngine("unused", ARGS)
        proc.stdin.write.assert_called_once_with("QUIT\n")

    def test_legacy_service_still_reloads_only_at_request_boundary(self):
        e = engine()
        e.spawn = ("unused", ARGS[:-1], None, None, None)
        svc = Service(e, ByteTokenizer(), None)
        svc.configure_memory({"enabled": True})
        svc.memory_pending = {"resident_budget_gib": 40, "vram_reserve_mib": 1536,
                              "reason": "sustained_pressure"}
        e.unload = mock.Mock(side_effect=lambda: setattr(e, "ended", True))
        old_applied = svc.memory_policy.last_applied
        svc._apply_memory_plan()
        e.unload.assert_called_once()
        self.assertEqual(e.spawn[1][1], "40")
        self.assertEqual(svc.memory_policy.last_applied, old_applied)
        e.restart = mock.Mock(side_effect=lambda: setattr(e, "ended", False))
        svc._ensure_loaded_native()
        self.assertEqual(svc.memory_policy.current["resident_budget_gib"], 40)
        self.assertIsNone(svc.memory_pending)
        self.assertEqual(svc.memory_status()["application"], "next request boundary")

    def test_real_policy_automatically_shrinks_and_grows_while_busy(self):
        svc = service()
        svc.status["busy"] = True
        for now in range(600, 661):
            svc.memory_snapshot.return_value = sample(now, used=61)
            with mock.patch("serve.server.time.time", return_value=now):
                svc.observe_memory()
        self.assertEqual(svc.engine.proc.stdin.getvalue(), "MEMORY 1 40448 1536\n")
        svc.engine.memory_acks.put((svc.engine.proc, StrataEngine._memory_ack(ack(resident=40448))))
        for now in range(661, 1262):
            svc.memory_snapshot.return_value = sample(now, used=25)
            with mock.patch("serve.server.time.time", return_value=now):
                svc.observe_memory()
        self.assertEqual(svc.engine.proc.stdin.getvalue().splitlines()[-1], "MEMORY 2 43008 1536")
        self.assertTrue(svc.memory_status()["pending"])

    def test_eager_load_reconciles_ram_once_while_busy_without_reload(self):
        e = engine()
        e.info["arena_mib"] = 17144
        svc = Service(e, ByteTokenizer(), None)
        with mock.patch("serve.server.time.time", return_value=100):
            svc.configure_memory({"enabled": True, "mode": "live"})
        svc.memory_snapshot = mock.Mock(return_value=sample(101, used=43.7421875))
        svc.status["busy"] = True
        e.unload = mock.Mock()
        e.restart = mock.Mock()
        with mock.patch("serve.server.time.time", return_value=101):
            svc.observe_memory()
        self.assertEqual(e.proc.stdin.getvalue(), "MEMORY 1 32256 1536\n")
        self.assertEqual(svc.memory_status()["reason"], "post_load_headroom")
        self.assertEqual(svc.memory_policy.current["resident_budget_gib"], 17144 / 1024)
        e.memory_acks.put((e.proc, StrataEngine._memory_ack(ack(resident=32256))))
        svc.memory_snapshot.return_value = sample(102, used=25)
        with mock.patch("serve.server.time.time", return_value=102):
            svc.observe_memory()
        self.assertEqual(svc.memory_policy.current["resident_budget_gib"], 31.5)
        self.assertEqual(svc.memory_status()["last_reason"], "post_load_headroom")
        self.assertFalse(svc.memory_status()["pending"])
        self.assertEqual(e.proc.stdin.getvalue().count("MEMORY"), 1)
        e.unload.assert_not_called()
        e.restart.assert_not_called()

    def test_successful_cold_reload_rearms_loaded_ram_reconciliation(self):
        svc = service()
        svc.memory_snapshot.return_value = sample(1, used=50)
        with mock.patch("serve.server.time.time", return_value=1):
            svc.observe_memory()
        self.assertEqual(svc.engine.proc.stdin.getvalue(), "")
        svc.engine.ended = True
        svc.memory_snapshot.return_value = sample(100, used=35)
        def restart():
            svc.engine.proc = engine().proc
            svc.engine.ended = False
            svc.engine.info["arena_mib"] = 17144
        svc.engine.restart = mock.Mock(side_effect=restart)
        with mock.patch("serve.server.time.time", return_value=100):
            svc.ensure_loaded()
        self.assertEqual(svc.memory_policy.current["resident_budget_gib"], 17144 / 1024)
        self.assertEqual(svc.memory_status()["last_reason"], "load_budget")
        svc.memory_snapshot.return_value = sample(101, used=43.7421875)
        with mock.patch("serve.server.time.time", return_value=101):
            svc.observe_memory()
        self.assertEqual(svc.engine.proc.stdin.getvalue(), "MEMORY 1 32256 1536\n")
        svc.engine.restart.assert_called_once()

    def test_partial_native_failure_keeps_retry_delay_after_loaded_reconciliation(self):
        e = engine()
        e.info["arena_mib"] = 17144
        svc = Service(e, ByteTokenizer(), None)
        with mock.patch("serve.server.time.time", return_value=100):
            svc.configure_memory({"enabled": True, "mode": "live"})
        svc.memory_snapshot = mock.Mock(return_value=sample(101, used=43.7421875))
        with mock.patch("serve.server.time.time", return_value=101):
            svc.observe_memory()
        self.assertEqual(e.proc.stdin.getvalue(), "MEMORY 1 32256 1536\n")
        e.memory_acks.put((e.proc, StrataEngine._memory_ack(
            ack(status="error", resident=20000).rstrip() + " error=ram_resize\n")))
        svc.memory_snapshot.return_value = sample(102, used=43.7421875)
        with mock.patch("serve.server.time.time", return_value=102):
            svc.observe_memory()
        self.assertEqual(svc.memory_policy.current["resident_budget_gib"], 20000 / 1024)
        self.assertEqual(svc.memory_status()["error"], "ram_resize")
        self.assertEqual(svc.memory_status()["retry_at"], 702)
        self.assertEqual(svc.memory_policy.last_applied, 100)
        svc.memory_snapshot.return_value = sample(103, used=43.7421875)
        with mock.patch("serve.server.time.time", return_value=103):
            svc.observe_memory()
        self.assertEqual(e.proc.stdin.getvalue().count("MEMORY"), 1)

    def test_partial_growth_error_allows_pressure_shrink_before_growth_retry(self):
        e = engine()
        e.info["arena_mib"] = 17144
        svc = Service(e, ByteTokenizer(), None)
        with mock.patch("serve.server.time.time", return_value=100):
            svc.configure_memory({"enabled": True, "mode": "live"})
        svc.status["busy"] = True
        e.unload = mock.Mock()
        e.restart = mock.Mock()
        svc.memory_snapshot = mock.Mock(return_value=sample(101, used=43.7421875))
        with mock.patch("serve.server.time.time", return_value=101):
            svc.observe_memory()
        self.assertEqual(e.proc.stdin.getvalue(), "MEMORY 1 32256 1536\n")
        e.memory_acks.put((e.proc, StrataEngine._memory_ack(ack(status="progress", resident=24576))))
        svc.memory_snapshot.return_value = sample(102, used=61)
        with mock.patch("serve.server.time.time", return_value=102):
            svc.observe_memory()
        self.assertTrue(svc.memory_status()["pending"])
        self.assertEqual(e.proc.stdin.getvalue().count("MEMORY"), 1)
        e.memory_acks.put((e.proc, StrataEngine._memory_ack(
            ack(status="error", resident=24576).rstrip() + " error=ram_resize\n")))
        for now in range(103, 164):
            svc.memory_snapshot.return_value = sample(now, used=61)
            with mock.patch("serve.server.time.time", return_value=now):
                svc.observe_memory()
            if now < 163:
                self.assertEqual(e.proc.stdin.getvalue().count("MEMORY"), 1)
        self.assertEqual(svc.memory_retry_at, 703)
        self.assertEqual(e.proc.stdin.getvalue().splitlines(),
                         ["MEMORY 1 32256 1536", "MEMORY 2 22016 1536"])
        self.assertEqual(svc.memory_live_pending["plan"]["reason"], "sustained_pressure")
        e.memory_acks.put((e.proc, StrataEngine._memory_ack(ack(request_id=2, resident=22016))))
        for now in range(164, 703):
            svc.memory_snapshot.return_value = sample(now, used=25)
            with mock.patch("serve.server.time.time", return_value=now):
                svc.observe_memory()
        self.assertEqual(e.proc.stdin.getvalue().count("MEMORY"), 2)
        self.assertEqual(svc.memory_policy.current["resident_budget_gib"], 21.5)
        self.assertFalse(svc.memory_status()["pending"])
        e.unload.assert_not_called()
        e.restart.assert_not_called()

    def test_pressure_write_failure_requires_new_pressure_window_before_retry(self):
        svc = service()
        svc.status["busy"] = True
        svc.engine.request_memory = mock.Mock(side_effect=OSError("pipe write failed"))
        for now in range(1, 62):
            svc.memory_snapshot.return_value = sample(now, used=61)
            with mock.patch("serve.server.time.time", return_value=now):
                svc.observe_memory()
        self.assertEqual(svc.engine.request_memory.call_count, 1)
        self.assertEqual(svc.memory_retry_at, 661)
        self.assertFalse(svc.memory_status()["pending"])
        for now in range(62, 122):
            svc.memory_snapshot.return_value = sample(now, used=61)
            with mock.patch("serve.server.time.time", return_value=now):
                svc.observe_memory()
            self.assertEqual(svc.engine.request_memory.call_count, 1)
        svc.memory_snapshot.return_value = sample(122, used=61)
        with mock.patch("serve.server.time.time", return_value=122):
            svc.observe_memory()
        self.assertEqual(svc.engine.request_memory.call_count, 2)
        self.assertEqual(svc.memory_retry_at, 722)
        self.assertFalse(svc.memory_status()["pending"])
        for now in range(123, 182):
            svc.memory_snapshot.return_value = sample(now, used=61)
            with mock.patch("serve.server.time.time", return_value=now):
                svc.observe_memory()
            self.assertEqual(svc.engine.request_memory.call_count, 2)
        self.assertEqual([call.args[1:3] for call in svc.engine.request_memory.call_args_list],
                         [(40448, 1536), (40448, 1536)])

    def test_real_pressure_cannot_supersede_pending_native_operation(self):
        svc = service()
        svc.status["busy"] = True
        svc.engine.unload = mock.Mock()
        for now in range(1, 62):
            svc.memory_snapshot.return_value = sample(now, used=61)
            with mock.patch("serve.server.time.time", return_value=now):
                svc.observe_memory()
        svc.engine.memory_acks.put((svc.engine.proc, StrataEngine._memory_ack(
            ack(status="progress", resident=40960))))
        for now in range(62, 124):
            svc.memory_snapshot.return_value = sample(now, used=63)
            with mock.patch("serve.server.time.time", return_value=now):
                svc.observe_memory()
        self.assertEqual(svc.engine.proc.stdin.getvalue(), "MEMORY 1 40448 1536\n")
        self.assertEqual(svc.memory_status()["request_id"], 1)
        self.assertTrue(svc.memory_status()["pending"])
        svc.engine.unload.assert_not_called()

    def test_busy_proposal_writes_control_without_reload_or_apply(self):
        svc = service()
        svc.status["busy"] = True
        svc.engine.unload = mock.Mock()
        svc.engine.restart = mock.Mock()
        propose(svc)
        self.assertEqual(svc.engine.proc.stdin.getvalue(), "MEMORY 1 40960 1536\n")
        self.assertEqual(svc.memory_policy.current["resident_budget_gib"], 42)
        self.assertEqual(svc.memory_policy.last_applied, 0)
        svc._apply_memory_plan()
        svc.engine.unload.assert_not_called()
        svc.engine.restart.assert_not_called()
        self.assertTrue(svc.memory_status()["pending"])

    def test_idle_completion_reports_actual_sizes(self):
        svc = service()
        propose(svc)
        deliver(svc, ack(resident=40000))
        self.assertEqual(svc.memory_policy.current["resident_budget_gib"], 40000 / 1024)
        self.assertEqual(svc.memory_policy.last_applied, 602)
        self.assertEqual(svc.engine.info["arena_mib"], 40000)
        self.assertEqual(svc.engine.info["expert_slots"], 128)
        self.assertFalse(svc.memory_status()["pending"])

    def test_capped_completion_exposes_limitation_without_failure_or_false_size(self):
        svc = service()
        propose(svc)
        deliver(svc, ack(resident=40000).rstrip() + " error=ram_capacity_or_rounding\n")
        status = svc.memory_status()
        self.assertEqual(status["limitation"], "ram_capacity_or_rounding")
        self.assertIsNone(status["error"])
        self.assertIsNone(status["retry_at"])
        self.assertFalse(status["pending"])
        self.assertEqual(status["current"]["resident_budget_gib"], 40000 / 1024)
        self.assertEqual(status["last_applied_at"], 602)
        propose(svc, 603)
        self.assertIsNone(svc.memory_status()["limitation"])

    def test_progress_and_error_keep_partial_actual_without_false_completion(self):
        svc = service()
        propose(svc)
        deliver(svc, ack(status="progress", resident=42000))
        self.assertTrue(svc.memory_status()["pending"])
        self.assertEqual(svc.memory_policy.last_applied, 0)
        deliver(svc, ack(status="error", resident=41500).rstrip() + " error=allocation_failed\n", now=603)
        self.assertEqual(svc.engine.info["arena_mib"], 41500)
        self.assertEqual(svc.memory_policy.current["resident_budget_gib"], 41500 / 1024)
        self.assertEqual(svc.memory_policy.last_applied, 0)
        self.assertEqual(svc.memory_status()["error"], "allocation_failed")
        self.assertFalse(svc.memory_status()["pending"])
        self.assertEqual(svc.memory_retry_at, 1203)
        svc.memory_policy.observe.return_value = {"resident_budget_gib": 42,
                                                  "vram_reserve_mib": 1536,
                                                  "reason": "stable_headroom"}
        with mock.patch("serve.server.time.time", return_value=604):
            svc.observe_memory()
        self.assertEqual(svc.engine.proc.stdin.getvalue().count("MEMORY"), 1)

    def test_telemetry_error_unknown_free_vram_keeps_actual_allocations(self):
        svc = service()
        propose(svc)
        deliver(svc, ack(status="error", resident=42000).replace("vram_free_mib=1024", "vram_free_mib=-1")
                .rstrip() + " error=telemetry\n")
        self.assertEqual(svc.memory_status()["error"], "telemetry")
        self.assertEqual(svc.engine.info["vram_free_mib"], -1)
        self.assertEqual(svc.memory_policy.last_applied, 0)

    def test_one_inflight_not_superseded(self):
        svc = service()
        propose(svc)
        propose(svc, 700)
        self.assertEqual(svc.engine.proc.stdin.getvalue().count("MEMORY"), 1)
        self.assertEqual(svc.memory_live_pending["id"], 1)

    def test_wrong_id_and_previous_process_ack_are_ignored(self):
        svc = service()
        propose(svc)
        deliver(svc, ack(request_id=2))
        deliver(svc, ack(), proc=object())
        self.assertEqual(svc.memory_policy.last_applied, 0)
        self.assertTrue(svc.memory_status()["pending"])

    def test_invalid_ack_numbers_and_status(self):
        valid = ack()
        for line in (valid.replace("status=applied", "status=ok"),
                     valid.replace("resident_mib=40960", "resident_mib=nan"),
                     valid.replace("resident_mib=40960", "resident_mib=-1"),
                     valid.replace("resident_mib=40960", "resident_mib=1.5"),
                     valid.replace("MEMORY 1 ", "MEMORY 0 "),
                     valid.replace("MEMORY 1 ", f"MEMORY {2**53 + 1} "),
                     valid.replace("MEMORY 1 ", "MEMORY " + "1" * 5000 + " "),
                     valid.replace("resident_mib=40960", "resident_mib=" + "1" * 5000),
                     valid + " status=error", valid.replace("expert_slots=128 ", "")):
            self.assertIsNone(StrataEngine._memory_ack(line), line)

    def test_token_burst_and_idle_ack_use_separate_queues(self):
        e = engine()
        e.proc.stdout = io.StringIO("PP 1 2\nT 65\n" + ack() + "T 66\nDONE 2 2 1 1 stop\n" + ack(2))
        e._pump()
        e.ended = False
        output = list(e.generate([1, 2], 2, {}, threading.Event()))
        self.assertEqual(output, [None, 65, 66])
        self.assertEqual(e.memory_acks.qsize(), 2)
        self.assertIsNone(e.lines.get_nowait())

    def test_malformed_control_output_never_contaminates_tokens(self):
        e = engine()
        e.proc.stdout = io.StringIO("MEMORY 1 status=invalid\nT 65\n")
        e._pump()
        self.assertEqual(e.memory_acks.qsize(), 0)
        self.assertEqual(e.lines.get_nowait(), "T 65\n")

    def test_pipe_writes_serialize_generation_and_memory(self):
        e = engine()
        entered = threading.Event()
        release = threading.Event()
        writes = []

        class Pipe:
            def write(self, line):
                writes.append(line)
                if line.startswith("GEN "):
                    entered.set()
                    self.assert_release = release.wait(2)
            def flush(self):
                writes.append("flush")

        e.proc.stdin = Pipe()
        e.lines.put("DONE 0 1 1 1 stop\n")
        t = threading.Thread(target=lambda: list(e.generate([1], 1, {}, threading.Event())))
        t.start()
        self.assertTrue(entered.wait(2))
        control = threading.Thread(target=lambda: e.request_memory(1, 40960, 1536, e.proc))
        control.start()
        self.assertEqual(len(writes), 1)
        release.set()
        t.join(2)
        control.join(2)
        self.assertFalse(t.is_alive() or control.is_alive())
        self.assertEqual(writes[1:], ["flush", "MEMORY 1 40960 1536\n", "flush"])

    def test_stale_telemetry_freezes_new_live_command(self):
        svc = service()
        svc.memory_snapshot.return_value = sample(1, used=61)
        with mock.patch("serve.server.time.time", return_value=601):
            svc.observe_memory()
        self.assertEqual(svc.engine.proc.stdin.getvalue(), "")
        self.assertEqual(svc.memory_policy.last_reason, "telemetry_unavailable")

    def test_pipe_death_and_replaced_process_invalidate_pending(self):
        for replacement in (False, True):
            svc = service()
            propose(svc)
            if replacement:
                svc.engine.proc = engine().proc
            else:
                svc.engine.ended = True
            svc.observe_memory()
            self.assertIsNone(svc.memory_live_pending)
            self.assertIsNone(svc.memory_policy.current)
            self.assertEqual(svc.memory_policy.last_applied, 0)

    def test_explicit_unload_invalidates_and_restart_reinitializes(self):
        svc = service()
        propose(svc)
        def unload():
            svc.engine.ended = True
        svc.engine.unload = unload
        self.assertEqual(svc.unload(), "unloaded")
        self.assertIsNone(svc.memory_live_pending)
        def restart():
            svc.engine.proc = engine().proc
            svc.engine.ended = False
            svc.engine.info["arena_mib"] = 20000
        svc.engine.restart = restart
        svc.memory_snapshot.return_value = sample(time.time(), used=30)
        svc.ensure_loaded()
        self.assertEqual(svc.memory_policy.current["resident_budget_gib"], 20000 / 1024)
        self.assertIsNone(svc.memory_live_pending)

    def test_reload_initializes_live_actual_even_without_fresh_startup_sample(self):
        svc = service()
        svc.engine.ended = True
        svc.memory_snapshot.return_value = {}
        svc.engine.restart = mock.Mock(side_effect=lambda: setattr(svc.engine, "ended", False))
        svc.ensure_loaded()
        self.assertEqual(svc.memory_policy.current["resident_budget_gib"], 42)
        self.assertIsNone(svc.memory_pending)


if __name__ == "__main__":
    unittest.main()
