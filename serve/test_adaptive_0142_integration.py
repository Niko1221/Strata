"""The .42 queue/restart/replica changes composed with adaptive ownership; no model/GPU."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from serve import server
from serve.frontend import ChatTemplate
from serve.resource_lease import ToolLeases
from serve.test_live_memory import ARGS, engine as native_engine
from serve.test_memory_policy_http import BudgetEngine
from serve.test_resource_lease import acquire_request, capacity


class TopologyCleanupTests(unittest.TestCase):
    def test_kfd_property_file_closes_after_success_and_read_failure(self):
        class BrokenReader(io.StringIO):
            def __next__(self):
                raise OSError("topology disappeared while reading")

        for properties, expected in (
            (io.StringIO("simd_count 128\nmax_engine_clk_fcompute 2350\n"), {0: 300800.0}),
            (io.StringIO("simd_count invalid\n"), None),
            (BrokenReader(), None),
        ):
            with self.subTest(expected=expected, reader=type(properties).__name__):
                with mock.patch.object(server.os, "listdir", return_value=["0"]), \
                        mock.patch("builtins.open", return_value=properties):
                    self.assertEqual(server.hip_speed_scores([0], root="topology"), expected)
                self.assertTrue(properties.closed)


class AdaptiveStartupTests(unittest.TestCase):
    def test_replicas_rejected_before_any_constructor(self):
        options = ({"memory_policy": {"enabled": True, "mode": "reload"}},
                   {"memory_policy": {"enabled": True, "mode": "live"}},
                   {"resource_presets": {"enabled": True}}, {"coadaptive": {"enabled": True}},
                   *({"coadaptive": {key: {"enabled": True}}}
                     for key in ("request_parking", "idle_parking", "tool_leases")),
                   {"args": ["--live-memory"]})
        with mock.patch.object(server, "StrataEngine") as native, \
                mock.patch.object(server, "Vision") as vision:
            for option in options:
                cfg = {"args": [], "gpu": [0, 1], "replicas": 2, **option}
                with self.subTest(option=option), self.assertRaisesRegex(ValueError, "replicas"):
                    server.start_replicas(cfg, [[0], [1]], "unused", [], False)
            native.assert_not_called()
            vision.assert_not_called()

    def test_disabled_replica_and_normalized_single_replica_paths_remain_available(self):
        cfg = {"gpu": [0, 1], "args": [], "replicas": 2,
               "memory_policy": {"enabled": False}, "coadaptive": {"enabled": False}}
        server.validate_adaptive_startup(cfg)
        for replica in (None, False, 1, [{"gpus": [0]}]):
            server.validate_adaptive_startup({"gpu": 0, "args": [], "replicas": replica,
                                               "memory_policy": {"enabled": True}})

    def test_helper_gpu_paths_rejected_without_changing_ordinary_args(self):
        for flag in ("--expert-cache-remote", "--expert-cache-device1", "--expert-cache-device2",
                     "--expert-cache-device3", "--remote-expert-opt"):
            args = [*ARGS[:-1], flag, "128"]
            for mode in ("reload", "live"):
                with self.subTest(flag=flag, mode=mode), self.assertRaisesRegex(ValueError, "helper"):
                    server.engine_args({"args": args, "memory_policy": {"enabled": True, "mode": mode}})
            self.assertEqual(server.engine_args({"args": args}), args)

    def test_async_swap_guard_matches_native_last_nonzero_option(self):
        for value in ("1", "2", "-1", " 1suffix"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "asynchronous expert swaps"):
                server.engine_args({"args": [*ARGS, "--adapt-async", value]})
        with self.assertRaisesRegex(ValueError, "needs a value"):
            server.engine_args({"args": [*ARGS, "--adapt-async"]})
        for args in ([*ARGS, "--adapt-async", "0"],
                     [*ARGS, "--adapt-async", "1", "--adapt-async", "0"],
                     [*ARGS[:-1], "--adapt-async", "1"]):
            self.assertEqual(server.engine_args({"args": args}), args)

    def test_main_rejects_before_vision_or_native_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, content in (("vocab.json", "{}"), ("merges.txt", ""), ("token_type.json", "[]")):
                (root / filename).write_text(content)
            cfg = {"args": [], "exe": "unused", "gpu": [0, 1], "replicas": 2,
                   "vision": {"exe": "unused"}, "memory_policy": {"enabled": True}}
            path = root / "run.json"
            path.write_text(json.dumps(cfg))
            with mock.patch.object(sys, "argv", ["server", "--engine", "strata", "--port", "0",
                    "--config", str(path), "--tokenizer", directory]), \
                    mock.patch.dict(sys.modules, {"strata_tokenizer": SimpleNamespace(
                        Tokenizer=lambda *args: server.ByteTokenizer())}), \
                    mock.patch.object(server, "hub_from_config", return_value=None), \
                    mock.patch.object(server, "StrataEngine") as native, \
                    mock.patch.object(server, "Vision") as vision:
                with self.assertRaisesRegex(SystemExit, "replicas"):
                    server.main()
                native.assert_not_called()
                vision.assert_not_called()


class RestartMetadataTests(unittest.TestCase):
    def test_live_policy_accepts_new_spawn_metadata(self):
        e = native_engine()
        self.assertEqual(len(e.spawn), 7)
        e.spawn = (*e.spawn[:6], [0, 2, 4])
        svc = server.Service(e, server.ByteTokenizer(), None)
        svc.configure_memory({"enabled": True, "mode": "live"})
        self.assertEqual(svc.memory_policy.mode, "live")

    def test_replan_preserves_cpu_and_other_restart_fields(self):
        e = BudgetEngine(server.ByteTokenizer())
        e.spawn = (*e.spawn, False, [0, 2, 4])
        original = e.spawn
        svc = server.Service(e, server.ByteTokenizer(), None)
        svc.configure_memory({"enabled": True})
        svc.memory_pending = {"resident_budget_gib": 35, "vram_reserve_mib": 1792,
                              "reason": "sustained_pressure"}
        svc._apply_memory_plan()
        self.assertEqual(e.spawn[1][1], "35")
        self.assertEqual(e.spawn[1][3], "1792")
        self.assertEqual(e.spawn[:1] + e.spawn[2:], original[:1] + original[2:])
        self.assertEqual(original[1][1], "42")

    def test_forwarding_proxy_does_not_gain_single_owner_capability(self):
        e = native_engine()
        class Proxy:
            def __getattr__(self, name):
                return getattr(e, name)
        svc = server.Service(Proxy(), server.ByteTokenizer(), None)
        with self.assertRaisesRegex(ValueError, "Strata engine"):
            svc.configure_memory({"enabled": True})

    def test_restart_preserves_cpu_assignment_and_does_not_store_cancel_event(self):
        e = server.StrataEngine("unused", [], lazy=True, cpus=[1, 3])
        saved = e.spawn
        cancel = threading.Event()
        calls = []
        def initialize(self, *args, startup_cancel=None):
            calls.append((args, startup_cancel))
            self.info = {"new": 1}
        with mock.patch.object(server.StrataEngine, "__init__", initialize):
            e.restart(tries=1, startup_cancel=cancel)
            e.restart(tries=1)
        self.assertEqual(calls, [(saved, cancel), (saved, None)])
        self.assertEqual(e.spawn[-2:], (False, [1, 3]))
        self.assertNotIn(cancel, e.spawn)
        self.assertFalse(e.starting)

    def test_already_cancelled_start_does_not_allocate(self):
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(server, "popen") as popen:
            with self.assertRaises(server.RequestParkCancelled):
                server.StrataEngine("unused", [], cpus=[1, 3], startup_cancel=cancel)
        popen.assert_not_called()


class RequestOwnershipTests(unittest.TestCase):
    def service(self):
        tok = server.ByteTokenizer()
        svc = server.Service(server.MockEngine(tok, "ok"), tok,
                             ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        with mock.patch.dict("os.environ", {"STRATA_RESOURCE_LEASE_TOKEN": "fixture-control-" * 4}):
            svc.tool_leases = ToolLeases({"enabled": True})
        svc.keep_awake = mock.Mock()
        return svc

    def test_run_composes_keepawake_and_actual_preparation_cleanup(self):
        svc = self.service()
        def events(*args):
            self.assertEqual(svc.preparing_requests, 1)
            yield "ping", None
        with mock.patch.object(svc, "_run", events):
            run = svc.run([1], False, None, 1, {}, threading.Event())
            self.assertEqual(next(run), ("ping", None))
            self.assertEqual(svc.preparing_requests, 1)
            run.close()
        self.assertEqual(svc.preparing_requests, 0)
        self.assertEqual(svc.request_owner.depth, 0)
        svc.keep_awake.acquire.assert_called_once()
        svc.keep_awake.release.assert_called_once()

    def test_close_during_lease_wait_releases_keepawake_without_negative_preparations(self):
        svc = self.service()
        svc.tool_leases.acquire(acquire_request(), capacity(), 100)
        with mock.patch.object(svc, "_run") as generation:
            run = svc.run([1], False, None, 1, {}, threading.Event())
            self.assertEqual(next(run), ("ping", None))
            run.close()
        self.assertEqual(svc.preparing_requests, 0)
        generation.assert_not_called()
        svc.keep_awake.release.assert_called_once()

    def test_fifo_handoff_cancel_never_reloads_and_counts_once(self):
        svc = self.service()
        cancel = threading.Event()
        class Handoff:
            released = 0
            def acquire(self, **kwargs):
                cancel.set()
                return True
            def release(self):
                self.released += 1
        svc.fifo = gate = Handoff()
        with mock.patch.object(svc, "ensure_loaded") as load, \
                mock.patch.object(svc, "_admit_loaded") as admission:
            result = list(svc.run([1], False, None, 1, {}, cancel))
        self.assertEqual(result[-1][1]["finish"], "cancel")
        self.assertEqual(svc.status["queued"], 0)
        self.assertEqual(svc.preparing_requests, 0)
        self.assertEqual(gate.released, 1)
        load.assert_not_called()
        admission.assert_not_called()

    def test_save_retry_keeps_fifo_and_lease_gate(self):
        svc = self.service()
        svc.slot_save_path = "."
        svc.engine.session_save_reclaim = True
        svc.engine.session_file = mock.Mock()
        def save(service, action, path, refused):
            self.assertTrue(service.fifo.locked())
            self.assertTrue(service.status["busy"])
            return {"tokens": 1, "bytes": 4, "ms": 1}
        with mock.patch("serve.session_save_retry.session_file", side_effect=save) as retry:
            status, _ = svc.slot_action("0", "save", "fixture.bin")
            self.assertEqual(status, 200)
            svc.tool_leases.acquire(acquire_request(), capacity(), 100)
            status, _ = svc.slot_action("0", "save", "fixture.bin")
            self.assertEqual(status, 503)
            retry.assert_called_once()
        self.assertFalse(svc.fifo.locked())
        self.assertEqual(svc.status["queued"], 0)

    def test_vram_waiter_rechecks_lease_before_native_or_deferred_mutation(self):
        for loaded in (False, True):
            svc = self.service()
            svc.telemetry = SimpleNamespace(capacity=lambda: capacity(time.time()))
            svc.engine.vram = mock.Mock()
            svc.vram_reserve = 350
            svc.loaded = lambda: loaded
            class Handoff:
                released = 0
                def acquire(self, **kwargs):
                    svc.resource_lease_action(acquire_request())
                    return True
                def release(self):
                    self.released += 1
            svc.fifo = gate = Handoff()
            with self.subTest(loaded=loaded), self.assertRaises(server.ModelBusy):
                svc.vram(999)
            svc.engine.vram.assert_not_called()
            self.assertEqual(svc.vram_reserve, 350)
            self.assertEqual(gate.released, 1)

    def test_vram_mutation_and_new_lease_acquire_are_ordered(self):
        svc = self.service()
        svc.telemetry = SimpleNamespace(capacity=lambda: capacity(time.time()))
        entered, release, lease_entered, lease_done = (threading.Event() for _ in range(4))
        errors = []
        def resize(value):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("fixture control timeout")
            return {"vram_free_mib": value}
        svc.engine.vram = resize
        def control():
            try:
                svc.vram(350)
            except BaseException as exc:
                errors.append(exc)
        def acquire():
            lease_entered.set()
            try:
                svc.resource_lease_action(acquire_request())
                lease_done.set()
            except BaseException as exc:
                errors.append(exc)
        a, b = threading.Thread(target=control), threading.Thread(target=acquire)
        a.start()
        try:
            self.assertTrue(entered.wait(2))
            b.start()
            self.assertTrue(lease_entered.wait(2))
            self.assertFalse(lease_done.wait(.05))
        finally:
            release.set()
            a.join(3)
            if b.ident is not None:
                b.join(3)
        self.assertFalse(a.is_alive() or b.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(lease_done.is_set())
        self.assertEqual(svc.vram_reserve, 350)
        self.assertFalse(svc.fifo.locked())


class VisionPipeTests(unittest.TestCase):
    def vision(self, dead=True):
        v = server.Vision.__new__(server.Vision)
        proc = mock.Mock()
        proc.poll.return_value = 0 if dead else None
        v.proc, v.stopped = proc, False
        return v, proc

    def test_exited_pipe_is_explicitly_closed(self):
        v, proc = self.vision()
        v.close()
        proc.stdin.close.assert_called_once()
        self.assertIsNone(v.proc)
        self.assertTrue(v.stopped)

    def test_failed_flush_and_close_still_release_a_confirmed_dead_handle(self):
        v, proc = self.vision()
        proc.stdin.flush.side_effect = OSError("closed pipe")
        proc.stdin.close.side_effect = OSError("buffer flush")
        v.close()
        proc.stdin.close.assert_called_once()
        self.assertIsNone(v.proc)
        self.assertTrue(v.stopped)

    def test_uncertain_death_retains_process_and_pipe(self):
        v, proc = self.vision(dead=False)
        proc.wait.side_effect = subprocess.TimeoutExpired("encoder", 10)
        with self.assertRaises(server.EngineStuck):
            v.close()
        proc.stdin.close.assert_not_called()
        self.assertIs(v.proc, proc)
        self.assertFalse(v.stopped)


if __name__ == "__main__":
    unittest.main()
