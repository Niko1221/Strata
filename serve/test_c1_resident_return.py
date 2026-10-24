"""AUTO return fallback reuses measured parking admission; no model or GPU."""
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from serve import server
from serve import test_c1_lease_service as fixtures


class ResidentReturnTests(unittest.TestCase):
    setUp = fixtures.ResidentLeaseTests.setUp
    sample = fixtures.ResidentLeaseTests.sample
    native = fixtures.ResidentLeaseTests.native
    acquire = fixtures.ResidentLeaseTests.acquire
    action = fixtures.ResidentLeaseTests.action

    def released(self, mode='auto'):
        row = self.acquire(mode=mode)
        self.svc.observe_tool_lease()
        self.action(row, 'start')
        self.action(row, 'release')
        self.gpu = 372.
        self.wall += 1
        self.native()
        self.svc.memory_limitation = 'prefill_cache_floor'
        return self.svc.tool_resume

    def processes(self, *, survive=False, cancel=None):
        svc = self.svc
        original = svc.engine.proc
        vision_proc = SimpleNamespace(poll=lambda: None)
        svc.vision = SimpleNamespace(proc=vision_proc, alive=lambda: svc.vision.proc is not None)
        def native_unload():
            if not survive:
                svc.engine.proc = None
            if cancel is not None:
                cancel.set()
        svc.engine.unload = mock.Mock(side_effect=native_unload)
        svc.vision.unload = mock.Mock(side_effect=lambda: setattr(svc.vision, 'proc', None))
        def reload(stopped):
            self.assertFalse(stopped.is_set())
            self.assertFalse(svc.loaded())
            self.assertFalse(svc.vision.alive())
            svc.engine.proc = SimpleNamespace(poll=lambda: None)
            svc.vision.proc = SimpleNamespace(poll=lambda: None)
        svc._idle_reload = mock.Mock(side_effect=reload)
        svc._ensure_loaded_base = mock.Mock(side_effect=AssertionError('unconditional load bypass'))
        svc._record_live_load = mock.Mock()
        return original, vision_proc

    def test_auto_floor_unloads_both_and_waits_for_full_reload_footprint(self):
        returning = self.released()
        original, old_vision = self.processes()
        self.mono = 10
        cancel = threading.Event()
        wait = self.svc._admit_loaded(cancel)
        # Current free capacity alone cannot reload the 36GiB/7GiB model.
        self.assertEqual(next(wait), ('ping', None))
        self.assertFalse(self.svc.loaded())
        self.assertFalse(self.svc.vision.alive())
        self.assertIsNone(self.svc.tool_resume)
        self.assertEqual(self.svc.memory_policy.reserve_floor, 320)
        self.svc._idle_reload.assert_not_called()
        self.assertEqual(self.svc.idle_parking_status['reason'], 'resident_return_auto_unload')
        admission = self.svc.idle_parked['admission']
        self.assertGreater(admission.required['ram_bytes'], 36*2**30)
        self.assertGreater(admission.required['gpu_bytes'], 7*2**30)
        self.ram = self.commit = 50.
        self.gpu = 8192.
        def recover(_):
            self.wall += 1
            self.mono += 1
            return False
        with mock.patch.object(cancel, 'wait', side_effect=recover):
            self.assertTrue(all(item == ('ping', None) for item in wait))
        self.assertIsNot(self.svc.engine.proc, original)
        self.assertIsNot(self.svc.vision.proc, old_vision)
        self.svc.engine.unload.assert_called_once()
        self.svc.vision.unload.assert_called_once()
        self.svc._idle_reload.assert_called_once()
        self.svc._ensure_loaded_base.assert_not_called()
        self.assertIsNone(self.svc.idle_parked)
        self.assertFalse(self.svc.tool_resuming)
        self.assertFalse(self.svc.memory_loading)
        self.assertEqual(self.svc.idle_parking_status['state'], 'ready')
        self.assertEqual(returning['mode'], 'auto')

    def test_auto_deadline_can_unload_without_fixed_floor(self):
        self.released()
        self.processes()
        self.svc.memory_limitation = None
        self.mono = 10
        wait = self.svc._admit_resident_return(threading.Event())
        self.assertEqual(next(wait), ('ping', None))
        self.mono = 41
        self.assertEqual(list(wait), [])
        self.assertIsNotNone(self.svc.idle_parked)
        self.svc.engine.unload.assert_called_once()
        self.svc._idle_reload.assert_not_called()

    def test_strict_relieve_never_unloads_at_floor_or_deadline(self):
        returning = self.released('relieve')
        self.processes()
        self.mono = 10
        wait = self.svc._admit_loaded(threading.Event())
        self.assertEqual(next(wait), ('ping', None))
        self.mono = 41
        with self.assertRaises(server.GpuBusy):
            next(wait)
        self.assertIs(self.svc.tool_resume, returning)
        self.assertIsNone(self.svc.idle_parked)
        self.svc.engine.unload.assert_not_called()
        self.svc._idle_reload.assert_not_called()

    def test_unverified_process_death_does_not_reload_or_attempt_second_lifecycle(self):
        returning = self.released()
        self.processes(survive=True)
        with self.assertRaises(server.EngineStuck):
            list(self.svc._admit_loaded(threading.Event()))
        self.assertIs(self.svc.tool_resume, returning)
        self.assertEqual(self.svc.idle_parking_status['state'], 'unavailable')
        with self.assertRaises(server.EngineStuck):
            list(self.svc._admit_loaded(threading.Event()))
        self.svc.engine.unload.assert_called_once()
        self.svc._idle_reload.assert_not_called()
        self.assertFalse(self.svc.memory_loading)

    def test_cancel_before_fallback_preserves_live_processes(self):
        returning = self.released()
        self.processes()
        cancel = threading.Event();cancel.set()
        with self.assertRaises(server.RequestParkCancelled):
            list(self.svc._park_resident_return(returning, cancel))
        self.svc.engine.unload.assert_not_called()
        self.assertIsNone(self.svc.idle_parked)
        self.assertIs(self.svc.tool_resume, returning)

    def test_cancel_during_teardown_joins_death_and_preserves_measured_next_load(self):
        self.released()
        cancel = threading.Event()
        self.processes(cancel=cancel)
        with self.assertRaises(server.RequestParkCancelled):
            list(self.svc._admit_loaded(cancel))
        self.assertFalse(self.svc.loaded())
        self.assertFalse(self.svc.vision.alive())
        self.assertIsNone(self.svc.tool_resume)
        self.assertIsNotNone(self.svc.idle_parked)
        self.assertEqual(self.svc.idle_parking_status['state'], 'suspended')
        self.assertFalse(self.svc.memory_loading)
        self.svc._idle_reload.assert_not_called()

    def test_new_lease_blocks_old_return_teardown(self):
        returning = self.released()
        self.processes()
        # Acquiring a successor cannot authorize old-owner destructive work.
        from serve.test_resource_lease import acquire_request
        self.svc.resource_lease_action(acquire_request(
            request_id='bf35d5c4-2a6e-4bdb-8cc4-9422e7403119', mode='auto'))
        with self.assertRaises(server.GpuBusy):
            list(self.svc._park_resident_return(returning, threading.Event()))
        self.svc.engine.unload.assert_not_called()
        self.svc._idle_reload.assert_not_called()
        self.assertIsNone(self.svc.idle_parked)

    def test_missing_footprint_never_authorizes_release(self):
        returning = self.released()
        self.processes()
        self.svc._parking_footprint.side_effect = ValueError('fresh footprint unavailable')
        with self.assertRaises(ValueError):
            list(self.svc._park_resident_return(returning, threading.Event()))
        self.svc.engine.unload.assert_not_called()
        self.assertIsNone(self.svc.idle_parked)
        self.assertFalse(self.svc.memory_loading)

    def test_changed_process_cannot_be_released_by_old_return_owner(self):
        returning = self.released()
        self.processes()
        self.svc.engine.proc = SimpleNamespace(poll=lambda: None)
        with self.assertRaises(server.GpuBusy):
            list(self.svc._park_resident_return(returning, threading.Event()))
        self.svc.engine.unload.assert_not_called()
        self.assertIsNone(self.svc.idle_parked)

    def test_pending_native_control_is_cleared_only_after_verified_teardown(self):
        returning = self.released()
        self.processes()
        pending = {'id': 91, 'proc': self.svc.engine.proc}
        self.svc.memory_live_pending = pending
        self.assertEqual(list(self.svc._park_resident_return(returning, threading.Event())), [])
        self.assertIsNone(self.svc.memory_live_pending)
        self.assertIsNone(self.svc.tool_resume)
        self.assertFalse(self.svc.loaded())

    def test_failed_teardown_keeps_pending_native_control(self):
        returning = self.released()
        self.processes(survive=True)
        pending = {'id': 91, 'proc': self.svc.engine.proc}
        self.svc.memory_live_pending = pending
        with self.assertRaises(server.EngineStuck):
            list(self.svc._park_resident_return(returning, threading.Event()))
        self.assertIs(self.svc.memory_live_pending, pending)
        self.assertIs(self.svc.tool_resume, returning)


if __name__ == '__main__':
    unittest.main()
