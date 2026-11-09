"""Resident handoff integration using the real service/policy/state machine."""
import threading
import time
import unittest
from unittest import mock
from types import SimpleNamespace

from serve import server
from serve.test_live_memory import service
from serve.test_resource_lease import acquire_request, capacity
from serve.resource_lease import ToolLeases, LeaseError


class ResidentLeaseTests(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.wall = 100.
        self.mono = 0.
        self.ram, self.gpu, self.commit = 14., 1024., 20.
        self.time_patch = mock.patch('serve.server.time.time', side_effect=lambda: self.wall)
        self.mono_patch = mock.patch('serve.server.time.monotonic', side_effect=lambda: self.mono)
        self.time_patch.start()
        self.mono_patch.start()
        self.addCleanup(self.time_patch.stop)
        self.addCleanup(self.mono_patch.stop)
        svc = self.svc
        svc.engine.can_stop = True
        svc.engine.info.update(memory_hold=1, background_control=1, memory_supersede=1,
                               expert_cache_mib=1024, arena_mib=32768)
        svc.engine.spawn[1].extend(['--pcie-frac', '.37'])
        svc.memory_policy.reserve_floor = svc.memory_policy.configured_reserve_floor = 320
        with mock.patch.dict('os.environ', {'STRATA_RESOURCE_LEASE_TOKEN': 'fixture-' * 8}):
            svc.configure_coadaptive({'enabled': True, 'mode': 'live',
                'idle_parking': {'enabled': True}, 'tool_leases': {'enabled': True}})
        svc.tool_leases.clock = lambda: self.mono
        svc.memory_policy.live_actual(32768, 320, self.wall, 'loaded', loaded=True)
        svc.engine.request_memory = mock.Mock()
        svc._parking_footprint = mock.Mock(return_value={
            'ram_bytes': 36*2**30, 'commit_bytes': 36*2**30, 'gpu_bytes': 7*2**30})
        svc._parking_identity = mock.Mock(return_value={'build': 'fixture'})
        svc.telemetry = SimpleNamespace(capacity=lambda: self.sample(), snapshot=lambda: {'now': self.sample()})
        del svc.memory_snapshot  # use the actual current-process native overlay
        self.native()

    def sample(self):
        return capacity(self.wall, ram=self.ram, gpu=self.gpu, commit=self.commit)

    def native(self, **overrides):
        self.svc.engine.native_capacity = dict(proc=self.svc.engine.proc, sampled_at=self.wall,
            free_mib=self.gpu, total_mib=8192, resident_mib=self.svc.engine.info['arena_mib'],
            cache_mib=self.svc.engine.info['expert_cache_mib'], memory_pending_id=0,
            memory_reserve_mib=320, memory_last_terminal_id=0, **overrides)

    def acquire(self, **kwargs):
        return self.svc.resource_lease_action(acquire_request(mode=kwargs.pop('mode', 'auto'), ram_headroom_gib=8,
            vram_headroom_mib=512, execution_ram_floor_gib=4, execution_vram_floor_mib=320, **kwargs))

    def action(self, row, action):
        return self.svc.resource_lease_action({'action': action, 'lease_token': row['lease_token']})

    def test_none_keeps_native_and_vision_identity_and_start_is_explicit(self):
        row = self.acquire()
        proc = self.svc.engine.proc
        self.svc.observe_tool_lease()
        state = self.action(row, 'status')
        self.assertEqual((state['state'], state['selected_action'], state['phase']), ('ready', 'none', 'admission'))
        self.assertEqual(self.action(row, 'start')['phase'], 'execution')
        self.assertEqual(self.action(row, 'start')['phase'], 'execution')
        self.assertIs(self.svc.engine.proc, proc)
        self.svc.engine.request_memory.assert_not_called()
        self.action(row, 'release')
        self.assertFalse(self.svc.tool_leases.blocked())
        self.assertIsNone(self.svc.memory_policy.lease_ceiling())
        self.assertEqual(list(self.svc._admit_loaded(threading.Event())), [])

    def test_start_retargets_without_double_charging_tool_allocation(self):
        row = self.acquire()
        self.svc.observe_tool_lease()
        self.action(row, 'start')
        # The admitted tool legitimately uses 6 GiB; its pre-dispatch target is
        # not maintained throughout execution. The configured engine floor is.
        self.ram = 7.
        self.wall += 1
        self.native()
        self.svc.observe_tool_lease()
        self.svc.observe_memory()
        self.svc.engine.request_memory.assert_not_called()
        self.assertEqual(self.svc.tool_leases.demand()['phase'], 'execution')

    def test_release_reactivates_normal_policy(self):
        row = self.acquire()
        self.svc.observe_tool_lease()
        self.action(row, 'release')
        self.ram = 3.
        self.wall += 1
        self.native()
        self.svc.observe_memory()
        self.svc.engine.request_memory.assert_called_once()
        self.assertNotIn('hold', self.svc.engine.request_memory.call_args.kwargs)

    def test_wrong_process_or_missing_native_never_grants_resident_ready(self):
        row = self.acquire()
        self.svc.engine.native_capacity['proc'] = object()
        self.svc.observe_tool_lease()
        self.assertNotEqual(self.action(row, 'status')['state'], 'ready')
        with self.assertRaises(LeaseError):
            self.action(row, 'start')

    def test_ram_relief_requires_terminal_ack_and_new_native_sample(self):
        row = self.acquire()
        self.ram = 7.
        self.svc.observe_tool_lease()
        self.svc.observe_memory()
        args = self.svc.engine.request_memory.call_args
        self.assertTrue(args.kwargs['hold'])
        pending = self.svc.memory_live_pending
        self.assertIsNotNone(pending)
        self.assertNotEqual(self.action(row, 'status')['state'], 'ready')
        self.wall += 1
        self.svc.engine.memory_acks.put((self.svc.engine.proc, {
            'id': pending['id'], 'status': 'applied', 'resident_mib': 31744,
            'vram_reserve_mib': 1024, 'expert_cache_mib': 1024, 'expert_slots': 100,
            'vram_free_mib': 1024, 'error': 'none'}))
        self.ram = 9.
        self.native()
        self.svc.observe_tool_lease()
        self.wall += 1
        self.native()
        self.svc.observe_tool_lease()
        self.assertEqual(self.action(row, 'status')['state'], 'ready')
        self.assertEqual(self.svc.engine.info['arena_mib'], 31744)

    def test_resident_return_fails_bounded_without_dispatch_when_space_missing(self):
        row = self.acquire(mode='relieve')
        self.svc.observe_tool_lease()
        self.action(row, 'release')
        self.gpu = 0.
        self.native()
        wait = self.svc._admit_loaded(threading.Event())
        self.mono = 10.
        self.assertEqual(next(wait), ('ping', None))
        self.mono = 45.
        with self.assertRaises(server.GpuBusy):
            next(wait)
        self.assertIsNotNone(self.svc.tool_resume)
        self.assertFalse(self.svc.tool_resuming)

    def test_resident_return_preserves_higher_configured_commit_floor(self):
        self.svc.idle_parking.pressure_commit_available_gib = 10
        row = self.acquire()
        self.svc.observe_tool_lease()
        self.action(row, 'release')
        self.assertEqual(self.svc.tool_resume['ram_gib'], 12)
        self.commit = 11
        self.native()
        self.mono = 10
        wait = self.svc._admit_resident_return(threading.Event())
        # This gate check supplies capacity externally; a mocked native resize
        # would otherwise remain unacknowledged and obscure the commit check.
        with mock.patch.object(self.svc, 'observe_memory'):
            self.assertEqual(next(wait), ('ping', None))
            self.commit = 12
            self.wall += 1
            self.native()
            with self.assertRaises(StopIteration):
                next(wait)
        self.assertIsNone(self.svc.tool_resume)


if __name__ == '__main__':
    unittest.main()
