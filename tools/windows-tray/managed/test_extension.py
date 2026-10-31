"""File IPC and lifecycle fakes only: no server, inference, downloads or UI."""
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4
from config import Settings, atomic_json
from controller import Controller
from extension import CONTEXTS, IDLES, ExtensionIPC, ExtensionRequest

REPO = Path(__file__).resolve().parents[3]


class Backend:
    alive, loaded = True, False
    def listener_owned(self, _port): return self.alive
    async def stop(self): self.alive = False


class Response:
    status = 200
    async def __aenter__(self): return self
    async def __aexit__(self, *_args): pass


class Client:
    def get(self, *_args, **_kwargs): return Response()


class ExtensionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='extension-fixture-', dir=Path(__file__).parent)
        self.root = Path(self.temp.name)
        atomic_json(self.root / 'model.json', {'model_name': 'fixture-physical-model', 'args': ['--max-context', '131072'], 'idle_unload_s': 1800})
        self.config = {'python': sys.executable, 'repository': str(REPO), 'profiles': {'Example Profile': 'model.json'},
                       'engine': 'mock', 'port': 8095, 'upstream_port': 8096}
        self.calls = []
        self.worker = None
        self.output = contextlib.redirect_stdout(io.StringIO()); self.output.__enter__()
        self.controller = self.make_controller()

    def make_controller(self, enabled=True):
        atomic_json(self.root / 'tray.json', {**self.config, 'app_extension': enabled})
        controller = Controller(Settings(self.root / 'tray.json'), Backend())
        controller.current_config = controller.settings.generated(controller.selected)
        controller.model, controller.phase, controller.api, controller.accepting = 'unloaded', 'ready', 'available', True
        controller.effective = dict(controller.selected)
        controller.update_checked = time.time()
        controller.client = Client()
        async def control(path, method='GET'):
            self.calls.append((path, method))
            if path == '/v1/load': controller.backend.loaded = True
            if path == '/v1/unload': controller.backend.loaded = False
            return {'loaded': controller.backend.loaded, 'activity': {'in_flight': 0}}
        controller.control = control
        controller.publish()
        return controller

    async def asyncTearDown(self):
        if self.worker:
            self.worker.cancel()
            with contextlib.suppress(asyncio.CancelledError): await self.worker
        self.output.__exit__(None, None, None)
        self.temp.cleanup()

    def request(self, action='load', **patches):
        return {'schema': 1, 'id': str(uuid4()), 'expectedModel': 'fixture-physical-model', 'action': action, **patches}

    def send(self, request, controller=None):
        c = controller or self.controller
        # Same no-overwrite publication as the Node consumer: complete file, then link.
        temporary = c.settings.state / 'fixture-request.tmp'
        atomic_json(temporary, request)
        try: os.link(temporary, c.settings.state / 'extension-request.json')
        finally: temporary.unlink()

    def status(self, controller=None):
        return json.loads(((controller or self.controller).settings.state / 'extension-status.json').read_text())

    async def execute(self, request):
        c = self.controller
        self.send(request); c.extension.claim(c.commands); c.publish()
        self.worker = asyncio.create_task(c.worker())
        await asyncio.wait_for(c.commands.join(), 2)
        self.worker.cancel()
        with contextlib.suppress(asyncio.CancelledError): await self.worker
        self.worker = None

    async def poll_once(self):
        c = self.controller
        async def finish(_seconds): c.exit.set()
        with patch('controller.asyncio.sleep', side_effect=finish): await c.poll()
        c.exit.clear()

    async def test_disabled_default_and_strict_boolean_leave_extension_files_untouched(self):
        self.controller = self.make_controller(False)
        c = self.controller
        for path in c.settings.state.glob('extension-*.json'): path.unlink()
        self.send(self.request(), c)
        before = (c.settings.state / 'extension-request.json').read_bytes()
        c.publish(); await self.poll_once()
        self.assertIsNone(c.extension)
        self.assertEqual((c.settings.state / 'extension-request.json').read_bytes(), before)
        self.assertFalse((c.settings.state / 'extension-status.json').exists())
        self.assertFalse((c.settings.state / 'extension-running.json').exists())
        atomic_json(self.root / 'tray.json', self.config)
        self.assertFalse(Settings(self.root / 'tray.json').app_extension)
        for invalid in (1, 'true', None):
            atomic_json(self.root / 'tray.json', {**self.config, 'app_extension': invalid})
            with self.assertRaises(ValueError): Settings(self.root / 'tray.json')

    async def test_snapshot_retains_contract_and_unknown_is_null(self):
        c = self.controller
        for state, loaded in [('loaded', True), ('unloaded', False), ('loading', None), ('unknown', None), ('failed', None)]:
            c.model, c.active = state, 2
            c.publish(); value = self.status()
            self.assertEqual(value['schema'], 1)
            self.assertEqual(value['modelId'], 'fixture-physical-model')
            self.assertEqual(value['apiBaseUrl'], 'http://127.0.0.1:8095/v1')
            self.assertIs(value['loaded'], loaded); self.assertEqual(value['model'], state); self.assertEqual(value['active'], 2)
            self.assertEqual(value['selected'], c.selected); self.assertEqual(value['effective'], c.effective)
            from datetime import datetime, timezone
            self.assertLess((datetime.now(timezone.utc) - datetime.fromisoformat(value['captured'])).total_seconds(), 10)
        c.current_config = None; c.publish(); self.assertIsNone(self.status()['modelId'])

    async def test_claim_is_once_and_terminal_ack_precedes_claim_removal(self):
        c = self.controller; request = self.request()
        self.send(request); c.extension.claim(c.commands); c.publish()
        running = c.settings.state / 'extension-running.json'
        self.assertTrue(running.exists()); self.assertEqual(self.status()['request'], {'id': request['id'], 'state': 'running'})
        c.extension.claim(c.commands); self.assertEqual(c.commands.qsize(), 1)
        queued = c.commands.get_nowait(); c.commands.task_done()
        c.extension.finish(queued)
        self.assertTrue(running.exists())
        with patch('extension.atomic_json', side_effect=OSError('PRIVATE_PATH_SENTINEL')): c.publish()
        self.assertTrue(running.exists()); self.assertFalse(c.extension.healthy)
        c.publish(); self.assertFalse(running.exists())
        self.assertEqual(self.status()['request'], {'id': request['id'], 'state': 'done'})
        c.publish(); self.assertEqual(self.status()['request']['state'], 'done')

    async def test_context_and_idle_map_to_existing_select_without_changing_other_selection(self):
        c = self.controller; original = (self.root / 'model.json').read_bytes()
        restarts = []
        async def restart():
            restarts.append(dict(c.selected)); c.effective = dict(c.selected)
        c.restart = restart
        for action, choices in [('context', CONTEXTS), ('idle', IDLES)]:
            for choice in choices:
                previous = dict(c.selected)
                await self.execute(self.request(action, value=choice))
                self.assertEqual(c.selected, {**previous, action: choice})
                self.assertEqual(self.status()['request']['state'], 'done')
        self.assertEqual(len(restarts), len(CONTEXTS) + len(IDLES))
        self.assertEqual((self.root / 'model.json').read_bytes(), original)
        self.assertEqual(json.loads((c.settings.state / 'selection.json').read_text()), c.selected)
        self.assertEqual(Settings(c.settings.path).selection(), c.selected)
        self.assertEqual(self.calls, [])

    async def test_load_and_unload_use_owner_drain_and_loaded_is_observed_on_next_poll(self):
        c = self.controller
        for action, expected in [('load', True), ('unload', False)]:
            await self.execute(self.request(action))
            self.assertIn(('/v1/status', 'GET'), self.calls)
            self.assertEqual(self.calls[-1], ('/v1/' + action, 'POST'))
            self.assertEqual(self.status()['request']['state'], 'done')
            await self.poll_once()
            self.assertIs(self.status()['loaded'], expected)
            self.assertEqual(self.status()['model'], 'loaded' if expected else 'unloaded')
            self.assertTrue(c.accepting)

    async def test_poll_does_not_turn_missing_or_nonboolean_loaded_into_unloaded(self):
        c = self.controller
        for raw, expected in [(None, None), (0, None), ('false', None), (True, True), (False, False)]:
            async def status(_path, _method='GET'): return {'loaded': raw}
            c.control = status
            await self.poll_once()
            self.assertIs(self.status()['loaded'], expected)
            self.assertEqual(self.status()['model'], 'unknown' if expected is None else 'loaded' if expected else 'unloaded')

    async def test_owner_waits_for_admitted_requests_before_load(self):
        c = self.controller; c.active = 1
        self.send(self.request()); c.extension.claim(c.commands)
        self.worker = asyncio.create_task(c.worker())
        await asyncio.sleep(0); self.assertEqual(c.phase, 'draining'); self.assertFalse(c.accepting)
        self.assertNotIn(('/v1/load', 'POST'), self.calls)
        async with c.condition: c.active = 0; c.condition.notify_all()
        await asyncio.wait_for(c.commands.join(), 2)
        self.assertEqual(self.calls.count(('/v1/load', 'POST')), 1)
        self.assertEqual(self.status()['request']['state'], 'done')

    async def test_identity_is_rechecked_after_queueing_before_any_lifecycle_call(self):
        c = self.controller; request = self.request()
        self.send(request); c.extension.claim(c.commands)
        c.current_config = {'model_name': 'different-physical-model'}
        self.worker = asyncio.create_task(c.worker()); await asyncio.wait_for(c.commands.join(), 2)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.status()['request'], {'id': request['id'], 'state': 'error', 'error': 'ValueError'})
        self.assertFalse((c.settings.state / 'extension-running.json').exists())

    async def test_changed_profile_file_cannot_silently_load_a_different_physical_model(self):
        c = self.controller
        profile = json.loads((self.root / 'model.json').read_text()); profile['model_name'] = 'different-physical-model'
        atomic_json(self.root / 'model.json', profile)
        await self.execute(self.request('context', value=65536))
        self.assertEqual(c.selected['context'], 131072); self.assertEqual(self.calls, [])
        self.assertEqual(self.status()['request']['error'], 'ValueError')

    async def test_invalid_fields_types_choices_and_json_never_enqueue_or_act(self):
        c = self.controller
        invalid = [self.request('context', value=True), self.request('context', value='65536'), self.request('context', value=200000),
                   self.request('context'), self.request('idle', value=60), self.request('idle', value=-1), self.request(value=0),
                   self.request('restart'), self.request('load', schema=True), self.request(extra='PRIVATE_SENTINEL'),
                   self.request(expectedModel=''), self.request(id='not-a-UUID'), [], None]
        for request in invalid:
            self.send(request); c.extension.claim(c.commands); c.publish()
            self.assertTrue(c.commands.empty()); self.assertEqual(self.status()['request']['state'], 'error')
            self.assertNotIn('PRIVATE_SENTINEL', json.dumps(self.status()))
            self.assertFalse((c.settings.state / 'extension-running.json').exists())
        for raw in ('{bad', '{"schema":1,"schema":1}', 'x' * 4097):
            (c.settings.state / 'extension-request.json').write_text(raw)
            c.extension.claim(c.commands); c.publish(); self.assertTrue(c.commands.empty())
        self.assertEqual(self.calls, [])

    async def test_existing_running_marker_is_not_overwritten_or_replayed_after_restart(self):
        c = self.controller; request = self.request()
        self.send(request); c.extension.claim(c.commands); c.publish()
        running = c.settings.state / 'extension-running.json'; before = running.read_bytes()
        newer = self.request('unload'); self.send(newer)
        restarted = self.make_controller(); restarted.extension.claim(restarted.commands); restarted.publish()
        self.assertTrue(restarted.commands.empty()); self.assertEqual(running.read_bytes(), before)
        self.assertEqual(self.status(restarted)['request'], {'id': request['id'], 'state': 'error', 'error': 'InterruptedError'})
        self.assertEqual(json.loads((c.settings.state / 'extension-request.json').read_text()), newer)
        self.assertEqual(self.calls, [])

    async def test_same_id_is_not_executed_twice_and_terminal_ack_survives_recreation(self):
        c = self.controller; request = self.request(); await self.execute(request)
        restarted = self.make_controller(); self.assertEqual(self.status(restarted)['request']['state'], 'done')
        self.send(request, restarted); restarted.extension.claim(restarted.commands); restarted.publish()
        self.assertTrue(restarted.commands.empty()); self.assertEqual(self.calls.count(('/v1/load', 'POST')), 1)
        self.assertEqual(self.status(restarted)['request']['error'], 'ValueError')

    async def test_cancelled_worker_leaves_an_unfinished_claim_and_never_reports_done(self):
        c = self.controller; entered = asyncio.Event()
        async def interrupted(_command): entered.set(); await asyncio.Event().wait()
        c.command = interrupted; request = self.request()
        self.send(request); c.extension.claim(c.commands)
        self.worker = asyncio.create_task(c.worker()); await asyncio.wait_for(entered.wait(), 2)
        self.worker.cancel()
        with contextlib.suppress(asyncio.CancelledError): await self.worker
        self.worker = None
        self.assertEqual(self.status()['request']['state'], 'running')
        self.assertTrue((c.settings.state / 'extension-running.json').exists())
        restarted = self.make_controller(); self.assertTrue(restarted.extension.blocked)
        self.assertEqual(self.status(restarted)['request']['error'], 'InterruptedError')
        self.assertEqual(self.calls, [])

    async def test_action_error_is_class_only_and_does_not_retry(self):
        c = self.controller
        async def failed(_command): raise RuntimeError('PRIVATE_PATH_AND_SECRET_SENTINEL')
        c.command = failed; request = self.request(); await self.execute(request)
        value = self.status(); self.assertEqual(value['request'], {'id': request['id'], 'state': 'error', 'error': 'RuntimeError'})
        self.assertNotIn('PRIVATE_PATH_AND_SECRET_SENTINEL', json.dumps(value))
        self.assertTrue(c.commands.empty()); self.assertFalse((c.settings.state / 'extension-running.json').exists())

    async def test_full_queue_or_failed_status_write_cannot_consume_a_request(self):
        c = self.controller; self.send(self.request())
        with patch('extension.atomic_json', side_effect=OSError('PRIVATE_SENTINEL')): c.publish()
        c.extension.claim(c.commands); self.assertTrue(c.commands.empty())
        c.publish()
        for _ in range(c.commands.maxsize): c.commands.put_nowait({'action': 'check'})
        c.extension.claim(c.commands)
        self.assertTrue((c.settings.state / 'extension-request.json').exists())
        self.assertFalse((c.settings.state / 'extension-running.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
