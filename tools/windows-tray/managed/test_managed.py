"""Run from repository root: python -B tools/windows-tray/managed/test_managed.py."""
import asyncio
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from aiohttp import ClientSession, web
from config import Settings, atomic_json
from controller import Controller
from updates import extract, version, fetch, stage
from windows_job import WindowsServer, JobError
import updates

REPO = Path(__file__).resolve().parents[3]


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def fixture(root):
    profile = {'model_name': 'test-model', 'args': ['--max-context', '65536'], 'idle_unload_s': 300}
    atomic_json(root / 'model.json', profile)
    value = {'python': sys.executable, 'repository': str(REPO), 'profiles': {'Test': 'model.json'},
             'engine': 'mock', 'port': port(), 'upstream_port': port(), 'installed_version': '0.1.43'}
    atomic_json(root / 'tray.json', value)
    return Settings(root / 'tray.json')


class ConfigTests(unittest.TestCase):
    def test_optional_desktop_app_does_not_change_existing_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = fixture(Path(temp))
            original = settings.selection()
            self.assertIsNone(settings.desktop_app)
            value = json.loads(settings.path.read_text())
            value['desktop_app'] = sys.executable
            atomic_json(settings.path, value)
            connected = Settings(settings.path)
            self.assertEqual(connected.selection(), original)
            self.assertEqual(connected.desktop_app, Path(sys.executable).resolve())
            value['desktop_app'] = 'missing.exe'
            atomic_json(settings.path, value)
            with self.assertRaisesRegex(ValueError, 'existing Windows App'):
                Settings(settings.path)

    def test_preservation_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = fixture(root)
            before = (root / 'model.json').read_bytes()
            selected = settings.selection()
            selected.update(context=262144, idle=0)
            generated = settings.generated(selected)
            self.assertEqual(generated['args'], ['--max-context', '262144'])
            self.assertEqual(generated['idle_unload_s'], 0)
            self.assertEqual(before, (root / 'model.json').read_bytes())
            atomic_json(settings.state / 'selection.json', selected)
            self.assertEqual(Settings(root / 'tray.json').selection(), selected)

    def test_removed_selected_profile_recovers_persisted_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = fixture(root)
            atomic_json(settings.state / 'selection.json', {**settings.selection(), 'idle': 600})
            value = json.loads(settings.path.read_text())
            value['profiles'] = {'Renamed': 'model.json'}
            atomic_json(settings.path, value)
            current = Settings(settings.path)
            selected = current.selection()
            self.assertEqual(selected['profile'], 'Renamed')
            self.assertEqual(selected['idle'], 300)
            self.assertEqual(json.loads((current.state / 'selection.json').read_text()), selected)
            self.assertEqual(Controller(current).selected, selected)

    def test_absent_idle_defaults_to_five_minutes_and_saved_thirty_wins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = fixture(root)
            profile = settings.profile('Test')
            profile.pop('idle_unload_s')
            atomic_json(root / 'model.json', profile)
            settings = Settings(root / 'tray.json')
            self.assertEqual(settings.selection()['idle'], 300)
            atomic_json(settings.state / 'selection.json', {**settings.selection(), 'idle': 1800})
            restored = Settings(root / 'tray.json')
            self.assertEqual(restored.selection()['idle'], 1800)
            self.assertEqual(restored.generated(restored.selection())['idle_unload_s'], 1800)

    def test_reject_invalid_and_changed_auth(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = fixture(root)
            for bad in (True, -1, '65536'):
                with self.assertRaises(ValueError):
                    settings.validate_selection({**settings.selection(), 'context': bad})
            profile = settings.profile('Test')
            profile['api_key'] = 'SECRET_SENTINEL'
            atomic_json(root / 'model.json', profile)
            with self.assertRaises(ValueError):
                settings.profile('Test')

    def test_archive_rejects_escape_collision_symlink_and_devices(self):
        for names in (['../escape'], ['C:/escape'], ['a\\escape'], ['NUL.txt'], ['foo.'], ['A', 'a']):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with zipfile.ZipFile(root / 'test.zip', 'w') as archive:
                    for name in names:
                        archive.writestr(name, b'bad')
                if names == ['a\\escape']:
                    archive_path = root / 'test.zip'
                    archive_path.write_bytes(archive_path.read_bytes().replace(b'a/escape', b'a\\escape'))
                with self.assertRaises(ValueError):
                    extract(root / 'test.zip', root / 'out')
                self.assertFalse((root / 'escape').exists())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with zipfile.ZipFile(root / 'test.zip', 'w') as archive:
                entry = zipfile.ZipInfo('link')
                entry.external_attr = 0o120777 << 16
                archive.writestr(entry, '../outside')
            with self.assertRaises(ValueError):
                extract(root / 'test.zip', root / 'out')

    def test_archive_and_versions(self):
        self.assertGreater(version('v0.1.43'), version('0.1.9'))
        for bad in ('0.1.44-rc1', 'latest', '../v1.2.3'):
            with self.assertRaises(ValueError):
                version(bad)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with zipfile.ZipFile(root / 'test.zip', 'w') as archive:
                archive.writestr('source/serve/server.py', 'ok')
            extract(root / 'test.zip', root / 'out', True)
            self.assertEqual((root / 'out/serve/server.py').read_text(), 'ok')


class UpdateTests(unittest.TestCase):
    def test_signed_component_release_is_deferred_to_the_official_updater(self):
        payload = {'tag_name': 'v0.2.0-beta', 'assets': [{'name': 'strata-feed.json'}]}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaises(updates.ComponentUpdateRequired):
                updates.latest()

    def test_digest_enforced(self):
        class Response(io.BytesIO):
            def geturl(self): return 'https://github.com/fixture'
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'download'
            with patch('urllib.request.urlopen', return_value=Response(b'corrupt')):
                with self.assertRaises(ValueError):
                    fetch('https://github.com/fixture', target, 100, '0' * 64, 7)

    def test_staging_preserves_original_and_refuses_dependency_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = fixture(root)
            engine = root / 'engine'
            engine.mkdir()
            (engine / 'strata.exe').write_bytes(b'original')
            (engine / 'runtime.dll').write_bytes(b'dll')
            (engine / 'strata.dll').write_bytes(b'old-engine-dll')
            atomic_json(engine / 'BUILD.json', {'version': '0.1.43', 'cuda': '13.0'})
            profile = settings.profile('Test')
            profile.update(exe=str(engine / 'strata.exe'))
            atomic_json(root / 'model.json', profile)
            settings.profiles['Second'] = settings.profiles['Test']
            settings.engine = 'strata'
            settings.asset = 'strata-windows-x64.zip'
            release = {'tag_name': 'v0.1.44', 'assets': [{
                'name': settings.asset, 'digest': 'sha256:' + '1' * 64, 'size': 10,
                'browser_download_url': 'https://github.com/Niko1221/Strata/releases/download/v0.1.44/' + settings.asset}]}
            changed = False
            def download(url, destination, *args):
                with zipfile.ZipFile(destination, 'w') as archive:
                    if 'codeload' in url:
                        archive.writestr('Strata/serve/server.py', '# fixture only')
                        archive.writestr('Strata/requirements.txt', b'changed' if changed else (REPO / 'requirements.txt').read_bytes())
                    else:
                        archive.writestr('strata.exe', b'new')
                        archive.writestr('strata.dll', b'new-engine-dll')
                        archive.writestr('BUILD.json', json.dumps({'version': '0.1.44', 'cuda': '13.0'}))
                return '2' * 64
            original = (root / 'model.json').read_bytes()
            with patch('updates.fetch', side_effect=download):
                result = stage(settings, release, settings.selection(), REPO)
                self.assertEqual((result / 'engine/strata.exe').read_bytes(), b'new')
                self.assertEqual((result / 'engine/runtime.dll').read_bytes(), b'dll')
                self.assertEqual((result / 'engine/strata.dll').read_bytes(), b'new-engine-dll')
                self.assertEqual((engine / 'strata.dll').read_bytes(), b'old-engine-dll')
                self.assertEqual((engine / 'strata.exe').read_bytes(), b'original')
                self.assertEqual((root / 'model.json').read_bytes(), original)
                # Release-bundled DLLs replace old bytes, but copied fallback DLLs
                # must still agree across all profile engines.
                other = root / 'other-engine'
                other.mkdir()
                (other / 'strata.exe').write_bytes(b'other-original')
                (other / 'runtime.dll').write_bytes(b'incompatible-runtime')
                atomic_json(other / 'BUILD.json', {'version': '0.1.43', 'cuda': '13.0'})
                atomic_json(root / 'second.json', {**profile, 'exe': str(other / 'strata.exe')})
                settings.profiles['Second'] = root / 'second.json'
                with self.assertRaisesRegex(ValueError, 'different native dependencies'):
                    stage(settings, release, settings.selection(), REPO)
                settings.profiles['Second'] = settings.profiles['Test']
                changed = True
                with self.assertRaises(ValueError):
                    stage(settings, release, settings.selection(), REPO)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = fixture(Path(self.temp.name))
        self.controller = Controller(self.settings)
        self.client = ClientSession(skip_auto_headers={'Content-Type'})
        self.controller.client = self.client
        self.controller.publish = lambda: None
        # Exercise lifecycle only; no release network calls or GPU workloads.

    async def asyncTearDown(self):
        await self.controller.backend.stop()
        await self.client.close()
        self.temp.cleanup()

    async def test_lazy_start_restart_and_update_do_not_preload(self):
        c = self.controller
        c.settings.engine = 'strata'
        class Backend:
            alive = True
            pid = 42
            def listener_owned(self, p): return self.alive
            async def start(self, argv, *args):
                self.alive = True
                self.argv = argv
            async def stop(self): self.alive = False
        c.backend = Backend()
        calls = []
        loaded = False
        async def control(path, method='GET'):
            nonlocal loaded
            calls.append((path, method))
            if path == '/health':
                return {'status': 'ok', 'service': 'strata', 'loaded': loaded,
                        'model': 'test-model'}
            if path == '/v1/load':
                loaded = True
                return {'status': 'loaded'}
            return {'loaded': loaded, 'model': 'test-model', 'engine': '0.1.43.1',
                    'cache_max_tokens': c.selected['context'], 'activity': {'in_flight': 0}}
        c.control = control
        c.settings.generated = lambda selection, runtime: {'model_name': 'test-model'}
        for expected in (None, 'v0.1.43.1'):
            await c.restart(expected_version=expected)
            self.assertEqual(c.phase, 'ready')
            self.assertEqual(c.model, 'unloaded')
            self.assertTrue(c.accepting)
            self.assertIn('--lazy', c.backend.argv)
        self.assertNotIn(('/v1/load', 'POST'), calls)
        await c.command({'action': 'load'})
        self.assertEqual(calls.count(('/v1/load', 'POST')), 1)

    async def test_development_version_starts_but_update_requires_exact_version(self):
        c = self.controller
        c.settings.engine = 'strata'
        class Backend:
            alive = True
            pid = 42
            def listener_owned(self, p): return True
            async def start(self, *args): pass
            async def stop(self): self.alive = False
        c.backend = Backend()
        status = {'loaded': True, 'model': 'test-model', 'cache_max_tokens': c.selected['context']}
        async def control(path, *args):
            return {'status': 'ok', 'service': 'strata'} if path == '/health' else status
        c.control = control
        for engine in ('0.1.43-dev', None, '0.1.43'):
            with self.subTest(engine=engine):
                status['engine'] = engine
                await c.launch({'model_name': 'test-model'}, None)
                self.assertEqual(c.phase, 'ready')
                self.assertEqual(c.version, '0.1.43' if engine == '0.1.43' else 'unknown')
        status['engine'] = '0.1.43-dev'
        with self.assertRaisesRegex(RuntimeError, 'version mismatch'):
            await c.launch({'model_name': 'test-model'}, None, '0.1.44')

    async def test_official_mock_lifecycle_auth_and_config(self):
        c = self.controller
        await c.command({'action': 'start'})
        self.assertTrue(c.owned())
        self.assertEqual(c.phase, 'ready')
        first_pid = c.backend.pid
        await c.command({'action': 'unload'})
        # Official MockEngine has no unload method and always reports loaded.
        self.assertTrue((await c.control('/v1/status'))['loaded'])
        await c.command({'action': 'load'})
        self.assertTrue((await c.control('/v1/status'))['loaded'])
        await c.command({'action': 'select', 'selection': {'context': 131072, 'idle': 600}})
        self.assertNotEqual(first_pid, c.backend.pid)
        self.assertEqual(c.selected['context'], 131072)
        self.assertEqual(c.effective['context'], 32768)
        original = self.settings.profiles['Test'].read_bytes()
        self.settings.profiles['Test'].write_text('{broken')
        pid = c.backend.pid
        with self.assertRaises(ValueError):
            await c.command({'action': 'restart'})
        self.assertEqual(pid, c.backend.pid)
        self.assertTrue(c.accepting)
        self.settings.profiles['Test'].write_bytes(original)
        await c.command({'action': 'stop'})
        self.assertFalse(c.backend.alive)
        self.assertEqual(c.phase, 'stopped')

    async def test_protected_proxy_does_not_add_credentials(self):
        path = self.settings.profiles['Test']
        profile = json.loads(path.read_text())
        profile['api_key'] = 'TEST_ONLY_KEY'
        atomic_json(path, profile)
        self.settings = Settings(self.settings.path)
        c = self.controller = Controller(self.settings)
        c.client, c.publish = self.client, lambda: None
        await c.command({'action': 'start'})
        proxy = web.AppRunner(c.app)
        await proxy.setup()
        await web.TCPSite(proxy, '127.0.0.1', self.settings.port).start()
        try:
            url = 'http://127.0.0.1:' + str(self.settings.port) + '/v1/models'
            async with self.client.get(url) as response:
                self.assertEqual(response.status, 401)
            async with self.client.get(url, headers={'Authorization': 'Bearer TEST_ONLY_KEY'}) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual((await response.json())['data'][0]['id'], 'test-model')
        finally:
            await proxy.cleanup()

    async def test_foreign_port_not_adopted(self):
        with socket.socket() as foreign:
            foreign.bind(('127.0.0.1', self.settings.upstream_port))
            foreign.listen()
            with self.assertRaises(JobError):
                await self.controller.command({'action': 'start'})
            self.assertFalse(self.controller.backend.alive)
            self.assertEqual(foreign.getsockname()[1], self.settings.upstream_port)

    async def test_stream_drains_before_stop_and_rejects_new_request(self):
        c = self.controller
        release, entered = asyncio.Event(), asyncio.Event()
        async def stream(request):
            response = web.StreamResponse()
            await response.prepare(request)
            await response.write(b'first\n')
            entered.set()
            await release.wait()
            await response.write(b'last\n')
            return response
        app = web.Application()
        app.router.add_get('/stream', stream)
        app.router.add_get('/v1/status', lambda r: web.json_response({'activity': {'in_flight': 0}}))
        upstream = web.AppRunner(app)
        await upstream.setup()
        await web.TCPSite(upstream, '127.0.0.1', self.settings.upstream_port).start()
        class FakeBackend:
            alive = True
            pid = 42
            def listener_owned(self, p): return self.alive
            async def stop(self): self.alive = False
        c.backend = FakeBackend()
        c.accepting, c.phase = True, 'ready'
        proxy = web.AppRunner(c.app)
        await proxy.setup()
        await web.TCPSite(proxy, '127.0.0.1', self.settings.port).start()
        try:
            async with self.client.get('http://127.0.0.1:' + str(self.settings.port) + '/stream') as response:
                self.assertEqual(await response.content.readline(), b'first\n')
                await entered.wait()
                stop = asyncio.create_task(c.command({'action': 'stop'}))
                for _ in range(100):
                    if c.phase == 'draining': break
                    await asyncio.sleep(.01)
                self.assertTrue(c.backend.alive)
                self.assertFalse(stop.done())
                async with self.client.get('http://127.0.0.1:' + str(self.settings.port) + '/stream') as refused:
                    self.assertEqual(refused.status, 503)
                release.set()
                self.assertEqual(await response.content.read(), b'last\n')
                await asyncio.wait_for(stop, 5)
                self.assertFalse(c.backend.alive)
                self.assertEqual(c.active, 0)
        finally:
            release.set()
            await proxy.cleanup()
            await upstream.cleanup()


if __name__ == '__main__':
    unittest.main(verbosity=2)
