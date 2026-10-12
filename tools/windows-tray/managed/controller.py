"""Managed Tray sidecar. Control uses inherited pipes, never an HTTP admin route."""
import asyncio
import contextlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from aiohttp import ClientSession, ClientTimeout, DummyCookieJar, web
from multidict import CIMultiDict
from yarl import URL
from config import Settings, atomic_json, read_json
from windows_job import WindowsServer
import updates

HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te',
       'trailer', 'transfer-encoding', 'upgrade', 'proxy-connection', 'expect'}


def headers(raw):
    values = [(k.decode('latin1'), v.decode('latin1')) for k, v in raw]
    omit = HOP | {part.strip().lower() for k, v in values if k.lower() == 'connection' for part in v.split(',')}
    return CIMultiDict((k, v) for k, v in values if k.lower() not in omit)


class Controller:
    def __init__(self, settings, backend=None):
        self.settings = settings
        self.backend = backend or WindowsServer()
        self.selected = settings.selection()
        self.effective = None
        self.phase, self.error, self.model = 'stopped', '', 'unknown'
        self.active, self.accepting = 0, False
        self.condition = asyncio.Condition()
        self.commands = asyncio.Queue(maxsize=32)
        self.exit = asyncio.Event()
        self.client = None
        self.runtime = None
        self.version = settings.version
        path = settings.state / 'active-runtime.json'
        if path.exists():
            active = read_json(path)
            runtime = Path(active['runtime']).resolve()
            if not runtime.is_relative_to((settings.state / 'runtimes').resolve()) or not (runtime / 'serve/server.py').is_file():
                raise ValueError('Invalid active runtime')
            self.runtime, self.version = runtime, active['version']
        self.release = None
        self.update_state, self.update_checked = 'not checked', 0
        self.api_checked, self.api = 0, 'unavailable'
        self.web_status = 'not checked'
        self.last_activity = time.monotonic()
        self.busy = False
        self.current_config = None
        self.app = web.Application(client_max_size=0, handler_args={'auto_decompress': False})
        self.app.router.add_route('*', '/{path:.*}', self.forward)

    def owned(self):
        return self.backend.alive and self.backend.listener_owned(self.settings.upstream_port)

    def snapshot(self):
        return {'phase': self.phase, 'error': self.error, 'api': self.api, 'model': self.model,
                'active': self.active, 'selected': self.selected, 'effective': self.effective,
                'web': self.web_status, 'busy': self.busy, 'version': self.version, 'update': self.update_state,
                'latest': self.release['tag_name'] if self.release else None,
                'update_checked': self.update_checked, 'api_checked': self.api_checked,
                'captured': datetime.now(timezone.utc).isoformat()}

    def publish(self):
        print(json.dumps(self.snapshot(), ensure_ascii=True), flush=True)

    async def control(self, path, method='GET'):
        if not self.owned():
            raise RuntimeError('Owned listener unavailable')
        key = (self.current_config or {}).get('api_key') or os.environ.get('STRATA_API_KEY', '')
        kwargs = {'headers': {'Authorization': 'Bearer ' + key} if key else {},
                  'allow_redirects': False, 'timeout': ClientTimeout(total=3 if method == 'GET' else 3600)}
        if method == 'POST':
            kwargs['headers']['Content-Type'] = 'application/json'
            kwargs['json'] = {}
        async with self.client.request(method, 'http://127.0.0.1:' + str(self.settings.upstream_port) + path, **kwargs) as response:
            raw = await response.content.read(1048577)
            if response.status != 200 or len(raw) > 1048576:
                raise RuntimeError('Official API rejected control')
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise RuntimeError('Invalid official response')
            return value

    async def drain(self):
        async with self.condition:
            self.accepting, self.phase = False, 'draining'
            self.publish()
            await asyncio.wait_for(self.condition.wait_for(lambda: self.active == 0), 300)
        if self.backend.alive:
            deadline = time.monotonic() + 300
            while True:
                activity = (await self.control('/v1/status')).get('activity', {})
                if activity.get('in_flight') == 0:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('Official activity did not drain')
                await asyncio.sleep(.2)

    async def launch(self, config, runtime, expected_version=None):
        root = runtime or self.settings.repository
        generated = self.settings.state / 'generated-profile.json'
        atomic_json(generated, config)
        self.current_config = config
        self.phase, self.model, self.api = 'starting', 'loading', 'checking'
        self.publish()
        await self.backend.start([str(self.settings.python), '-B', str(root / 'serve/server.py'),
                                  '--engine', self.settings.engine, '--config', str(generated),
                                  '--host', '127.0.0.1', '--port', str(self.settings.upstream_port), '--lazy'],
                                 root, self.settings.logs / 'server.log', self.settings.upstream_port)
        deadline = time.monotonic() + 120
        while True:
            if not self.backend.alive:
                raise RuntimeError('Owned server exited during startup')
            if self.owned():
                try:
                    health = await self.control('/health')
                    if health.get('status') == 'ok' and health.get('service') == 'strata':
                        break
                except (OSError, RuntimeError, ValueError):
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Server startup timed out')
            await asyncio.sleep(.2)
        await self.control('/v1/load', 'POST')
        status = await self.control('/v1/status')
        if not status.get('loaded') or status.get('model') != config['model_name']:
            raise RuntimeError('Loaded model identity mismatch')
        if self.settings.engine != 'mock':
            if status.get('cache_max_tokens') != self.selected['context']:
                raise RuntimeError('Effective context mismatch')
            if expected_version and status.get('engine') not in (expected_version.lstrip('v'), 'v' + expected_version.lstrip('v')):
                raise RuntimeError('Effective engine version mismatch')
        if self.settings.engine != 'mock':
            actual_version = status.get('engine', '')
            try:
                updates.version(actual_version)
            except ValueError:
                if expected_version:
                    raise
                self.version = 'unknown'
            else:
                self.version = actual_version.lstrip('v')
        self.effective = {**self.selected, 'context': status.get('cache_max_tokens')}
        self.phase, self.model, self.api, self.error = 'ready', 'loaded', 'available', ''
        self.accepting = True
        self.last_activity = time.monotonic()

    async def restart(self, runtime=None, expected_version=None):
        runtime = self.runtime if runtime is None else runtime
        config = self.settings.generated(self.selected, runtime)
        # Validate original config before interrupting the previous service.
        await self.drain()
        await self.backend.stop()
        self.effective = None
        try:
            await self.launch(config, runtime, expected_version)
        except BaseException:
            await self.backend.stop()
            self.accepting = False
            raise

    async def check_update(self):
        self.update_state = 'checking'
        self.publish()
        try:
            release = await asyncio.to_thread(updates.latest)
            self.release = release
            self.update_state = 'available' if updates.version(release['tag_name']) > updates.version(self.version) else 'current'
        except Exception:
            self.update_state = 'check failed or installed version unknown'
        self.update_checked = time.time()

    async def command(self, command):
        action = command.get('action')
        if action == 'check':
            await self.check_update()
            return
        if action == 'select':
            proposed = {**self.selected, **command.get('selection', {})}
            self.settings.validate_selection(proposed)
            self.settings.generated(proposed, self.runtime)
            self.selected = proposed
            atomic_json(self.settings.state / 'selection.json', self.selected)
            await self.restart()
        elif action == 'auto':
            if type(command.get('enabled')) is not bool:
                raise ValueError('Boolean required')
            self.selected['auto'] = command['enabled']
            atomic_json(self.settings.state / 'selection.json', self.selected)
        elif action in ('start', 'restart'):
            await self.restart()
        elif action in ('stop', 'exit'):
            await self.drain()
            await self.backend.stop()
            self.effective = None
            self.phase, self.model, self.api = 'stopped', 'unloaded', 'unavailable'
            if action == 'exit':
                self.exit.set()
        elif action in ('load', 'unload'):
            await self.drain()
            try:
                await self.control('/v1/' + action, 'POST')
            finally:
                self.accepting = self.owned()
                self.phase = 'ready' if self.accepting else 'failed'
        elif action == 'update':
            if self.update_state != 'available' or self.release is None:
                raise ValueError('Check for a compatible update first')
            self.update_state = 'staging'
            self.publish()
            try:
                staged = await asyncio.to_thread(updates.stage, self.settings, self.release, self.selected,
                                                 self.runtime or self.settings.repository)
                await self.restart(staged, self.release['tag_name'])
                try:
                    atomic_json(self.settings.state / 'active-runtime.json',
                                {'runtime': str(staged), 'version': self.release['tag_name'].lstrip('v')})
                except Exception:
                    self.accepting = False
                    await self.backend.stop()
                    self.effective = None
                    raise
                self.runtime, self.version = staged, self.release['tag_name'].lstrip('v')
                self.update_state = 'current'
            except Exception:
                self.update_state = 'update failed; manual action required'
                self.selected['auto'] = False
                atomic_json(self.settings.state / 'selection.json', self.selected)
                raise
        else:
            raise ValueError('Unknown command')

    async def worker(self):
        while not self.exit.is_set():
            command = await self.commands.get()
            self.busy, self.error = True, ''
            self.publish()
            try:
                await self.command(command)
            except Exception as error:
                with (self.settings.logs / 'controller.log').open('a', encoding='utf-8') as log:
                    traceback.print_exc(file=log)
                # Class only: errors may carry keys, request bodies or private paths.
                self.error = 'Action failed (' + type(error).__name__ + '); check configuration and local logs'
                if not self.backend.alive:
                    self.phase, self.api, self.model, self.effective = 'failed', 'unavailable', 'unknown', None
                    self.accepting = False
                elif self.owned():
                    self.phase, self.accepting = 'ready', True
            finally:
                self.busy = False
                self.publish()
                self.commands.task_done()

    async def poll(self):
        while not self.exit.is_set():
            if not self.busy and self.backend.alive:
                try:
                    status = await self.control('/v1/status')
                    self.api = 'available'
                    try:
                        async with self.client.get('http://127.0.0.1:' + str(self.settings.upstream_port) + '/',
                                                   allow_redirects=False, timeout=ClientTimeout(total=3)) as response:
                            self.web_status = 'reachable' if response.status == 200 else 'auth protected' if response.status in (401, 403) else 'HTTP ' + str(response.status)
                    except Exception:
                        self.web_status = 'unavailable'
                    self.model = 'loaded' if status.get('loaded') else 'unloaded'
                    if status.get('activity', {}).get('in_flight'):
                        self.last_activity = time.monotonic()
                except Exception:
                    self.api, self.model, self.web_status = 'unavailable', 'unknown', 'unavailable'
                self.api_checked = time.time()
            elif not self.backend.alive:
                self.api = 'unavailable'
                if self.phase == 'ready':
                    self.phase, self.model, self.effective, self.accepting = 'failed', 'unknown', None, False
            if not self.busy and self.commands.empty():
                if time.time() - self.update_checked >= 86400:
                    await self.commands.put({'action': 'check'})
                elif self.selected['auto'] and self.update_state == 'available' and self.phase == 'ready' and not self.active and time.monotonic() - self.last_activity >= 60:
                    await self.commands.put({'action': 'update'})
            self.publish()
            await asyncio.sleep(2)

    async def forward(self, request):
        async with self.condition:
            if not self.accepting or not self.owned():
                return web.json_response({'error': {'message': 'Tray server is stopped or draining'}}, status=503)
            self.active += 1
            self.last_activity = time.monotonic()
        response = None
        try:
            url = URL('http://127.0.0.1:' + str(self.settings.upstream_port) + request.raw_path, encoded=True)
            async with self.client.request(request.method, url, headers=headers(request.raw_headers),
                                           data=request.content.iter_any(), allow_redirects=False) as upstream:
                response = web.StreamResponse(status=upstream.status, headers=headers(upstream.raw_headers))
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        except Exception:
            if response is not None and response.prepared:
                if request.transport:
                    request.transport.close()
                return response
            return web.json_response({'error': {'message': 'Owned upstream transport unavailable'}}, status=502)
        finally:
            async with self.condition:
                self.active -= 1
                self.last_activity = time.monotonic()
                self.condition.notify_all()


async def run(settings):
    controller = Controller(settings)
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    listener.bind(('127.0.0.1', settings.port))
    listener.listen(128)
    listener.setblocking(False)
    runner = web.AppRunner(controller.app, access_log=None, handler_cancellation=True, auto_decompress=False)
    tasks = []
    async def stdin():
        while True:
            line = await asyncio.to_thread(sys.stdin.buffer.readline, 4097)
            if not line:
                controller.exit.set()
                return
            if len(line) > 4096:
                controller.error = 'Control message exceeds bound'
                continue
            try:
                command = json.loads(line)
                if not isinstance(command, dict):
                    raise ValueError('Object required')
                await controller.commands.put(command)
                if command.get('action') == 'exit':
                    await controller.commands.join()
                    if controller.exit.is_set():
                        return
            except (ValueError, TypeError):
                controller.error = 'Invalid control message'
    try:
        async with ClientSession(timeout=ClientTimeout(total=None, sock_connect=10),
                                 auto_decompress=False, cookie_jar=DummyCookieJar(), trust_env=False,
                                 skip_auto_headers={'Accept', 'Accept-Encoding', 'User-Agent', 'Content-Type'}) as client:
            controller.client = client
            await runner.setup()
            await web.SockSite(runner, listener).start()
            tasks = [asyncio.create_task(controller.worker()), asyncio.create_task(controller.poll()), asyncio.create_task(stdin())]
            await controller.commands.put({'action': 'start'})
            await controller.exit.wait()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await controller.backend.stop()
        await runner.cleanup()
        listener.close()


if __name__ == '__main__':
    # The parent assigns this exact child to its Job before releasing this gate.
    if sys.stdin.buffer.readline(32).strip() != b'START':
        raise SystemExit(2)
    try:
        asyncio.run(run(Settings(sys.argv[1])))
    except Exception as error:
        print(json.dumps({'phase': 'failed', 'error': 'Controller failed (' + type(error).__name__ + ')'}), flush=True)
        raise SystemExit(1)
