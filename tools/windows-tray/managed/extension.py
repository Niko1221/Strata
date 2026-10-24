"""Opt-in file IPC for the companion local-model extension. No independent lifecycle."""
from dataclasses import dataclass
import json
import os
import re
import stat
from uuid import UUID
from config import atomic_json, read_json

CONTEXTS = (8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288)
IDLES = (0, 300, 600, 1800)


def request_id(value):
    if not isinstance(value, str):
        return None
    try:
        return value if str(UUID(value)) == value.lower() else None
    except ValueError:
        return None


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate request field')
        value[key] = item
    return value


@dataclass(frozen=True)
class ExtensionRequest:
    id: str
    expected_model: str
    command: dict

    @staticmethod
    def parse(value):
        if not isinstance(value, dict) or type(value.get('schema')) is not int or value['schema'] != 1 or not request_id(value.get('id')):
            raise ValueError('Invalid extension request')
        action = value.get('action')
        fields = {'schema', 'id', 'expectedModel', 'action'}
        if action in ('context', 'idle'):
            fields.add('value')
        expected = value.get('expectedModel')
        if set(value) != fields or not isinstance(expected, str) or not expected or len(expected) > 1024 or any(ord(c) < 32 for c in expected):
            raise ValueError('Invalid extension fields')
        if action in ('context', 'idle'):
            choice = value.get('value')
            if type(choice) is not int or choice not in (CONTEXTS if action == 'context' else IDLES):
                raise ValueError('Invalid extension selection')
            command = {'action': 'select', 'selection': {action: choice}}
        elif action in ('load', 'unload'):
            command = {'action': action}
        else:
            raise ValueError('Unsupported extension action')
        return ExtensionRequest(value['id'], expected, command)

    def prepare(self, controller):
        # Recheck here, on the serial worker, after earlier Tray commands finish.
        if (controller.current_config or {}).get('model_name') != self.expected_model:
            raise ValueError('Extension model identity changed')
        if self.command['action'] == 'select':
            proposed = {**controller.selected, **self.command['selection']}
            if controller.settings.generated(proposed, controller.runtime).get('model_name') != self.expected_model:
                raise ValueError('Extension profile identity changed')
        return self.command


class ExtensionIPC:
    def __init__(self, directory, port):
        self.status_path = directory / 'extension-status.json'
        self.input_path = directory / 'extension-request.json'
        self.running_path = directory / 'extension-running.json'
        self.api_base = f'http://127.0.0.1:{port}/v1'
        self.ack, self.active, self.healthy = None, False, False
        self.seen = set()
        if self.status_path.exists():
            try:
                previous = read_json(self.status_path).get('request')
                if isinstance(previous, dict) and request_id(previous.get('id')) and previous.get('state') in ('done', 'error', 'running'):
                    self.ack = {key: previous[key] for key in ('id', 'state')}
                    if previous['state'] == 'error':
                        self.ack['error'] = previous.get('error') if re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', str(previous.get('error', ''))) else 'RuntimeError'
                    self.seen.add(previous['id'].lower())
            except (OSError, ValueError):
                pass
        self.blocked = self.running_path.exists() or bool(self.ack and self.ack['state'] == 'running')
        if self.blocked and (not self.ack or self.ack['state'] == 'running'):
            # A previous owner may have acted. Report interruption, never replay or
            # remove its marker. Recovery requires inspecting local state manually.
            if not self.ack:
                try:
                    self.ack = {'id': request_id(read_json(self.running_path).get('id'))}
                except (OSError, ValueError):
                    self.ack = {'id': None}
            self.ack.update(state='error', error='InterruptedError')

    def publish(self, snapshot, current_config):
        loaded = True if snapshot['model'] == 'loaded' else False if snapshot['model'] == 'unloaded' else None
        value = {**snapshot, 'schema': 1, 'modelId': (current_config or {}).get('model_name'),
                 'apiBaseUrl': self.api_base, 'loaded': loaded}
        if self.ack is not None:
            value['request'] = self.ack
        try:
            atomic_json(self.status_path, value)
            self.healthy = True
            # A durable terminal acknowledgment must precede removal of the claim.
            if self.active and self.ack['state'] in ('done', 'error'):
                self.running_path.unlink()
                self.active = False
        except OSError:
            self.healthy = False

    def claim(self, commands):
        if self.blocked or self.active or not self.healthy or commands.full():
            return
        try:
            os.link(self.input_path, self.running_path, follow_symlinks=False)
        except (FileExistsError, FileNotFoundError):
            return
        except OSError:
            self.healthy = False
            return
        self.active = True
        self.ack = {'id': None, 'state': 'running'}
        try:
            self.input_path.unlink()
        except OSError:
            self.blocked, self.active = True, False
            self.ack.update(state='error', error='OSError')
            return
        try:
            info = self.running_path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
                raise ValueError('Invalid extension file')
            value = json.loads(self.running_path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)
            self.ack['id'] = request_id(value.get('id')) if isinstance(value, dict) else None
            if self.ack['id'] and self.ack['id'].lower() in self.seen:
                raise ValueError('Extension request already consumed')
            if self.ack['id']:
                self.seen.add(self.ack['id'].lower())
            request = ExtensionRequest.parse(value)
            commands.put_nowait(request)
        except (OSError, ValueError, TypeError) as error:
            self.ack.update(state='error', error=type(error).__name__)

    def finish(self, request, error=None):
        if self.active and self.ack['id'] == request.id:
            self.ack = {'id': request.id, 'state': 'done' if error is None else 'error'}
            if error is not None:
                self.ack['error'] = type(error).__name__
