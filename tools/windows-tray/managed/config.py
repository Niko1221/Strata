"""Explicit local configuration for the optional managed Tray."""
import json
import os
import re
from pathlib import Path
import tempfile

IDLES = (0, 300, 600, 1800)
ASSETS = ('strata-windows-x64.zip', 'strata-windows-x64-cuda12.zip', 'strata-windows-x64-hip.zip')


def read_json(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(1048577)
    if len(raw) > 1048576:
        raise ValueError('Configuration exceeds 1 MiB')
    value = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Configuration must be an object')
    return value


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('Invalid integer setting')
    return value


def resolve(base, value):
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('Invalid path setting')
    return (base / value).resolve()


class Settings:
    def __init__(self, path):
        self.path = Path(path).resolve()
        value = read_json(self.path)
        if set(value) - {'python', 'repository', 'profiles', 'port', 'upstream_port', 'engine',
                         'automatic_upgrade', 'update_asset', 'installed_version', 'desktop_app'}:
            raise ValueError('Unknown Tray configuration field')
        self.desktop_app = resolve(self.path.parent, value['desktop_app']) if 'desktop_app' in value else None
        if self.desktop_app is not None and (not self.desktop_app.is_file() or self.desktop_app.suffix.lower() != '.exe'):
            raise ValueError('desktop_app must name an existing Windows App executable')
        self.python = resolve(self.path.parent, value['python'])
        self.repository = resolve(self.path.parent, value['repository'])
        if not self.python.is_file() or not (self.repository / 'serve/server.py').is_file():
            raise ValueError('Python or official server is missing')
        self.port = integer(value.get('port', 8095), 1, 65535)
        self.upstream_port = integer(value.get('upstream_port', 8096), 1, 65535)
        if self.port == self.upstream_port:
            raise ValueError('Ports must differ')
        self.engine = value.get('engine', 'strata')
        if self.engine not in ('strata', 'mock'):
            raise ValueError('Unsupported engine')
        self.auto = value.get('automatic_upgrade', False)
        if type(self.auto) is not bool:
            raise ValueError('automatic_upgrade must be boolean')
        self.asset = value.get('update_asset')
        if self.asset is not None and self.asset not in ASSETS:
            raise ValueError('Unsupported update asset')
        self.version = value.get('installed_version', 'unknown')
        if self.version != 'unknown' and (not isinstance(self.version, str) or not re.fullmatch(r'v?\d+\.\d+\.\d+(?:\.\d+)?', self.version)):
            raise ValueError('Invalid installed version')
        profiles = value['profiles']
        if not isinstance(profiles, dict) or not 1 <= len(profiles) <= 32:
            raise ValueError('One to 32 profiles required')
        self.profiles = {}
        for name, source in profiles.items():
            if not isinstance(name, str) or not name.strip() or len(name) > 64 or any(ord(c) < 32 for c in name):
                raise ValueError('Invalid profile label')
            self.profiles[name] = resolve(self.path.parent, source)
        self.state = self.path.parent / (self.path.stem + '.state')
        self.state.mkdir(exist_ok=True)
        self.logs = self.state / 'logs'
        self.logs.mkdir(exist_ok=True)
        self.policy = None
        for name in self.profiles:
            self.profile(name)

    def profile(self, name):
        if name not in self.profiles:
            raise ValueError('Unknown profile')
        value = read_json(self.profiles[name])
        if not isinstance(value.get('model_name'), str) or not value['model_name']:
            raise ValueError('Profile needs model_name')
        args = value.get('args')
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError('Profile needs string args')
        if args.count('--max-context') != 1:
            raise ValueError('Profile needs exactly one --max-context')
        index = args.index('--max-context') + 1
        if index >= len(args) or not args[index].isdigit():
            raise ValueError('Invalid profile context')
        integer(int(args[index]), 1, 2147483647)
        integer(value.get('idle_unload_s', 300), 0, 2147483647)
        policy = {key: value.get(key) for key in ('api_key', 'trusted_origins', 'cors_origins', 'allowed_hosts')}
        if self.policy is not None and self.policy != policy:
            raise ValueError('Profiles must preserve authentication and origin policy; restart Tray to change it')
        self.policy = policy
        cwd = resolve(self.repository, value.get('cwd', str(self.repository)))
        if not cwd.is_dir():
            raise ValueError('Profile working directory is missing')
        value['cwd'] = str(cwd)
        if self.engine == 'strata':
            exe = resolve(cwd, value['exe'])
            if not exe.is_file():
                raise ValueError('Profile engine is missing')
            value['exe'] = str(exe)
        return value

    def selection(self):
        first = next(iter(self.profiles))
        profile = self.profile(first)
        value = {'profile': first, 'context': int(profile['args'][profile['args'].index('--max-context') + 1]),
                 'idle': profile.get('idle_unload_s', 300), 'auto': self.auto}
        path = self.state / 'selection.json'
        self.validate_selection(value)
        if path.exists():
            saved = read_json(path)
            try:
                self.validate_selection(saved)
            except ValueError:
                # A renamed/removed profile must not make every future launch fail.
                # Reset only persisted selection; invalid source config still fails.
                atomic_json(path, value)
            else:
                value = saved
        self.validate_selection(value)
        return value

    def validate_selection(self, value):
        if set(value) != {'profile', 'context', 'idle', 'auto'} or value['profile'] not in self.profiles:
            raise ValueError('Invalid saved selection')
        integer(value['context'], 1, 2147483647)
        integer(value['idle'], 0, 2147483647)
        if type(value['auto']) is not bool:
            raise ValueError('Invalid automatic update preference')

    def generated(self, selected, runtime=None):
        self.validate_selection(selected)
        value = self.profile(selected['profile'])
        value['args'][value['args'].index('--max-context') + 1] = str(selected['context'])
        value['idle_unload_s'] = selected['idle']
        if runtime:
            if value.get('vision') or value.get('lib_dirs') not in (None, [], [str(Path(value['exe']).parent)]):
                raise ValueError('Custom vision/library profiles need a manual update')
            value['exe'] = str(Path(runtime) / 'engine/strata.exe')
            value['lib_dirs'] = [str(Path(runtime) / 'engine')]
        return value
