"""Verified, immutable official runtime staging; never runs setup or changes Git."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
import urllib.request
import uuid
import zipfile
from config import atomic_json, read_json

RELEASES = 'https://api.github.com/repos/Niko1221/Strata/releases'


class ComponentUpdateRequired(ValueError):
    """Signed component releases are installed only by the official updater."""


def version(value):
    if not isinstance(value, str) or not re.fullmatch(r'v?\d+\.\d+\.\d+(?:\.\d+)?', value):
        raise ValueError('Stable version required')
    parts = tuple(map(int, value.lstrip('v').split('.')))
    return parts + (0,) * (4 - len(parts))


def fetch(url, destination, limit, digest=None, size=None):
    request = urllib.request.Request(url, headers={'User-Agent': 'Strata-Windows-Tray', 'Accept': 'application/vnd.github+json'})
    sha, total = hashlib.sha256(), 0
    with urllib.request.urlopen(request, timeout=60) as response, Path(destination).open('xb') as output:
        if not response.geturl().startswith('https://'):
            raise ValueError('HTTPS required')
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                raise ValueError('Download exceeds bound')
            output.write(chunk)
            sha.update(chunk)
    if size is not None and total != size or digest is not None and sha.hexdigest() != digest:
        raise ValueError('Release digest or size mismatch')
    return sha.hexdigest()


def latest():
    request = urllib.request.Request(RELEASES + '/latest', headers={'User-Agent': 'Strata-Windows-Tray'})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(1048577)
    if len(raw) > 1048576:
        raise ValueError('Release metadata exceeds bound')
    value = json.loads(raw)
    if any(asset.get('name') == 'strata-feed.json' for asset in value.get('assets', []) if isinstance(asset, dict)):
        raise ComponentUpdateRequired('Use the official signed component updater')
    version(value['tag_name'])
    if value.get('draft') or value.get('prerelease'):
        raise ValueError('Stable release required')
    return value


def extract(archive, destination, strip_root=False):
    destination = Path(destination)
    destination.mkdir()
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if len(entries) > 20000 or sum(e.file_size for e in entries) > 4 * 1024**3:
            raise ValueError('Archive exceeds bound')
        seen, roots = set(), set()
        for entry in entries:
            name = entry.orig_filename
            parts = PurePosixPath(name).parts
            if (not parts or name.startswith('/') or '\\' in name or ':' in name or '..' in parts
                    or any(p.endswith((' ', '.')) or p.split('.')[0].upper() in
                           {'CON', 'PRN', 'AUX', 'NUL', *('COM' + str(i) for i in range(1, 10)),
                            *('LPT' + str(i) for i in range(1, 10))} for p in parts)
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError('Unsafe archive entry')
            roots.add(parts[0])
            if strip_root:
                parts = parts[1:]
            if not parts:
                continue
            key = '/'.join(parts).casefold()
            if key in seen:
                raise ValueError('Duplicate archive entry')
            seen.add(key)
            target = destination.joinpath(*parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Archive escapes destination')
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as stream, target.open('xb') as output:
                    shutil.copyfileobj(stream, output)
        if strip_root and len(roots) != 1:
            raise ValueError('Expected one source root')


def stage(settings, release, selected, current_repository):
    if settings.engine != 'strata' or settings.asset is None:
        raise ValueError('Select the matching official update asset in Tray config first')
    tag = release['tag_name']
    version(tag)
    assets = [a for a in release['assets'] if a.get('name') == settings.asset]
    if len(assets) != 1:
        raise ValueError('Matching backend asset unavailable; use official setup manually')
    asset = assets[0]
    digest = asset.get('digest', '')
    expected_url = 'https://github.com/Niko1221/Strata/releases/download/' + tag + '/' + settings.asset
    if asset.get('browser_download_url') != expected_url or not re.fullmatch(r'sha256:[a-f0-9]{64}', digest):
        raise ValueError('Official release asset checksum required')
    if type(asset.get('size')) is not int or not 0 < asset['size'] <= 2 * 1024**3:
        raise ValueError('Invalid release size')
    base = settings.state / 'runtimes'
    base.mkdir(exist_ok=True)
    root = base / (tag + '-' + uuid.uuid4().hex)
    root.mkdir()
    # Official source URL is fixed, not supplied by release metadata. TLS authenticates
    # source; the asset has an additional digest supplied by GitHub's releases API.
    source_sha = fetch('https://codeload.github.com/Niko1221/Strata/zip/refs/tags/' + tag,
                       root / 'source.zip', 256 * 1024**2)
    fetch(expected_url, root / 'engine.zip', 2 * 1024**3, digest[7:], asset['size'])
    runtime = root / 'runtime'
    extract(root / 'source.zip', runtime, strip_root=True)
    extract(root / 'engine.zip', runtime / 'engine')
    if not (runtime / 'serve/server.py').is_file() or not (runtime / 'engine/strata.exe').is_file():
        raise ValueError('Official runtime layout changed; manual update required')
    # No implicit pip install or execution of newly downloaded installer code.
    if (runtime / 'requirements.txt').read_bytes() != (Path(current_repository) / 'requirements.txt').read_bytes():
        raise ValueError('Python dependencies changed; manual update required')
    build = read_json(runtime / 'engine/BUILD.json')
    bundled_dlls = {p.name.casefold() for p in (runtime / 'engine').glob('*.dll')}
    for name in settings.profiles:
        profile = settings.profile(name)
        old_engine = Path(profile['exe']).parent
        old_build = read_json(old_engine / 'BUILD.json')
        # Reject changed build/runtime families rather than guessing compatible DLLs.
        for key in ('backend', 'cuda', 'cuda_version', 'rocm', 'rocm_version'):
            if old_build.get(key) != build.get(key):
                raise ValueError('Engine backend/runtime changed; manual update required')
        if not any(k in build for k in ('backend', 'cuda', 'cuda_version', 'rocm', 'rocm_version')):
            raise ValueError('Backend metadata unavailable; manual update required')
        settings.generated({**selected, 'profile': name}, runtime)
        for dll in old_engine.glob('*.dll'):
            target = runtime / 'engine' / dll.name
            # Official release DLLs supersede old engine bytes. Only fallback
            # dependencies copied from installed engines must agree by profile.
            if dll.name.casefold() in bundled_dlls:
                continue
            if target.exists():
                if target.read_bytes() != dll.read_bytes():
                    raise ValueError('Profiles use different native dependencies')
            else:
                shutil.copyfile(dll, target)
    atomic_json(root / 'manifest.json', {'tag': tag, 'asset_sha256': digest[7:], 'source_sha256': source_sha,
                                       'staged_at': time.time()})
    return runtime
