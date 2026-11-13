"""Install, enable, disable and roll back the independent Local Model Extension.

Only its dedicated runtime directory is written. App files, the shared carrier,
model files and saved context/idle/client settings are never deployment inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import uuid

SUFFIXES = {'.html', '.mjs', '.js', '.css', '.svg', '.png', '.json', '.woff2'}


def no_links(path):
    for current in (path, *path.parents):
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError('Linked Extension Paths Are Not Supported')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    no_links(path)
    if path.stat().st_size > 65536:
        raise ValueError('Extension Metadata Is Too Large')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def atomic(path, data):
    no_links(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def runtime_root(path):
    path = Path(path)
    if not path.is_absolute() or path.name != 'strata-local-models':
        raise ValueError('Use An Absolute Dedicated strata-local-models Directory')
    no_links(path)
    return path


def asset_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'ui/[A-Za-z0-9_./-]+', value):
        raise ValueError('Invalid Extension Asset Name')
    path = PurePosixPath(value)
    if path.as_posix() != value or any(part in ('', '.', '..') for part in value.split('/')) or path.suffix not in SUFFIXES:
        raise ValueError('Invalid Extension Asset Name')
    return value


def manifest_fields(value):
    if not isinstance(value, dict) or value.get('schema') != 1 or value.get('id') != 'strata-local-models':
        raise ValueError('Invalid Local Model Extension Manifest')
    if not isinstance(value.get('version'), str) or not re.fullmatch(r'\d+\.\d+\.\d+', value['version']):
        raise ValueError('Invalid Extension Version')
    if value.get('bridgeApi') != 1 or value.get('app') != {'major': 0, 'minor': 2}:
        raise ValueError('Unsupported Extension Compatibility Contract')
    asset_name(value.get('entry'))
    asset_name(value.get('panel'))
    fields = {key: value[key] for key in ('schema', 'id', 'version', 'bridgeApi', 'app', 'entry', 'panel')}
    if 'features' in value:
        raise ValueError('Unsupported Extension Feature Contract')
    return fields


def verify_release(root, record):
    root = runtime_root(root)
    if (not isinstance(record, dict) or set(record) != {'release', 'sha256'}
            or not re.fullmatch(r'\d+\.\d+\.\d+-[a-f0-9]{12}', record.get('release', ''))
            or not re.fullmatch(r'[a-f0-9]{64}', record.get('sha256', ''))
            or not record['release'].endswith(record['sha256'][:12])):
        raise ValueError('Invalid Extension Release Record')
    release = root/'releases'/record['release']
    no_links(release)
    manifest_path = release/'manifest.json'
    value = read_json(manifest_path)
    fields = manifest_fields(value)
    if digest(manifest_path.read_bytes()) != record['sha256'] or not record['release'].startswith(fields['version']+'-'):
        raise ValueError('Changed Extension Manifest')
    files = value.get('files')
    if not isinstance(files, list) or not 1 <= len(files) <= 32:
        raise ValueError('Invalid Extension Inventory')
    names, total = set(), 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {'path', 'bytes', 'sha256'}:
            raise ValueError('Invalid Extension File Record')
        name = asset_name(item['path'])
        if name in names or type(item['bytes']) is not int or not 0 <= item['bytes'] <= 2097152:
            raise ValueError('Invalid Extension File Record')
        names.add(name)
        path = release/name
        no_links(path)
        data = path.read_bytes()
        total += len(data)
        if total > 8388608 or len(data) != item['bytes'] or digest(data) != item['sha256']:
            raise ValueError('Changed Extension Asset')
    if not {fields['entry'], fields['panel']} <= names:
        raise ValueError('Missing Extension Entry')
    return value


def state(root):
    path = runtime_root(root)/'active.json'
    if not path.exists():
        return {'schema': 1, 'enabled': False, 'active': None, 'previous': None}
    value = read_json(path)
    if set(value) != {'schema', 'enabled', 'active', 'previous'} or value['schema'] != 1 or type(value['enabled']) is not bool:
        raise ValueError('Invalid Extension Activation State')
    return value


def install(source, root, *, activate=False):
    source, root = Path(source), runtime_root(root)
    no_links(source)
    fields = manifest_fields(read_json(source/'manifest.json'))
    payload, records = {}, []
    for path in sorted((source/'ui').rglob('*')):
        no_links(path)
        if path.is_dir() or path.name.endswith('.test.mjs'):
            continue
        name = asset_name(path.relative_to(source).as_posix())
        data = path.read_bytes()
        if len(data) > 2097152:
            raise ValueError('Extension Asset Is Too Large')
        payload[name] = data
        records.append({'path': name, 'bytes': len(data), 'sha256': digest(data)})
    if not 1 <= len(records) <= 32 or sum(len(data) for data in payload.values()) > 8388608:
        raise ValueError('Invalid Extension Inventory')
    manifest = encode({**fields, 'files': records})
    checksum = digest(manifest)
    record = {'release': fields['version']+'-'+checksum[:12], 'sha256': checksum}
    release = root/'releases'/record['release']
    if not release.exists():
        release.mkdir(parents=True)
        for name, data in payload.items():
            target = release/name
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic(target, data)
        atomic(release/'manifest.json', manifest)
    verify_release(root, record)
    preferences = root/'preferences.json'
    if not preferences.exists():
        atomic(preferences, encode({'schema': 1, 'uiCopyStyle': 'title_case'}))
    if activate:
        activate_release(root, record)
    return record


def activate_release(root, record):
    root = runtime_root(root)
    verify_release(root, record)
    old = state(root)
    previous = old['previous'] if old['active'] == record else old['active']
    atomic(root/'active.json', encode({'schema': 1, 'enabled': True, 'active': record, 'previous': previous}))


def set_enabled(root, enabled):
    root = runtime_root(root)
    value = state(root)
    if enabled:
        verify_release(root, value['active'])
    value['enabled'] = enabled
    root.mkdir(parents=True, exist_ok=True)
    atomic(root/'active.json', encode(value))
    return value


def rollback(root):
    root = runtime_root(root)
    value = state(root)
    verify_release(root, value['previous'])
    value['active'], value['previous'] = value['previous'], value['active']
    value['enabled'] = True
    atomic(root/'active.json', encode(value))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('install', 'status', 'enable', 'disable', 'rollback'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args()
    if args.command == 'install':
        if args.source is None:
            parser.error('install requires --source')
        result = install(args.source, args.root, activate=args.activate)
    elif args.command == 'status':
        result = state(args.root)
    elif args.command == 'rollback':
        result = rollback(args.root)
    else:
        result = set_enabled(args.root, args.command == 'enable')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
