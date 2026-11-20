import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packaging'))
spec=importlib.util.spec_from_file_location('public_shell',ROOT/'packaging/public_shell.py')
shell=importlib.util.module_from_spec(spec)
spec.loader.exec_module(shell)


def test_stage_excludes_unqualified_resources_and_upstream_updater(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'src').mkdir()
    for name in ['Cargo.toml','Cargo.lock','build.rs']:
        (source/name).write_text('source')
    (source/'src/main.rs').write_text('fn main() {}')
    (source/'tauri.conf.json').write_text(json.dumps({'build':{'frontendDist':'old'},'bundle':{'resources':{'private':'private'},'windows':{'nsis':{'headerImage':'private.bmp'}}},'plugins':{'updater':{'pubkey':'upstream'}}}))
    (source/'unqualified.png').write_bytes(b'do not copy')
    icons=tmp_path/'owned';icons.mkdir()
    for name in ['icon.png','icon.ico']:(icons/name).write_bytes(name.encode())
    output=tmp_path/'stage'
    pins=shell.stage(source,icons,output)
    config=json.loads((output/'tauri.conf.json').read_text())
    assert config['build']['frontendDist']=='minimal-ui'
    assert config['bundle']['resources']=={} and config['plugins']=={}
    assert config['bundle']['windows']['nsis']=={'installMode':'currentUser'}
    assert not (output/'unqualified.png').exists()
    assert (output/'minimal-ui/index.html').read_text()==shell.MINIMAL_HTML
    assert 'src/main.rs' in pins
    with pytest.raises(ValueError):shell.stage(source,icons,output)
