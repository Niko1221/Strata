"""Wrap a verified full ZIP in a per-user NSIS setup executable."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from native_preview import no_reparse, sha256


def build(archive: Path, output: Path, nsis: Path, icon: Path) -> dict:
    for path in [archive, output, nsis, icon]:
        no_reparse(path)
    if output.exists() or not output.parent.is_dir() or archive.stat().st_size >= 2**31-1024**2:
        raise ValueError('setup output must be fresh and payload below the NSIS 2 GiB limit')
    with archive.open('rb') as stream:
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
        archive_hash=digest.hexdigest()
        stream.seek(0)
        with zipfile.ZipFile(stream) as package:
            raw=package.read('full-windows.json')
            entries=package.infolist()
    manifest=json.loads(raw)
    if manifest['schema']!='strata-full-windows-v1':
        raise ValueError('wrong full-package schema')
    expected={item['path']:item for item in manifest['files']}
    names=[item.filename for item in entries]
    if len(set(name.casefold() for name in names))!=len(names) or set(names)!=set(expected)|{'full-windows.json','Install-Strata.ps1','START-HERE.cmd'}:
        raise ValueError('ZIP membership differs from full-package manifest')
    scripts=Path(__file__).parent
    program_bytes=sum(item['bytes'] for name,item in expected.items() if name.startswith('programs/'))
    component_bytes=sum(item['bytes'] for name,item in expected.items() if name.startswith('components/'))
    unpack_bytes=sum(item.file_size for item in entries)+64*1024**2
    definitions={
        'PAYLOAD_ZIP':str(archive.resolve()),'OUTPUT_EXE':str(output.resolve()),
        'OWNED_ICON':str(icon.resolve()),'BOOTSTRAP_PS1':str((scripts/'full_installer_bootstrap.ps1').resolve()),
        'ZIP_SHA256':archive_hash,'MANIFEST_SHA256':hashlib.sha256(raw).hexdigest(),
        'ZIP_BYTES':str(archive.stat().st_size),'UNPACK_BYTES':str(unpack_bytes),
        'PROGRAM_BYTES':str(program_bytes),'COMPONENT_BYTES':str(component_bytes),
    }
    command=[str(nsis),'/V3',*[f'/D{k}={v}' for k,v in definitions.items()],str(scripts/'full_installer.nsi')]
    log=output.with_suffix('.build.log')
    with log.open('x',encoding='utf-8') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'NSIS build failed; retained log {log}')
    if sha256(archive)!=archive_hash:
        raise ValueError('full ZIP changed during setup build')
    receipt={'installer':output.name,'sha256':sha256(output),'bytes':output.stat().st_size,
        'archive_sha256':archive_hash,'manifest_sha256':definitions['MANIFEST_SHA256'],
        'source_commit':manifest['public']['source_commit'],
        'wrapper_source_sha256':{name:sha256(scripts/name) for name in ['build_full_installer.py','full_installer.nsi','full_installer_bootstrap.ps1']},
        'unpacked_budget_bytes':unpack_bytes,'program_bytes':program_bytes,'component_bytes':component_bytes,
        'signature':'unsigned preview','fresh_install_tested':False}
    output.with_suffix('.build.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['archive','output','nsis','icon']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.archive,a.output,a.nsis,a.icon),indent=2))
