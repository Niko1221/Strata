"""Fetch the Node24 security patch into private evidence, pinned to official release digests."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen
import zipfile

BASE = Path('H:/Strata/evidence/upstream-sync-20261116')
OUT = BASE / 'node-24.22.3-v1'
URL = 'https://nodejs.org/dist/v24.22.3/'
ARCHIVE = 'node-v24.22.3-win-x64.zip'
ZIP_SHA = 'e612af99f626ab415ea225af6bce5c971feb448cb5cae250bb1c3fa50c5f7d9f'
EXE_SHA = '86fc4785327db5f5c7558dc6156c543b433c313e813a222e088086222ab4afa6'


def download(name, maximum):
    target = OUT / name
    request = Request(URL + name, headers={'User-Agent': 'Strata-runtime-verification/0.6.0'})
    count = 0
    with urlopen(request, timeout=45) as response, target.open('xb') as stream:
        while data := response.read(1024 * 1024):
            count += len(data)
            if count > maximum:
                raise ValueError('Runtime download exceeds bounded size')
            stream.write(data)
    return target


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    OUT.mkdir()
    sums = download('SHASUMS256.txt', 2 * 1024**2)
    rows = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
            for line in sums.read_text().splitlines() if line.strip()}
    assert rows[ARCHIVE] == ZIP_SHA and rows['win-x64/node.exe'] == EXE_SHA
    archive = download(ARCHIVE, 200 * 1024**2)
    assert sha(archive) == ZIP_SHA
    prefix = 'node-v24.22.3-win-x64/'
    with zipfile.ZipFile(archive) as package:
        for name in ('node.exe', 'LICENSE'):
            member = prefix + name
            assert package.namelist().count(member) == 1
            (OUT / name).write_bytes(package.read(member))
    assert sha(OUT / 'node.exe') == EXE_SHA
    version = subprocess.check_output([str(OUT / 'node.exe'), '--version'], text=True, timeout=15).strip()
    assert version == 'v24.22.3'
    report = {'version': version, 'release': 'https://nodejs.org/en/blog/release/v24.22.3/',
              'source': URL, 'archiveSHA256': ZIP_SHA, 'nodeSHA256': EXE_SHA,
              'licenseSHA256': sha(OUT / 'LICENSE'), 'shasumsSHA256': sha(sums),
              'primaryChecksumPinsMatched': True, 'stagedIntoApp': False,
              'oldInstalledRuntimeChanged': False, 'pgpSignatureVerified': False}
    (OUT / 'receipt.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
