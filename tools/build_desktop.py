"""Build the Windows desktop app using a verified, project-local Microsoft SDK."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import urllib.request
from urllib.parse import urlsplit
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.local' / 'desktop-build'
SDK_VERSION = '10.0.401'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def sdk():
    target = CACHE / 'dotnet'
    binary = target / 'dotnet.exe'
    if (target / 'sdk' / SDK_VERSION).is_dir() and binary.is_file():
        return binary
    CACHE.mkdir(parents=True, exist_ok=True)
    with OPENER.open('https://builds.dotnet.microsoft.com/dotnet/release-metadata/10.0/releases.json', timeout=30) as response:
        metadata = json.load(response)
    package = next(item for release in metadata['releases'] for entry in release.get('sdks', [])
                   if entry['version'] == SDK_VERSION for item in entry['files']
                   if item['rid'] == 'win-x64' and item['name'].endswith('.zip'))
    url = package['url']
    if urlsplit(url).hostname != 'builds.dotnet.microsoft.com':
        raise RuntimeError('Unexpected SDK download host')
    archive = CACHE / ('dotnet-sdk-' + SDK_VERSION + '.zip')
    expected = package['hash'].lower()
    if not archive.is_file() or hashlib.file_digest(archive.open('rb'), 'sha512').hexdigest() != expected:
        temporary = archive.with_suffix('.partial')
        print('Downloading Microsoft SDK to the project cache...', flush=True)
        with OPENER.open(url, timeout=120) as response, temporary.open('wb') as output:
            while data := response.read(1024 * 1024):
                output.write(data)
        with temporary.open('rb') as source:
            if hashlib.file_digest(source, 'sha512').hexdigest() != expected:
                raise RuntimeError('Microsoft SDK SHA512 verification failed')
        temporary.replace(archive)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if not (target / member.filename).resolve().is_relative_to(target.resolve()):
                raise RuntimeError('Unsafe SDK archive path')
        source.extractall(target)
    (CACHE / 'sdk-receipt.json').write_text(json.dumps({'version': SDK_VERSION, 'url': url,
        'sha512': expected, 'verified': True}, indent=2), encoding='utf-8')
    return binary


def icon():
    # Small geometric application icon, generated locally without image dependencies.
    size = 256
    pixels = bytearray()
    for y in range(size):
        pixels.append(0)
        for x in range(size):
            outside = (x < 28 and y < 28 and (x-28)**2+(y-28)**2 > 28**2) or \
                (x > 227 and y < 28 and (x-227)**2+(y-28)**2 > 28**2) or \
                (x < 28 and y > 227 and (x-28)**2+(y-227)**2 > 28**2) or \
                (x > 227 and y > 227 and (x-227)**2+(y-227)**2 > 28**2)
            color = (15, 20, 23, 0 if outside else 255)
            for top, green in ((61, 226), (106, 190), (151, 150)):
                relative = y - top
                if 0 <= relative <= 37 and abs(x - 128) <= 77 - abs(relative - 18):
                    color = (46, green, 169, 255)
            pixels.extend(color)
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 6, 0, 0, 0)) + \
        chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')
    directory = ROOT / 'desktop' / 'Strata.Desktop'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'strata.ico').write_bytes(struct.pack('<HHH', 0, 1, 1) +
        struct.pack('<BBBBHHII', 0, 0, 0, 0, 1, 32, len(png), 22) + png)


def main():
    if os.name != 'nt':
        raise RuntimeError('This desktop build targets Windows x64')
    binary = sdk()
    if '--sdk-only' in sys.argv:
        print('Microsoft SDK ready (SHA512 verified).')
        return
    icon()
    env = dict(os.environ, DOTNET_ROOT=str(binary.parent), DOTNET_CLI_HOME=str(CACHE / 'cli'),
               DOTNET_CLI_TELEMETRY_OPTOUT='1', DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1', DOTNET_GENERATE_ASPNET_CERTIFICATE='false',
               NUGET_PACKAGES=str(CACHE / 'nuget'))
    subprocess.run([str(binary), 'publish', str(ROOT / 'desktop/Strata.Desktop/Strata.Desktop.csproj'),
                    '-c', 'Release', '-r', 'win-x64', '--self-contained', 'true',
                    '-o', str(ROOT / 'dist/Strata')], cwd=ROOT, env=env, check=True)
    print('Built: ' + str(ROOT / 'dist/Strata/Strata.exe'))


if __name__ == '__main__':
    main()
