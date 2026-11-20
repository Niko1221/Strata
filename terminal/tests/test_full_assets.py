import importlib.util
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET
import zlib

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('full_assets', ROOT/'packaging/full_assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def test_sprite_covers_every_ui_icon_without_reusing_supplied_artwork():
    ui = (ROOT/'apps/host/src/components/ui.tsx').read_text(encoding='utf-8')
    definition = ui.split('export type IconName =', 1)[1].split(';', 1)[0]
    expected = set(re.findall(r"'([a-z0-9-]+)'", definition))
    assert set(assets.ICONS) == expected


@pytest.mark.parametrize('svg', [
    b'<!DOCTYPE svg><svg viewBox="0 0 24 24"/>',
    b'<svg viewBox="0 0 24 24" onload="alert(1)"/>',
    b'<svg viewBox="0 0 24 24"><script/></svg>',
    b'<svg viewBox="0 0 24 24"><image href="https://example.com"/></svg>',
    b'<svg viewBox="0 0 24 24"><path fill="url(https://example.com)"/></svg>',
])
def test_active_or_external_svg_is_rejected(svg):
    with pytest.raises(ValueError):
        assets.symbol('test', svg)


def test_symbol_retains_shape_but_uses_safe_inherited_presentation():
    node = assets.symbol('test', b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M1 2h3"/></svg>')
    assert node.attrib['id'] == 'i-test'
    assert node.attrib['stroke'] == 'currentColor'
    assert node[0].tag == 'path'
    assert node[0].attrib == {'d': 'M1 2h3'}


def test_owned_png_and_ico_are_complete_and_resolve_brand_requests(tmp_path):
    assets.write_owned(tmp_path)
    raw = (tmp_path/'icon.png').read_bytes()
    assert raw[:8] == b'\x89PNG\r\n\x1a\n'
    pos = 8
    compressed = bytearray()
    while pos < len(raw):
        size, kind = struct.unpack('>I4s', raw[pos:pos+8])
        payload = raw[pos+8:pos+8+size]
        crc = struct.unpack('>I', raw[pos+8+size:pos+12+size])[0]
        assert zlib.crc32(kind+payload) & 0xffffffff == crc
        if kind == b'IHDR':
            assert struct.unpack('>II', payload[:8]) == (256, 256)
        if kind == b'IDAT':
            compressed.extend(payload)
        pos += 12+size
    assert len(zlib.decompress(compressed)) == 256*(1+256*4)
    ico = (tmp_path/'icon.ico').read_bytes()
    assert struct.unpack('<HHH', ico[:6]) == (0, 1, 6)
    for index in range(6):
        entry = struct.unpack('<BBBBHHII', ico[6+16*index:22+16*index])
        assert ico[entry[-1]:entry[-1]+entry[-2]].startswith(b'\x89PNG')
    for theme in ['light', 'dark']:
        for variant in ['mark', 'lockup']:
            ET.parse(tmp_path/'brand'/f'strata-{variant}-{theme}.svg')
