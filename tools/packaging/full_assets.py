"""Owned shell artwork and commit-pinned ISC Lucide symbols for public packages."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import xml.etree.ElementTree as ET
import zlib

LUCIDE_COMMIT = '03f11e2d6adc319ea21e9a1d5820a99cd3767257'
LUCIDE = 'https://raw.githubusercontent.com/lucide-icons/lucide/' + LUCIDE_COMMIT + '/'
ICONS = {
    'pause':'pause','play':'play','activity':'activity','plus':'plus','shield':'shield-check',
    'computer':'monitor','panel':'panel-left','search':'search','sliders':'sliders-horizontal',
    'folder':'folder','folder-plus':'folder-plus','file':'file','ellipsis':'ellipsis',
    'pencil':'pencil','pin':'pin','x':'x','attach':'paperclip','bolt':'zap','branch':'git-branch',
    'chat':'message-circle','check':'check','chevron':'chevron-down','clock':'clock','copy':'copy',
    'cpu':'cpu','disk':'hard-drive','download':'download','error':'circle-x','experts':'network',
    'gauge':'gauge','gpu':'circuit-board','image':'image','info':'info','expand':'maximize',
    'contract':'minimize','layers':'layers','models':'boxes','skills':'sparkles','engine':'cog',
    'family-qwen':'hexagon','model-file':'file-code','link':'link','memory':'brain','mic':'mic',
    'moon':'moon','new-chat':'message-circle-plus','refresh':'refresh-cw','send':'send',
    'settings':'settings','stop':'square','sun':'sun','thermometer':'thermometer',
    'thinking':'ellipsis','tool':'wrench','trash':'trash','warning':'triangle-alert',
}
ALLOWED_TAGS = {'svg','path','line','polyline','polygon','rect','circle','ellipse','g'}
ALLOWED_ATTRS = {'xmlns','width','height','viewBox','fill','stroke','stroke-width',
                 'stroke-linecap','stroke-linejoin','d','x','y','x1','x2','y1','y2',
                 'cx','cy','r','rx','ry','points','class'}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(name: str) -> tuple[str, bytes]:
    with urllib.request.urlopen(LUCIDE+name, timeout=30) as response:
        data=response.read(128*1024+1)
    if len(data)>128*1024:
        raise ValueError('asset exceeds bounded size')
    return name,data


def symbol(name: str, raw: bytes) -> ET.Element:
    if b'<!' in raw:
        raise ValueError('SVG declarations are forbidden')
    root=ET.fromstring(raw)
    for node in root.iter():
        node.tag=node.tag.rsplit('}',1)[-1]
        if node.tag not in ALLOWED_TAGS or set(node.attrib)-ALLOWED_ATTRS:
            raise ValueError('SVG has active or unsupported content')
        if any('url(' in value or '://' in value for key,value in node.attrib.items() if key!='xmlns'):
            raise ValueError('SVG external reference')
    if root.tag!='svg' or root.get('viewBox')!='0 0 24 24':
        raise ValueError('unexpected icon coordinate system')
    out=ET.Element('symbol',{'id':'i-'+name,'viewBox':'0 0 24 24',
        'fill':'none','stroke':'currentColor','stroke-width':'2',
        'stroke-linecap':'round','stroke-linejoin':'round'})
    out.extend(root)
    return out


def mark_rectangles(size: int):
    """An original square-grid letter S, independent of supplied brand artwork."""
    k=size/16
    return [(int(x*k),int(y*k),int(w*k),int(h*k)) for x,y,w,h in
            [(4,3,8,2),(4,5,2,2),(4,7,8,2),(10,9,2,2),(4,11,8,2)]]


def png(size: int) -> bytes:
    rows=[]
    rectangles=mark_rectangles(size)
    for y in range(size):
        row=bytearray()
        for x in range(size):
            foreground=any(a<=x<a+w and b<=y<b+h for a,b,w,h in rectangles)
            row.extend((248,250,252,255) if foreground else (20,48,66,255))
        rows.append(b'\0'+row)
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',size,size,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(rows)))+chunk(b'IEND',b'')


def mark_svg() -> str:
    paths=''.join(f'<rect x="{x}" y="{y}" width="{w}" height="{h}"/>' for x,y,w,h in mark_rectangles(16))
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect width="16" height="16" rx="2" fill="#143042"/><g fill="#f8fafc">'+paths+'</g></svg>'


def write_owned(root: Path) -> None:
    root.mkdir(parents=True,exist_ok=True)
    (root/'icon.png').write_bytes(png(256))
    sizes=[16,32,48,64,128,256]
    offset=6+16*len(sizes); entries=[]; blobs=[]
    for size in sizes:
        raw=png(size)
        entries.append(struct.pack('<BBBBHHII',size%256,size%256,0,0,1,32,len(raw),offset))
        blobs.append(raw);offset+=len(raw)
    (root/'icon.ico').write_bytes(struct.pack('<HHH',0,1,len(sizes))+b''.join(entries+blobs))
    (root/'favicon.svg').write_text(mark_svg(),encoding='utf-8')
    for size in [16,32,48]:
        for prefix in ['favicon-','favicon-dark-']:(root/f'{prefix}{size}.png').write_bytes(png(size))
    for name,size in [('apple-touch-icon.png',180),('icon-192.png',192),('icon-512.png',512)]:
        (root/name).write_bytes(png(size))
    brand=root/'brand';brand.mkdir(exist_ok=True)
    for theme in ['light','dark']:
        (brand/f'strata-mark-{theme}.svg').write_text(mark_svg(),encoding='utf-8')
        colour='#f8fafc' if theme=='dark' else '#143042'
        lockup=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 182 48"><text x="1" y="35" fill="{colour}" font-family="system-ui,Segoe UI,sans-serif" font-weight="700" font-size="36">Strata</text></svg>'
        (brand/f'strata-lockup-{theme}.svg').write_text(lockup,encoding='utf-8')
    (root/'site.webmanifest').write_text(json.dumps({'name':'Strata','short_name':'Strata','display':'standalone','start_url':'/','icons':[{'src':'/icon-192.png','sizes':'192x192','type':'image/png'},{'src':'/icon-512.png','sizes':'512x512','type':'image/png'}]}),encoding='utf-8')


def prepare(output: Path, sources: Path) -> dict:
    if output.exists() or sources.exists():
        raise ValueError('asset outputs must be fresh')
    sources.mkdir(parents=True)
    names=['LICENSE']+['icons/'+name+'.svg' for name in sorted(set(ICONS.values()))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloaded=dict(pool.map(fetch,names))
    license_text=downloaded['LICENSE'].decode()
    if 'ISC' not in license_text or 'Permission' not in license_text:
        raise ValueError('unexpected Lucide license')
    for name,data in downloaded.items():
        path=sources/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    write_owned(output)
    sprite=ET.Element('svg',{'xmlns':'http://www.w3.org/2000/svg'})
    for name,upstream in ICONS.items():sprite.append(symbol(name,downloaded['icons/'+upstream+'.svg']))
    icons=output/'icons';icons.mkdir()
    ET.ElementTree(sprite).write(icons/'sprite.svg',encoding='utf-8',xml_declaration=False)
    license_dir=output/'licenses';license_dir.mkdir()
    (license_dir/'Lucide-LICENSE.txt').write_bytes(downloaded['LICENSE'])
    (license_dir/'OWNED-ASSETS.md').write_text('The square-grid S icon and system-font Strata lockup were authored for this full package. Distributed under the repository MIT license. No supplied UI ZIP artwork or font files are used. Lucide symbols are pinned to '+LUCIDE_COMMIT+' and carry their ISC license.\n',encoding='utf-8')
    receipt={'lucide_commit':LUCIDE_COMMIT,'inputs':{name:digest(data) for name,data in downloaded.items()},'icons':len(ICONS),'owned_assets':'square-grid S and system-font wordmark; MIT','files':{p.relative_to(output).as_posix():digest(p.read_bytes()) for p in output.rglob('*') if p.is_file()}}
    (sources/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.output,a.sources),indent=2))
