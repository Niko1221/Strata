"""Recreate the exact measured profile (32 original-ranked experts per layer)."""
import hashlib,json,struct
from pathlib import Path
root=Path(__file__).resolve().parent
r=json.loads((root/'profile.json').read_text())
pairs=r['pairs']; assert len(pairs)==1536 and len(set(map(tuple,pairs)))==1536
assert [sum(l==i for l,e in pairs) for i in range(48)]==[32]*48
rank=[-1]*(48*512)
for i,(l,e) in enumerate(pairs):rank[l*512+e]=i
b=b'STRP'+struct.pack('<5I',1,48,512,len(pairs),len(pairs))
b+=b''.join(struct.pack('<HH',*p) for p in pairs)
b+=struct.pack('<24576i',*rank)
assert hashlib.sha256(b).hexdigest()==r['sha256']
Path('matched-expert-profile.bin').write_bytes(b)
print(r['sha256'])
