import gzip, hashlib, json, re, subprocess, urllib.request
from pathlib import Path
root = Path.home() / 'workspace/strata-validation'
packages = {}
for block in (gzip.open(root/'downloads/intel-Packages.gz', 'rt').read() + '\n\n' + gzip.open(root/'downloads/intel-Packages-all.gz', 'rt').read()).split('\n\n'):
    d = dict(line.split(': ', 1) for line in block.splitlines() if ': ' in line and not line.startswith(' '))
    if 'Package' not in d: continue
    name = d['Package']
    if name not in packages or subprocess.run(['dpkg','--compare-versions', d['Version'], 'gt', packages[name]['Version']]).returncode == 0:
        packages[name] = d
selected = {}
external = set()
def add(name):
    if name in selected: return
    if not name.startswith('intel-oneapi-'):
        external.add(name)
        return
    d = packages[name]
    selected[name] = d
    for dep in d.get('Depends','').split(','):
        dep = dep.strip()
        if dep: add(re.split(r'\s|\|', dep)[0])
for name in ('intel-oneapi-compiler-dpcpp-cpp-2026.1', 'intel-oneapi-mkl-devel-2026.1'):
    add(name)
print('External system dependencies:', sorted(external), flush=True)
print('Download GiB:', sum(int(d['Size']) for d in selected.values())/2**30, flush=True)
(root/'logs/intel-package-manifest.json').write_text(json.dumps(list(selected.values()), indent=2))
prefix = root/'toolchains/intel-root'
prefix.mkdir(parents=True, exist_ok=True)
for name,d in selected.items():
    target=root/'downloads'/Path(d['Filename']).name
    print('Installing',name,d['Version'],flush=True)
    if not target.exists(): urllib.request.urlretrieve('https://apt.repos.intel.com/oneapi/'+d['Filename'],target)
    with target.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(stream.read()).hexdigest()
    if digest != d['SHA256']: raise RuntimeError('Checksum mismatch: '+str(target))
    subprocess.run(['dpkg-deb','--extract',str(target),str(prefix)],check=True)
print('Installation complete',flush=True)
