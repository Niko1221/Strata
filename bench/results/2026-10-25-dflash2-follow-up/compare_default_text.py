"""Compare SM89 machine sections, ignoring the added opt-in template argument.

Supply the two preserved baseline executables and the candidate. This performs
static inspection only; it does not execute GPU code or establish other-backend
compatibility. CUDA cuobjdump and c++filt must be installed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline-native', type=Path, required=True)
p.add_argument('--baseline-topk', type=Path, required=True)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cuobjdump', default='/usr/local/cuda/bin/cuobjdump')
a = p.parse_args()


def sections(folder, executable, kind, normalize):
    subprocess.run([a.cuobjdump, '--extract-text', kind, str(executable.resolve())],
                   cwd=folder, check=True, stdout=subprocess.DEVNULL)
    files = sorted(f for f in folder.glob('*.sm_89.elf.bin') if kind in f.name)
    assert files, f'no SM89 sections: {kind}'
    names = [f.name[2:].removesuffix('.sm_89.elf.bin') for f in files]
    demangled = subprocess.check_output(['c++filt', '-n'],
        input='\n'.join(names) + '\n', text=True).splitlines()
    result = {}
    for file, name in zip(files, demangled):
        name = name.removeprefix('void ')
        if normalize and kind == 'native_mmvq_multi_kernel':
            end = name.find('>(')
            if end < 0 or not name[:end].endswith(', false'):
                continue  # new opt-in specialization
            name = name[:end - 7] + name[end:]
        elif normalize:
            if '<false>' not in name:
                continue
            name = name.replace('topk_parallel_kernel<false>', 'topk_parallel_kernel', 1)
        assert name not in result, name
        result[name] = file.read_bytes()
    return result


report = {'architecture': 'sm_89', 'executables': {}, 'kernels': {}}
for label, exe in [('baseline_native', a.baseline_native),
                   ('baseline_topk', a.baseline_topk), ('candidate', a.candidate)]:
    report['executables'][label] = {'name': exe.name,
        'sha256': hashlib.sha256(exe.read_bytes()).hexdigest()}
with tempfile.TemporaryDirectory(prefix='dflash-default-text-') as temporary:
    for kind, baseline in [('native_mmvq_multi_kernel', a.baseline_native),
                           ('topk_parallel_kernel', a.baseline_topk)]:
        before_dir = Path(temporary) / (kind + '-before')
        after_dir = Path(temporary) / (kind + '-after')
        before_dir.mkdir(); after_dir.mkdir()
        before = sections(before_dir, baseline, kind, False)
        after = sections(after_dir, a.candidate, kind, True)
        missing = [name for name in before if name not in after]
        different = [name for name in before if name in after and before[name] != after[name]]
        report['kernels'][kind] = {'baseline_functions': len(before),
            'candidate_functions': len(after), 'missing': missing, 'different': different,
            'identical_functions': len(before) - len(missing) - len(different),
            'identical_bytes': sum(len(code) for name, code in before.items()
                                   if name in after and code == after[name])}
a.output.write_text(json.dumps(report, indent=2) + '\n')
assert all(not value['missing'] and not value['different']
           for value in report['kernels'].values()), 'default machine sections differ'
print('PASS:', {kind: value['identical_functions'] for kind, value in report['kernels'].items()})
