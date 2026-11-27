"""Fixed-input window logits regression; runs existing built binary, no downloads."""
import argparse
import hashlib
import itertools
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "terminal"))
from strata_control.hy4_flash import environment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', default='3')
    p.add_argument('--build-dir', type=Path, default=ROOT/'build')
    p.add_argument('--model', type=Path, required=True, help='Target GGUF file or first shard')
    p.add_argument('--pack', type=Path, help='Defaults to pack beside the first GGUF shard')
    p.add_argument('--expert-vram', type=float, default=48.0)
    p.add_argument('--cpu-oracle', action='store_true',
                   help='Compare window and T1 CPU logits with all trunk expert ranks on CPU')
    storage = p.add_mutually_exclusive_group()
    storage.add_argument('--resident-experts', action='store_true')
    storage.add_argument('--compare-storage', action='store_true', help='Compare full-host and split storage with the same T1 reference')
    p.add_argument('--tokens', type=Path, required=True, help='Whitespace-separated token IDs')
    p.add_argument('--prefix', type=int, default=38)
    p.add_argument('--windows', default='1,2,3,4,7,8')
    p.add_argument('--env', action='append', default=[])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    for name in ('hy4_flash_verify_rows', 'libstrata_kq.so'):
        shutil.copy2(a.build_dir/name, a.out/name)
    pack = a.pack or a.model.parent/'pack'
    base={**os.environ, **environment(), 'CUDA_VISIBLE_DEVICES': a.gpu, 'CUDA_MODULE_LOADING':'EAGER',
          'STRATA_HY4_FLASH_CPU_MISS_MAX':'0', 'STRATA_HY4_FLASH_VERIFY_DIRECT_DMA':'1',
          'STRATA_HY4_FLASH_VERIFY_BATCH_DENSE':'1', 'STRATA_HY4_FLASH_VERIFY_HEADS':'1'}
    results={}
    base.update(dict(e.split('=', 1) for e in a.env))
    if a.cpu_oracle:
        # Identical arithmetic assignments are necessary for comparing every logit bit.
        a.expert_vram = 0
        base.update(STRATA_HY4_FLASH_ASYNC_MOE='1', STRATA_HY4_FLASH_CPU_MISS_MAX='16',
                    STRATA_HY4_FLASH_CPU_CACHE_FIRST='0', STRATA_HY4_FLASH_CPU_SPLIT='fixed')
    reference=None
    modes = [False, True] if a.compare_storage else [a.resident_experts]
    widths = [int(value) for value in a.windows.split(',')]
    if not widths or widths[0] != 1 or any(w < 1 or w > 8 for w in widths):
        p.error('--windows must start with 1 and contain widths from 1 to 8')
    for resident, window in itertools.product(modes, widths):
        label = f'T{window}'
        if a.compare_storage:
            label += '-split' if resident else '-full'
        env={**base,'STRATA_HY4_FLASH_WINDOW':'1','STRATA_HY4_FLASH_WINDOW_GRAPH':'1'}
        if window == 1:
            env.update(STRATA_HY4_FLASH_WINDOW='0', STRATA_HY4_FLASH_ASYNC_MOE='1' if a.cpu_oracle else '0',
                       STRATA_HY4_FLASH_GRAPHS='0', STRATA_HY4_FLASH_TOKEN_GRAPH='0')
        output=a.out/f'{label}.f32'
        cmd=[str((a.out/'hy4_flash_verify_rows').resolve()), str(a.model.resolve()),
             str(pack.resolve()),str(a.tokens.resolve()),str(output.resolve()),str(a.prefix),str(window),
             str(a.expert_vram)]
        if resident:
            cmd.append('--resident-experts')
        with (a.out/f'{label}.log').open('w') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        if a.cpu_oracle:
            log = (a.out / f'{label}.log').read_text()
            if window == 1:
                assert re.search(r'HY4_FLASH_CPU_STATS experts=[1-9]\d*', log), 'T1 reference did not run CPU experts'
            else:
                cpu = re.search(r'HY4_FLASH_WINDOW_CPU_STATS .*? entries=(\d+)', log)
                total = re.search(r'HY4_FLASH_WINDOW_STATS .*? entries=(\d+).*? transfers=(\d+)', log)
                assert cpu and total and int(cpu[1]) > 0, 'window did not run CPU experts'
                assert cpu[1] == total[1] and int(total[2]) == 0, 'target expert assignments differ'
        digest=hashlib.sha256(output.read_bytes()).hexdigest()
        if reference is None: reference=digest
        results[label]={'sha256':digest,'bytes':output.stat().st_size,'exact':digest==reference}
        (a.out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
        print(label,results[label],flush=True)
        assert digest==reference, f'{label} logits differ'


if __name__=='__main__': main()
