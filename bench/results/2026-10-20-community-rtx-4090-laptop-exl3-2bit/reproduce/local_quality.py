#!/usr/bin/env python3
"""Run Strata's fixed quality yardstick while retaining the tested configuration.

The stock quant_quality runner replaces KV precision and turns GPU/PCIe experts
off. This adapter deliberately keeps them; only teacher-forcing and cache/suffix
controls are identical for both sides. KL is the stock top-64 approximation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import quant_quality as q

def args_for_quality(cfg, _gpu, seg_tokens, prompt=False):
    args = list(cfg['args'])
    # Force the same decode-window scoring path on both configurations.
    controls = [('--prompt-cache', '0'), ('--suffix-draft', '0')]
    if not prompt:
        controls.append(('--short-read', str(seg_tokens + 64)))
    for flag, value in controls:
        if flag in args:
            i = args.index(flag); args[i + 1] = value
        else:
            args += [flag, value]
    return [cfg['exe'], '--serve', *args]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ids', default=str(q.SET_DIR / 'set.ids.json'))
    p.add_argument('--env', action='append', default=[])
    p.add_argument('--min-free-gb', type=float, default=4)
    p.add_argument('--prompt', action='store_true', help='Stock 20-prompt full-vocabulary logits check, preserving configured prefill path')
    a = p.parse_args(); a.wrap = None; a.gpu_experts = False
    if any(proc.info['name']=='strata' for proc in psutil.process_iter(['name'])):
        raise SystemExit('Another Strata engine is running; stop it before quality checks.')
    q.engine_args = lambda c,g,n: args_for_quality(c,g,n,prompt=a.prompt)
    q.FIXED_ENV = {'PYTHONDONTWRITEBYTECODE': '1'}
    a.keep_config = True
    rc = q.cmd_prompt(a) if a.prompt else q.cmd_run(a)
    cfg = json.loads(Path(a.config).read_text())
    meta = Path(a.out)/'index.json' if a.prompt else Path(a.out + '.seg.json')
    x = json.loads(meta.read_text())
    x.update(config_snapshot=cfg, ids_sha256=hashlib.sha256(Path(a.ids).read_bytes()).hexdigest(),
             adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             metric_note=('20 prompt prefixes, full-vocabulary first-token logits; baseline-relative, not BF16' if a.prompt else
                          'Teacher-forced fixed set, top-64-renormalized approximate KL; baseline-relative, not BF16'))
    meta.write_text(json.dumps(x, indent=2) + '\n')
    return rc

if __name__ == '__main__':
    raise SystemExit(main())
