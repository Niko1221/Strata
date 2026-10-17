"""Run a cache-latency campaign serially and preserve progress across SSH disconnects.

The campaign JSON lists exact source commits, source paths, base configs and modes.
Launch under a detached supervisor (systemd-run/tmux); no inference runs in parallel.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--samples-per-block', type=int, default=20)
    ap.add_argument('--blocks', type=int, default=10)
    args = ap.parse_args()
    if args.blocks < 1 or args.samples_per_block < 1:
        ap.error('sample counts must be positive')
    cfg = json.loads(args.config.read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(config=cfg, blocks=args.blocks, samples_per_block=args.samples_per_block,
                    started=time.time(), python=sys.version, results=[])
    for cmd, name in [(['nvidia-smi', '-q'], 'gpu.txt'), (['lscpu'], 'cpu.txt'),
                      (['free', '-b'], 'ram.txt'), (['lsblk', '-o', 'NAME,MODEL,SIZE,ROTA,MOUNTPOINTS'], 'disk.txt')]:
        try:
            (out / name).write_text(subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT))
        except (OSError, subprocess.CalledProcessError) as exc:
            (out / name).write_text(str(exc))
    jobs = [(build, mode) for build in cfg['builds'] for mode in build['modes']]
    manifest_path = out / 'campaign.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['config'] != cfg or old['blocks'] != args.blocks or old['samples_per_block'] != args.samples_per_block:
            raise RuntimeError('Cannot resume with different campaign settings')
        manifest = old

    def save():
        temp = manifest_path.with_suffix('.tmp')
        temp.write_text(json.dumps(manifest, indent=2))
        os.replace(temp, manifest_path)

    for block in range(args.blocks):
        # Reverse mode/build order on alternate blocks to reduce simple thermal/time drift.
        for build, mode in (jobs if block % 2 == 0 else list(reversed(jobs))):
            name = f"block-{block:02d}-{build['label']}-{mode}"
            if any(r['name'] == name for r in manifest['results']):
                continue
            folder = out / name
            if folder.exists():
                # An interrupted unit's evidence is kept; a clean restart gets a new directory.
                folder = out / (name + '-retry-' + str(int(time.time())))
            cmd = [sys.executable, str(Path(__file__).with_name('cache_latency.py')),
                   '--source', build['source'], '--config', build['config'], '--output', str(folder),
                   '--source-commit', build['commit'], '--build-label', build['label'], '--mode', mode,
                   '--block', str(block), '--samples', str(args.samples_per_block),
                   '--warmups', str(cfg.get('warmups', 3)),
                   '--trial-start', str(block * args.samples_per_block), '--targets', cfg['targets']]
            manifest['active'] = name
            save()
            begin = time.time()
            print('START', name, flush=True)
            with (out / (name + '.log')).open('a') as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            manifest['results'].append(dict(name=name, output=str(folder), exit_code=result.returncode,
                                            elapsed_s=time.time()-begin))
            save()
            print('FINISH', name, result.returncode, round(time.time()-begin, 1), flush=True)
            # Only generated test snapshots in this completed job may be reclaimed.
            for path in (folder / 'kv').rglob('*'):
                if path.is_file() and path.suffix in ('.sess', '.meta'):
                    if not path.resolve().is_relative_to(folder.resolve()):
                        raise RuntimeError('Cache path escapes the job directory')
                    path.unlink()
            if result.returncode:
                manifest['failed'] = name
                save()
                return result.returncode
    manifest.update(active=None, finished=time.time())
    save()
    print('CAMPAIGN_COMPLETE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
