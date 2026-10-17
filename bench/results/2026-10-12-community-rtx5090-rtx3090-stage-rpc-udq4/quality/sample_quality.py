#!/usr/bin/env python3
"""Native metrics-only collector: /api/requests is optional and disabled here."""
import argparse
import json
import time
from pathlib import Path
from run_quality import fetch, command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    errors, count = [], 0
    first = time.time()
    with (args.out / 'telemetry-recovered.jsonl').open('w') as log:
        while True:
            try:
                metrics = fetch(args.url, '/metrics?requests=all')
                starts = list((args.out / 'coding').glob('*/started.json'))
                phase = max(starts, key=lambda p: p.stat().st_mtime).parent.name if starts else 'needles'
                row = {'time': time.time(), 'phase': phase, 'live': metrics.get('live'),
                       'totals': metrics.get('totals'), 'requests': metrics.get('requests'),
                       'worker_process_vram': command(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits']),
                       'nodes': [{k: node.get(k) for k in ('id', 'role', 'layers', 'hardware_static', 'hardware', 'freshness', 'engine_cache')}
                                 for node in metrics.get('nodes', [])]}
                log.write(json.dumps(row) + '\n')
                log.flush()
                count += 1
            except Exception as error:
                errors.append({'time': time.time(), 'error': str(error)})
            if (args.out / 'summary.json').exists() or time.time() - first > 14400:
                break
            time.sleep(2)
    (args.out / 'collector-summary.json').write_text(json.dumps({'started': first, 'finished': time.time(),
        'samples': count, 'errors': errors, 'api_monitor': 'disabled; metrics-only collection',
        'incident': 'Initial collector required optional /api/requests (404); request timings recovered from retained native history, no test replay'}, indent=2) + '\n')
    print('METRICS_COLLECTOR_FINISHED samples=' + str(count) + ' errors=' + str(len(errors)), flush=True)


if __name__ == '__main__':
    main()
