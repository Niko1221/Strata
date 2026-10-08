"""Render the completed sweep cells without treating failed capacity as supported."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    data = json.loads((a.directory / 'matrix.json').read_text())
    print('| Context | Engine | Status | Min available GiB | 172-token decode/s | 557-token decode/s | 1069-token decode/s | Near-limit prompt tokens / prompt/s / decode/s |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for cell in sorted(data['cells'], key=lambda c: (c['context_limit'], c['label'])):
        rates = []
        for repeat in (32, 128, 256):
            rows = [r['response']['timings']['predicted_per_second']
                    for r in cell['requests'] if r['round'] < 99 and r['pattern_repeats'] == repeat]
            rates.append(f'{statistics.median(rows):.1f} ({min(rows):.1f}–{max(rows):.1f})' if rows else '—')
        near = [r for r in cell['requests'] if r['round'] == 99]
        near_text = '—'
        if near:
            response = near[0]['response']; t = response['timings']
            near_text = f"{response['usage']['prompt_tokens']} / {t['prompt_per_second']:.1f} / {t['predicted_per_second']:.1f}"
        memory = cell.get('memory', {}).get('minimum_available_bytes')
        available = f'{memory/2**30:.2f}' if memory is not None else '—'
        status = cell['status']
        if cell.get('headroom_preserved') is False:
            status += '; below headroom'
        print('| ' + ' | '.join([str(cell['context_limit']), cell['label'], status, available, *rates, near_text]) + ' |')
        if cell.get('error'):
            print(f"\n{cell['label']} {cell['context_limit']}: `{cell['error']}`\n")


    print('\nMatched prompt throughput: median (minimum–maximum), three runs per size.\n')
    print('| Context | Engine | 172-token prompt/s | 557-token prompt/s | 1069-token prompt/s |')
    print('| --- | --- | --- | --- | --- |')
    for cell in sorted(data['cells'], key=lambda c: (c['context_limit'], c['label'])):
        rates = []
        for repeat in (32, 128, 256):
            rows = [r['response']['timings']['prompt_per_second']
                    for r in cell['requests'] if r['round'] < 99 and r['pattern_repeats'] == repeat]
            rates.append(f'{statistics.median(rows):.1f} ({min(rows):.1f}–{max(rows):.1f})' if rows else '—')
        print('| ' + ' | '.join([str(cell['context_limit']), cell['label'], *rates]) + ' |')

    print('\nPhysical memory and expert cache allocation (samples include startup):\n')
    print('| Context | Engine | Expert cache GiB / slots | Max swap GiB | Host disk reads GiB |')
    print('| --- | --- | --- | --- | --- |')
    import re
    for cell in sorted(data['cells'], key=lambda c: (c['context_limit'], c['label'])):
        directory = a.directory / cell['label'] / str(cell['context_limit'])
        log = (directory / 'engine.txt').read_text()
        cache = re.search(r'expert cache (\d+) slots, ([\d.]+) GiB', log)
        cache_text = f'{cache[2]} / {cache[1]}' if cache else '—'
        memory = cell.get('memory', {})
        swap = memory.get('maximum_swap_used_bytes', 0) / 2**30
        disk = memory.get('host_disk_read_bytes', 0) / 2**30
        print('| ' + ' | '.join([str(cell['context_limit']), cell['label'], cache_text,
                                f'{swap:.2f}', f'{disk:.2f}']) + ' |')


if __name__ == '__main__':
    main()
