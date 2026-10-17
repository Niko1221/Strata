#!/usr/bin/env python3
"""Offline validation of the measured package. No model, network or GPU access."""
import csv
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), (a, b)

def main():
    rows, summary = read('requests.json'), read('summary.json')
    assert len(rows) == 360
    with (ROOT / 'requests.csv').open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 360
    for original, exported in zip(rows, csv_rows):
        assert all(exported[k] == ('' if v is None else str(v)) for k, v in original.items())
    assert all(r['ok'] and not r['err'] and r['decode_tps_source'] == r['prompt_tps_source'] == 'engine' for r in rows)
    counts = Counter((r['block'], r['arm']) for r in rows)
    assert len(counts) == 8 and set(counts.values()) == {45}
    assert sum(r['workload'] == 'calibration' for r in rows) == summary['calibration_requests'] == 72
    assert sum(r['workload'] != 'calibration' for r in rows) == summary['official_requests'] == 288
    for arm, groups in summary['arms'].items():
        for workload, metrics in groups.items():
            records = [r for r in rows if r['arm'] == arm and r['workload'] == workload]
            for field in ('prompt_tps', 'decode_tps', 'ttft_s', 'prompt_tokens', 'generated_tokens'):
                values = [r[field] for r in records]
                actual = {'n': len(values), 'median': statistics.median(values), 'min': min(values), 'max': max(values)}
                for key, value in actual.items():
                    close(value, metrics[field][key])
            close(sum(r['generated_tokens'] for r in records) / sum(r['decode_ms'] for r in records) * 1000,
                  metrics['weighted_decode_tps'])
            assert sum(r['generated_tokens'] for r in records) == metrics['tokens_generated_total']
            if workload in ('4K', '16K', '32K'):
                assert all(r['reused_tokens'] == 0 for r in records)
                assert {r['prompt_tokens'] for r in records} == {dict(zip(('4K', '16K', '32K'), (4020, 16058, 32049)))[workload]}
        cells = []
        for workload in ('chat', '4K', '16K', '32K', 'agent', 'calibration'):
            field = 'prompt_tps' if workload in ('4K', '16K', '32K') else 'decode_tps'
            stat = groups[workload][field]
            cells.append(f"{stat['median']:.2f} ({stat['min']:.2f}–{stat['max']:.2f})")
        assert '| ' + arm + ' | ' + ' | '.join(cells) + ' |' in (ROOT / 'README.md').read_text()
    pairs = [('dual_vs_single_baseline', 'DUAL-BASE', 'SINGLE-BASE'),
             ('dual_tuning_vs_baseline', 'DUAL-TUNED', 'DUAL-BASE'),
             ('single_tuning_vs_baseline', 'SINGLE-TUNED', 'SINGLE-BASE')]
    for label, candidate, reference in pairs:
        for workload, value in summary['comparisons'][label].items():
            field = 'prompt_tps' if workload in ('4K', '16K', '32K') else 'decode_tps'
            close((summary['arms'][candidate][workload][field]['median'] /
                   summary['arms'][reference][workload][field]['median'] - 1) * 100, value)
    for evidence in summary['evidence']:
        assert evidence['cgroup_events']['high'] > 0
        assert all(evidence['cgroup_events'][k] == 0 for k in ('max', 'oom', 'oom_kill'))
        assert evidence['telemetry_summary']['service_swap_max_bytes'] == 0
        assert evidence['effective_limits_unchanged']
        telemetry = [json.loads(line) for line in gzip.decompress((ROOT / evidence['folder'] / 'telemetry.jsonl.gz').read_bytes()).decode().splitlines()]
        metrics = evidence['telemetry_summary']
        close(min(t['mem_available_bytes'] for t in telemetry) / 2**30, metrics['min_available_gib'])
        close(max(int(t['cgroup']['memory.peak']) for t in telemetry) / 2**30, metrics['peak_service_gib'])
        assert max(t['swap_used_bytes'] for t in telemetry) == metrics['global_swap_max_bytes']
        assert max(int(t['cgroup']['memory.swap.current']) for t in telemetry) == metrics['service_swap_max_bytes']
        for event, expected in evidence['cgroup_events'].items():
            assert max(int(dict(line.split() for line in t['cgroup']['memory.events'].splitlines())[event]) for t in telemetry) == expected
        device = summary['model_mount']['counter_device']
        assert telemetry[-1]['disk_io'][device]['read_bytes'] - telemetry[0]['disk_io'][device]['read_bytes'] == evidence['disk_reads_bytes'][device]
        for probe in read(evidence['folder'] + '/calibration-probes.json'):
            assert hashlib.sha256(probe['response']['text'].encode()).hexdigest() == probe['answer_sha256']
    assert summary['all_effective_limits_unchanged'] and summary['oc_configuration_constant']
    for md in ROOT.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)', md.read_text()):
            if not re.match(r'(https?://|#)', link):
                assert (md.parent / link).exists(), (md.name, link)
    for file in ROOT.rglob('*'):
        if not file.is_file() or '__pycache__' in file.parts:
            continue
        contents = gzip.decompress(file.read_bytes()).decode() if file.suffix == '.gz' else file.read_text()
        # Export must have no actual private home path, UUID, credentials or LAN IP.
        assert not re.search(r'/(?:home|Users)/[^\s<>/]+|GPU-[0-9a-fA-F-]{16,}|sk-[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]{8,}|\b192\.168\.\d+\.\d+\b', contents), file
    checksum = ROOT / 'SHA256SUMS'
    assert checksum.exists(), 'Missing SHA256SUMS'
    for line in checksum.read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    print('PASS: 360 requests / 8 services; pooled and weighted rates; comparisons; token/reuse counts; probe answer hashes; safety/OC receipts; relative links; privacy patterns; package checksums.')

if __name__ == '__main__':
    main()
