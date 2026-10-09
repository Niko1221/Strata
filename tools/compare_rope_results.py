"""Compare saved migration measurements locally, without inference or quality gates.

python tools/compare_rope_results.py --output report.html results1.json results2.json
Raw input files are never changed. Identical task answers are displayed once.
"""
import argparse
import difflib
import hashlib
import html
import json
from pathlib import Path

TASKS = ('ordinary-task', 'A-fresh-yarn', 'C-exact-yarn-replay',
         'B-migrated-yarn', 'D-ordinary-large-capacity')


def compare_file(path):
    raw = path.read_bytes()
    rows = json.loads(raw)
    tasks = {r['label']: r for r in rows if r.get('label') in TASKS and r['kind'] == 'generate'}
    groups = {}
    for label, row in tasks.items():
        groups.setdefault(row['text'], []).append(label)
    baseline = tasks.get('ordinary-task')
    differences = []
    if baseline:
        for label, row in tasks.items():
            if row['text'] != baseline['text']:
                differences.append(dict(label=label, diff='\n'.join(difflib.unified_diff(
                    baseline['text'].splitlines(), row['text'].splitlines(),
                    fromfile='ordinary-task', tofile=label, lineterm=''))))
    qualities = [r for r in rows if r['kind'] == 'quality']
    return dict(source=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                complete=any(r['kind'] == 'complete' for r in rows),
                input_tokens=sorted({r['input_tokens'] for r in tasks.values()}),
                same_input_for_task_paths=len({r['input_sha256'] for r in tasks.values()}) == 1 if tasks else None,
                missing_task_paths=[label for label in TASKS if label not in tasks],
                output_groups=[dict(text=text, paths=paths) for text, paths in groups.items()],
                differences=differences, quality=qualities,
                timings=[{k: v for k, v in r.items() if k not in ('text','correct')}
                         for r in rows if r['kind'] == 'B-migration' or
                         r['kind'] == 'generate' and (r.get('label') in TASKS or r.get('label','').endswith('-prefix'))],
                failures=[r for r in rows if r['kind'] == 'failure'])


def render(cases):
    e = html.escape
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>RoPE measurements</title>',
             '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#17212d}',
             'table{border-collapse:collapse;width:100%;margin:16px 0}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}',
             'pre{white-space:pre-wrap;background:#f1f4f8;padding:16px}small{color:#526173}section{margin:40px 0}</style>',
             '<h1>RoPE / YaRN: measured differences</h1>',
             '<p>No automatic quality verdict. These are three-code retrieval probes with repeated filler,',
             'not broad coding or conversation evaluations. Different hardware and profile settings must remain separate.</p>',
             '<p>A: fresh YaRN; B: ordinary cache migrated to YaRN; C: exact YaRN replay;',
             ' D: ordinary RoPE with a larger allocation. Allocation size is not tested sequence length.</p>']
    for case in cases:
        parts += ['<section><h2>'+e(Path(case['source']).name)+'</h2>',
                  f'<p>Actual task input: {e(str(case["input_tokens"]))} tokens. Run complete: {case["complete"]}.',
                  f' Same input hash across task paths: {case["same_input_for_task_paths"]}.</p>']
        if case['missing_task_paths']:
            parts.append('<p>Unmeasured paths: '+e(', '.join(case['missing_task_paths']))+'</p>')
        for group in case['output_groups']:
            parts += ['<p>Exact same output: '+e(', '.join(group['paths']))+'</p>', '<pre>'+e(group['text'])+'</pre>']
        for diff in case['differences']:
            parts.append('<pre>'+e(diff['diff'])+'</pre>')
        parts.append('<table><tr><th>Comparison</th><th>Positions</th><th>Mean KL</th><th>Max KL</th><th>Top-token agreement</th><th>Target-text PPL change</th></tr>')
        for q in case['quality']:
            parts.append(f'<tr><td>{e(q["reference"])} → {e(q["candidate"])}</td><td>{q["rows"]}</td>'
                         f'<td>{q["mean_kl"]:.6f}</td><td>{q["max_kl"]:.6f}</td>'
                         f'<td>{q["top_agreement"]:.2%}</td><td>{(q["perplexity_ratio"]-1)*100:+.2f}%</td></tr>')
        parts.append('</table><p>KL measures distribution differences; zero means identical. PPL change concerns this fixed continuation only.</p>')
        parts.append('<details><summary>Per-position measurements, timings and failures</summary><pre>'+e(json.dumps(
            {k:case[k] for k in ('quality','timings','failures')},indent=2))+'</pre></details>')
        parts.append('<small>Source SHA-256: '+case['sha256']+'</small></section>')
    return ''.join(parts)+'</html>'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('results', nargs='+', type=Path)
    a = p.parse_args()
    if any(a.output.resolve() == x.resolve() or a.output.with_suffix('.json').resolve() == x.resolve() for x in a.results):
        p.error('report must not overwrite an input result')
    cases = [compare_file(path) for path in a.results]
    a.output.write_text(render(cases), encoding='utf-8')
    a.output.with_suffix('.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
    for c in cases:
        print(Path(c['source']).name, 'tokens=',c['input_tokens'], 'unique answers=',len(c['output_groups']),
              'same inputs=',c['same_input_for_task_paths'], 'missing=',c['missing_task_paths'])
    print(a.output.resolve())


if __name__ == '__main__':
    main()
