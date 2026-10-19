"""Summarize raw results, including every output mismatch; no performance filtering."""
import json
import math
import re
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
baseline = json.loads((OUT / 'ar.json').read_text())
reference = {r['task']: r['tokens'] for r in baseline['rows']}
summary = {}
for tag in ('ar', 'mtp', 'bf16', 'Q4_0', 'Q8_0', 'Q5_0', 'mtp-final'):
    path = OUT / f'{tag}.json'
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    log = (OUT / f'{tag}.log').read_text()
    match = re.search(r'dflash: ([\d.]+) MiB of VRAM held', log)
    if match is None:
        match = re.search(r'dflash mtp: draft layer loaded, ([\d.]+) MiB of VRAM', log)
    s = dict(info=data['info'], draft_vram_mib=float(match[1]) if match else None, tasks={},
             measured_requests=len(data['rows']),
             expert_slots=int(data['info']['expert_slots']),
             max_gpu_used_mib=max(int(v['gpu'].split(',')[1]) for v in data['vram']),
             min_gpu_free_mib=min(int(v['gpu'].split(',')[2]) for v in data['vram']),
             quality_controls=[{k: c[k] for k in ('task', 'matches_speculative', 'matches_auto_ar')}
                               for c in data.get('quality_controls', [])])
    for task in reference:
        rows = [r for r in data['rows'] if r['task'] == task]
        if not rows:
            continue
        mismatches = []
        for r in rows:
            if r['tokens'] != reference[task]:
                first = next((i for i, (x, y) in enumerate(zip(r['tokens'], reference[task])) if x != y),
                             min(len(r['tokens']), len(reference[task])))
                mismatches.append(dict(repeat=r['repeat'], first_token=first,
                                       actual=r['tokens'][first:first+4], target=reference[task][first:first+4]))
        offered = sum(r['offered'] for r in rows)
        s['tasks'][task] = dict(median_tok_s=statistics.median(r['tok_s'] for r in rows),
                               min_tok_s=min(r['tok_s'] for r in rows),
                               max_tok_s=max(r['tok_s'] for r in rows),
                               median_wall_tok_s=statistics.median(len(r['tokens'])*1000/r['wall_ms'] for r in rows),
                               accepted=sum(r['accepted'] for r in rows), offered=offered,
                               acceptance=sum(r['accepted'] for r in rows)/offered if offered else None,
                               matching_tokens=sum(len(r['tokens']) for r in rows if r['tokens'] == reference[task]),
                               compared_tokens=sum(len(r['tokens']) for r in rows), mismatches=mismatches)
    if len(s['tasks']) == len(reference):
        s['geomean_tok_s'] = math.exp(statistics.mean(math.log(t['median_tok_s']) for t in s['tasks'].values()))
    summary[tag] = s
if all(tag in summary and 'geomean_tok_s' in summary[tag] for tag in ('mtp', 'mtp-final')):
    combined = [r for tag in ('mtp', 'mtp-final')
                for r in json.loads((OUT / f'{tag}.json').read_text())['rows']]
    pooled = dict(tasks={}, expert_slots_range=[summary[tag]['expert_slots'] for tag in ('mtp', 'mtp-final')],
                  measured_requests=len(combined), draft_vram_mib=None,
                  end_vs_start_pct=100*(summary['mtp-final']['geomean_tok_s']/summary['mtp']['geomean_tok_s']-1))
    for task in reference:
        rows = [r for r in combined if r['task'] == task]
        pooled['tasks'][task] = dict(median_tok_s=statistics.median(r['tok_s'] for r in rows),
                                    min_tok_s=min(r['tok_s'] for r in rows),
                                    max_tok_s=max(r['tok_s'] for r in rows),
                                    median_wall_tok_s=statistics.median(len(r['tokens'])*1000/r['wall_ms'] for r in rows),
                                    mismatches=[dict(repeat=r['repeat'], first_token=next(
                                        (i for i, (x, y) in enumerate(zip(r['tokens'], reference[task])) if x != y),
                                        min(len(r['tokens']), len(reference[task]))))
                                        for r in rows if r['tokens'] != reference[task]],
                                    matching_tokens=sum(len(r['tokens']) for r in rows if r['tokens'] == reference[task]),
                                    compared_tokens=sum(len(r['tokens']) for r in rows))
    pooled['geomean_tok_s'] = math.exp(statistics.mean(math.log(t['median_tok_s']) for t in pooled['tasks'].values()))
    summary['mtp-pooled'] = pooled
mtp = summary.get('mtp-pooled', summary.get('mtp', {}))
for tag, s in summary.items():
    if 'geomean_tok_s' in s and 'geomean_tok_s' in mtp:
        s['vs_mtp_pct'] = 100*(s['geomean_tok_s']/mtp['geomean_tok_s']-1)
    if 'geomean_tok_s' in s and 'bf16' in summary and 'geomean_tok_s' in summary['bf16']:
        s['vs_bf16_pct'] = 100*(s['geomean_tok_s']/summary['bf16']['geomean_tok_s']-1)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
for tag, s in summary.items():
    print(tag, 'VRAM', s['draft_vram_mib'], 'tasks',
          {task: round(t['median_tok_s'], 3) for task, t in s['tasks'].items()},
          'geo', round(s.get('geomean_tok_s', 0), 3),
          'mismatches', sum(len(t['mismatches']) for t in s['tasks'].values()))
