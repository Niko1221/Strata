"""Freeze compact fresh benchmark evidence; retain hashes of bulky local reports."""
from pathlib import Path
import hashlib
import json
import re
import statistics

root=Path(__file__).resolve().parents[1]
raw=root/'build-validation'; out=root/'docs/evidence/v0404-performance'
out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p95=lambda xs:sorted(xs)[round(.95*(len(xs)-1))] if xs else None
rows=json.loads((raw/'summary.json').read_text())
assert len(rows)==12, 'Finish all six pairs first'
for row in rows:
    report=raw/(row['name']+'.json'); r=json.loads(report.read_text())
    log=(raw/(row['name']+'.stderr.log')).read_text()
    gaps=[b-a for t in r['turns'] for a,b in zip(t['times_s'],t['times_s'][1:])]
    ttft=[t['times_s'][0]-t['requested_s'] for t in r['turns'] if t['times_s']]
    row.update(report_sha256=sha(report),binary_sha256=r['exe_sha256'],
        actual_experts=int(re.search(r'expert cache (\d+) slots',log)[1]),
        gap_p95_ms=1000*p95(gaps),ttft_p95_s=p95(ttft),
        gap_over_250ms_request_seconds=sum(g for g in gaps if g>.250),
        reserve_shrink='shrinking the expert cache' in log,
        cache_retry='trying a smaller expert cache' in log,
        stable_seconds=r['c4_no_admission_s'],
        max_prompt_tokens=max(len(t['prompt_ids']) for t in r['turns']),
        engine_timings=[s for s in log.splitlines() if re.match(r'strata batch: \d+ windows,|strata batch output:',s)])
summary={}
for work in ['essay','agentic']:
    arms={a:{key:statistics.median(r[key] for r in rows if r['arm']==a and r['workload']==work)
        for key in ['decode_tps','whole_run_tps','gap_p95_ms','ttft_p95_s']} for a in ['baseline','candidate']}
    arms['ratios']={key:arms['candidate'][key]/arms['baseline'][key] for key in ['decode_tps','whole_run_tps']}
    summary[work]=arms
(out/'summary.json').write_text(json.dumps(dict(baseline='6674a0065f',candidate='ae084bc',medians=summary,runs=rows),indent=2)+'\n')
manifest={p.name:sha(p) for p in sorted(raw.iterdir()) if p.is_file()}
(out/'raw-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
for arm in ['baseline','candidate']:
    (out/(arm+'.json')).write_bytes((raw/(arm+'.json')).read_bytes())
(out/'.gitattributes').write_text('*.json -text -whitespace\n*.log -text -whitespace\n')
print(json.dumps(summary,indent=2))
