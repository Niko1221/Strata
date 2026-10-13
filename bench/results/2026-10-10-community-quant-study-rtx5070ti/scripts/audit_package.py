#!/usr/bin/env python3
"""Audit public package coverage, credentials, links and recommendation parity."""
from pathlib import Path
import argparse,gzip,hashlib,json,re,subprocess
BASE=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--verify-manifest',action='store_true');args=ap.parse_args()
D=json.loads((BASE/'data/ALL-METRICS.json').read_text());M=D['models'];REF=D['hosted_api_reference'];CL=json.loads((BASE/'claims.json').read_text());SB=json.loads((BASE/'data/FINAL-SCOREBOARD.json').read_text())
checks=[]
def check(label,ok):
 checks.append({'claim':label,'passed':bool(ok)})
 if not ok:raise AssertionError(label)
check('10 models / 20 HE arms / 18 local tool arms',len(M)==10 and sum(len(m['heplus']) for m in M.values())==20 and sum(len(m.get('trials')or{}) for m in M.values())==18)
ref_path=BASE/REF['artifact'];detail=json.loads(ref_path.read_text()) if ref_path.is_file() else {}
check('Sanitized hosted-reference artifact exists and matches pinned SHA-256',ref_path.is_file() and hashlib.sha256(ref_path.read_bytes()).hexdigest()==REF['artifact_sha256'])
check('Hosted reference identity/role/recommendation are explicit',detail.get('role')=='external_baseline_reference_only' and detail.get('recommendation') is None and detail.get('eligible_for_local_composite') is False and detail.get('display_name')==REF['display_name'] and 'GPT-6-luna' not in M)
source_artifacts=detail.get('source_artifacts',{})
check('Hosted reference provenance includes source hashes',len(source_artifacts)==9 and all(re.fullmatch(r'[0-9a-f]{64}',v.get('sha256','')) for v in source_artifacts.values()))
forbidden={'prompt','prompt_text','response','output','completion','code','code_text','thinking','thoughts','reasoning_trace','messages','bridge_payload','authorization','api_key','access_token','password'}
def forbidden_keys(v):
 if isinstance(v,dict):return any(str(k).lower() in forbidden or forbidden_keys(x) for k,x in v.items())
 if isinstance(v,list):return any(forbidden_keys(x) for x in v)
 return False
check('Hosted detail omits prompts, outputs, code/thought traces and credentials',not forbidden_keys(detail))
check('Hosted sample metrics equal the merged comparison summary',detail.get('metrics',{}).get('human_eval_plus',{}).get('final_percent')==REF['human_eval_plus']['final_percent'] and detail.get('metrics',{}).get('tool_eval_bench',{}).get('pass_hat_5_percent')==REF['tool_eval_bench']['pass_hat_5_percent'] and detail.get('metrics',{}).get('five_bugs',{}).get('hidden_passes')==REF['five_bugs']['hidden_passes'] and detail.get('metrics',{}).get('performance',{}).get('valid') is False and REF['performance']['valid'] is False)
rawtool=[];rawperf=[]
for k,m in M.items():
 for e in ['low','xhigh']:
  hp=BASE/'data/heplus'/f'{k}-{e}.json.gz';check(f'Coding raw {k}/{e}',hp.is_file() and len(json.loads(gzip.decompress(hp.read_bytes()))['tasks'])==164)
  if m.get('trials'):
   fn=f'gyro-m-teb-{e}.json' if k=='gyro-m' else (f'campaign11-{k}-trials-{e}.json' if k in ['ista-iq2_xs','swift-q2_0'] else f'trials-{k}-{e}.json')
   p=BASE/'data'/fn;check(f'Tool raw {k}/{e}',p.is_file());j=json.loads(p.read_text());check(f'Tool full 92 x 5 {k}/{e}',len(j['trial_statistics']['per_scenario'])==92 and all(len(r['points'])==5 for r in j['trial_statistics']['per_scenario'].values()));check(f'Tool source matches every canonical point {k}/{e}',j['trial_statistics']['per_scenario']==m['trials'][e]['per_scenario']);rawtool.append(p)
  if m.get('perf'):
   fn=f'gyro-m-perf-{e}.json' if k=='gyro-m' else (f'campaign11-{k}-perf-{e}.perf.json' if k in ['ista-iq2_xs','swift-q2_0'] else f'rebase-{k}-perf-{e}.perf.json')
   p=BASE/'data'/fn;check(f'Performance raw {k}/{e}',p.is_file());pj=json.loads(p.read_text());check(f'Performance values reproduce rollup {k}/{e}',len(pj['benchmarks'])==3 and all(b['tg_throughput']['values']==m['perf'][e][str(b['context_size'])]['tg_values'] and b['tg_throughput']['mean']==m['perf'][e][str(b['context_size'])]['tg_mean'] and b['e2e_ttft']['mean']==m['perf'][e][str(b['context_size'])]['e2e_ttft_ms'] for b in pj['benchmarks']));rawperf.append(p)
check('All tool and performance arms have unique source files',len(set(rawtool))==18 and len(set(rawperf))==18)
check('Excluded malformed Gyro-S performance is not packaged',not list((BASE/'data').glob('*gyro-s*perf*')))
fb=[json.loads(x) for x in (BASE/'data/gyro-m-fivebugs.jsonl').read_text().splitlines() if x.strip()];fs=json.loads((BASE/'data/gyro-m-fivebugs-score.json').read_text());check('25 new debugging records and exact visible/hidden scores',len(fb)==25 and fs['score']=={'hidden':'18/25','visible':'23/25'})
readme=(BASE/'README.md').read_text();body=(BASE/'PR-BODY.md').read_text();html=(BASE/'community-updated.html').read_text();inv=json.loads((BASE/'data/TABLE-INVENTORY.json').read_text())
check('Hosted reference is visible as non-recommendation on README, PR body and PDF source',all('GPT-6-luna MAX' in x and 'OpenAI' in x and 'not a recommendation' in x for x in [readme,body,html]))
check('Hosted reference stays outside local model inventory and recommendation registry',len(M)==10 and all(r['key']!='gpt-6-luna' for r in SB) and CL.get('hosted_api_reference',{}).get('eligible_for_local_composite') is False and CL.get('hosted_api_reference',{}).get('recommendation') is None)
for r in CL['recommendations']:
 target=next(x for x in SB if x['key']==r['key'] and x['arm']==r['effort']);check(f'Recommendation registry rank {r["role"]}',r['rank']==target['rank'])
 label=f"{M[r['key']]['short']} ({M[r['key']]['lab']}) {r['effort']}";check(f'GitHub recommendation reflects same label/rank {r["role"]}',label in readme and ('| #'+str(r['rank'])+' |' in readme if r['rank'] else '| unranked |' in readme))
top=CL['stock_composite_leader'];winner=f"{M[top['key']]['short']} ({M[top['key']]['lab']}) {top['arm']}"
check('README and PR body use same #1 stock all-rounder',winner in readme and winner in body and 'is #1' in readme and 'is #1' in body)
check('No model-specific report chapter',not re.search(r'<h2[^>]*>[^<]*Gyro',html))
check('All tables have their caption and end marker',all(html.count(t['id']+' ·')==2 for t in inv))
check('Critical validity caveats on every surface',all(s in readme and s in body and s in html for s in ['completed_with_invalid_slices','ple_parity','157/164','153/162']))
check('Versioned model-only comparison caveat on every surface',all('model-only' in x for x in [readme,body,html]))
for url in re.findall(r'\]\(([^)]+)\)',readme):
 if url.startswith(('http','mailto:','#')):continue
 check('README relative link '+url,(BASE/url.split('#')[0]).exists())
files=[p for p in BASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not(p.parent.name=='qa' and p.suffix=='.png')]
patterns=[r'ghp_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{30,}',r'sk-[A-Za-z0-9_-]{20,}',r'Bearer\s+[A-Za-z0-9_-]{20,}',r'/home/[A-Za-z0-9_-]+/',r'192\.168\.\d+\.\d+',r'[A-Z]:\\Users\\[^\\\s]+']
secret_fields=[];hits=[]
def walk(v,where):
 if isinstance(v,dict):
  for k,x in v.items():
   if re.fullmatch(r'(api_?key|access_token|password|secret|authorization)',k,re.I) and isinstance(x,str) and x not in ['', '[REDACTED]', '***','redacted','REDACTED']:secret_fields.append(where+':'+k)
   walk(x,where)
 elif isinstance(v,list):
  for x in v:walk(x,where)
for p in files:
 if p.suffix in ['.pdf','.png']:continue
 try:t=gzip.decompress(p.read_bytes()).decode() if p.suffix=='.gz' else p.read_text()
 except UnicodeError:continue
 for pat in patterns:
  if re.search(pat,t):hits.append((str(p.relative_to(BASE)),pat))
 if p.suffix=='.json' or p.name.endswith('.json.gz'):
  walk(json.loads(t),str(p.relative_to(BASE)))
 elif p.suffix=='.jsonl':
  for line in t.splitlines():
   if line.strip():walk(json.loads(line),str(p.relative_to(BASE)))
check('No credential patterns or private rig paths',not hits and not secret_fields)
# Verify the actual distributed PDF separately from plain-text artifacts.
qa=json.loads((BASE/'qa/AUDIT.json').read_text());check('Distributed PDF is the audited artifact',qa['pdf_sha256']==hashlib.sha256((BASE/'strata-community.pdf').read_bytes()).hexdigest() and qa['failed']==0)
size=sum(p.stat().st_size for p in files);check('Package below 30 MiB',size<30*1024*1024)
if args.verify_manifest:
 manifest=json.loads((BASE/'SOURCE-MANIFEST.json').read_text())
 for row in manifest['files']:
  p=BASE/row['path'];check('Manifest '+row['path'],p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] and p.stat().st_size==row['bytes'])
result={'passed':len(checks),'failed':0,'package_bytes':size,'tool_arms':len(rawtool),'performance_arms':len(rawperf),'coding_arms':20,'tables':len(inv),'manifest_verified':args.verify_manifest,'credential_hits':0}
print('PACKAGE_AUDIT',json.dumps(result))
