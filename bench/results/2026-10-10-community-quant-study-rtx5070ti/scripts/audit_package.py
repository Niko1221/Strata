#!/usr/bin/env python3
"""Audit public package coverage, credentials, links and recommendation parity."""
from pathlib import Path
import argparse,gzip,hashlib,json,math,re,subprocess
BASE=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--verify-manifest',action='store_true');args=ap.parse_args()
D=json.loads((BASE/'data/ALL-METRICS.json').read_text());M=D['models'];REF=D['hosted_api_reference'];SOL_REF=D['hosted_api_reference_sol'];CL=json.loads((BASE/'claims.json').read_text());SB=json.loads((BASE/'data/FINAL-SCOREBOARD.json').read_text())
checks=[]
def check(label,ok):
 checks.append({'claim':label,'passed':bool(ok)})
 if not ok:raise AssertionError(label)
check('10 models / 20 HE arms / 20 local tool arms',len(M)==10 and sum(len(m['heplus']) for m in M.values())==20 and sum(len(m.get('trials')or{}) for m in M.values())==20)
forbidden={'prompt','prompt_text','response','output','completion','code','code_text','thinking','thoughts','reasoning_trace','messages','bridge_payload','authorization','api_key','access_token','password'}
def forbidden_keys(v):
 if isinstance(v,dict):return any(str(k).lower() in forbidden or forbidden_keys(x) for k,x in v.items())
 if isinstance(v,list):return any(forbidden_keys(x) for x in v)
 return False
details={}
for key,summary in [('luna',REF),('sol',SOL_REF)]:
 ref_path=BASE/summary['artifact'];detail=json.loads(ref_path.read_text()) if ref_path.is_file() else {}
 details[key]=detail
 check(f'{key} hosted-reference artifact exists and matches pinned SHA-256',ref_path.is_file() and hashlib.sha256(ref_path.read_bytes()).hexdigest()==summary['artifact_sha256'])
 check(f'{key} identity/role/recommendation are explicit',detail.get('role')=='external_baseline_reference_only' and detail.get('recommendation') is None and detail.get('eligible_for_local_composite') is False and detail.get('display_name')==summary['display_name'])
 check(f'{key} detail omits prompts, outputs, code/thought traces and credentials',not forbidden_keys(detail))
 check(f'{key} metrics equal the merged comparison summary',detail.get('metrics',{}).get('human_eval_plus',{}).get('final_percent')==summary['human_eval_plus']['final_percent'] and detail.get('metrics',{}).get('tool_eval_bench',{}).get('pass_hat_5_percent',detail.get('metrics',{}).get('tool_eval_bench',{}).get('pass_all5_percent'))==summary['tool_eval_bench'].get('pass_hat_5_percent',summary['tool_eval_bench'].get('pass_all5_percent')) and detail.get('metrics',{}).get('five_bugs',{}).get('hidden_passes')==summary['five_bugs']['hidden_passes'] and detail.get('metrics',{}).get('performance',{}).get('valid') is False and summary['performance']['valid'] is False)
luna_detail=details['luna'];luna_sources=luna_detail.get('source_artifacts',{})
check('Luna hosted-reference provenance includes source hashes',len(luna_sources)==9 and all(re.fullmatch(r'[0-9a-f]{64}',v.get('sha256','')) for v in luna_sources.values()))
sol_detail=details['sol'];sol_sources=sol_detail.get('source_artifacts',{});sol_first=sol_sources.get('first_upload_benchmark_scores',{});sol_readme=sol_sources.get('first_upload_readme',{})
check('SOL provenance pins first upload README/scores hashes and source commit',sol_detail.get('source_commit')=='a48292f5766f66efcdb1e23396569b0907c44c04' and sol_first.get('sha256')=='bb040b5e27d0602730638dc3f511a02f1f9e9abd0b46969c543becffd7b1d7d8' and sol_readme.get('sha256')=='7fb22abfb40253ca835c4a4d4009a51c0051debf8868e2258ed8ffb17470b84b' and all(re.fullmatch(r'[0-9a-f]{64}',v.get('sha256','')) for v in sol_sources.get('suite_inputs',{}).values()))
check('SOL Five Bugs uses first-upload independent runs only, not post-feedback validation',sol_detail.get('metrics',{}).get('five_bugs',{}).get('visible_passes')==25 and sol_detail.get('metrics',{}).get('five_bugs',{}).get('hidden_passes')==20 and sol_detail.get('metrics',{}).get('five_bugs',{}).get('post_feedback_validation_included') is False)
rawtool=[];rawperf=[]
for k,m in M.items():
 for e in ['low','xhigh']:
  hp=BASE/'data/heplus'/f'{k}-{e}.json.gz';check(f'Coding raw {k}/{e}',hp.is_file() and len(json.loads(gzip.decompress(hp.read_bytes()))['tasks'])==164)
  if m.get('trials'):
   fn=f'gyro-m-teb-{e}.json' if k=='gyro-m' else (f'gyro-s-teb-{e}.json' if k=='gyro-s' else (f'campaign11-{k}-trials-{e}.json' if k in ['ista-iq2_xs','swift-q2_0'] else f'trials-{k}-{e}.json'))
   p=BASE/'data'/fn;check(f'Tool raw {k}/{e}',p.is_file());j=json.loads(p.read_text());check(f'Tool full 92 x 5 {k}/{e}',len(j['trial_statistics']['per_scenario'])==92 and all(len(r['points'])==5 for r in j['trial_statistics']['per_scenario'].values()));check(f'Tool source matches every canonical point {k}/{e}',j['trial_statistics']['per_scenario']==m['trials'][e]['per_scenario']);rawtool.append(p)
  if m.get('perf'):
   if k=='gyro-s':
    p=BASE/'data/gyro-s-speed-audit.json';sj=json.loads(p.read_text()) if p.is_file() else {};sid=M['gyro-s']['setup_identity'];src=sid.get('source_sha256',{})
    check('Gyro-S rc2 speed audit is hash-pinned to setup and engine-log provenance',p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==src.get('speed-audit') and sj.get('identity',{}).get('model_revision')==sid.get('model_revision') and sj.get('identity',{}).get('model_sha256')==sid.get('model_sha256') and sj.get('identity',{}).get('engine_commit')==sid.get('engine_commit') and sj.get('identity',{}).get('engine_sha256')==sid.get('engine_sha256') and sj.get('source_metadata',{}).get('engine_log',{}).get('sha256')==src.get('engine-log'))
    check('Gyro-S rc2 speed audit is engine-log eligible',sj.get('status')=='eligible_engine_log_speed_cells')
    def same_numbers(actual,expected):return isinstance(actual,list) and len(actual)==len(expected) and all(math.isclose(float(a),float(b),rel_tol=1e-6,abs_tol=1e-6) for a,b in zip(actual,expected))
    for ctx,v in m['perf'][e].items():
     source=sj['profiles'][e][ctx]
     reproduced=(v['requested_depth_tokens']==source['requested_depth_tokens'] and same_numbers(v['tg_values'],source['decode_tps']['samples']) and math.isclose(v['tg_mean'],source['decode_tps']['mean'],rel_tol=1e-6,abs_tol=1e-6) and math.isclose(v['tg_std'],source['decode_tps']['sample_sd'],rel_tol=1e-6,abs_tol=1e-6) and same_numbers(v['e2e_ttft_values'],source['e2e_ttft_ms']['samples']) and math.isclose(v['e2e_ttft_ms'],source['e2e_ttft_ms']['mean'],rel_tol=1e-6,abs_tol=1e-6) and same_numbers(v['prefill_e2e_values'],source['effective_e2e_first_token_tps']['samples']) and math.isclose(v['prefill_e2e_tps'],source['effective_e2e_first_token_tps']['mean'],rel_tol=1e-6,abs_tol=1e-6) and same_numbers(v['engine_prefill_tps_values'],source['engine_prefill_tps']['samples']) and math.isclose(v['engine_prefill_tps_mean'],source['engine_prefill_tps']['mean'],rel_tol=1e-6,abs_tol=1e-6) and same_numbers(v['actual_prompt_tokens_values'],source['observed_full_prompt_tokens']['samples']) and v['runs']==source['n_measured_runs']==3)
     check(f'Gyro-S engine-log profile matches canonical rollup {e}/{ctx}',reproduced)
   else:
    fn=f'gyro-m-perf-{e}.json' if k=='gyro-m' else (f'campaign11-{k}-perf-{e}.perf.json' if k in ['ista-iq2_xs','swift-q2_0'] else f'rebase-{k}-perf-{e}.perf.json')
    p=BASE/'data'/fn;check(f'Performance raw {k}/{e}',p.is_file());pj=json.loads(p.read_text());check(f'Performance values reproduce rollup {k}/{e}',len(pj['benchmarks'])==3 and all(b['tg_throughput']['values']==m['perf'][e][str(b['context_size'])]['tg_values'] and b['tg_throughput']['mean']==m['perf'][e][str(b['context_size'])]['tg_mean'] and b['e2e_ttft']['mean']==m['perf'][e][str(b['context_size'])]['e2e_ttft_ms'] for b in pj['benchmarks']))
   rawperf.append(p)
check('All 20 local tool/performance arms are covered by 20 tool files and 19 performance sources',len(rawtool)==20 and len(set(rawtool))==20 and len(rawperf)==20 and len(set(rawperf))==19)
check('Retired malformed Gyro-S rc1 speed files are excluded; the audited rc2 source is present',not list((BASE/'data').glob('gyro-s-perf-*')) and (BASE/'data/gyro-s-speed-audit.json').is_file())
fb=[json.loads(x) for x in (BASE/'data/gyro-m-fivebugs.jsonl').read_text().splitlines() if x.strip()];fs=json.loads((BASE/'data/gyro-m-fivebugs-score.json').read_text());check('25 new debugging records and exact visible/hidden scores',len(fb)==25 and fs['score']=={'hidden':'18/25','visible':'23/25'})
readme=(BASE/'README.md').read_text();body=(BASE/'PR-BODY.md').read_text();html=(BASE/'community-updated.html').read_text();inv=json.loads((BASE/'data/TABLE-INVENTORY.json').read_text())
check('Knowledge/debugging table explains the 20/25 three-way tie and local-first order on all public surfaces','Top hidden Five Bugs score: 20/25, tied among' in html and 'not a unique win' in html and 'top score is tied at 20/25' in readme and 'top hidden Five Bugs score is tied at 20/25' in body and 'not a unique win' in body)
check('Both hosted references are visible and non-recommendations on README, PR body and PDF source',all(name in x for x in [readme,body,html] for name in ['GPT-6-luna MAX','GPT-6.1-SOL ultra','OpenAI']) and all('recommendation' in x for x in [readme,body,html]))
check('PR body publishes exactly five fixes and names the audit', '### We publish our fixes' in body and 'five report corrections' in body and 'scripts/audit_revision.py' in body and all(term in body for term in ['partial coding denominators','five-run means','prompt-prefill rates','Pass@5 ceiling minus Pass^5 floor','64 C','78 C']))
hosted_claims=CL.get('hosted_api_references',[])
check('Both hosted references stay outside local model inventory and recommendation registry',len(M)==10 and all(not any(name in str(r.get('key','')) for name in ['gpt-6-luna','gpt-6.1-sol']) for r in SB) and len(hosted_claims)==2 and all(x.get('eligible_for_local_composite') is False and x.get('recommendation') is None for x in hosted_claims))
policy=CL['daily_driver_policy']
check('Full decision matrix has no arbitrary master score',policy['weights'] is None and len(SB)==20 and all('rank' not in r and 'composite' not in r for r in SB))
for r in CL['recommendations']:
 label=f"{M[r['key']]['short']} ({M[r['key']]['lab']}) {r['effort']}"
 check(f'Recommendation appears on public surfaces {r["role"]}',label in readme and 'rank' not in r and any(x['key']==r['key'] and x['arm']==r['effort'] for x in SB))
check('Same daily-default recommendation on all public surfaces',all('Swift IQ3_S' in x and 'recommended daily-driver default' in x.lower() and 'no master score' in x.lower() for x in [readme,body,html]))
check('No model-specific report chapter',not re.search(r'<h2[^>]*>[^<]*Gyro',html))
check('All tables have their caption and end marker',all(html.count(t['id']+' ·')==2 for t in inv))
check('Critical validity caveats on every surface',all(s in readme and s in body and s in html for s in ['completed_with_invalid_slices','aborted_or_invalid','plugin-mmlu','ple_parity','157/164','153/162']))
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
