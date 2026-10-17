"""Import retained valid Gyro-S rc2 auxiliary slices without rerunning inference.
Usage: python scripts/import_gyro_s_auxiliary.py --source PATH [--apply]
The source is the full-battery run directory containing results/ and logs/.
"""
from pathlib import Path
import argparse,copy,hashlib,json,re,shutil

ROOT=Path(__file__).resolve().parent.parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
 if isinstance(value,list):return [clean(v) for v in value]
 if isinstance(value,str):return re.sub(r'/home/[^/]+/','<study-home>/',value)
 return value

def collect(source,data):
 results=source/'results';campaign=data['gyro_s_campaign'];model=data['models']['gyro-s'];manifest=json.loads((results/'manifest.json').read_text());done=json.loads((results/'done.json').read_text())
 expected={r['name']:r['sha256'] for r in campaign['source_hashes']}
 for name,path in [('manifest.json',results/'manifest.json'),('results/done.json',results/'done.json'),('results/steps.jsonl',results/'steps.jsonl')]:
  assert sha(path)==expected[name],(name,sha(path),expected[name])
 for k in ['model_revision','model_sha256','engine_commit','engine_sha256']:
  assert manifest[k]==model['setup_identity'][k],k
 assert manifest['context']==model['setup_identity']['context']==131072 and manifest['kv']=='int8'
 assert 'plugin and FiveBugs axes use xhigh' in manifest['comparator_notes']
 steps=[json.loads(l) for l in (results/'steps.jsonl').read_text().splitlines() if l.strip()]
 files={};metrics={};checks={}
 for suite in ['gsm8k','ifeval','needle']:
  name='plugin-'+suite;assert done['slice_audit']['slices'][name]=={'present':True,'valid':True}
  step=next(x for x in reversed(steps) if x.get('name')==name and x.get('valid') and x.get('metrics'))
  path=Path(step['metrics']['report']);text=path.read_text()
  assert 'gyro-s-rc2-ranking' in text and manifest['campaign_id'] in text and '**Completion**: 100.0%' in text
  if suite=='gsm8k':assert '(96/100)' in text and step['metrics']['accuracy']==96.0
  if suite=='ifeval':assert '(38/50)' in text and '(63/76)' in text and round(63/76*100,1)==step['metrics']['instruction_accuracy']==82.9 and step['metrics']['prompt_accuracy']==76.0
  if suite=='needle':assert '(25/25)' in text and step['metrics']['retrieval_accuracy']==100.0 and step['metrics']['effective_context']==129024
  metrics[suite]={k:v for k,v in step['metrics'].items() if k!='report'}|{'_file':path.name,'wall_s':step['wall_s'],'engine_cohort':'rc2','run_id':manifest['campaign_id'],'sample_effort':'xhigh'}
  files['data/'+path.name]=path
  checks[name]={'valid':True,'report_sha256':sha(path),'wall_s':step['wall_s']}
 fb=next(x for x in reversed(steps) if x.get('name')=='fivebugs' and x.get('score_valid') and x.get('valid'))
 jsonl=Path(fb['jsonl']);records=[json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()];tag='gyro-s-rc2-'+manifest['campaign_id']
 matrix_path=jsonl.parent/'fivebugs_matrix.json';matrix=json.loads(matrix_path.read_text())[tag]
 assert len(records)==fb['unique_records']==25 and {(x['task'],x['run']) for x in records}=={(f't{i}',j) for i in range(1,6) for j in range(1,6)}
 assert all(x['model']==tag and x['meta']['reasoning_effort']=='xhigh' for x in records)
 scored=[r for rr in matrix['per_task'].values() for r in rr];assert len(scored)==25
 for rec in records:
  score=next(s for s in matrix['per_task'][rec['task']] if s['file']==f"{rec['task']}_r{rec['run']}.py")
  assert score['fp']==rec['fp']
 visible=sum(s['vis'] for s in scored);hidden=sum(s['hid'] for s in scored)
 assert matrix['vis']==fb['score']['visible']==f'{visible}/25'=='21/25'
 assert matrix['hid']==fb['score']['hidden']==f'{hidden}/25'=='16/25'
 metrics['fivebugs']={'vis':matrix['vis'],'hid':matrix['hid'],'n_runs':25,'wall_s':fb['last_pass']['wall_s'],'tokens':sum(x['meta']['usage']['completion_tokens'] for x in records),'engine_cohort':'rc2','run_id':manifest['campaign_id'],'sample_effort':'xhigh'}
 assert fb['last_pass']['exit_code']==0 and fb['last_pass']['engine_event'] is False
 checks['fivebugs']={'valid':True,'verdict_source':'retained checker matrix; no generated code re-executed','visible':visible,'hidden':hidden,'records':25,'wall_s':fb['last_pass']['wall_s'],'jsonl_sha256':sha(jsonl),'matrix_sha256':sha(matrix_path)}
 files['data/gyro-s-rc2-fivebugs.jsonl']=jsonl;files['data/gyro-s-rc2-fivebugs-matrix.json']=matrix_path
 failure=source/'logs/plugin-mmlu-retry1.log';assert done['slice_audit']['slices']['plugin-mmlu']=={'present':True,'valid':False}
 assert 'Failed to download MMLU dataset' in failure.read_text() and 'HTTP Error 429' in failure.read_text()
 files['data/gyro-s-rc2-mmlu-failure.log']=failure
 metrics['mmlu']={'accuracy':None,'_file':'gyro-s-rc2-mmlu-failure.log','engine_cohort':'rc2','status':'invalid_dataset_download_http_429','reason':'No scored rc2 algebra result; legacy rc1 76% archived separately, not substituted.'}
 provenance={'run_id':manifest['campaign_id'],'identity':{k:manifest[k] for k in ['model_revision','model_sha256','engine_commit','engine_sha256','context','kv']},'source_hashes':{n:expected[n] for n in ['manifest.json','results/done.json','results/steps.jsonl']},'validated_slices':checks,'unavailable_slice':{'name':'plugin-mmlu','reason':'dataset download HTTP 429','failure_log_sha256':sha(failure)},'overall_outcome_preserved':done['outcome'],'sample_protocol':'xhigh once per file; not two independent low/xhigh suite repeats','artifacts':{dest:{'original_sha256':sha(path)} for dest,path in files.items()}}
 return metrics,provenance,files

def apply(source,root=ROOT):
 data=json.loads((root/'data/ALL-METRICS.json').read_text());metrics,audit,files=collect(source,data);m=data['models']['gyro-s']
 m.setdefault('legacy_rc1_auxiliary',copy.deepcopy({k:m[k] for k in ['knowledge','fivebugs','needle']}))
 for suite in ['gsm8k','ifeval','mmlu']:m['knowledge'][suite]=metrics[suite]
 m['needle']=metrics['needle'];m['fivebugs']=metrics['fivebugs'];m['auxiliary_cohort']='rc2';m['auxiliary_engine']='rc2; algebra unavailable (HTTP 429)'
 m['auxiliary_source_cohorts']={s:('unavailable rc2' if s=='mmlu' else 'rc2') for s in ['gsm8k','mmlu','ifeval','fivebugs','needle']}
 m['auxiliary_audit']='data/gyro-s-rc2-auxiliary-audit.json'
 for dest,path in files.items():
  content=path.read_text();content=re.sub(r'/home/[^/]+/','<study-home>/',content)
  (root/dest).write_text(content)
  audit['artifacts'][dest]['packaged_sha256']=sha(root/dest)
 (root/'data/gyro-s-rc2-auxiliary-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 data['gyro_s_campaign']['auxiliary_audit']={'path':'data/gyro-s-rc2-auxiliary-audit.json','sha256':sha(root/'data/gyro-s-rc2-auxiliary-audit.json'),'valid_slices':list(audit['validated_slices']),'unavailable':['plugin-mmlu']}
 (root/'data/ALL-METRICS.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
 return audit

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--apply',action='store_true');args=ap.parse_args()
 if args.apply:a=apply(args.source)
 else:_,a,_=collect(args.source,json.loads((ROOT/'data/ALL-METRICS.json').read_text()))
 print(json.dumps(a,indent=2))
