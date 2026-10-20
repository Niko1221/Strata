import json,time,urllib.request,re,sys,statistics,shlex
from pathlib import Path
import os
root=Path(os.environ.get('BENCH_OUTPUT_DIR', '.')); root.mkdir(parents=True,exist_ok=True)
label,port,log=sys.argv[1:4]
key=os.environ.get('STRATA_API_KEY','')
def generate(prompt,limit):
 path=Path(log); offset=path.stat().st_size if path.exists() else 0
 body={'model':'qwen3.8-flash-next-iq3_s','messages':[{'role':'user','content':prompt}], 'max_tokens':limit,'temperature':0,'presence_penalty':0,'top_p':1.0,'top_k':0,'min_p':0.0,'repetition_penalty':1.0,'frequency_penalty':0.0,'seed':42,'reasoning_effort':'none','stream':True,'stream_options':{'include_usage':True}}
 req=urllib.request.Request('http://127.0.0.1:'+port+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 start=time.perf_counter();first=None;usage=None;parts=[]
 with urllib.request.urlopen(req,timeout=240) as response:
  for line in response:
   if not line.startswith(b'data:'):continue
   raw=line[5:].strip()
   if raw==b'[DONE]':break
   data=json.loads(raw)
   if 'error' in data:raise RuntimeError(str(data['error']))
   if data.get('usage'):usage=data['usage']
   for choice in data.get('choices',[]):
    text=choice.get('delta',{}).get('content','')
    if text:
     if first is None:first=time.perf_counter()
     parts.append(text)
 end=time.perf_counter()
 if not parts:raise RuntimeError('Empty output')
 time.sleep(.2)
 with path.open('rb') as f:f.seek(offset);tail=f.read().decode(errors='replace')
 stats=re.findall(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms \(([\d.]+) tok/s\), (\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\)',tail)
 return {'ttft_ms':round((first-start)*1000,2),'wall_s':round(end-start,3),'usage':usage,'engine_stats':stats,'text':''.join(parts)}
prompts={'japanese':'初心者が家庭菜園を始めるときの注意点を日本語で10項目、具体例とともに詳しく教えてください。','code':'Build a single-file HTML task manager. It should support adding, completing, and deleting tasks. Save tasks in localStorage and make the layout work on mobile. Return the complete HTML in one code block.','long':'次の架空の会議記録を日本語で要約してください。\n'+''.join(f'記録{i}: 三姉妹AIの配信では、視聴者への返答を優先し、音声合成とLINEの返信を並行して処理します。会話履歴を整理し、待ち時間とメモリ使用量を測定します。\n' for i in range(30))}
result={'label':label,'rows':[]};dest=root/(label+'.json')
try:
 generate('短く挨拶してください。',16)
 for kind,prompt in prompts.items():
  for i in range(3):
   row=generate(f'比較試験 {kind} {i}。\n'+prompt,256 if kind!='long' else 64)
   row.update(kind=kind,repeat=i);result['rows'].append(row)
   dest.write_text(json.dumps(result,ensure_ascii=False,indent=2))
   print(json.dumps({k:v for k,v in row.items() if k!='text'},ensure_ascii=False),flush=True)
except Exception as e:
 result['error']=str(e);dest.write_text(json.dumps(result,ensure_ascii=False,indent=2));raise
