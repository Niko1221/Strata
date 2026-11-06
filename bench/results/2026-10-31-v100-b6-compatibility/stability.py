import concurrent.futures,json,os,subprocess,threading,time,urllib.request
from pathlib import Path
B=Path(os.environ.get('BENCH_OUTPUT_DIR','./results')); B.mkdir(parents=True,exist_ok=True)
key=os.environ['STRATA_API_KEY']
headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
result={'rows':[],'status':'running'};done=threading.Event()
def save(): (B/'current-stability.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
def request(path,data,timeout=600):
 req=urllib.request.Request('http://127.0.0.1:8080'+path,data=json.dumps(data).encode(),headers=headers)
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
def monitor():
 with (B/'current-stability-gpu.jsonl').open('w') as f:
  while not done.is_set():
   try:
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,temperature.gpu,temperature.memory,memory.used,memory.free,power.draw,power.limit','--format=csv,noheader,nounits'],text=True,timeout=5)
    f.write(json.dumps({'time':time.time(),'gpus':s.strip().splitlines()})+'\n');f.flush()
   except Exception as e:f.write(json.dumps({'error':str(e)})+'\n')
   done.wait(2)
def make_prompt(target,label):
 prefix=f'安定性試験 {label}。冒頭の確認符号は START7391 です。以下は架空の記録です。\n'
 suffix='\n末尾の確認符号は END2846 です。最初の1行に必ず冒頭と末尾の確認符号を両方そのまま書いてください。その後、記録を日本語で1000文字以上詳しく要約してください。'
 records=''.join(f'記録{i}: 喫茶店では注文を受け、紅茶とコーヒーを用意します。午後には読書と会話を楽しめる時間を設け、席の予約と在庫の確認を行います。\n' for i in range(1100))
 lo,hi=0,len(records);best=None
 while lo<=hi:
  mid=(lo+hi)//2;prompt=prefix+records[:mid]+suffix
  count=request('/v1/messages/count_tokens',{'model':'qwen3.8-flash-next-iq3_s','messages':[{'role':'user','content':prompt}],'max_tokens':512,'thinking':{'type':'disabled'}})['input_tokens']
  if count<=target:best=(prompt,count);lo=mid+1
  else:hi=mid-1
 return best
def run(item):
 label,prompt,count=item; start=time.monotonic()
 d=request('/v1/chat/completions',{'model':'qwen3.8-flash-next-iq3_s','messages':[{'role':'user','content':prompt}],'max_tokens':512,'temperature':0,'seed':42,'reasoning_effort':'none'})
 text=d['choices'][0]['message']['content'];usage=d.get('usage',{})
 row={'label':label,'count_estimate':count,'seconds':round(time.monotonic()-start,3),'usage':usage,'finish_reason':d['choices'][0].get('finish_reason'),'text':text}
 if not text.strip() or not all(x in text for x in ['START7391','END2846']):raise RuntimeError('Missing sentinel/content '+label+': '+text[:180])
 return row
try:
 threading.Thread(target=monitor,daemon=True).start()
 for repeat in range(2):
  items=[]
  for j in range(2):
   prompt,count=make_prompt(32000,f'pair-{repeat}-{j}');items.append((f'pair-{repeat}-{j}',prompt,count))
  with concurrent.futures.ThreadPoolExecutor(2) as pool:rows=list(pool.map(run,items))
  result['rows'].extend(rows);save();print('PAIR',repeat,[(r['seconds'],r['usage']) for r in rows],flush=True)
 result['status']='passed';save()
except BaseException as e:
 result['status']='failed';result['error']=str(e);save();raise
finally:done.set()
