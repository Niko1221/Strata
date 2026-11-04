"""One explicit streaming request; preserves raw SSE, request and observed timings."""
import argparse,json,pathlib,time,urllib.request,threading,sys
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from metrics import sample
def run(base,request,out,pid=None):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=False)
 request=dict(request,stream=True);request['stream_options']={'include_usage':True}
 (out/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
 stop=threading.Event()
 def monitor():
  with (out/'resources.jsonl').open('x') as f:
   while not stop.is_set():f.write(json.dumps(sample(pid))+'\n');f.flush();stop.wait(1)
 worker=threading.Thread(target=monitor);worker.start();start=time.monotonic()
 result={'started_unix':time.time(),'ttft_any_s':None,'ttft_content_s':None,'content':'','reasoning':'','usage':None,'timings':None,'events':0,'status':'running'}
 try:
  req=urllib.request.Request(base.rstrip('/')+'/v1/chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=14400) as response,(out/'events.jsonl').open('x') as log:
   for raw in response:
    if not raw.startswith(b'data:'):continue
    data=raw[5:].strip()
    if data==b'[DONE]':result['done_marker']=True;break
    event=json.loads(data);elapsed=time.monotonic()-start;log.write(json.dumps({'elapsed_s':elapsed,'event':event},ensure_ascii=False)+'\n');log.flush();result['events']+=1
    if event.get('usage'):result['usage']=event['usage']
    if event.get('timings'):result['timings']=event['timings']
    for choice in event.get('choices',[]):
     delta=choice.get('delta',{});content=delta.get('content') or '';reasoning=delta.get('reasoning_content') or delta.get('reasoning') or ''
     if (content or reasoning) and result['ttft_any_s'] is None:result['ttft_any_s']=elapsed
     if content and result['ttft_content_s'] is None:result['ttft_content_s']=elapsed
     result['content']+=content;result['reasoning']+=reasoning
     if choice.get('finish_reason'):result['finish_reason']=choice['finish_reason']
  result['status']='complete' if result.get('finish_reason') or result.get('done_marker') else 'incomplete_stream'
 except Exception as exc:result['status']='error';result['error_type']=type(exc).__name__;result['error']=str(exc)
 finally:
  result['elapsed_s']=time.monotonic()-start;stop.set();worker.join();(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--request',required=True);p.add_argument('--out',required=True);p.add_argument('--pid',type=int);a=p.parse_args()
 result=run(a.base,json.loads(pathlib.Path(a.request).read_text()),a.out,a.pid)
 print(json.dumps({k:result.get(k) for k in ['status','elapsed_s','ttft_any_s','ttft_content_s','usage','timings']}))
