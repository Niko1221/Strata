"""Exported API/vision/queue checks; coding is reproduced separately."""
import argparse,base64,concurrent.futures,io,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
rows=[]
def save(name,v):(a.out/name).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def get(path):
    with urllib.request.urlopen(a.url+path,timeout=15) as r:return json.load(r)
def post(name,path,body):
    save(name+'-request.json',body);start=time.perf_counter()
    req=urllib.request.Request(a.url+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','anthropic-version':'2023-06-01'})
    with urllib.request.urlopen(req,timeout=600) as r:value=json.load(r)
    elapsed=time.perf_counter()-start
    save(name+'-response.json',dict(response=value,client_seconds=elapsed))
    return value,elapsed
def check(name,fn):
    try:detail=fn();rows.append(dict(name=name,passed=True,detail=detail))
    except Exception as e:rows.append(dict(name=name,passed=False,error=repr(e)))
    save('checks.json',rows);print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
health=get('/health');assert health['images'] and health['max_context']==262144
save('health.json',health)
def tool():
    common=dict(model=health['model'],max_tokens=256,temperature=0,thinking={'type':'disabled'},tools=[dict(name='add_numbers',description='Add two integers.',input_schema=dict(type='object',properties={'a':{'type':'integer'},'b':{'type':'integer'}},required=['a','b']))])
    ms=[dict(role='user',content='Use add_numbers to add 19 and 23. After receiving the result, answer only the result.')]
    r,_=post('tool-call','/v1/messages',dict(**common,messages=ms));c=[x for x in r['content'] if x['type']=='tool_use'];assert len(c)==1 and c[0]['input']=={'a':19,'b':23} and c[0]['name']=='add_numbers',r
    ms += [dict(role='assistant',content=r['content']),dict(role='user',content=[dict(type='tool_result',tool_use_id=c[0]['id'],content='42')])]
    r,t=post('tool-result','/v1/messages',dict(**common,messages=ms));s=''.join(x.get('text','') for x in r['content']).strip().strip('.');assert s=='42',s
    return dict(expected=42,actual=s)
def openai_tool():
    common=dict(model=health['model'],temperature=0,reasoning_effort='none',max_tokens=256,tools=[dict(type='function',function=dict(name='add_numbers',description='Add two integers.',parameters=dict(type='object',properties={'a':{'type':'integer'},'b':{'type':'integer'}},required=['a','b'])))])
    ms=[dict(role='user',content='Use add_numbers to add 19 and 23. After receiving the result, answer only the result.')]
    r,_=post('openai-tool-call','/v1/chat/completions',dict(**common,messages=ms));m=r['choices'][0]['message'];c=m['tool_calls'];assert len(c)==1 and c[0]['function']['name']=='add_numbers' and json.loads(c[0]['function']['arguments'])=={'a':19,'b':23},r
    ms += [m,dict(role='tool',tool_call_id=c[0]['id'],content='42')]
    r,t=post('openai-tool-result','/v1/chat/completions',dict(**common,messages=ms));s=r['choices'][0]['message']['content'].strip().strip('.');assert s=='42',s
    return dict(expected=42,actual=s)
def card(code,nonce):
    im=Image.new('RGB',(1000,700),'white');d=ImageDraw.Draw(im);font=ImageFont.truetype(str(Path(os.environ['WINDIR'])/'Fonts'/'arial.ttf'),42)
    for pos,t in [((55,40),'BENCHMARK CARD'),((55,120),'Code: '+code),((55,200),'Total: 37'),((55,580),'Trial '+nonce)]:d.text(pos,t,font=font,fill='black')
    for x in (160,420,680):d.rectangle((x,370,x+100,470),fill='blue')
    buf=io.BytesIO();im.save(buf,format='PNG');(a.out/(nonce+'.png')).write_bytes(buf.getvalue());return base64.b64encode(buf.getvalue()).decode()
def parse(s):
    s=s.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip();return json.loads(s)
def single(i):
    code=f'VX-{4827+i}';im=card(code,f'single-{i}')
    content=[dict(type='text',text='Read the card. Return only JSON with keys code (string), total (integer), blue_squares (integer).'),dict(type='image',source=dict(type='base64',media_type='image/png',data=im))]
    r,t=post(f'single-{i}','/v1/messages',dict(model=health['model'],messages=[dict(role='user',content=content)],max_tokens=128,temperature=0,thinking={'type':'disabled'}))
    actual=parse(''.join(x.get('text','') for x in r['content']));expected=dict(code=code,total=37,blue_squares=3);assert actual==expected,actual
    metrics=get('/metrics')['requests'][0];save(f'single-{i}-metrics.json',metrics)
    return dict(actual=actual,expected=expected,client_seconds=t,engine=metrics)
def multi(i):
    codes=[f'VT-{9300+i*4+j}' for j in range(4)];content=[dict(type='text',text='Return only a JSON array containing the Code from each of the four cards, in image order.')]
    for j,code in enumerate(codes):content.append(dict(type='image_url',image_url={'url':'data:image/png;base64,'+card(code,f'batch-{i}-{j}')}))
    r,t=post(f'four-{i}','/v1/chat/completions',dict(model=health['model'],messages=[dict(role='user',content=content)],max_tokens=128,temperature=0,reasoning_effort='none'))
    actual=parse(r['choices'][0]['message']['content']);assert actual==codes,actual
    metrics=get('/metrics')['requests'][0];save(f'four-{i}-metrics.json',metrics)
    return dict(expected=codes,actual=actual,client_seconds=t,engine=metrics)
def queue():
    def task(i):
        prompt=f'Independent request {i}. '+('Verify simple integer addition. '*500)+f'\nReply with exactly QUEUE-{i}.'
        r,t=post(f'queue-{i}','/v1/chat/completions',dict(model=health['model'],messages=[dict(role='user',content=prompt)],max_tokens=32,temperature=0,reasoning_effort='none'))
        text=r['choices'][0]['message']['content'].strip();assert text==f'QUEUE-{i}',text;return dict(request=i,answer=text,client_seconds=t)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:values=list(ex.map(task,(1,2)))
    assert get('/health')['loaded'];return dict(results=values,meaning='Two client requests, engine parallel=1; both completed. Not a throughput/concurrency scaling benchmark.')
check('anthropic_tool_roundtrip',tool)
check('openai_tool_roundtrip',openai_tool)
for i in range(3):check(f'single_image_{i}',lambda i=i:single(i))
for i in range(3):check(f'four_images_{i}',lambda i=i:multi(i))
check('two_clients_serial_queue',queue)
save('final-health.json',get('/health'))
raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
