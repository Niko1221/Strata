"""Small synthetic tool and image checks; no external actions or private documents."""
import base64,json,time,urllib.request
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
out=Path(__file__).resolve().parent.parent/'correctness';out.mkdir(exist_ok=True)
records=[]
def post(name,body):
    (out/(name+'-request.json')).write_text(json.dumps(body,indent=2),encoding='utf-8')
    req=urllib.request.Request('http://127.0.0.1:8080/v1/messages',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','anthropic-version':'2023-06-01'})
    t=time.perf_counter()
    with urllib.request.urlopen(req,timeout=600) as r:value=json.load(r)
    (out/(name+'-response.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
    return value,time.perf_counter()-t
def check(name,fn):
    try:detail=fn();records.append(dict(name=name,passed=True,detail=detail))
    except Exception as e:records.append(dict(name=name,passed=False,error=repr(e)))
    (out/'checks.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(records[-1],flush=True)
def tool():
    messages=[dict(role='user',content='Use add_numbers to add 19 and 23. After receiving the tool result, reply with only the result.')]
    common=dict(model='qwen3.8-flash-next-iq2_xs',max_tokens=256,temperature=0,thinking={'type':'disabled'},tools=[dict(name='add_numbers',description='Add two integers.',input_schema=dict(type='object',properties={'a':{'type':'integer'},'b':{'type':'integer'}},required=['a','b'],additionalProperties=False))])
    r,_=post('tool-call',dict(**common,messages=messages));calls=[x for x in r['content'] if x['type']=='tool_use'];assert len(calls)==1 and calls[0]['name']=='add_numbers' and calls[0]['input']=={'a':19,'b':23}
    messages += [dict(role='assistant',content=r['content']),dict(role='user',content=[dict(type='tool_result',tool_use_id=calls[0]['id'],content='42')])]
    r2,_=post('tool-result',dict(**common,messages=messages));answer=''.join(x.get('text','') for x in r2['content']).strip().strip('.');assert answer=='42',answer
    return dict(expected_tool='add_numbers',expected_arguments={'a':19,'b':23},expected_final='42',actual_final=answer)
def vision():
    img=Image.new('RGB',(1000,700),'white');d=ImageDraw.Draw(img);font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',42)
    d.text((55,40),'BENCHMARK CARD',font=font,fill='black');d.text((55,120),'Code: VX-4827',font=font,fill='black');d.text((55,200),'Total: 37',font=font,fill='black')
    for x in (160,420,680):d.rectangle((x,370,x+100,470),fill='blue')
    img.save(out/'vision-card.png');data=base64.b64encode((out/'vision-card.png').read_bytes()).decode()
    body=dict(model='qwen3.8-flash-next-iq2_xs',max_tokens=128,temperature=0,thinking={'type':'disabled'},messages=[dict(role='user',content=[dict(type='text',text='Read the card. Return only JSON with keys code (string), total (integer), blue_squares (integer).'),dict(type='image',source=dict(type='base64',media_type='image/png',data=data))])])
    r,elapsed=post('vision',body);text=''.join(x.get('text','') for x in r['content']).strip();text=text.removeprefix('```json').removesuffix('```').strip();v=json.loads(text);expected={'code':'VX-4827','total':37,'blue_squares':3};assert v==expected,v
    return dict(expected=expected,actual=v,client_elapsed_s=elapsed)
check('anthropic_tool_roundtrip',tool);check('vision_card',vision)
raise SystemExit(0 if all(r['passed'] for r in records) else 1)
