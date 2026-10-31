#!/usr/bin/env python3
"""One cold 122K code-edit request against a freshly started Strata server."""
import argparse, ast, gzip, hashlib, json, os, re, time, urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8080');p.add_argument('--arm',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
r=Path(__file__).resolve().parent
body=json.loads(gzip.decompress((r/'edit-task.json.gz').read_bytes()))
body.update(stream=True,stream_options={'include_usage':True})
headers={'Content-Type':'application/json'}
if os.environ.get('STRATA_API_KEY'):headers['Authorization']='Bearer '+os.environ['STRATA_API_KEY']
req=urllib.request.Request(a.url.rstrip('/')+'/v1/chat/completions',data=json.dumps(body).encode(),headers=headers)
t0=time.monotonic();first=None;text='';usage=None;timings=None;finish=None
with urllib.request.urlopen(req,timeout=1800) as response:
 for line in response:
  if not line.startswith(b'data: '):continue
  raw=line[6:].strip()
  if raw==b'[DONE]':break
  x=json.loads(raw)
  if x.get('error'):raise RuntimeError(x['error'])
  for c in x.get('choices',[]):
   d=c.get('delta',{})
   if first is None and (d.get('content') or d.get('reasoning_content')):first=time.monotonic()-t0
   text+=d.get('content') or '';finish=c.get('finish_reason') or finish
  usage=x.get('usage') or usage;timings=x.get('timings') or timings
wall=time.monotonic()-t0
expected=ast.dump(ast.parse((r/'expected.py.txt').read_text()))
m=re.search(r'```(?:python)?\s*\n(.*?)```',text,re.S)
try:correct=ast.dump(ast.parse(m[1] if m else text))==expected
except SyntaxError:correct=False
row={'arm':a.arm,'wall_seconds':wall,'ttft_seconds':first,'usage':usage,'timings':timings,'finish_reason':finish,'correct':correct,'output_sha256':hashlib.sha256(text.encode()).hexdigest()}
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row,indent=2))
if not timings or timings['cache_n']!=0:raise SystemExit('Not a cold run: restart the engine before the next arm.')
if usage['prompt_tokens']!=122451:raise SystemExit('Tokenizer/template differs from the measured fixture.')
if not correct:raise SystemExit('Code-edit correctness failed; keep this result.')
