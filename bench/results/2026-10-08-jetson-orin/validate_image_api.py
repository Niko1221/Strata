"""Exercise image embedding transport through the real OpenAI API."""
import argparse,base64,json,time
from pathlib import Path
import requests
root=Path(__file__).resolve().parents[3];base=Path('/home/calvin/models/strata-orin-validation')
ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(base/'image-api.json'));options=ap.parse_args()
fixture=root/'bench/results/2026-10-07-jetson-orin/vision-fixture.png'
url='http://127.0.0.1:18081'
health=requests.get(url+'/health',timeout=10).json()
body=dict(model=health['model'],messages=[dict(role='user',content=[dict(type='text',text='Briefly describe the colors and patterns in this image.'),dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(fixture.read_bytes()).decode()))])],temperature=0,max_tokens=48,chat_template_kwargs=dict(enable_thinking=False))
start=time.monotonic();r=requests.post(url+'/v1/chat/completions',json=body,timeout=600);r.raise_for_status();response=r.json()
assert response['usage']['completion_tokens']>0 and response['choices'][0]['message']['content'].strip()
Path(options.out).write_text(json.dumps(dict(elapsed_s=time.monotonic()-start,response=response),indent=2)+'\n');print(json.dumps(response,indent=2))
