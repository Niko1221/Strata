import argparse, json, pathlib, random, secrets, statistics, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent

def request(cfg, route, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}{route}",
        headers={'Authorization': 'Bearer '+cfg['api_key'], 'Content-Type':'application/json'},
        data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)

def benchmark(cfg, label, sizes=(4096,16384), repeats=3):
    model = request(cfg, '/v1/models')['data'][0]['id']
    rows=[]
    chars_per_token=4.4
    for size in sizes:
        for repeat in range(repeats):
            rng=random.Random(size*100+repeat)
            paragraphs=[]
            while sum(map(len,paragraphs)) < size*chars_per_token:
                paragraphs.append(f"Record {rng.randrange(10000,99999)}: The research team measured water flow through the northern valley. "
                    f"On day {rng.randrange(1,366)}, station {rng.randrange(1,100)} reported {rng.randrange(1,900)} units. "
                    "They checked the instruments and recorded observations before comparing them with the previous season.")
            prompt='Reference '+secrets.token_hex(12)+'.\n'+'\n'.join(paragraphs)+'\nSummarize the subject in one sentence.'
            start=time.monotonic()
            data=request(cfg,'/v1/chat/completions',{'model':model,'messages':[{'role':'user','content':prompt}],
                'max_tokens':8,'temperature':0,'stream':False,'strata_checkpoint':False})
            timing=data.get('timings',{})
            row={'label':label,'target':size,'repeat':repeat+1,'wall_s':round(time.monotonic()-start,3),
                 'usage':data.get('usage'), 'timings':timing}
            rows.append(row)
            print(json.dumps(row),flush=True)
            actual=(data.get('usage') or {}).get('prompt_tokens')
            if actual: chars_per_token=max(2,min(8,len(prompt)/actual))
            (ROOT/(label+'.json')).write_text(json.dumps(rows,indent=2))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('config');p.add_argument('label');p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    benchmark(json.loads(pathlib.Path(a.config).read_text(encoding='utf-8')),a.label,repeats=a.repeats)
