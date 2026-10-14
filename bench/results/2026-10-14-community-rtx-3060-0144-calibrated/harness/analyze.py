"""Aggregate retained benchmark JSON and sampled telemetry; no inference."""
import argparse,csv,json,statistics
from pathlib import Path

def stat(xs):
    xs=[x for x in xs if isinstance(x,(int,float))]
    return {'median':statistics.median(xs),'min':min(xs),'max':max(xs)} if xs else None
def load_measurements(source):
    if (source/'rows.jsonl').exists():
        rows=[json.loads(x) for x in (source/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
        telemetry=[json.loads(x) for x in (source/'telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
    else:
        rows=json.loads((source/'per-request.json').read_text(encoding='utf-8'))
        with (source/'telemetry.csv').open(encoding='utf-8',newline='') as f:
            telemetry=[]
            for row in csv.DictReader(f):
                telemetry.append({key:(value if key in ('session','case','source_stage','utc') else
                    json.loads(value) if value else None) for key,value in row.items()})
    return rows,telemetry

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('source',type=Path)
    ap.add_argument('--out',type=Path,required=True,help='New output directory; source evidence is not modified')
    a=ap.parse_args();rows,telemetry=load_measurements(a.source);out=a.out
    out.mkdir(parents=True,exist_ok=False)
    sessions={r['session'] for r in rows};resource={};requests=[]
    for row in rows:
        e=row.get('engine',{});samples=[t for t in telemetry if row['started_epoch']<=t['epoch']<=row['ended_epoch']]
        item=dict(row)
        item['pp_tok_s']=(e['prompt_tokens']-e.get('reused',0))*1000/e['prompt_ms'] if e.get('prompt_ms') else None
        item['decode_tok_s']=e.get('decode_tok_s')
        item['resource']={k:stat([t.get(k) for t in samples]) for k in
            ('used_ram_bytes','available_ram_bytes','commit_fraction','commit_total_bytes','server_tree_rss_bytes','vram_mib',
             'temperature_c','graphics_mhz','memory_mhz','power_w','pcie_gen','pcie_width')}
        if len(samples)>1:
            item['system_disk_read_bytes']=samples[-1].get('disk_read_bytes',0)-samples[0].get('disk_read_bytes',0)
        requests.append(item)
    for session in sorted(sessions):
        samples=[t for t in telemetry if t['session']==session]
        resource[session]={k:stat([t.get(k) for t in samples]) for k in
            ('used_ram_bytes','available_ram_bytes','commit_fraction','commit_total_bytes','server_tree_rss_bytes','vram_mib',
             'temperature_c','graphics_mhz','memory_mhz','power_w','pcie_gen','pcie_width')}
    groups=[]
    for session in sorted(sessions):
        for target in sorted({r['target'] for r in requests if r['session']==session and r['group']=='speed'}):
            group=[r for r in requests if r['session']==session and r['target']==target and r['group']=='speed']
            successful=[r for r in group if r['ok']]
            groups.append({'session':session,'model':group[0]['model'],'context':group[0]['context'],'target':target,
                'attempts':len(group),'successful':len(successful),'prompt_tokens':[r['engine']['prompt_tokens'] for r in successful],
                'generated_tokens':[r['engine']['engine_generated'] for r in successful],
                'pp_tok_s':stat([r['pp_tok_s'] for r in successful]),'decode_tok_s':stat([r['decode_tok_s'] for r in successful]),
                'ttft_s':stat([r['ttft_s'] for r in successful]),'latency_s':stat([r['client_elapsed_s'] for r in successful]),
                'draft_accept_fraction':stat([r['engine'].get('drafts_accepted',0)/max(1,r['engine'].get('drafts_offered',0)) for r in successful])})
    needles=[r for r in requests if r['group']=='needle'];conversation=[r for r in requests if r['group']=='conversation']
    soak=[r for r in requests if r['group']=='soak'];soak_summary=[]
    if soak:
        begin=min(r['started_epoch'] for r in soak);end=max(r['ended_epoch'] for r in soak);third=(end-begin)/3
        for target in sorted({r['target'] for r in soak}):
            same=[r for r in soak if r['target']==target and r['ok']]
            early=[r for r in same if r['started_epoch']<=begin+third];late=[r for r in same if r['started_epoch']>=end-third]
            soak_summary.append({'target':target,'successful':len(same),'pp':stat([r['pp_tok_s'] for r in same]),
                'decode':stat([r['decode_tok_s'] for r in same]),'early_decode':stat([r['decode_tok_s'] for r in early]),
                'late_decode':stat([r['decode_tok_s'] for r in late]),'early_runs':len(early),'late_runs':len(late)})
    result={'speed':groups,'resources':resource,'needles':[{'session':r['session'],'depth_percent':r['repetition'],
        'ok':r['ok'],'recall_pass':r.get('recall_pass'), 'expected':r.get('expected_code_word'),'answer':r.get('answer'),
        'prompt_tokens':r.get('engine',{}).get('prompt_tokens')} for r in needles],
        'conversation':[{'session':r['session'],'turn':r['repetition'],'ok':r['ok'],'prompt_tokens':r.get('engine',{}).get('prompt_tokens'),
        'reused':r.get('engine',{}).get('reused'),'fresh_tokens':r.get('engine',{}).get('prompt_read'),'ttft_s':r.get('ttft_s'),
        'latency_s':r.get('client_elapsed_s'),'answer':r.get('answer')} for r in conversation],
        'soak':soak_summary,'placement_trials':[r for r in requests if r['group']=='placement_trial'],
        'failures':[r for r in requests if not r['ok']]}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'enriched-requests.json').write_text(json.dumps(requests,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'requests.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['number','session','group','target','repetition','ok','prompt_tokens','reused','generated','pp_tok_s','decode_tok_s','ttft_s','latency_s']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in requests:
            e=r.get('engine',{});w.writerow({k:v for k,v in dict(number=r['number'],session=r['session'],group=r['group'],target=r['target'],
                repetition=r['repetition'],ok=r['ok'],prompt_tokens=e.get('prompt_tokens'),reused=e.get('reused'),generated=e.get('engine_generated'),
                pp_tok_s=r['pp_tok_s'],decode_tok_s=r['decode_tok_s'],ttft_s=r.get('ttft_s'),latency_s=r['client_elapsed_s']).items() if k in fields})
    print(json.dumps({'speed_groups':len(groups),'speed_successes':sum(g['successful'] for g in groups),'needle_passes':sum(bool(r.get('recall_pass')) for r in needles),
        'needle_attempts':len(needles),'conversation_requests':len(conversation),'soak_requests':len(soak),'failures':len(result['failures'])}))
if __name__=='__main__':main()
