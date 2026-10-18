#!/usr/bin/env python3
"""Independent OFFLINE random100 analysis; no engine/controller imports or network.
Usage after final archive:
 python3 -B analyze.py /path/run-01/results.json --out-dir /new/analysis-dir
Also accepts remote-results.json, but that alone is not an archive provenance check.
During collection: --schema-only (NO performance estimates).
Interrupted final run: --final-incomplete explicitly permits descriptive partial
artifacts; no CIs/performance qualification. Frozen controller/manifest untouched.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

HERE=Path(__file__).resolve().parent
DEFAULT_MANIFEST=HERE/'manifest.json'
MANIFEST_SHA="f0a325ae610b0c00981d5f48f631122888854f0d859e14d7822b4fdbe5b16fc5"
MASTER=2026101802591
BOOTSTRAP_SEED=2026101802592
BOOTSTRAP_DRAWS=20000
CELLS=('english@0.6','english@1.0','source@0.6','source@1.0')
NORMAL=('length','stop','eos')


def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def cell(r):return r['fixture']+'@'+format(r['temperature'],'.1f')
def finite(x):return type(x) in (int,float) and math.isfinite(x)
def geomean(values):
    return math.exp(math.fsum(math.log(x) for x in values)/len(values)) if values and all(x is not None and finite(x) and x>0 for x in values) else None

def manifest_check(m):
    if m.get('schema')!='augmented-random100-v1' or m.get('master_seed')!=MASTER:raise ValueError('wrong frozen manifest design')
    if len(m['blocks'])!=20:raise ValueError('expected20 blocks')
    counts=Counter();orders=Counter();seeds=[];groups=[]
    for i,b in enumerate(m['blocks']):
        if b['block']!=i or b['order'] not in (['U','A'],['A','U']):raise ValueError('block identity/order')
        orders[''.join(b['order'])]+=1;seeds.append(b['warmup']['seed'])
        if b['warmup']['group_id'] is not None or b['warmup']['limit']!=16:raise ValueError('warmup')
        cc=Counter()
        for j in b['requests']:
            c=cell(j);cc[c]+=1;counts[c]+=1;seeds.append(j['seed']);groups.append(j['group_id'])
            if j['workload']!=j['fixture'] or j['limit']!=(64 if j['fixture']=='english' else 67):raise ValueError('fixture/cap')
        if set(cc)!=set(CELLS) or sorted(cc.values())!=[1,1,1,2]:raise ValueError('block cell coverage')
    if counts!=Counter({c:25 for c in CELLS}) or orders!=Counter(UA=10,AU=10):raise ValueError('cell/order balance')
    if len(groups)!=100 or len(set(groups))!=100 or len(seeds)!=120 or len(set(seeds))!=120:raise ValueError('unique seeds/groups')
    if any(type(x) is not int or not 0<x<2**63 for x in seeds):raise ValueError('seed range')


def validate(results,manifest,manifest_file_sha=None):
    """Schema/completeness only: no effect estimates. Returns audit + raw pair records."""
    manifest_check(manifest)
    errors=[];missing=[];received=defaultdict(list);runseen=Counter();record_errors={};allrecords=[];incomplete_requests=[]
    bygroup={j['group_id']:(b,j) for b in manifest['blocks'] for j in b['requests']}
    fixtures={f['name']:f for f in manifest['fixtures']}
    def err(text):errors.append(text)
    if results.get('manifest')!=manifest:err('embedded manifest absent/different')
    if manifest_file_sha and manifest_file_sha!=MANIFEST_SHA:err('supplied manifest differs from pre-data frozen SHA')
    if manifest_file_sha and results.get('manifest_sha256')!=manifest_file_sha:err('manifest raw SHA mismatch/absent')
    if results.get('binary_hashes')!=manifest['binary_sha256']:err('run-level binary receipts differ/absent')
    cfg=results.get('config_sha256',{})
    if cfg.get('U')!=manifest['config_sha256']['U'] or cfg.get('A')!=manifest['config_sha256']['A']:err('config receipt mismatch/absent')
    if results.get('fixtures')!=manifest['fixtures'] or results.get('sampling')!=manifest['sampling']:err('fixture/sampling mismatch/absent')
    runs=results.get('runs',[])
    expected_order=[(b['block'],a) for b in manifest['blocks'] for a in b['order']]
    actual_order=[(r.get('block'),r.get('arm')) for r in runs]
    if actual_order!=expected_order[:len(actual_order)]:err('engine order differs from frozen schedule')
    for run in runs:
        bid,arm=run.get('block'),run.get('arm');runseen[bid,arm]+=1;label=f'block{bid}/{arm}'
        if type(bid) is not int or not 0<=bid<20 or arm not in ('U','A'):err(label+': unknown engine');continue
        b=manifest['blocks'][bid]
        if run.get('binary_sha256')!=manifest['binary_sha256'][arm]:err(label+': executable receipt')
        if run.get('traced') is not False or run.get('state_hash') is not False:err(label+': trace/state instrumentation not disabled')
        if run.get('override')!= (manifest['A_overrides'] if arm=='A' else {}):err(label+': arm overrides')
        reqs=run.get('requests',[]);expected=[b['warmup'],*b['requests']]
        for pos,r in enumerate(reqs):
            allrecords.append(dict(block=bid,arm=arm,ordinal=pos,record=r));bad=[]
            if pos>=len(expected):bad.append('unexpected extra request');job=None
            else:job=expected[pos]
            if job is not None:
                for k in ('group_id','fixture','workload','temperature','seed','limit'):
                    if r.get(k)!=job[k]:bad.append('identity/history '+k)
            gid=r.get('group_id')
            if gid is not None:
                received[gid,arm].append((bid,r))
                if gid not in bygroup:bad.append('unknown group')
                elif bygroup[gid][0]['block']!=bid:bad.append('group in wrong block')
            n=r.get('produced');ids=r.get('output_ids')
            if type(n) is not int or n<0 or not isinstance(ids,list) or len(ids)!=n or any(type(x) is not int or x<0 for x in (ids or [])):bad.append('output IDs/count')
            if type(n) is int and job is not None and n>job['limit']:bad.append('count exceeds cap')
            for k in ('decode_ms','prompt_ms','wall_seconds','ttft_seconds'):
                if not finite(r.get(k)) or r[k]<0:bad.append('invalid time '+k)
            if r.get('finish') not in NORMAL or r.get('status','ok') not in ('ok','completed'):bad.append('failed/abnormal finish')
            try:
                done=r['raw_done'].split()
                if len(done)!=16 or done[0]!='DONE':raise ValueError('DONE shape')
                if int(done[1])!=n or done[5]!=r['finish']:bad.append('DONE output/finish')
                if job and int(done[2])!=len(fixtures[job['fixture']]['ids']):bad.append('DONE prompt count')
                if float(done[3])!=r['prompt_ms'] or float(done[4])!=r['decode_ms']:bad.append('DONE time mismatch')
                accepted,offered,lookups=int(done[6]),int(done[7]),int(done[10])
                if accepted!=r['draft_accepted'] or offered!=r['draft_offered'] or not 0<=accepted<=offered:bad.append('DONE draft counts')
                if lookups<0 or lookups%480 or int(done[15])!=0:bad.append('fixed row geometry/offload')
                rows=lookups//480;windows=rows-offered;extra=windows+accepted-n
                if windows<1 or not 0<=extra<=5:bad.append('row/terminal accounting')
                actualwork=dict(computed_rows=rows,windows=windows,offered=offered,accepted=accepted,unused=offered-accepted,terminal_extra=extra,lookups=lookups)
                if 'work' in r and any(r['work'].get(k)!=v for k,v in actualwork.items()):bad.append('stored work not consistent with DONE')
            except (KeyError,TypeError,ValueError):bad.append('missing/malformed DONE or metrics')
            if 'host_start_monotonic' in r or 'host_end_monotonic' in r:
                if not finite(r.get('host_start_monotonic')) or not finite(r.get('host_end_monotonic')) or r['host_end_monotonic']<r['host_start_monotonic']:bad.append('host timestamp ordering')
            if bad:err(label+f'/ordinal{pos}: '+', '.join(bad))
            if gid is not None:record_errors[gid,arm,pos,bid]=bad
        if len(reqs)!=6:missing.append(label+': expected warmup+5 requests, got '+str(len(reqs)))
        if run.get('incomplete_request') is not None:
            incomplete_requests.append(dict(block=bid,arm=arm,record=run['incomplete_request']))
            err(label+': interrupted request retained')
    for ident in expected_order:
        if runseen[ident]!=1:missing.append(str(ident)+': engine count '+str(runseen[ident]))
    pairs=[]
    for gid,(b,j) in bygroup.items():
        p=dict(group_id=gid,block=b['block'],cell=cell(j),slot=b['requests'].index(j),seed=j['seed'],limit=j['limit'],order=''.join(b['order']),
               status='complete_normal',U=None,A=None,issues=[])
        for arm in ('U','A'):
            items=received.get((gid,arm),[])
            if len(items)!=1:
                p['status']='incomplete_or_duplicate';p['issues'].append(arm+': record count '+str(len(items)));missing.append(gid+'/'+arm+': count '+str(len(items)))
                continue
            bid,r=items[0];p[arm]=r
            reasons=[x for (g,a,_,_),x in record_errors.items() if g==gid and a==arm]
            if any(reasons):p['status']='invalid_or_failed';p['issues']+=sum(reasons,[])
        pairs.append(p)
    normal=sum(p['status']=='complete_normal' for p in pairs)
    complete=not errors and not missing and len(runs)==40 and normal==100 and results.get('completed') is True
    return dict(matrix_complete=complete,controller_completed=results.get('completed') is True,run_count=len(runs),planned_pairs=100,
                complete_normal_pairs=normal,errors=errors,missing=missing,all_received_records=allrecords,
                incomplete_requests=incomplete_requests,controller_error=results.get('error'),request_timeout=results.get('request_timeout'),
                partial_status='complete' if complete else 'incomplete_or_invalid',performance_qualification=False),pairs


def decorate(p):
    out=dict(p)
    if p['status']!='complete_normal':return out
    u,a=p['U'],p['A']
    out.update(same_output=u['output_ids']==a['output_ids'],same_length=u['produced']==a['produced'],
               both_full_cap=u['produced']==a['produced']==p['limit'],delta_tokens=a['produced']-u['produced'])
    valid=all(r['produced']>0 and r['decode_ms']>0 for r in (u,a))
    out['throughput_ratio']=(a['produced']/a['decode_ms'])/(u['produced']/u['decode_ms']) if valid else None
    out['ratio_issue']=None if valid else 'zero tokens or decode time: all-pair log estimator undefined; no epsilon/drop'
    out['throughput_change_percent']=100*(out['throughput_ratio']-1) if valid else None
    out['delta_wall_seconds']=a['wall_seconds']-u['wall_seconds'];out['delta_ttft_seconds']=a['ttft_seconds']-u['ttft_seconds']
    for arm in ('U','A'):
        r=p[arm];d=r['raw_done'].split();rows=int(d[10])//480;o=int(d[7]);acc=int(d[6])
        out[arm+'_work']=dict(rows=rows,windows=rows-o,offered=o,accepted=acc,unused=o-acc,terminal_extra=rows-o+acc-r['produced'])
    return out


def values(ps):
    if not ps:return dict(geomean=None,token_weighted=None,median=None)
    ratios=[p.get('throughput_ratio') for p in ps]
    totals={a:(sum(p[a]['produced'] for p in ps),math.fsum(p[a]['decode_ms'] for p in ps)) for a in ('U','A')}
    nu,du=totals['U'];na,da=totals['A']
    weighted=(na/da)/(nu/du) if nu>0 and du>0 and da>0 else None
    return dict(geomean=geomean(ratios),token_weighted=weighted,median=statistics.median(ratios) if all(x is not None for x in ratios) else None)

def quantile(xs,q):
    xs=sorted(xs);idx=(len(xs)-1)*q;i=int(idx);j=min(i+1,len(xs)-1)
    return xs[i]+(xs[j]-xs[i])*(idx-i)

def ci(xs):
    bad=sum(x is None or not finite(x) for x in xs)
    if not xs or bad:return dict(lower=None,upper=None,undefined_draws=bad,reason='incomplete run or undefined bootstrap draws; no draws dropped')
    lo,hi=quantile(xs,.025),quantile(xs,.975)
    return dict(lower=lo,upper=hi,lower_change_percent=100*(lo-1),upper_change_percent=100*(hi-1),undefined_draws=0)

def boot(pairs,enabled=True):
    estimates={c:defaultdict(list) for c in CELLS};overall=defaultdict(list)
    if enabled:
        rng=random.Random(BOOTSTRAP_SEED)
        cache={c:{b:[p for p in pairs if p['cell']==c and p['block']==b] for b in range(20)} for c in CELLS}
        for _ in range(BOOTSTRAP_DRAWS):
            bids=[rng.randrange(20) for _ in range(20)];v=[]
            for c in CELLS:
                ps=[p for b in bids for p in cache[c][b]];m=values(ps);v.append(m)
                for k in ('geomean','token_weighted','median'):estimates[c][k].append(m[k])
                estimates[c]['full_cap_geomean'].append(values([p for p in ps if p['both_full_cap']])['geomean'])
            for k in ('geomean','token_weighted'):overall[k].append(geomean([x[k] for x in v]))
    return ({c:{k:ci(estimates[c][k]) for k in ('geomean','token_weighted','median','full_cap_geomean')} for c in CELLS},
            {k:ci(overall[k]) for k in ('geomean','token_weighted')})


def describe(ps):
    if not ps:return dict(n=0,blocks=0,metrics=values([]))
    ratios=[p.get('throughput_ratio') for p in ps]
    return dict(n=len(ps),blocks=len({p['block'] for p in ps}),metrics=values(ps),
        ratio_min=min(x for x in ratios if x is not None) if any(x is not None for x in ratios) else None,
        ratio_max=max(x for x in ratios if x is not None) if any(x is not None for x in ratios) else None,
        undefined_groups=[p['group_id'] for p in ps if p.get('throughput_ratio') is None],
        changed_output_n=sum(not p['same_output'] for p in ps),different_length_n=sum(not p['same_length'] for p in ps),
        same_output_n=sum(p['same_output'] for p in ps),full_cap_n=sum(p['both_full_cap'] for p in ps),
        lengths={a:dict(min=min(p[a]['produced'] for p in ps),max=max(p[a]['produced'] for p in ps),mean=statistics.mean(p[a]['produced'] for p in ps)) for a in ('U','A')},
        finishes={a:dict(Counter(p[a]['finish'] for p in ps)) for a in ('U','A')},
        totals={a:{k:math.fsum(p[a][k] for p in ps) for k in ('produced','decode_ms','prompt_ms','wall_seconds','ttft_seconds')} for a in ('U','A')},
        work={a:{k:sum(p[a+'_work'][k] for p in ps) for k in ('rows','windows','offered','accepted','unused','terminal_extra')} for a in ('U','A')},
        paired_wall_delta_mean=statistics.mean(p['delta_wall_seconds'] for p in ps),
        paired_ttft_delta_mean=statistics.mean(p['delta_ttft_seconds'] for p in ps))


def analyze(results,manifest,manifest_file_sha=None,final_incomplete=False):
    audit,rawpairs=validate(results,manifest,manifest_file_sha)
    if not audit['matrix_complete'] and not final_incomplete:raise ValueError('Incomplete/invalid matrix: use --schema-only while active; only explicit --final-incomplete permits final partial artifacts.')
    pairs=[decorate(p) for p in rawpairs];normal=[p for p in pairs if p['status']=='complete_normal']
    intervals,overallci=boot(normal,audit['matrix_complete'])
    cells={}
    for c in CELLS:
        ps=[p for p in normal if p['cell']==c]
        cells[c]=dict(primary=describe(ps),primary_scope='all25 normal pairs' if audit['matrix_complete'] else 'AVAILABLE normal pairs only; partial descriptive, NOT full primary',
            planned=25,received_pairs=sum(p['U'] is not None and p['A'] is not None for p in pairs if p['cell']==c),
            bootstrap95=intervals[c],full_cap_secondary=describe([p for p in ps if p['both_full_cap']]),
            same_output_subset=describe([p for p in ps if p['same_output']]),
            order_strata={order:describe([p for p in ps if p['order']==order]) for order in ('UA','AU')},
            per_block=[dict(block=b,**describe([p for p in ps if p['block']==b])) for b in range(20)])
    overall={k:geomean([cells[c]['primary']['metrics'][k] for c in CELLS]) for k in ('geomean','token_weighted')}
    overall.update(bootstrap95=overallci,scope='descriptive four-cell equal geometric weighting; NOT a general application estimate')
    return dict(schema='independent-random100-v2',audit=audit,cells=cells,descriptive_overall=overall,raw_pairs=pairs,
        bootstrap=dict(unit='whole paired engine block',blocks=20,draws=BOOTSTRAP_DRAWS if audit['matrix_complete'] else 0,seed=BOOTSTRAP_SEED,method='percentile95 linear interpolation at (n-1)*q; common resampled block IDs across all cells/metrics'),
        engine_startup_records=[dict(block=x.get('block'),arm=x.get('arm'),startup_seconds=x.get('engine_startup_seconds')) for x in results.get('runs',[])],
        protocol='Prespecified method: ALL valid-pair geomean primary; controller median descriptive secondary only.',
        quality=dict(GSM_score=None,reason='Two fixed prompts only. No GSM questions scored; output equality/repetition are not semantic correctness.',repeat_template_conformity=None,template_reason='No independent pre-frozen template oracle supplied; do not derive one from U/A outputs.'),
        performance_passed=None,adoption_qualified=False,original_controller_decision=results.get('decision'),service_recovery_receipt=results.get('recovery'),
        limits=['20 dependence clusters, not100 independent tasks. Block independence/exchangeability is assumed; order strata and per-block observations remain visible.',
                'CIs are marginal95%, not simultaneous/familywise95%. Overall is descriptive for these four fixed cells.',
                'EOS, stop, output-length changes and slow samples retained. Token/s is generated-work throughput, not equal-work latency.',
                'Natural output-conditioned history/planner state can differ between arms despite identical prescribed input history.',
                'Full-cap and same-output subsets are selected by outcomes, secondary only. No survivor-only performance claim on incomplete runs.',
                'Rows use the frozen48-layer/10-expert counter geometry; they do not predict milliseconds.',
                'This parser checks stored receipts/records. ZIP/stdout/binary provenance needs the independent final artifact audit.',
                'Inference settings and the matrix are fixed in manifest.json.'])


def fmt(x,percent=False):
    if x is None:return 'undefined'
    return f'{100*(x-1):+.3f}%' if percent else f'{x:.6f}'

def markdown(r):
    a=r['audit'];lines=['# Random100 independent final analysis','',
        '**'+('Complete fixed matrix.' if a['matrix_complete'] else 'INCOMPLETE/INVALID: descriptive partial output only; no performance claim or inferential CI.')+'**',
        'Primary follows the prespecified method: all-normal-pair geometric mean, including EOS. Controller median is secondary. No GSM score; two fixed prompts only.','',
        '| Cell | Normal N / blocks | Geomean change | Block-bootstrap95% | Token-weighted change | Median change | Full-cap N | Same-output N |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for c,x in r['cells'].items():
        p=x['primary'];m=p['metrics'];ci_=x['bootstrap95']['geomean'];interval=fmt(ci_['lower'],True)+' .. '+fmt(ci_['upper'],True)
        lines.append(f"| {c} | {p['n']}/{p['blocks']} | {fmt(m['geomean'],True)} | {interval} | {fmt(m['token_weighted'],True)} | {fmt(m['median'],True)} | {x['full_cap_secondary']['n']} | {x['same_output_subset']['n']} |")
    o=r['descriptive_overall'];lines+=['',f"Equal-four-cell descriptive overall: {fmt(o['geomean'],True)},95% interval {fmt(o['bootstrap95']['geomean']['lower'],True)} .. {fmt(o['bootstrap95']['geomean']['upper'],True)}. Not a general-application claim.",
        '', '## Totals, order and secondary support']
    for c,x in r['cells'].items():
        p=x['primary'];lines+=['',f'### {c}']
        if not p['n']:lines+=['No normal pair available.'];continue
        for arm in ('U','A'):
            t=p['totals'][arm];lines.append(f"- {arm}: tokens={t['produced']:g}; decode={t['decode_ms']/1000:.6f}s; wall={t['wall_seconds']:.6f}s; rows={p['work'][arm]['rows']}; finishes={p['finishes'][arm]}; lengths={p['lengths'][arm]}.")
        for order,v in x['order_strata'].items():lines.append(f"- Order {order}: N={v['n']},blocks={v['blocks']}, geomean={fmt(v['metrics']['geomean'],True)}.")
        ci_=x['bootstrap95']['token_weighted'];lines.append(f"- Token-weighted95% CI: {fmt(ci_['lower'],True)} .. {fmt(ci_['upper'],True)}.")
        full=x['full_cap_secondary'];ci_=x['bootstrap95']['full_cap_geomean'];lines.append(f"- Full-cap secondary: N={full['n']}, geomean={fmt(full['metrics']['geomean'],True)},95% CI={fmt(ci_['lower'],True)} .. {fmt(ci_['upper'],True)}; undefined resamples={ci_['undefined_draws']}.")
        lines.append(f"- Changed output pairs={p['changed_output_n']}; different-length pairs={p['different_length_n']}; undefined log-ratio groups={p['undefined_groups']}.")
    lines+=['','## Completeness and limits',f"Runs={a['run_count']}/40, normal pairs={a['complete_normal_pairs']}/100; controller completed={a['controller_completed']}."]
    lines += ['- '+x for x in a['errors']+a['missing']+r['limits']]
    lines += ['', 'All100 planned pair entries, all received records (including warmups/errors), raw timing/work and original output IDs are retained in JSON. CSV contains raw pair diagnostics. No favorable filtering, winsorization, epsilon ratios or dropped bootstrap draws.']
    return '\n'.join(lines)+'\n'


def write_artifacts(r,out):
    out.mkdir(parents=True,exist_ok=True)
    names=('analysis.json','raw_pairs.json','raw_pairs.csv','RESULTS.md')
    if any((out/n).exists() for n in names):raise FileExistsError('refuse overwriting independent analysis artifacts')
    (out/'analysis.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (out/'raw_pairs.json').write_text(json.dumps(r['raw_pairs'],indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    fields=['group_id','block','slot','cell','order','seed','status','throughput_ratio','throughput_change_percent','same_output','same_length','both_full_cap','delta_tokens']
    fields += [arm+'_'+k for arm in ('U','A') for k in ('produced','finish','decode_ms','prompt_ms','wall_seconds','ttft_seconds','rows','windows','unused','terminal_extra')]
    with (out/'raw_pairs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for p in r['raw_pairs']:
            row={k:p.get(k) for k in fields}
            for arm in ('U','A'):
                req=p.get(arm) or {};work=p.get(arm+'_work',{})
                for k in ('produced','finish','decode_ms','prompt_ms','wall_seconds','ttft_seconds','rows','windows','unused','terminal_extra'):row[arm+'_'+k]=req.get(k,work.get(k))
            writer.writerow(row)
    (out/'RESULTS.md').write_text(markdown(r),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('results',type=Path);ap.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    ap.add_argument('--out-dir',type=Path,default=HERE);ap.add_argument('--schema-only',action='store_true');ap.add_argument('--final-incomplete',action='store_true');args=ap.parse_args()
    data=args.results.read_bytes();mb=args.manifest.read_bytes();results=json.loads(data.decode('utf-8-sig'));manifest=json.loads(mb.decode('utf-8-sig'))
    if args.schema_only:
        audit,_=validate(results,manifest,sha_bytes(mb));audit.pop('all_received_records',None)
        print(json.dumps(audit,indent=2,ensure_ascii=False));return
    r=analyze(results,manifest,sha_bytes(mb),args.final_incomplete)
    r['provenance']=dict(input=str(args.results.resolve()),input_sha256=sha_bytes(data),manifest=str(args.manifest.resolve()),manifest_sha256=sha_bytes(mb),
                         analyzer_sha256=sha_bytes(Path(__file__).read_bytes()),analysis_protocol_sha256=sha_bytes((HERE/'ANALYSIS-PROTOCOL.md').read_bytes()))
    write_artifacts(r,args.out_dir)
    print(json.dumps(dict(output=str(args.out_dir),matrix_complete=r['audit']['matrix_complete'],bootstrap_draws=r['bootstrap']['draws']),indent=2))
if __name__=='__main__':main()
