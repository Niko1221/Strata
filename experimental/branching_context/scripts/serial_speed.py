"""Bounded serial-request comparison: immutable API versus untouched serving.

Uses native stdio only. Timings describe whole paths, not isolated cache speedup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def compare(baseline, candidate):
    equal = bool(baseline) and len(baseline) == len(candidate) and all(
        (a['name'], a['prompt'], a['ids']) == (b['name'], b['prompt'], b['ids'])
        for a, b in zip(baseline, candidate))
    return dict(tokens_equal=equal, observed_latency_ratio=(
        sum(r['seconds'] for r in baseline)/sum(r['seconds'] for r in candidate)
        if equal and sum(r['seconds'] for r in candidate) > 0 else None),
        full_logit_equivalence=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--upstream-root', type=Path, required=True)
    ap.add_argument('--upstream-engine', type=Path, required=True)
    ap.add_argument('--modified-engine', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    sys.path.insert(0, str(a.upstream_root))
    from context_time_travel.real import BranchingModel, FreshOracle
    from context_time_travel.numerics import assert_logits_close
    from serve.server import StrataEngine
    p = json.loads(a.profile.read_text())
    if p['calibration_status'] != 'FROZEN':
        raise ValueError('Frozen numerical profile required')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    prefix = tuple(p['fixtures']['prefix'])
    x, y = tuple(p['fixtures']['suffix_a']), tuple(p['fixtures']['suffix_b'])
    requests = [('A',x), ('B',y), ('A-extended',x+y), ('B-extended',y+x),
                ('A-revisit',x), ('rewind-to-prefix',()), ('A-disk',x), ('B-evicted',y)]
    report = dict(schema=1,status='RUNNING',scope='One bounded serial run per path; no performance certification',
        profile_sha256=sha(a.profile),model_weights_sha256=p['model_weights_sha256'],
        model_contract_id=p['model_contract_id'],execution_profile_id=p['execution_profile_id'],
        native_build=p['native_build'],hardware=p['hardware'],
        script_sha256=sha(__file__),upstream_binary_sha256=sha(a.upstream_engine),
        modified_serving_binary_sha256=sha(a.modified_engine),sides={},oracle=[],
        limitations=['One run per path, fixed order; OS/expert cache warmth may affect timing',
            'Serving uses prefill=128; branching uses pinned token-level operator',
            'Branch API consumes and durably publishes the last generated token; serving may leave it pending',
            'Greedy output equality is weaker than full-logit equivalence',
            'Synthetic exact-token fixture, two generated tokens maximum per call',
            'Evaluation, grammar, reseeding and history GC are covered by the separate smoke, not this timing'])
    def save():
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    backend = oracle = sut = None
    try:
        # Ordinary engine settings match between the pristine and modified builds.
        args = list(p['execution']['oracle_command'][1:])
        for flag in ('--max-new',):
            if flag in args:
                i=args.index(flag);del args[i:i+2]
        args += ['--mtp-max-t','1','--adapt-swaps','0','--prompt-cache','6',
                 '--conversation-cache-mib','8192','--conversation-cache-slots','4',
                 '--conversation-cache-min-tokens','0','--spec-min-p','0']
        for label, engine_path in [('upstream-serving',a.upstream_engine),
                                   ('modified-serving',a.modified_engine)]:
            start=time.perf_counter()
            engine=StrataEngine(str(engine_path),args,cwd=str(a.upstream_root),
                log=str(a.output/(label+'.log')),env=dict(os.environ,**p['execution']['environment']))
            side=dict(startup_seconds=time.perf_counter()-start,args=args,engine_info=engine.info,requests=[])
            report['sides'][label]=side;save()
            try:
                for name,suffix in requests:
                    prompt=list(prefix+suffix)
                    start=time.perf_counter();first=None
                    ids=[]
                    for token in engine.generate(prompt,2,{'temperature':0,'repetition_penalty':1.0},threading.Event()):
                        if token is not None:
                            if first is None:first=time.perf_counter()-start
                            ids.append(token)
                    row=dict(name=name,prompt=prompt,ids=ids,seconds=time.perf_counter()-start,
                             first_token_seconds=first,engine_last=dict(engine.last))
                    if not ids: raise AssertionError('Missing generated output')
                    side['requests'].append(row);save()
                    print(label,name,round(row['seconds'],3),ids,flush=True)
            finally:
                engine.close()
        start=time.perf_counter()
        backend=BranchingModel.load_backend(p)
        sut=BranchingModel(a.output/'store',backend,p)
        side=dict(startup_seconds=time.perf_counter()-start,requests=[])
        report['sides']['branching-api']=side;save()
        root=None;tips={};checks=[];root_bytes=None
        for name,suffix in requests:
            start=time.perf_counter()
            before=dict(backend._refresh_stats())
            if root is None:
                root=sut.create(prefix);root_bytes=sut.snapshot_bytes(root)
                sut.store.fork('main',root)
            if name=='A-disk':
                sut.cache.clear();backend.release_checkpoints()
            if name=='B-evicted':
                sut.evict_all()
            if suffix not in tips:
                # Extend from the already materialized A/B ancestor when possible.
                parent=next((s for s in (x,y) if s in tips and suffix[:len(s)]==s),())
                tip=sut.feed(tips[parent] if parent else root,suffix[len(parent):])
                tips[suffix]=tip
                sut.store.fork(name,tip)
            tip=tips[suffix]
            # Resolve the exact checkpoint before attaching a new controller scope.
            sut.next_logits(tip)
            head=sut.store.head('main')
            moved=sut.store.move('main',head.revision,tip,'publish-'+name)
            if sut.store.move('main',head.revision,tip,'publish-'+name)!=moved:
                raise AssertionError('Retry receipt changed')
            scoped=sut.start_generation(tip,{'temperature':0},'serial-speed')
            first=None;ids=[];last=scoped
            for last,token in sut.stream(scoped,2):
                if first is None:first=time.perf_counter()-start
                ids.append(token)
            elapsed=time.perf_counter()-start
            after=dict(backend._refresh_stats())
            if name=='A-disk' and after['loaded_file_bytes']<=before['loaded_file_bytes']:
                raise AssertionError('Disk checkpoint was not actually loaded')
            if name=='B-evicted' and after['forward_tokens']-before['forward_tokens']!=len(prefix+suffix)+len(ids):
                raise AssertionError('Cold replay work differs')
            if sut.history(last)!=prefix+suffix+tuple(ids) or sut.snapshot_bytes(root)!=root_bytes:
                raise AssertionError('Consumed history or immutable prefix changed')
            row=dict(name=name,prompt=list(prefix+suffix),ids=ids,seconds=elapsed,
                first_token_seconds=first,counters_before=before,counters_after=after,
                snapshot=last,consumed=len(sut.history(last)))
            side['requests'].append(row)
            checks.append((name,sut.history(last),tuple(sut.next_logits(last))))
            if name=='A-revisit':
                begin=time.perf_counter();files=sut.checkpoint(tips[x])
                report['checkpoint']=dict(seconds=time.perf_counter()-begin,
                    files={str(v):dict(bytes=v.stat().st_size,sha256=sha(v)) for v in files.values()})
            save();print('branching-api',name,round(elapsed,3),ids,flush=True)
        side['final_counters']=dict(backend._refresh_stats())
        # Oracle executions are deliberately outside timed requests and after engine teardown.
        backend.close();backend=None
        oracle=FreshOracle(p)
        for name,history,logits in checks:
            expected=oracle.forward(history)
            result=assert_logits_close(logits,expected,atol=p['atol'],rtol=p['rtol'])
            report['oracle'].append(dict(name=name,tokens=len(history),
                max_absolute_error=result.max_absolute_error,computed_tokens=oracle.last['computed_tokens']))
            save();print('fresh-oracle',name,result.max_absolute_error,flush=True)
        report['comparisons']={label:compare(report['sides']['upstream-serving']['requests'],v['requests'])
            for label,v in report['sides'].items() if label!='upstream-serving'}
        report['status']='MEASURED'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:
        if backend is not None:backend.close()
        if oracle is not None:oracle.close()
        save()


if __name__=='__main__':main()
