import argparse
import ast
import importlib.util
import json
import statistics
import time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--model',choices=['Q2_0','IQ2_XS'],required=True);p.add_argument('--out',type=Path);a=p.parse_args()
root=(a.out or Path(r'<OPS_ROOT>\results')/a.model.lower())/'code-task'
# These small algorithm modules must not access the filesystem/network or launch processes.
for path in root.glob('*.py'):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            modules=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module]
            assert all(x in ('collections','typing','re') for x in modules), modules
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            assert node.func.id not in ('open','eval','exec','compile','__import__','getattr','setattr'),node.func.id
def load(name):
    spec=importlib.util.spec_from_file_location('eval_'+name,root/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
s,c,r=load('settings'),load('cache'),load('records')
results=[]
def check(name,fn):
    try:fn();results.append(dict(name=name,passed=True))
    except Exception as e:results.append(dict(name=name,passed=False,error=repr(e)))
def eq(actual,expected):assert actual==expected,(actual,expected)
def reject(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Expected ValueError')
for value,expected in [(None,30),('',30),('  ',30),(0,0),(12,12),(' 007 ',7)]:
    check('timeout-valid-'+repr(value),lambda v=value,e=expected:eq(s.parse_timeout(v),e))
check('timeout-default',lambda:eq(s.parse_timeout(None,9),9))
for value in [True,False,1.5,-1,'-1','+2','2.0','１２',[],{}]:
    check('timeout-invalid-'+repr(value),lambda v=value:reject(lambda:s.parse_timeout(v)))
for value in [0,-1,True,1.5,'2']:
    check('cache-capacity-'+repr(value),lambda v=value:reject(lambda:c.LRUCache(v)))
def cache_promote():
    q=c.LRUCache(2);q.put('a',1);q.put('b',2);eq(q.get('a'),1);q.put('c',3);eq(q.get('b','missing'),'missing');eq(q.get('a'),1)
def cache_update():
    q=c.LRUCache(2);q.put('a',1);q.put('b',2);q.put('a',4);eq(q.get('b'),2);eq(q.get('a'),4)
def cache_falsy():
    q=c.LRUCache(2);q.put('a',None);q.put('b',0);eq(q.get('a','missing'),None);q.put('c',3);eq(q.get('b','missing'),'missing');eq(q.get('a','missing'),None)
def cache_one():
    q=c.LRUCache(1);q.put('a',1);q.put('a',2);eq(q.get('a'),2);q.put('b',3);eq(q.get('a','absent'),'absent')
for fn in [cache_promote,cache_update,cache_falsy,cache_one]:check(fn.__name__,fn)
def records_first():
    x=[{'id':1,'v':'first'},{'id':2},{'id':1,'v':'later'}];z=r.stable_unique(iter(x));eq(z,x[:2]);assert z[0] is x[0]
def records_missing():
    x=[{}, {'v':3}, {'id':None}, {'id':None,'v':4}, {}];z=r.stable_unique(iter(x));eq(z,[x[0],x[1],x[2],x[4]]);assert all(a is b for a,b in zip(z,[x[0],x[1],x[2],x[4]]));eq(x,[{}, {'v':3}, {'id':None}, {'id':None,'v':4}, {}])
check('records-first-generator-identity',records_first)
check('records-missing-none-mutation',records_missing)
check('records-empty',lambda:eq(r.stable_unique([]),[]))
summary=dict(model=a.model,passed=sum(x['passed'] for x in results),total=len(results),checks=results,scope='One Claude Code repair task across three algorithm modules; deterministic contract checks, not a broad coding benchmark.')
(root.parent/'code-quality.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
rows=[]
for capacity in (128,4096,65536):
    samples=[]
    for trial in range(3):
        cache=c.LRUCache(capacity)
        for i in range(capacity):cache.put(i,i)
        started=time.perf_counter()
        for i in range(32768):cache.put(capacity+i,i)
        samples.append((time.perf_counter()-started)*1e9/32768)
    rows.append(dict(capacity=capacity,median_ns_per_eviction=statistics.median(samples),samples=samples))
ratio=rows[-1]['median_ns_per_eviction']/rows[0]['median_ns_per_eviction']
scaling=dict(rows=rows,large_to_small_ratio=ratio,flag_for_complexity_review=ratio>5,
    method='3 repeats, 32768 new insertions each, prefilled cache; measured implementation scaling, not a proof of complexity.')
(root.parent/'code-cache-scaling.json').write_text(json.dumps(scaling,indent=2),encoding='utf-8')
print(json.dumps(scaling,indent=2))
