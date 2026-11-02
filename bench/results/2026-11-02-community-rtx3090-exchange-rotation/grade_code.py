#!/usr/bin/env python3
# Fixed task contracts; generated modules run in a separate resource-limited process.
import ast,builtins,json,math,random,re,resource,sys
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(5,5));resource.setrlimit(resource.RLIMIT_AS,(384*1024**2,384*1024**2))
rec=json.loads(sys.stdin.read() if sys.argv[1]=='-' else Path(sys.argv[1]).read_text());blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',rec['content'],re.S);code=blocks[0] if blocks else rec['content'];tree=ast.parse(code)
allowed_modules={'math','json','typing','collections','functools','heapq'}
for node in ast.walk(tree):
 if isinstance(node,ast.Import) and any(x.name not in allowed_modules for x in node.names):raise ValueError('unexpected import')
 if isinstance(node,ast.ImportFrom) and (node.module not in allowed_modules or node.level):raise ValueError('unexpected from import')
 if isinstance(node,ast.Name) and node.id.startswith('__'):raise ValueError('dunder forbidden')
 if isinstance(node,ast.Attribute) and node.attr.startswith('_'):raise ValueError('private attributes forbidden')
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {'open','eval','exec','compile','getattr','setattr','delattr','globals','locals','vars','breakpoint','input'}:raise ValueError('unsafe call')
safe={k:getattr(builtins,k) for k in ['abs','all','any','bool','dict','enumerate','filter','float','int','isinstance','issubclass','iter','len','list','map','max','min','next','range','reversed','round','set','sorted','str','sum','tuple','type','zip','OverflowError','ArithmeticError','ZeroDivisionError','ValueError','TypeError','Exception','KeyError','StopIteration','__build_class__']}
def import_safe(name,*args,**kwargs):
 if name not in allowed_modules:raise ImportError(name)
 return __import__(name,*args,**kwargs)
safe['__import__']=import_safe;ns={'__builtins__':safe,'__name__':'candidate'};exec(compile(tree,'candidate','exec'),ns)
tests=[]
def check(name,fn):
 try:fn();tests.append({'name':name,'passed':True})
 except Exception as e:tests.append({'name':name,'passed':False,'error':repr(e)})
def eq(got,want):assert got==want,(got,want)
def raises(t,fn):
 try:fn()
 except t:return
 raise AssertionError('expected '+t.__name__)
id=rec.get('case_id',rec['id'].split('-')[-1])
if id=='code_merge':
 f=ns['merge_intervals']
 for i,(a,b) in enumerate([([],[]),([[2,3],[1,2]],[[1,3]]),([[1,2],[3,4]],[[1,2],[3,4]]),([[5,5],[5,5]],[[5,5]]),([[1,10],[2,3],[-5,-2]], [[-5,-2],[1,10]])]):check('fixed-'+str(i),lambda a=a,b=b:eq(f(a),b))
 check('invalid',lambda:raises(ValueError,lambda:f([[3,1]])))
 def immutable():
  a=[[5,7],[1,3],[2,6]];old=json.loads(json.dumps(a));f(a);eq(a,old)
 check('immutable',immutable)
 rng=random.Random(7301)
 for i in range(60):
  a=[sorted([rng.randrange(-20,21),rng.randrange(-20,21)]) for _ in range(rng.randrange(20))];wanted=[]
  for lo,hi in sorted(a):
   if wanted and lo<=wanted[-1][1]:wanted[-1][1]=max(hi,wanted[-1][1])
   else:wanted.append([lo,hi])
  check('random-'+str(i),lambda a=a,w=wanted:eq(f(a),w))
elif id=='code_retry':
 f=ns['retry_delays']
 for name,args,want in [('empty',(0,1,8),[]),('normal',(5,.5,4),[.5,1,2,4,4]),('zero-base',(5,0,4),[0]*5),('zero-cap',(5,1,0),[0]*5),('base-over-cap',(3,10,4),[4]*3),('large',(10000,.5,4),[.5,1,2]+[4]*9997)]:check(name,lambda args=args,w=want:eq(f(*args),w))
 for value in [True,False,1.5,'3',None]:check('type-'+str(value),lambda v=value:raises(TypeError,lambda:f(v,1,2)))
 for args in [(-1,1,2),(1,-1,2),(1,1,-2)]:check('negative-'+str(args),lambda args=args:raises(ValueError,lambda:f(*args)))
elif id=='code_log':
 f=ns['summarize']
 def obj(s,status,lat):return json.dumps({'service':s,'status':status,'latency_ms':lat})
 check('empty',lambda:eq(f([]),[]));check('basic',lambda:eq(f([obj('z',500,12),obj('a',200,8),obj('z',201,4)]),[{'service':'a','count':1,'errors':0,'p95_ms':8},{'service':'z','count':2,'errors':1,'p95_ms':12}]))
 invalid=['',' ', '{','null','[]','3','"str"','{}',obj('',200,3),obj('a',True,3),obj('a',99,3),obj('a',600,3),obj('a',200,-1),obj('a',200,True),obj('a',200,'3'),obj('a',200,None),obj('a',200.0,3)]
 check('invalid-lines',lambda:eq(f(invalid),[]))
 for n in [1,2,19,20,21,40]:check('nearest-rank-'+str(n),lambda n=n:eq(f(obj('x',200,i) for i in range(1,n+1)),[{'service':'x','count':n,'errors':0,'p95_ms':math.ceil(.95*n)}]))
 check('extra-fields',lambda:eq(f(['{"service":"a","status":599,"latency_ms":0,"foo":42}']),[{'service':'a','count':1,'errors':1,'p95_ms':0}]))
else:raise ValueError(id)
if '--exploratory' in sys.argv and id=='code_retry':check('posthoc-large-zero-base',lambda:eq(ns['retry_delays'](10000,0,4),[0]*10000))
print(json.dumps({'grader_revision':'safe-arithmetic-exceptions-v2','id':id,'passed':sum(t['passed'] for t in tests),'total':len(tests),'all_passed':all(t['passed'] for t in tests),'tests':tests},ensure_ascii=False))
