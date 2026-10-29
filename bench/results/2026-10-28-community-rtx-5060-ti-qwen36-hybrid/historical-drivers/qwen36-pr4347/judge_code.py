"""Bounded child-process checker for four tiny, pure generated functions."""
import ast,json,re,sys
payload=json.load(sys.stdin);text=payload['text'].strip()
fence=re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```',text,re.S)
if fence:text=fence.group(1)
try:
    if len(text)>8192:raise ValueError('code too large')
    tree=ast.parse(text)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!='solve':
        raise ValueError('expected one solve function')
    fn=tree.body[0]
    if fn.decorator_list:raise ValueError('decorators refused')
    calls={'min':min,'max':max,'sum':sum,'abs':abs,'len':len,'sorted':sorted,'reversed':reversed,
           'list':list,'tuple':tuple,'int':int,'float':float,'str':str,'bool':bool}
    denied=(ast.Import,ast.ImportFrom,ast.Attribute,ast.ClassDef,ast.Lambda,ast.Global,ast.Nonlocal,
            ast.While,ast.With,ast.AsyncWith,ast.AsyncFor,ast.Try,ast.Raise,ast.Delete,ast.Yield,ast.YieldFrom,ast.Pow)
    if sum(isinstance(x,ast.FunctionDef) for x in ast.walk(tree))!=1:raise ValueError('nested function')
    for node in ast.walk(tree):
        if isinstance(node,denied):raise ValueError('unsupported construct: '+type(node).__name__)
        if isinstance(node,ast.Name) and '__' in node.id:raise ValueError('dunder refused')
        if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id not in calls):
            raise ValueError('unsupported call')
        if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)) and abs(node.value)>100000:
            raise ValueError('oversized numeric constant')
    namespace={'__builtins__':calls}
    exec(compile(tree,'generated-pure-function','exec'),namespace)
    actual=[namespace['solve'](*args) for args in payload['inputs']]
    print(json.dumps({'passed':actual==payload['expected'],'actual':actual}))
except BaseException as exc:print(json.dumps({'passed':False,'error':str(exc)}))
