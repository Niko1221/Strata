"""Local ctypes port of Strata tune_hipblaslt.cpp, using the packaged DLLs unchanged.

Only public C APIs and the DLL's exported getIndexFromAlgo(reference) are called.
Each candidate must pass the upstream numerical and padding gates before timing.
No kernels, drivers, foreign-GPU solution IDs, or model tensors are modified.
"""
import argparse,ctypes as c,json,os,pathlib,statistics,time,hashlib
import numpy as np
from inspect_dll import exports
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'optimization'/'calibration';OUT.mkdir(parents=True,exist_ok=True)
ENGINE=ROOT/'Strata'/'engine';BIN=ENGINE/'rocm'/'bin'
os.environ['HIP_VISIBLE_DEVICES']='1'
os.environ['PATH']=str(ENGINE)+os.pathsep+str(BIN)+os.pathsep+os.environ['PATH']
dirs=[os.add_dll_directory(str(x)) for x in [ENGINE,BIN]]
P=c.c_void_p;I=c.c_int;Z=c.c_size_t;U=c.c_uint64
class Algo(c.Structure):_fields_=[('data',c.c_ubyte*16),('workspace',Z)]
class Heuristic(c.Structure):_fields_=[('algo',Algo),('workspace',Z),('state',I),('waves',c.c_float),('reserved',I*4)]
assert c.sizeof(Algo)==24 and c.sizeof(Heuristic)==56
def bind(lib,name,types):
    f=lib[name];f.argtypes=types;f.restype=I;return f
def check(code):
    if code:raise RuntimeError('GPU API status '+str(code))
hip=c.CDLL(str(ENGINE/'amdhip64_7.dll'));lt=c.CDLL(str(BIN/'libhipblaslt.dll'));blas=c.CDLL(str(BIN/'hipblas.dll'))
setdev=bind(hip,'hipSetDevice',[I]);sync=bind(hip,'hipDeviceSynchronize',[])
malloc=bind(hip,'hipMalloc',[c.POINTER(P),Z]);free=bind(hip,'hipFree',[P]);copy=bind(hip,'hipMemcpy',[P,P,Z,I])
evcreate=bind(hip,'hipEventCreate',[c.POINTER(P)]);evrecord=bind(hip,'hipEventRecord',[P,P]);evsync=bind(hip,'hipEventSynchronize',[P]);evms=bind(hip,'hipEventElapsedTime',[c.POINTER(c.c_float),P,P]);evfree=bind(hip,'hipEventDestroy',[P])
create=bind(lt,'hipblasLtCreate',[c.POINTER(P)]);destroy=bind(lt,'hipblasLtDestroy',[P]);version=bind(lt,'hipblasLtGetVersion',[P,c.POINTER(I)])
archfn=bind(lt,'hipblasLtGetArchName',[c.POINTER(c.c_char_p)])
desc=bind(lt,'hipblasLtMatmulDescCreate',[c.POINTER(P),I,I]);attr=bind(lt,'hipblasLtMatmulDescSetAttribute',[P,I,P,Z]);descfree=bind(lt,'hipblasLtMatmulDescDestroy',[P])
layout=bind(lt,'hipblasLtMatrixLayoutCreate',[c.POINTER(P),I,U,U,c.c_int64]);layoutfree=bind(lt,'hipblasLtMatrixLayoutDestroy',[P])
prefcreate=bind(lt,'hipblasLtMatmulPreferenceCreate',[c.POINTER(P)]);prefattr=bind(lt,'hipblasLtMatmulPreferenceSetAttribute',[P,I,P,Z]);preffree=bind(lt,'hipblasLtMatmulPreferenceDestroy',[P])
heur=bind(lt,'hipblasLtMatmulAlgoGetHeuristic',[P,P,P,P,P,P,P,I,c.POINTER(Heuristic),c.POINTER(I)])
matmul=bind(lt,'hipblasLtMatmul',[P,P,P,P,P,P,P,P,P,P,P,P,c.POINTER(Algo),P,Z,P])
getindex=bind(lt,next(x for x in exports(BIN/'libhipblaslt.dll') if 'getIndexFromAlgo@' in x),[c.POINTER(Algo)])
blascreate=bind(blas,'hipblasCreate',[c.POINTER(P)]);blasfree=bind(blas,'hipblasDestroy',[P])
gemm=bind(blas,'hipblasGemmEx',[P,I,I,I,I,I,P,P,I,I,P,I,I,P,P,I,I,I,I])
class Buffer:
    def __init__(self,size):self.p=P();check(malloc(c.byref(self.p),size))
    def close(self):
        if self.p:check(free(self.p));self.p=P()
def timed(f):
    a=P();b=P();check(evcreate(c.byref(a)));check(evcreate(c.byref(b)))
    try:
        check(evrecord(a,None));check(f());check(evrecord(b,None));check(evsync(b));check(sync());ms=c.c_float();check(evms(c.byref(ms),a,b));return ms.value
    finally:check(evfree(a));check(evfree(b))
def hostbits(shape,bf16,rng):
    x=rng.uniform(-1,1,size=shape).astype(np.float32)
    if bf16:
        u=x.view(np.uint32);return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
    return x.astype(np.float16).view(np.uint16)
def run_case(shape,h,b,caseid,log):
    dtype,t,n,k,ldy=shape;itype=14 if dtype=='bf16' else 2
    rng=np.random.default_rng(0x5A17C3+caseid)
    ha=hostbits(k*n,dtype=='bf16',rng);hb=hostbits(k*t,dtype=='bf16',rng)
    canary=np.full((t,ldy),123456.25,dtype=np.float32)
    buffers=[Buffer(ha.nbytes),Buffer(hb.nbytes),Buffer(canary.nbytes),Buffer(canary.nbytes),Buffer(32*2**20)]
    da,db,dc,dy,dws=[x.p for x in buffers];op=P();ad=P();bd=P();cd=P();pref=P()
    try:
        for dst,src in [(da,ha),(db,hb),(dc,canary),(dy,canary)]:check(copy(dst,P(src.ctypes.data),src.nbytes,1))
        check(desc(c.byref(op),2,0));ta=I(112);tb=I(111)
        check(attr(op,0,c.byref(ta),4));check(attr(op,1,c.byref(tb),4))
        check(layout(c.byref(ad),itype,k,n,k));check(layout(c.byref(bd),itype,k,t,k));check(layout(c.byref(cd),0,n,t,ldy))
        check(prefcreate(c.byref(pref)));ws=U(32*2**20);check(prefattr(pref,1,c.byref(ws),8))
        alpha=c.c_float(1);beta=c.c_float(0)
        base=lambda:gemm(b,112,111,n,t,k,c.byref(alpha),da,itype,k,db,itype,k,c.byref(beta),dc,0,ldy,2,160)
        for _ in range(2):check(base());check(sync())
        base_times=[timed(base) for _ in range(3)]
        ref=np.empty_like(canary);check(copy(P(ref.ctypes.data),dc,ref.nbytes,2))
        hs=(Heuristic*16)();count=I();code=heur(h,op,ad,bd,cd,cd,pref,16,hs,c.byref(count))
        meta=dict(dtype=dtype,T=t,N=n,K=k,ldy=ldy,base_ms=base_times,base_median_ms=statistics.median(base_times),heuristic_status=code,count=count.value)
        log.write(json.dumps(dict(kind='shape',**meta))+'\n');log.flush()
        if code or count.value<=0:return None
        best=None
        for i in range(min(count.value,16)):
            entry=hs[i]
            if entry.state or entry.workspace>ws.value:continue
            algo=entry.algo;solution=getindex(c.byref(algo))
            run=lambda:matmul(h,op,c.byref(alpha),da,ad,db,bd,c.byref(beta),dy,cd,dy,cd,c.byref(algo),dws,ws,None)
            check(copy(dy,P(canary.ctypes.data),canary.nbytes,1));code=run()
            if code:
                log.write(json.dumps(dict(kind='unsupported',solution=solution,status=code,**meta))+'\n');log.flush();continue
            check(sync());got=np.empty_like(canary);check(copy(P(got.ctypes.data),dy,got.nbytes,2))
            finite=bool(np.isfinite(got[:,:n]).all());d=got[:,:n].astype(np.float64)-ref[:,:n]
            l2=float(np.linalg.norm(d.ravel())/max(np.linalg.norm(ref[:,:n].astype(np.float64).ravel()),1e-150))
            maxabs=float(np.max(np.abs(d)));padding=int(np.count_nonzero(got[:,n:]!=123456.25))
            accuracy=finite and l2<=1e-4 and maxabs<=1e-2 and padding==0
            row=dict(kind='candidate',solution_id=solution,heuristic_index=i,relative_l2=l2,max_abs=maxabs,padding_writes=padding,accuracy_pass=accuracy,algo_hex=bytes(algo.data).hex(),**meta)
            del d,got
            if accuracy:
                for _ in range(2):check(run());check(sync())
                times=[timed(run) for _ in range(3)];row.update(lt_ms=times,lt_median_ms=statistics.median(times),speedup=meta['base_median_ms']/statistics.median(times))
                if best is None or row['lt_median_ms']<best['lt_median_ms']:best=row
            log.write(json.dumps(row)+'\n');log.flush()
        print(json.dumps({**best,'kind':'best'} if best else dict(kind='no_valid_candidate',**meta)),flush=True)
        return best
    finally:
        for fn,obj in [(preffree,pref),(layoutfree,cd),(layoutfree,bd),(layoutfree,ad),(descfree,op)]:
            if obj:check(fn(obj))
        for x in buffers:x.close()
def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');p.add_argument('--quick',action='store_true');a=p.parse_args()
    if not a.probe and not (ROOT/'pipeline-hold.json').exists():raise RuntimeError('Wait for evaluation pipeline to stop')
    check(setdev(0));h=P();b=P();check(create(c.byref(h)));check(blascreate(c.byref(b)))
    try:
        ver=I();check(version(h,c.byref(ver)));arch=c.c_char_p();check(archfn(c.byref(arch)));archstr=arch.value.decode().split(':')[0]
        meta=dict(arch=archstr,hipblaslt_version=ver.value,library_sha256=hashlib.sha256((BIN/'libhipblaslt.dll').read_bytes()).hexdigest(),at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),method='public C APIs + exported getIndexFromAlgo; port of pinned upstream tool',workspace_mib=32,repetitions=3)
        assert archstr=='gfx1101' and ver.value==100500,meta
        (OUT/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
        if a.probe:return
        shapes=[]
        for line in (ROOT/'Strata'/'tools'/'hip'/'gfx1100-hipblaslt-100500.txt').read_text().splitlines()[2:]:
            dtype,n,k,ldy,t,_=line.split()
            # Only reuse geometry, never another GPU's solution ID.
            for bucket in [int(t),6144,1024]:
                row=(dtype,bucket,int(n),int(k),int(ldy))
                if row not in shapes:shapes.append(row)
        for bucket in [64,256,1024,4096,6144,8192]:shapes.append(('f16',bucket,1280,2560,1280))
        shapes.sort(key=lambda x:x[1]*x[2]*x[3],reverse=True)
        if a.quick:shapes=shapes[:1]
        bests=[]
        with (OUT/('quick.jsonl' if a.quick else 'candidates.jsonl')).open('w') as log:
            for i,s in enumerate(shapes):
                result=run_case(s,h,b,i,log)
                if result:bests.append(result)
        (OUT/('quick-best.json' if a.quick else 'best.json')).write_text(json.dumps(bests,indent=2))
        text='# Calibrated locally; geometry reused, solution IDs measured on RX 7800 XT\nSTRATA_HIPBLASLT_TUNING_V1 gfx1101 100500\n'
        text+=''.join(f"{r['dtype']} {r['N']} {r['K']} {r['ldy']} {r['T']} {r['solution_id']}\n" for r in bests)
        (OUT/('quick-table.txt' if a.quick else 'gfx1101-hipblaslt-100500.txt')).write_text(text)
    finally:check(destroy(h));check(blasfree(b))
if __name__=='__main__':main()
