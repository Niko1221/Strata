import sys,json,traceback
from pathlib import Path
r=Path('<POC_ROOT>');sys.path.insert(0,str(r/'bench/ascend_a2_s2/moe'));import probe as p
p.torch.npu.set_device(0);p.torch_npu.npu.matmul.allow_hf32=False
report={'passed':False,'quant':[]}
try:
 rng=p.np.random.default_rng(20261021)
 xs=[p.np.zeros((1,640),p.np.float32)]+[rng.normal(size=(t,2560)).astype(p.np.float32) for t in (1,2,4,8)]
 tie=p.np.zeros((1,64),p.np.float32);tie[0,:7]=[127,.5,-.5,1.5,-1.5,126.5,-126.5];xs.append(tie)
 for x in xs:
  q,s=p.quant_npu(p.nd(p.torch.from_numpy(x).to('npu:0')));cq,cs=p.quant_cpu(x);a={'shape':list(x.shape),'codes_equal':bool(p.np.array_equal(p.cpu(q),cq)),'scales_equal':bool(p.np.array_equal(p.cpu(s),cs))};report['quant'].append(a);assert a['codes_equal'] and a['scales_equal'],a
 logits,ids,scores=p.route_npu(p.torch.zeros((2,2560),device='npu:0'),p.torch.zeros((512,2560),device='npu:0'));report['tie_ids']=p.cpu(ids).tolist();assert report['tie_ids']==[list(range(10))]*2
 report['passed']=True
except Exception as e:report['error']=repr(e);traceback.print_exc()
finally:
 (r/'real-moe-evidence/precheck-1.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
