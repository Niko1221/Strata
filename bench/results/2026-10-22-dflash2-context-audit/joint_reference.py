"""Read-only evaluation of the published BF16 implementation; no optimization/training."""
from pathlib import Path
import sys,json,importlib.util,time
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import Qwen3Config
root=Path.cwd();out=(root/'../Strata-data/dflash2-context-audit-20261022/joint').resolve();official=root/'../Strata-data/dflash2-official'
sys.path.insert(0,str(root/'tools'));from gguf_reader import GGUFFile
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(1234)
spec=importlib.util.spec_from_file_location('published_dflash',official/'reference_model.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
cfg=Qwen3Config(**json.loads((official/'config.json').read_text()));cfg._attn_implementation='sdpa'
old=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16);model=ref.DFlash2DraftModel(cfg);torch.set_default_dtype(old)
state=load_file(str(official/'model.safetensors'))
for n in ('predecessor_codebook','successor_codebook'):
 k='candidate_selector.'+n
 if k in state:state[k+'.weight']=state.pop(k)
model.load_state_dict(state,strict=True);del state;model=model.cuda().eval().requires_grad_(False)
def tensor(path,name,fp32=False):
 g=GGUFFile(path);t=next(t for t in g.tensors if t.name==name)
 if t.type_id==8:
  raw=np.memmap(path,mode='r',dtype='u1',offset=g.data_start+t.offset,shape=(t.shape[1],t.shape[0]//32,34))
  scale=np.array(raw[:,:,:2]).view('<f2').astype(np.float32)
  values=np.array(raw[:,:,2:]).view('i1').astype(np.float32)*scale
  return torch.from_numpy(values.reshape(t.shape[1],t.shape[0])).to('cuda',torch.float32 if fp32 else torch.bfloat16)
 assert t.type_id==30
 raw=np.memmap(path,mode='r',dtype='<u2',offset=g.data_start+t.offset,shape=tuple(reversed(t.shape)))
 return torch.from_numpy(np.array(raw)).view(torch.bfloat16).cuda()
base=root/'../Strata-data/models/Maximilian228--Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata'
head=tensor(base/'qwen-nvidia-nvfp4-dense.gguf','output.weight',True)
head16=head.to(torch.bfloat16);embedding=tensor(base/'token-embd-bf16.gguf','token_embd.weight')

native=json.loads((out/'native-dense.json').read_text());report={}
with torch.inference_mode():
 for task in ['code_short','code_long','cache_long']:
  m=json.loads((out/(task+'.json')).read_text());n=len(m['prompt']);ids=torch.tensor(m['prompt']+m['output'],device='cuda');taps=torch.from_numpy(np.load(out/(task+'-taps.npy'))).to('cuda',torch.bfloat16)
  result=[]
  for trace in native[task]['traces']:
   p=trace['position'];lo=max(0,p-2048);noise=torch.full((1,8),model.mask_token_id,device='cuda',dtype=torch.long);noise[0,0]=ids[p]
   hidden=model(position_ids=torch.arange(lo,p+8,device='cuda')[None],noise_embedding=F.embedding(noise,embedding),target_hidden=taps[lo:p][None],use_cache=False)[:,1:]
   block=hidden.float().reshape(1,7,-1,32);scale=block.abs().amax(-1,keepdim=True)/127;safe=scale.clamp_min(1e-20)
   quant=(block/safe).abs().add(.5).floor().mul((block/safe).sign()).clamp(-127,127)*scale.to(torch.float16).float()
   logits=F.linear(quant.reshape_as(hidden),head)
   pred,candidates,_=model.candidate_selector.select(hidden,logits,ids[p:p+1],0)
   pred=pred[0].tolist();gold=ids[p+1:min(p+8,len(ids))].tolist()
   result.append(dict(position=p,native=trace['ids'],reference=pred,gold=gold,first_gold_in_top16=bool((candidates[0,0]==ids[p+1]).any().item())))
  summary=dict(positions=len(result),first_native_correct=sum(r['native'][0]==r['gold'][0] for r in result),first_reference_correct=sum(r['reference'][0]==r['gold'][0] for r in result),first_disagreements=sum(r['native'][0]!=r['reference'][0] for r in result),all_rows_disagreements=sum(a!=b for r in result for a,b in zip(r['native'],r['reference'])),first_gold_in_top16=sum(r['first_gold_in_top16'] for r in result))
  report[task]=dict(summary=summary,rows=result);(out/'reference-dense.json').write_text(json.dumps(report,indent=2));print(task,summary,flush=True)
