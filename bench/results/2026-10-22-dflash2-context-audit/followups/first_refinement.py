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


reference=json.loads((out/'reference-dense.json').read_text());report={}
with torch.inference_mode():
 for task in ['code_short','code_long','cache_long']:
  m=json.loads((out/(task+'.json')).read_text());n=len(m['prompt']);ids=torch.tensor(m['prompt']+m['output'],device='cuda');taps=torch.from_numpy(np.load(out/(task+'-taps.npy'))).to('cuda',torch.bfloat16);rows=[]
  for saved in reference[task]['rows']:
   p=saved['position'];lo=max(0,p-2048);first=saved['reference'][0]
   noise=torch.full((1,8),model.mask_token_id,device='cuda',dtype=torch.long);noise[0,0]=ids[p];noise[0,1]=first
   hidden=model(position_ids=torch.arange(lo,p+8,device='cuda')[None],noise_embedding=F.embedding(noise,embedding),target_hidden=taps[lo:p][None],use_cache=False)[:,2:]
   block=hidden.float().reshape(1,6,-1,32);scale=block.abs().amax(-1,keepdim=True)/127;safe=scale.clamp_min(1e-20)
   quant=(block/safe).abs().add(.5).floor().mul((block/safe).sign()).clamp(-127,127)*scale.to(torch.float16).float();logits=F.linear(quant.reshape_as(hidden),head)
   pred,_,_=model.candidate_selector.select(hidden,logits,torch.tensor([first],device='cuda'),0);pred=[first]+pred[0].tolist();prefixes=[]
   for tokens in [saved['reference'],pred]:
    length=0
    for a,b in zip(tokens,saved['gold']):
     if a!=b:break
     length+=1
    prefixes.append(length)
   rows.append(dict(position=p,original=saved['reference'],refined=pred,gold=saved['gold'],original_prefix=prefixes[0],refined_prefix=prefixes[1]))
  summary=dict(positions=len(rows),original_prefix_sum=sum(r['original_prefix'] for r in rows),refined_prefix_sum=sum(r['refined_prefix'] for r in rows),better=sum(r['refined_prefix']>r['original_prefix'] for r in rows),worse=sum(r['refined_prefix']<r['original_prefix'] for r in rows),equal=sum(r['refined_prefix']==r['original_prefix'] for r in rows))
  report[task]=dict(summary=summary,rows=rows);(out.parent/'first-refinement.json').write_text(json.dumps(report,indent=2));print(task,summary,flush=True)
