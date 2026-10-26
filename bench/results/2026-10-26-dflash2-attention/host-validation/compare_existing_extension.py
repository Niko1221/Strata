"""Static SM89 check of existing DFlash2 kernel sections before adding staging."""
from pathlib import Path
import tempfile,subprocess,hashlib,json,sys
before,after,output=map(Path,sys.argv[1:4]);patterns=['attention128','select_kernel','select_bf16_parallel','selector_lattice','topk_parallel_kernel','topk_merge_heads','topk_register_kernel','bf16_project_tiles','conv_kernel','hc_mean_kernel']
def read(exe,folder):
 result={}
 for i,pattern in enumerate(patterns):
  d=folder/str(i);d.mkdir()
  subprocess.run(['/usr/local/cuda/bin/cuobjdump','--extract-text',pattern,str(exe.resolve())],cwd=d,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  for f in d.glob('*.sm_89.elf.bin'):
   value=f.read_bytes()
   if f.name in result:assert result[f.name]==value,f.name
   result[f.name]=value
 assert result
 return result
with tempfile.TemporaryDirectory() as tmp:
 b=Path(tmp)/'b';a=Path(tmp)/'a';b.mkdir();a.mkdir();x=read(before,b);y=read(after,a)
 result={'architecture':'sm_89','baseline_sha256':hashlib.sha256(before.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(after.read_bytes()).hexdigest(),'baseline_sections':len(x),'candidate_sections':len(y),'identical_bytes':sum(len(v) for k,v in x.items() if y.get(k)==v),'different':[k for k in x if y.get(k)!=x[k]]};output.write_text(json.dumps(result,indent=2)+'\n');assert not result['different'],result;print(result)
