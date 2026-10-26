from pathlib import Path
import subprocess,tempfile,hashlib,json,argparse
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
def read(exe,folder,normalize):
 subprocess.run(['/usr/local/cuda/bin/cuobjdump','--extract-text','attention128_kernel',str(exe.resolve())],cwd=folder,stdout=subprocess.DEVNULL,check=True)
 files=sorted(folder.glob('*.sm_89.elf.bin'));assert files
 names=subprocess.check_output(['c++filt','-n'],input='\n'.join(f.name[2:].removesuffix('.sm_89.elf.bin') for f in files)+'\n',text=True).splitlines();out={}
 for f,name in zip(files,names):
  if normalize:
   end=name.find('>(')
   if not name[:end].endswith(', false'):continue
   name=name[:end-7]+name[end:]
  assert name not in out;out[name]=f.read_bytes()
 return out
with tempfile.TemporaryDirectory() as d:
 before=Path(d)/'before';after=Path(d)/'after';before.mkdir();after.mkdir();x=read(a.baseline,before,False);y=read(a.candidate,after,True)
 result={'architecture':'sm_89','baseline_sha256':hashlib.sha256(a.baseline.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'baseline_kernels':len(x),'candidate_default_kernels':len(y),'identical_bytes':sum(len(v) for k,v in x.items() if y.get(k)==v),'different':[k for k in x if x[k]!=y.get(k)]};a.output.write_text(json.dumps(result,indent=2)+'\n');assert not result['different'],result;print(result)
