"""An ordinary-initialized verifier must consume all forced prompt tokens."""
import array
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unittest

class OrdinaryStepTests(unittest.TestCase):
    def test_n02_step_consumes_forced_history(self):
        profile=json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        evidence=Path(os.environ['CTT_SHARING_MEASUREMENTS']).parent/'ordinary-step'
        evidence.mkdir()
        for length in (2,1,3,4):
            ids=[100,117,134,151][:length]
            vectors={}
            for kind,exe in [('step',os.environ['CTT_STEP_EXECUTABLE']),('fresh',profile['execution']['oracle_command'][0])]:
                label=f'{kind}-{length}'
                logits=evidence/(label+'.f32')
                command=[exe]+profile['execution']['oracle_command'][1:]
                if kind=='step':command[command.index('--prefill')+1]='0'
                command+=['--tokens',','.join(map(str,ids))]
                result=subprocess.run(command,env=dict(os.environ,**profile['execution']['environment'],STRATA_DUMP_FIRST_LOGITS=str(logits)),capture_output=True,text=True,timeout=120)
                (evidence/(label+'.stdout')).write_text(result.stdout)
                (evidence/(label+'.stderr')).write_text(result.stderr)
                self.assertEqual(result.returncode,0,'Ordinary native control must consume forced history: '+result.stderr[-1000:])
                counters=json.loads(result.stdout)
                self.assertEqual(counters['consumed'],length)
                self.assertEqual(counters['forward_tokens'],length)
                self.assertEqual(counters['position_max'],length-1)
                self.assertEqual(counters['windows'],length)
                raw=logits.read_bytes();self.assertEqual(len(raw),248320*4)
                values=array.array('f');values.frombytes(raw);vectors[kind]=values
                row=dict(test=self.id(),operator=kind,length=length,command=command,counters=counters,binary_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest(),logits_sha256=hashlib.sha256(raw).hexdigest())
                if kind=='fresh':row['max_absolute_error']=max(abs(a-b) for a,b in zip(values,vectors['step']))
                with Path(os.environ['CTT_SHARING_MEASUREMENTS']).open('a') as f:f.write(json.dumps(row)+'\n')
            self.assertEqual(vectors['step'],vectors['fresh'],'Same T=1 operator must agree after every forced history')

if __name__=='__main__':unittest.main(verbosity=2)
