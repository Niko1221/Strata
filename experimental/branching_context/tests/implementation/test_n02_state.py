"""Read-only diagnostic of populated state after one consumed token."""
import array, hashlib, json, math, os, struct, subprocess, unittest
from pathlib import Path

class StateDiagnosticTests(unittest.TestCase):
    def test_n02_state_observes_consumed_buffers(self):
        profile=json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        evidence=Path(os.environ['CTT_SHARING_MEASUREMENTS']).parent/'state-comparison';evidence.mkdir()
        records={}
        for name,chunk in [('step','0'),('prefill','1')]:
            prefix=evidence/name
            command=[os.environ['CTT_STEP_EXECUTABLE']]+profile['execution']['oracle_command'][1:]
            command[command.index('--prefill')+1]=chunk
            command+=['--tokens','100,117']
            result=subprocess.run(command,env=dict(os.environ,**profile['execution']['environment'],STRATA_DUMP_FIRST_LOGITS=str(prefix)+'.logits',STRATA_STEP_STATE=str(prefix)),capture_output=True,text=True,timeout=180)
            Path(str(prefix)+'.stdout').write_text(result.stdout);Path(str(prefix)+'.stderr').write_text(result.stderr)
            self.assertEqual(result.returncode,0,result.stderr[-1000:])
            manifest=Path(str(prefix)+'.json')
            self.assertTrue(manifest.exists(),'Consumed-state instrumentation must publish a descriptor and real payloads')
            record=json.loads(manifest.read_text());self.assertEqual(record['consumed'],1);self.assertEqual(record['ple_prev'],[-1,100]);self.assertTrue(record['parts'])
            records[name]=record
        self.assertEqual(records['step']['parts'],records['prefill']['parts'])
        comparisons=[]
        for part in records['step']['parts']:
            vectors=[];hashes=[]
            for name in ('step','prefill'):
                raw=Path(str(evidence/name)+'.'+part['name']).read_bytes();self.assertEqual(len(raw),part['bytes']);hashes.append(hashlib.sha256(raw).hexdigest())
                if part['dtype']=='f16':values=[x[0] for x in struct.iter_unpack('<e',raw)]
                else:values=array.array('f' if part['dtype']=='f32' else 'i');values.frombytes(raw)
                self.assertTrue(all(math.isfinite(x) for x in values));vectors.append(values)
            comparisons.append(dict(part=part,sha256=hashes,max_absolute_error=max((abs(a-b) for a,b in zip(*vectors)),default=0),different_values=sum(a!=b for a,b in zip(*vectors))))
        report=dict(test=self.id(),scope='State measurements only, not numerical equivalence',comparisons=comparisons)
        Path(str(evidence/'comparison')+'.json').write_text(json.dumps(report,indent=2))
        with Path(os.environ['CTT_SHARING_MEASUREMENTS']).open('a') as f:f.write(json.dumps(report)+'\n')

if __name__=='__main__':unittest.main(verbosity=2)
