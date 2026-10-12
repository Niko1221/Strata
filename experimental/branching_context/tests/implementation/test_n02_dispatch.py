"""Resident experts must contribute through the ordinary verifier work plan."""
import array
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class DispatchParityTests(unittest.TestCase):
    def test_normal_and_private_drivers_include_resident_experts(self):
        profile=json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        evidence=Path(os.environ['CTT_SHARING_MEASUREMENTS']).parent/'dispatch-parity'
        evidence.mkdir()
        errors=[]
        for fraction in ('0','0.55'):
            vectors={}
            for kind,executable in [('normal',os.environ['CTT_NORMAL_EXECUTABLE']),
                                    ('fresh',os.environ['CTT_ORACLE_EXECUTABLE']),
                                    ('branch',os.environ['CTT_SUT_EXECUTABLE'])]:
                label=kind+'-'+fraction
                logits=evidence/(label+'.f32')
                command=[executable]+profile['execution']['oracle_command'][1:]+['--pcie-frac',fraction,'--tokens','100']
                protocol='feed 100\nlogits '+str(logits)+'\nstats\nquit\n' if kind=='branch' else None
                result=subprocess.run(command,input=protocol,text=True,capture_output=True,
                    env=dict(os.environ,STRATA_DUMP_FIRST_LOGITS=str(logits)),timeout=120)
                (evidence/(label+'.stdout')).write_text(result.stdout)
                (evidence/(label+'.stderr')).write_text(result.stderr)
                self.assertEqual(result.returncode,0,result.stderr[-2000:])
                raw=logits.read_bytes()
                self.assertEqual(len(raw),248320*4)
                values=array.array('f');values.frombytes(raw);vectors[kind]=values
                counters={}
                if kind=='fresh':
                    counters=json.loads(result.stdout)
                    self.assertEqual(counters['forward_tokens'],1)
                elif kind=='branch':
                    transcript=[json.loads(row) for row in result.stdout.splitlines()]
                    counters=next(row for row in transcript if 'forward_tokens' in row)
                    self.assertEqual(counters['forward_tokens'],1)
                row=dict(test=self.id(),kind='dispatch-parity',operator=kind,pcie_fraction=fraction,
                    command=command,binary_sha256=hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
                    logits_sha256=hashlib.sha256(raw).hexdigest(),counters=counters)
                if kind!='normal':
                    error=max(abs(a-b) for a,b in zip(values,vectors['normal']))
                    row['max_absolute_error']=error
                    errors.append(error)
                with Path(os.environ['CTT_SHARING_MEASUREMENTS']).open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
        self.assertEqual(errors,[0.0]*4,'Private drivers must execute the same resident and PCIe experts as normal T=1')


if __name__=='__main__': unittest.main(verbosity=2)
