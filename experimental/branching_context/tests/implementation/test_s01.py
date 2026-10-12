"""Real branch-slot FIFO execution binds canonical controllers and CAS heads."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from context_time_travel.real import BranchingModel, FreshOracle
from context_time_travel.store import InvalidOperation


class SerializedSlotTests(unittest.TestCase):
    def test_s01_invariant(self):
        profile=json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        backend=BranchingModel.load_backend(profile)
        self.addCleanup(backend.close)
        oracle=FreshOracle(profile)
        self.addCleanup(oracle.close)
        with tempfile.TemporaryDirectory() as directory:
            model=BranchingModel(directory,backend,profile)
            prefix=tuple(profile['fixtures']['prefix'][:9])
            a,b=profile['fixtures']['suffix_a'][0],profile['fixtures']['suffix_b'][0]
            self.assertNotEqual(a,b)
            root=model.create(prefix)
            original=model.snapshot_bytes(root)
            def compare(snapshot,history):
                expected=oracle.forward(history)
                actual=model.next_logits(snapshot)
                self.assertEqual(actual,expected)
                with Path(os.environ['CTT_SHARING_MEASUREMENTS']).open('a') as stream:
                    stream.write(json.dumps({'test':self.id(),'kind':'slot-numerics',
                        'max_absolute_error':max(abs(x-y) for x,y in zip(actual,expected)),
                        'oracle_computed_tokens':oracle.last['computed_tokens']})+'\n')
            compare(root,prefix)
            factory=getattr(model,'slot_queue',None)
            self.assertTrue(callable(factory),'Canonical branch-slot FIFO execution must be available')
            slots=factory(2)
            starts=[]
            for index,token in enumerate((a,b)):
                ready=model.start_generation(root,{'allowed':[token],'temperature':1.0},'slot-'+str(index))
                starts.append(ready)
                model.store.fork('slot-'+str(index),ready)
                slots.bind(index,'slot-'+str(index))
                slots.submit(index,'generate',2,'slot-work-'+str(index))
            with self.assertRaises(InvalidOperation): slots.bind(0,'slot-1')
            results=[slots.run_next(),slots.run_next()]
            self.assertEqual([r.head.name for r in results],['slot-0','slot-1'])
            for index,result in enumerate(results):
                token=(a,b)[index]
                self.assertEqual(result.output.accepted_tokens,(token,token))
                self.assertEqual(result.head,model.store.head('slot-'+str(index)))
                self.assertEqual(result.head.revision,1)
                self.assertEqual(model.controller(starts[index])['next_draw'],0)
                self.assertGreater(model.controller(result.head.tip)['next_draw'],0)
                for step,snapshot in enumerate(result.output.boundaries):
                    compare(snapshot,prefix+(token,)*(step+1))
            self.assertIsNone(slots.run_next())
            self.assertEqual(slots.stats['completed'],2)
            self.assertEqual(slots.stats['failed'],0)
            self.assertEqual(slots.stats['execution_mode'],'serialized')
            self.assertEqual(backend.stats['lease_acquires'],backend.stats['lease_releases'])
            self.assertEqual(model.snapshot_bytes(root),original)
            with Path(os.environ['CTT_SHARING_MEASUREMENTS']).open('a') as stream:
                stream.write(json.dumps({'test':self.id(),'kind':'slot-work',**slots.stats})+'\n')


if __name__=='__main__': unittest.main(verbosity=2)
