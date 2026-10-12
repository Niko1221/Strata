"""Cancellation, stale jobs and cold replay preserve canonical branch state."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from context_time_travel.real import BranchingModel, FreshOracle
from context_time_travel.store import Conflict


class SlotRecoveryTests(unittest.TestCase):
    def test_s02_invariant(self):
        profile=json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        backend=BranchingModel.load_backend(profile)
        self.addCleanup(backend.close)
        oracle=FreshOracle(profile)
        self.addCleanup(oracle.close)
        with tempfile.TemporaryDirectory() as directory:
            model=BranchingModel(directory,backend,profile)
            prefix=tuple(profile['fixtures']['prefix'][:9])
            a,b=profile['fixtures']['suffix_a'][0],profile['fixtures']['suffix_b'][0]
            root=model.create(prefix)
            original=model.snapshot_bytes(root)
            def compare(snapshot,history):
                expected=oracle.forward(history)
                actual=model.next_logits(snapshot)
                self.assertEqual(actual,expected)
                with Path(os.environ['CTT_SHARING_MEASUREMENTS']).open('a') as stream:
                    stream.write(json.dumps({'test':self.id(),'kind':'slot-recovery-numerics',
                        'max_absolute_error':max(abs(x-y) for x,y in zip(actual,expected)),
                        'oracle_computed_tokens':oracle.last['computed_tokens']})+'\n')
            compare(root,prefix)
            for name in ('a','b','replacement'): model.store.fork(name,root)
            slots=model.slot_queue(2)
            slots.bind(0,'a')
            slots.bind(1,'b')
            cancelled=slots.submit(0,'feed',(a,),'cancelled')
            slots.submit(1,'feed',(b,),'survivor')
            cancel=getattr(slots,'cancel',None)
            self.assertTrue(callable(cancel),'Queued cancellation must release its slot without executing work')
            before=backend.stats['consumed_tokens']
            self.assertTrue(cancel(cancelled))
            self.assertFalse(cancel(cancelled))
            self.assertEqual(backend.stats['consumed_tokens'],before)
            self.assertEqual(model.store.head('a').revision,0)
            slots.bind(0,'replacement')
            survivor=slots.run_next()
            self.assertEqual(survivor.head.name,'b')
            self.assertEqual(survivor.head.revision,1)
            compare(survivor.head.tip,prefix+(b,))
            saved=model.snapshot_bytes(survivor.head.tip)
            slots.submit(1,'feed',(b,),'stale')
            moved=model.store.move('b',1,root,'external-rewind')
            before=backend.stats['consumed_tokens']
            with self.assertRaises(Conflict): slots.run_next()
            self.assertEqual(backend.stats['consumed_tokens'],before)
            self.assertEqual(model.store.head('b'),moved)
            self.assertIsNone(slots.active)
            self.assertEqual(model.store.move('b',0,survivor.head.tip,'survivor'),survivor.head)
            ready=model.start_generation(root,{'allowed':[a],'temperature':1.0},'recover')
            ready_head=model.store.move('replacement',0,ready,'ready')
            slots.submit(0,'generate',3,'interrupted')
            publications=0
            def interrupt(point):
                nonlocal publications
                if point=='before_snapshot_publish':
                    publications+=1
                    if publications==2: raise InterruptedError('stop before second durable frontier')
            model.store.fault_hook=interrupt
            try:
                with self.assertRaises(InterruptedError): slots.run_next()
            finally: model.store.fault_hook=None
            self.assertEqual(publications,2)
            self.assertEqual(model.store.head('replacement'),ready_head)
            self.assertEqual(model.controller(ready)['next_draw'],0)
            self.assertIsNone(slots.active)
            self.assertEqual(backend.stats['lease_acquires'],backend.stats['lease_releases'])
            slots.submit(0,'generate',1,'recovered')
            recovered=slots.run_next()
            self.assertEqual(recovered.output.accepted_tokens,(a,))
            compare(recovered.head.tip,prefix+(a,))
            model.checkpoint(root)
            model.evict_all()
            self.assertEqual(len(model.cache),0)
            self.assertEqual(list(model.checkpoint_directory.iterdir()),[])
            self.assertEqual(backend._request('ram_stats')['bytes'],0)
            before=backend.stats['consumed_tokens']
            compare(root,prefix)
            self.assertEqual(backend.stats['consumed_tokens']-before,len(prefix))
            compare(survivor.head.tip,prefix+(b,))
            self.assertEqual(model.snapshot_bytes(root),original)
            self.assertEqual(model.snapshot_bytes(survivor.head.tip),saved)
            self.assertEqual(slots.stats['completed'],2)
            self.assertEqual(slots.stats['failed'],2)
            self.assertEqual(backend.stats['lease_acquires'],backend.stats['lease_releases'])


if __name__=='__main__': unittest.main(verbosity=2)
