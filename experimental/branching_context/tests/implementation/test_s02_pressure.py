"""Cancellation, stale jobs and cold replay preserve canonical branch state."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from context_time_travel.real import BranchingModel, FreshOracle
from context_time_travel.store import Conflict


class SlotPressureTests(unittest.TestCase):
    def test_real_ram_admission_pressure_preserves_evicted_history(self):
        profile=json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        backend=BranchingModel.load_backend(profile)
        self.addCleanup(backend.close)
        oracle=FreshOracle(profile)
        self.addCleanup(oracle.close)
        with tempfile.TemporaryDirectory() as directory:
            model=BranchingModel(directory,backend,profile)
            prefix=tuple(profile['fixtures']['prefix'][:9])
            roots=[]
            originals=[]
            histories=[]
            for index in range(10):
                history=((prefix[0]+index)%backend.vocab_size,)+prefix[1:]
                root=model.create(history)
                roots.append(root)
                originals.append(model.snapshot_bytes(root))
                histories.append(history)
            pressure=backend._request('ram_stats')
            self.assertLess(pressure['entries'],len(roots))
            self.assertGreater(pressure['entries'],0)
            self.assertLessEqual(pressure['bytes'],1024**3)
            model.cache.clear()
            before=backend.stats['consumed_tokens']
            expected=oracle.forward(histories[0])
            actual=model.next_logits(roots[0])
            self.assertEqual(actual,expected)
            replay=backend.stats['consumed_tokens']-before
            self.assertEqual(replay,len(histories[0]))
            for root,original in zip(roots,originals):
                self.assertEqual(model.snapshot_bytes(root),original)
            self.assertEqual(backend.stats['lease_acquires'],backend.stats['lease_releases'])
            with Path(os.environ['CTT_SHARING_MEASUREMENTS']).open('a') as stream:
                stream.write(json.dumps({'test':self.id(),'kind':'real-cache-admission-pressure',
                    'max_absolute_error':max(abs(x-y) for x,y in zip(actual,expected)),
                    'oracle_computed_tokens':oracle.last['computed_tokens'],
                    'sut_replay_tokens':replay,'retained_histories':len(roots),
                    'ram_cache':pressure})+'\n')


if __name__=='__main__': unittest.main(verbosity=2)
