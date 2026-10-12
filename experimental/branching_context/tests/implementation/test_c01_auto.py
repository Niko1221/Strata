"""Canonical materialization uses real RAM checkpoints and truthful replay misses."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from context_time_travel.real import BranchingModel, FreshOracle


class AutomaticRamTests(unittest.TestCase):
    def test_canonical_ram_hit_miss_and_complete_eviction(self):
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
                self.assertEqual(oracle.last['computed_tokens'],len(history))
                with Path(os.environ['CTT_SHARING_MEASUREMENTS']).open('a') as stream:
                    stream.write(json.dumps({'test':self.id(),'kind':'automatic-ram-numerics',
                        'history':history,'max_absolute_error':max(abs(x-y) for x,y in zip(actual,expected)),
                        'oracle_computed_tokens':oracle.last['computed_tokens']})+'\n')
            model.cache.clear()  # evict only hot images; real native RAM must supply this boundary
            before=backend.stats['consumed_tokens']
            compare(root,prefix)
            self.assertEqual(backend.stats['consumed_tokens']-before,0,
                             'Canonical cache miss must restore exact parked history without prefix replay')
            self.assertGreater(backend._request('ram_stats')['hits'],0)
            for suffix in ((a,a),(b,b)):
                compare(root,prefix)
                tip=root
                for i,token in enumerate(suffix):
                    before=backend.stats['consumed_tokens']
                    tip=model.feed(tip,(token,))
                    self.assertEqual(backend.stats['consumed_tokens']-before,1)
                    compare(tip,prefix+suffix[:i+1])
            other_prefix=((prefix[0]+1)%backend.vocab_size,)+prefix[1:]
            other=model.create(other_prefix)
            backend.clear_cached_histories()
            with model.materialize(root) as (lease,_): backend.park(lease)
            model.cache.clear()
            before=backend.stats['consumed_tokens']
            misses=backend._request('ram_stats')['misses']
            compare(other,other_prefix)
            self.assertEqual(backend.stats['consumed_tokens']-before,len(other_prefix))
            self.assertEqual(backend._request('ram_stats')['misses']-misses,1)
            model.checkpoint(root)
            model.evict_all()
            self.assertEqual(len(model.cache),0)
            self.assertEqual(list(model.checkpoint_directory.iterdir()),[])
            self.assertEqual(backend._request('ram_stats')['bytes'],0)
            before=backend.stats['consumed_tokens']
            compare(root,prefix)
            self.assertEqual(backend.stats['consumed_tokens']-before,len(prefix))
            self.assertEqual(model.snapshot_bytes(root),original)


if __name__=='__main__': unittest.main(verbosity=2)
