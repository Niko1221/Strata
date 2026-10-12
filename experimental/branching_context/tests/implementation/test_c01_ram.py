"""Actual native conversation-cache restoration under canonical history leases."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from context_time_travel.real import BranchingModel, FreshOracle


class NativeRamCacheTests(unittest.TestCase):
    def test_native_ram_forced_siblings_and_exact_miss(self):
        profile = json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        backend = BranchingModel.load_backend(profile)
        self.addCleanup(backend.close)
        oracle = FreshOracle(profile)
        self.addCleanup(oracle.close)
        with tempfile.TemporaryDirectory() as directory:
            model = BranchingModel(directory, backend, profile)
            prefix = tuple(profile['fixtures']['prefix'][:9])
            a,b = profile['fixtures']['suffix_a'][0],profile['fixtures']['suffix_b'][0]
            self.assertNotEqual(a,b)
            root = model.create(prefix)
            original = model.snapshot_bytes(root)
            self.assertTrue(callable(getattr(backend,'park',None)),
                            'Native RAM cache capture must be implemented and measured')
            with model.materialize(root) as (lease,_):
                parked = backend.park(lease)
                self.assertGreater(parked['bytes'], 0)
                self.assertGreater(parked['payload_bytes'], 0)
                self.assertEqual(parked['consumed'], len(prefix))
            for suffix in ((a,a),(b,b)):
                with backend.lease() as lease:
                    before = backend.stats['consumed_tokens']
                    self.assertTrue(backend.restore_cached(lease,prefix))
                    self.assertEqual(backend.stats['consumed_tokens']-before,0)
                    history = prefix
                    for step in range(3):
                        expected = oracle.forward(history)
                        actual = backend.logits(lease)
                        self.assertEqual(actual,expected)
                        self.assertEqual(oracle.last['computed_tokens'],len(history))
                        with Path(os.environ['CTT_SHARING_MEASUREMENTS']).open('a') as stream:
                            stream.write(json.dumps({'test':self.id(),'kind':'native-ram-numerics',
                                'history':history,'max_absolute_error':max(abs(x-y) for x,y in zip(actual,expected)),
                                'oracle_computed_tokens':oracle.last['computed_tokens'],
                                'ram_cache_bytes':parked['bytes'],'ram_payload_bytes':parked['payload_bytes']})+'\n')
                        if step<2:
                            before=backend.stats['consumed_tokens']
                            backend.consume(lease,(suffix[step],))
                            self.assertEqual(backend.stats['consumed_tokens']-before,1)
                            history += (suffix[step],)
            with backend.lease() as lease:
                before=backend._refresh_stats()['consumed']
                wrong = ((prefix[0]+1)%backend.vocab_size,) + prefix[1:]
                self.assertFalse(backend.restore_cached(lease,wrong))
                self.assertEqual(backend._refresh_stats()['consumed'],before)
            self.assertEqual(model.snapshot_bytes(root),original)
            stats=backend._request('ram_stats')
            self.assertEqual(stats['hits'],2)
            self.assertEqual(stats['misses'],1)
            self.assertEqual(stats['entries'],1)
            self.assertGreater(stats['payload_allocations'],0)
            self.assertEqual(backend.clear_cached_histories(),1)
            empty=backend._request('ram_stats')
            self.assertEqual(empty['bytes'],0)
            self.assertEqual(empty['payload_allocations'],0)


if __name__=='__main__':
    unittest.main(verbosity=2)
