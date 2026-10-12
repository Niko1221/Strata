"""Real inference observation with exact comparison; no numerical-profile admission."""
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import time
import unittest

import test_e01 as storage_test
from context_time_travel.real import BranchingModel, FreshOracle
from context_time_travel.store import Store


class EvaluationNoninterferenceTests(unittest.TestCase):
    def test_e01_real_record_noninterference(self):
        profile = json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        backend = BranchingModel.load_backend(profile)
        self.addCleanup(backend.close)
        oracle = FreshOracle(profile)
        self.addCleanup(oracle.close)
        with tempfile.TemporaryDirectory() as directory:
            model = BranchingModel(directory, backend, profile)
            prefix = tuple(profile['fixtures']['prefix'])
            suffix = tuple(profile['fixtures']['suffix_a'])
            root = model.create(prefix)
            head = model.store.fork('main', root)
            before = model.snapshot_bytes(root)
            frontier = model.next_logits(root)
            expected = oracle.forward(prefix)
            self.assertEqual(frontier, expected)
            self.assertEqual(oracle.last['computed_tokens'], len(prefix))
            started, cpu_started = time.monotonic(), time.process_time()
            value = storage_test.record(root, root)
            value['evaluator'] = dict(name='consumed-token-count', version='1',
                                     source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            value['fixtures'] = {'tokens': hashlib.sha256(json.dumps(prefix).encode()).hexdigest()}
            value['environment'] = {'execution_profile_id': profile['execution_profile_id'],
                                    'resource_scope': 'Python process; RSS lifetime high-water'}
            value['metric'] = {'name': 'consumed_tokens', 'order': 'maximize'}
            value['outcomes'] = dict(mandatory={'consumed': 'PASS'}, score=model.consumed(root), unresolved=[])
            self.assertEqual(value['outcomes']['score'], len(prefix))
            value['resources'] = dict(wall_seconds=time.monotonic()-started,
                                      cpu_seconds=time.process_time()-cpu_started,
                                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            identity = model.store.put_evaluation(value)
            self.assertEqual(Store(directory).get_evaluation(identity), value)
            self.assertEqual(model.snapshot_bytes(root), before)
            self.assertEqual(model.store.head('main'), head)
            self.assertEqual(model.next_logits(root), frontier)
            tip = model.feed(root, suffix)
            future = model.next_logits(tip)
            fresh = oracle.forward(prefix+suffix)
            self.assertEqual(future, fresh)
            self.assertEqual(oracle.last['computed_tokens'], len(prefix+suffix))
            self.assertEqual(model.snapshot_bytes(root), before)
            self.assertEqual(model.next_logits(root), frontier)
            report = dict(test=self.id(), scope='Supporting exact real-backend observation, not G4 admission',
                          execution_profile_id=profile['execution_profile_id'],
                          oracle_requests=oracle.requests, oracle_tokens=oracle.tokens_computed,
                          max_absolute_error=max(abs(a-b) for a,b in zip(future,fresh)),
                          evaluation_id=identity, record=value, branch_head_unchanged=True,
                          snapshot_unchanged=True, prefix_tokens=len(prefix), suffix_tokens=len(suffix))
            Path(os.environ['CTT_SHARING_MEASUREMENTS']).write_text(json.dumps(report,indent=2)+'\n')
            model.evict_all()


if __name__ == '__main__':
    unittest.main(verbosity=2)
