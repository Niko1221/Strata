"""Real execution leases reject foreign-thread use and release before mutation."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import unittest

from context_time_travel.real import BranchingModel, FreshOracle
from context_time_travel.store import InvalidOperation


class ExecutionOwnerTests(unittest.TestCase):
    def test_foreign_owner_cannot_use_or_release_execution_lease(self):
        profile=json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        backend=BranchingModel.load_backend(profile)
        self.addCleanup(backend.close)
        oracle=FreshOracle(profile)
        self.addCleanup(oracle.close)
        with tempfile.TemporaryDirectory() as directory:
            model=BranchingModel(directory,backend,profile)
            prefix=tuple(profile['fixtures']['prefix'][:9])
            token=profile['fixtures']['suffix_a'][0]
            root=model.create(prefix)
            original=model.snapshot_bytes(root)
            context=backend.lease()
            lease=context.__enter__()
            try:
                self.assertTrue(backend.restore_cached(lease,prefix))
                expected=oracle.forward(prefix)
                actual=backend.logits(lease)
                self.assertEqual(actual,expected)
                with Path(os.environ['CTT_SHARING_MEASUREMENTS']).open('a') as stream:
                    stream.write(json.dumps({'test':self.id(),'kind':'owner-numerics',
                        'max_absolute_error':max(abs(a-b) for a,b in zip(actual,expected)),
                        'oracle_computed_tokens':oracle.last['computed_tokens']})+'\n')
                def outcome(action):
                    try: return action()
                    except Exception as error: return error
                before=backend.stats['consumed_tokens']
                with ThreadPoolExecutor(max_workers=1) as worker:
                    foreign=worker.submit(outcome,lambda:backend.position(lease)).result(timeout=10)
                    self.assertIsInstance(foreign,InvalidOperation,
                                          'Foreign thread must not use an active execution lease')
                    foreign=worker.submit(outcome,lambda:backend.consume(lease,(token,))).result(timeout=10)
                    self.assertIsInstance(foreign,InvalidOperation)
                    foreign=worker.submit(outcome,lambda:context.__exit__(None,None,None)).result(timeout=10)
                    self.assertIsInstance(foreign,InvalidOperation)
                self.assertEqual(backend.stats['consumed_tokens'],before)
                self.assertEqual(backend.position(lease),len(prefix)-1)
                backend.consume(lease,(token,))
                self.assertEqual(backend.stats['consumed_tokens']-before,1)
                expected=oracle.forward(prefix+(token,))
                actual=backend.logits(lease)
                self.assertEqual(actual,expected)
                with Path(os.environ['CTT_SHARING_MEASUREMENTS']).open('a') as stream:
                    stream.write(json.dumps({'test':self.id(),'kind':'owner-numerics',
                        'max_absolute_error':max(abs(a-b) for a,b in zip(actual,expected)),
                        'oracle_computed_tokens':oracle.last['computed_tokens']})+'\n')
            finally:
                context.__exit__(None,None,None)
            with self.assertRaises(InvalidOperation): backend.position(lease)
            self.assertEqual(backend.stats['lease_acquires'],backend.stats['lease_releases'])
            self.assertEqual(model.snapshot_bytes(root),original)


if __name__=='__main__': unittest.main(verbosity=2)
