"""T22: abort-safe private leases, actual allocation failure, retained history."""
from pathlib import Path
import json
import subprocess
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'G1'))
import test_continuation as continuation
from context_time_travel.store import Store

class LeaseTests(unittest.TestCase):
    setUpClass = classmethod(continuation.ContinuationTests.setUpClass.__func__)
    tearDownClass = classmethod(continuation.ContinuationTests.tearDownClass.__func__)
    setUp = continuation.ContinuationTests.setUp
    check = continuation.ContinuationTests.check

    def test_g2_012_kill_during_decode(self):
        head = self.sut.store.fork('main',self.root)
        before = set(self.sut.store.objects.glob('*.json'))
        result = subprocess.run([sys.executable,str(Path(__file__).resolve().parents[3]/'scripts/crash_worker.py'),
            self.tmp.name,'feed',self.root,'after_token_consume'],capture_output=True)
        self.assertEqual(result.returncode,73,result.stderr[-1000:])
        self.assertEqual(Store(self.tmp.name).head('main'),head)
        self.assertEqual(set(self.sut.store.objects.glob('*.json')),before)
        self.check(self.root,self.P)
        recovered = self.sut.feed(self.root,self.B)
        self.check(recovered,self.P+self.B)

    def test_g2_013_out_of_memory_transaction(self):
        history = (self.P*100)[:1024]
        root = self.sut.create(history)
        head = self.sut.store.fork('main',root)
        sibling = self.sut.feed(root,self.B)
        before = set(self.sut.store.objects.glob('*.json'))
        result = subprocess.run([sys.executable,str(Path(__file__).resolve().parents[3]/'scripts/oom_worker.py'),
            self.tmp.name,root],capture_output=True,check=True,text=True)
        measured = json.loads(result.stdout)
        self.assertTrue(measured['allocation_failed'],measured)
        self.assertGreater(measured['allocation_bytes'][0],1024*1024)
        self.assertEqual(measured['counters']['contexts_created'],measured['counters']['contexts_freed'])
        self.assertEqual(Store(self.tmp.name).head('main'),head)
        self.assertEqual(set(self.sut.store.objects.glob('*.json')),before)
        self.check(root,history)
        self.check(sibling,history+self.B)

    def test_g2_014_eviction_vs_deletion(self):
        child = self.sut.feed(self.root,self.A)
        self.sut.store.fork('retained',child)
        snapshots = {p.name:p.read_bytes() for p in self.sut.store.objects.glob('*.json')}
        self.sut.checkpoint(child)
        self.sut.evict_all()
        self.assertEqual(snapshots,{p.name:p.read_bytes() for p in self.sut.store.objects.glob('*.json')})
        self.assertEqual(self.sut.store.head('retained').tip,child)
        self.check(child,self.P+self.A)
        self.check(self.root,self.P)
