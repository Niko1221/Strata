"""T21: real histories with actual independent-process termination and CAS."""
from pathlib import Path
import json
import subprocess
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'G1'))
import test_continuation as continuation
from context_time_travel.store import Store, Conflict, InvalidOperation

WORKER = str(Path(__file__).resolve().parents[3]/'scripts/crash_worker.py')

class CrashTests(unittest.TestCase):
    setUpClass = classmethod(continuation.ContinuationTests.setUpClass.__func__)
    tearDownClass = classmethod(continuation.ContinuationTests.tearDownClass.__func__)
    check = continuation.ContinuationTests.check

    def setUp(self):
        continuation.ContinuationTests.setUp(self)
        self.head = self.sut.store.fork('main',self.root)

    def worker(self, action, target, point):
        return subprocess.run([sys.executable,WORKER,self.tmp.name,action,target,point],capture_output=True)

    def test_g2_006_crash_before_snapshot_publish(self):
        before = set(self.sut.store.objects.glob('*.json'))
        result = self.worker('feed',self.root,'before_snapshot_publish')
        self.assertEqual(result.returncode,73,result.stderr[-2000:])
        self.assertEqual(Store(self.tmp.name).head('main'),self.head)
        self.assertEqual(set(self.sut.store.objects.glob('*.json')),before)
        self.check(self.root,self.P)

    def test_g2_007_crash_before_branch_rename(self):
        tip = self.sut.feed(self.root,self.A)
        result = self.worker('move',tip,'after_refs_stage_fsync')
        self.assertEqual(result.returncode,73)
        reopened = Store(self.tmp.name)
        self.assertEqual(reopened.head('main'),self.head)
        self.assertEqual(len(reopened.events()),1)
        self.check(tip,self.P+self.A)

    def test_g2_008_crash_after_branch_rename(self):
        tip = self.sut.feed(self.root,self.A)
        result = self.worker('move',tip,'after_refs_rename_before_dir_fsync')
        self.assertEqual(result.returncode,73)
        reopened = Store(self.tmp.name)
        committed = reopened.head('main')
        self.assertEqual((committed.tip,committed.revision),(tip,1))
        self.assertEqual(reopened.move('main',0,tip,'attempt'),committed)
        self.assertEqual(reopened.move('main',0,tip,'attempt'),committed)
        self.assertEqual(len(reopened.events()),2)
        self.check(tip,self.P+self.A)

    def test_g2_009_idempotency_payload_mismatch(self):
        tip = self.sut.feed(self.root,self.A)
        self.sut.store.move('main',0,tip,'attempt')
        with self.assertRaises(InvalidOperation):
            self.sut.store.move('main',0,self.root,'attempt')
        self.assertEqual(self.sut.store.head('main').tip,tip)
        self.check(tip,self.P+self.A)

    def test_g2_010_aba_pointer_revision(self):
        tip = self.sut.feed(self.root,self.A)
        self.sut.store.move('main',0,tip,'out')
        self.sut.store.move('main',1,self.root,'back')
        with self.assertRaises(Conflict):
            self.sut.store.move('main',0,tip,'stale')
        self.assertEqual(self.sut.store.head('main').revision,2)
        self.check(self.root,self.P)

    def test_g2_011_racing_cas_processes(self):
        a = self.sut.feed(self.root,self.A)
        b = self.sut.feed(self.root,self.B)
        workers = [subprocess.Popen([sys.executable,WORKER,self.tmp.name,'race',tip,op],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            for tip,op in ((a,'racerA'),(b,'racerB'))]
        (Path(self.tmp.name)/'go').touch()
        results = [worker.communicate(timeout=30) for worker in workers]
        self.assertEqual(sorted(worker.returncode for worker in workers),[0,74],results)
        self.assertEqual(Store(self.tmp.name).head('main').revision,1)
        self.assertEqual(len(self.sut.store.events()),2)
        self.check(a,self.P+self.A)
        self.check(b,self.P+self.B)
