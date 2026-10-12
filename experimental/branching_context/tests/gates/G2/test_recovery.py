"""T20: real serialized checkpoints and independent-process cold reconstruction."""
from pathlib import Path
import json
import os
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'G1'))
import test_continuation as continuation
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError
from context_time_travel.numerics import assert_logits_close


class RecoveryTests(unittest.TestCase):
    setUpClass = classmethod(continuation.ContinuationTests.setUpClass.__func__)
    tearDownClass = classmethod(continuation.ContinuationTests.tearDownClass.__func__)
    setUp = continuation.ContinuationTests.setUp
    check = continuation.ContinuationTests.check

    def test_g2_001_cold_process_reopen(self):
        tip = self.sut.feed(self.root,self.A)
        self.sut.checkpoint(tip)
        result = subprocess.run([sys.executable,str(Path(__file__).resolve().parents[3]/'scripts/recovery_worker.py'),
            self.tmp.name,tip,'--feed',json.dumps(self.B)],env=os.environ,capture_output=True,check=True,text=True)
        child = json.loads(result.stdout)
        self.assertNotEqual(child['pid'],os.getpid())
        self.assertEqual(child['history'],list(self.P+self.A))
        assert_logits_close(child['logits'],self.oracle.forward(self.P+self.A),atol=self.profile['atol'],rtol=self.profile['rtol'])
        self.assertEqual(child['final_history'],list(self.P+self.A+self.B))
        report = assert_logits_close(child['final_logits'],self.oracle.forward(self.P+self.A+self.B),
            atol=self.profile['atol'],rtol=self.profile['rtol'])
        self.sut.measure(report,tip,self.P+self.A+self.B)

    def test_g2_002_remove_all_kv_checkpoints(self):
        start = self.sut.start_generation(self.root,dict(allowed=sorted(set(self.A+self.B))),'cold-controller')
        mid = self.sut.generate(start,7).snapshot_id
        expected = self.sut.generate(mid,3)
        before = self.sut.snapshot_bytes(mid)
        self.sut.checkpoint(mid)
        self.sut.evict_all()
        self.assertEqual(self.sut.cache,{})
        self.assertEqual(list(self.sut.checkpoint_directory.iterdir()),[])
        work = self.backend.stats['consumed_tokens']
        self.check(mid,self.sut.history(mid))
        self.assertEqual(self.backend.stats['consumed_tokens']-work,len(self.sut.history(mid)))
        self.assertEqual(self.sut.generate(mid,3).accepted_tokens,expected.accepted_tokens)
        self.assertEqual(before,self.sut.snapshot_bytes(mid))

    def test_g2_003_checkpoint_payload_corruption(self):
        manifest = self.sut.checkpoint(self.root)
        data = bytearray(manifest['payload_path'].read_bytes())
        data[len(data)//2] ^= 1
        manifest['payload_path'].write_bytes(data)
        self.sut.cache.clear()
        with self.assertRaises(IntegrityError):
            self.sut.next_logits(self.root)
        self.assertEqual(self.sut.history(self.root),self.P)

    def test_g2_004_checkpoint_manifest_corruption(self):
        manifest = self.sut.checkpoint(self.root)
        record = json.loads(manifest['manifest_path'].read_text())
        record['manifest']['consumed'] += 1
        manifest['manifest_path'].write_text(json.dumps(record))
        self.sut.cache.clear()
        with self.assertRaises(IntegrityError):
            self.sut.next_logits(self.root)

    def test_g2_005_partial_checkpoint_write(self):
        self.sut._checkpoint_fault = lambda point: (_ for _ in ()).throw(OSError('interrupted')) if point=='before_manifest_publish' else None
        with self.assertRaises(OSError):
            self.sut.checkpoint(self.root)
        self.sut.cache.clear()
        self.assertFalse((self.sut.checkpoint_directory/(self.root+'.json')).exists())
        self.check(self.root,self.P)
