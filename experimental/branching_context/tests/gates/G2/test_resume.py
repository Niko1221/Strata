"""T23: compatibility, complete cold controller replay, effects and scope."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'G1'))
import test_continuation as continuation
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError, NotFound, _json_bytes
from context_time_travel.numerics import assert_logits_close

class ResumeTests(unittest.TestCase):
    setUpClass = classmethod(continuation.ContinuationTests.setUpClass.__func__)
    tearDownClass = classmethod(continuation.ContinuationTests.tearDownClass.__func__)
    setUp = continuation.ContinuationTests.setUp
    check = continuation.ContinuationTests.check

    def test_g2_015_checkpoint_from_wrong_model(self):
        files = self.sut.checkpoint(self.root)
        original = json.loads(files['manifest_path'].read_text())
        for field,value in [('model_contract_id','wrong-weights'),('positions','offset-100')]:
            envelope = json.loads(json.dumps(original))
            envelope['manifest'][field] = value
            envelope['checksum'] = hashlib.sha256(_json_bytes(envelope['manifest'])).hexdigest()
            files['manifest_path'].write_bytes(_json_bytes(envelope))
            self.sut.cache.clear()
            with self.assertRaises(IntegrityError):
                self.sut.next_logits(self.root)
        self.sut.evict_all()
        self.check(self.root,self.P)

    def test_g2_016_checkpoint_restore_grammar_rng(self):
        a,b = sorted(set(self.A+self.B))[:2]
        grammar = dict(start='left',transitions={'left':{str(a):'right'},'right':{str(b):'left'}})
        start = self.sut.start_generation(self.root,dict(allowed=[a,b],grammar=grammar,
            repetition_penalty=1.1,rejection_threshold=.4,stop_sequences=[[a,b,a,b,a,b,a,b,a,b,a,b]]),'persisted-stream')
        mid = self.sut.generate(start,7).snapshot_id
        expected = self.sut.generate(mid,3)
        state = self.sut.controller(mid)
        self.assertGreater(state['next_draw'],7)
        self.sut.checkpoint(mid)
        result = subprocess.run([sys.executable,str(Path(__file__).resolve().parents[3]/'scripts/recovery_worker.py'),
            self.tmp.name,mid,'3'],capture_output=True,check=True,text=True)
        child = json.loads(result.stdout)
        self.assertEqual(child['controller'],state)
        self.assertEqual(tuple(child['generated']),expected.accepted_tokens)
        self.assertEqual(child['final_controller'],self.sut.controller(expected.snapshot_id))
        report = assert_logits_close(child['final_logits'],self.oracle.forward(self.sut.history(expected.snapshot_id)),
            atol=self.profile['atol'],rtol=self.profile['rtol'])
        self.sut.measure(report,expected.snapshot_id,self.sut.history(expected.snapshot_id))

    def test_g2_017_missing_input_artifact(self):
        tip = self.sut.feed(self.root,self.A)
        self.sut.checkpoint(tip)
        self.sut.cache.clear()
        (self.sut.store.objects/(self.root+'.json')).unlink()
        with self.assertRaises(NotFound):
            self.sut.next_logits(tip)
        with self.assertRaises(IntegrityError):
            self.sut.create(self.P,contract=dict(self.sut.contract,input_kind='image'))

    def test_g2_018_external_effects_not_replayed(self):
        marker = Path(self.tmp.name)/'external-tool-invocations'
        subprocess.run([sys.executable,'-c','import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("1")',str(marker)],check=True)
        # The caller records exact tokenized tool output once. Replay consumes these
        # retained inputs; it has no external tool execution interface.
        tip = self.sut.feed(self.root,self.A)
        self.sut.evict_all()
        self.check(tip,self.P+self.A)
        reopened = BranchingModel(self.tmp.name,self.backend,self.profile)
        report = assert_logits_close(reopened.next_logits(tip),self.oracle.forward(self.P+self.A),
            atol=self.profile['atol'],rtol=self.profile['rtol'])
        self.sut.measure(report,tip,self.P+self.A)
        self.assertEqual(marker.read_text(),'1')

    def test_g2_019_power_loss_scope_honest(self):
        self.assertEqual(self.sut.durability_scope,dict(platform='local-posix',
            process_crash_tested=True,power_loss_tested=False,
            publication='fsync-file; atomic-link-or-rename; fsync-directory'))
        self.sut.store.fork('main',self.root)
        tip = self.sut.feed(self.root,self.A)
        result = subprocess.run([sys.executable,str(Path(__file__).resolve().parents[3]/'scripts/crash_worker.py'),
            self.tmp.name,'move',tip,'after_refs_rename_before_dir_fsync'],capture_output=True)
        self.assertEqual(result.returncode,73)
        self.assertEqual(self.sut.store.move('main',0,tip,'attempt').revision,1)
        self.check(tip,self.P+self.A)
