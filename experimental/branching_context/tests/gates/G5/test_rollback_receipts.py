"""Real-model histories survive ambiguous acknowledgement and refused formats."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import test_access_control as access
from context_time_travel.store import Store, IntegrityError, InvalidOperation, Conflict, _json_bytes


class RollbackTests(unittest.TestCase):
    setUpClass = classmethod(access.AccessControlTests.setUpClass.__func__)
    tearDownClass = classmethod(access.AccessControlTests.tearDownClass.__func__)
    setUp = access.AccessControlTests.setUp
    check = access.AccessControlTests.check

    def test_g5_010_restart_with_journal_ambiguity(self):
        candidate = self.sut.feed(self.root, self.A)
        before = self.sut.snapshot_bytes(candidate)
        self.sut.store.fork('main', self.root)
        worker = Path(__file__).resolve().parents[3]/'scripts/crash_worker.py'
        result = subprocess.run([sys.executable, str(worker), self.tmp.name, 'move', candidate,
                                 'after_refs_rename_before_dir_fsync'], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 73, result.stderr[-2000:])
        store = Store(self.tmp.name)
        head = store.head('main')
        self.assertEqual((head.tip, head.revision), (candidate, 1))
        self.assertEqual(store.move('main', 0, candidate, 'attempt'), head)
        with self.assertRaises(InvalidOperation): store.move('main', 0, self.root, 'attempt')
        store.move('main', 1, self.root, 'back')
        self.assertEqual(store.head('main').revision, 2)
        with self.assertRaises(Conflict): store.move('main', 0, candidate, 'stale')
        self.sut.evict_all()
        self.check(candidate, self.P+self.A)
        self.assertEqual(self.sut.snapshot_bytes(candidate), before)
        self.assertEqual(Store(self.tmp.name).move('main', 0, candidate, 'attempt'), head)

    def test_g5_012_release_rollback(self):
        candidate = self.sut.feed(self.root, self.A)
        head = self.sut.store.fork('rollback', candidate)
        before = self.sut.snapshot_bytes(candidate)
        self.sut.checkpoint(candidate)
        marker = self.sut.store.refs_path
        original = marker.read_bytes()
        value = json.loads(original)
        value['schema'] = 3
        marker.write_bytes(_json_bytes(value))
        refused = {str(p.relative_to(self.tmp.name)):p.read_bytes()
                   for p in Path(self.tmp.name).rglob('*') if p.is_file()}
        with self.assertRaises(IntegrityError): Store(self.tmp.name)
        self.assertEqual({str(p.relative_to(self.tmp.name)):p.read_bytes()
                          for p in Path(self.tmp.name).rglob('*') if p.is_file()}, refused)
        marker.write_bytes(original)
        self.assertEqual(Store(self.tmp.name).head('rollback'), head)
        self.sut.evict_all()
        work = self.backend.stats['consumed_tokens']
        self.check(candidate, self.P+self.A)
        self.assertEqual(self.backend.stats['consumed_tokens']-work, len(self.P+self.A))
        self.assertEqual(self.sut.snapshot_bytes(candidate), before)
        self.assertEqual(Store(self.tmp.name).head('rollback'), head)


if __name__ == '__main__':
    unittest.main(verbosity=2)
