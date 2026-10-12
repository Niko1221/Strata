"""Reject old numerical layouts, then replay unchanged canonical history."""
import copy
import hashlib
import json
import unittest

import test_access_control as access
from context_time_travel.store import IntegrityError, Store, _json_bytes


class UpgradeTests(unittest.TestCase):
    setUpClass = classmethod(access.AccessControlTests.setUpClass.__func__)
    tearDownClass = classmethod(access.AccessControlTests.tearDownClass.__func__)
    setUp = access.AccessControlTests.setUp
    check = access.AccessControlTests.check

    def test_g5_003_replay_after_upgrade(self):
        start = self.sut.start_generation(self.root, dict(allowed=sorted(set(self.A+self.B))), 'upgrade')
        candidate = self.sut.generate(start, 2).snapshot_id
        head = self.sut.store.fork('upgrade', candidate)
        saved = self.sut.snapshot_bytes(candidate)
        history = self.sut.history(candidate)
        files = self.sut.checkpoint(candidate)
        original = json.loads(files['manifest_path'].read_bytes())
        for field, value in [('schema', 2), ('schema', True),
                             ('execution_profile_id', 'old-layout'),
                             ('tensor_layout', 'unrecognized'),
                             ('model_contract_id', 'other-model'), ('positions', 'relative')]:
            with self.subTest(field=field, value=value):
                envelope = copy.deepcopy(original)
                envelope['manifest'][field] = value
                envelope['checksum'] = hashlib.sha256(_json_bytes(envelope['manifest'])).hexdigest()
                files['manifest_path'].write_bytes(_json_bytes(envelope))
                self.sut.cache.clear()
                before = self.backend.stats['checkpoint_import_bytes']
                with self.assertRaises(IntegrityError):
                    self.sut.next_logits(candidate)
                self.assertEqual(self.backend.stats['checkpoint_import_bytes'], before)
                self.assertEqual(self.sut.snapshot_bytes(candidate), saved)
                self.assertEqual(Store(self.tmp.name).head('upgrade'), head)
        self.sut.evict_all()
        work = self.backend.stats['consumed_tokens']
        self.check(candidate, history)
        self.assertEqual(self.backend.stats['consumed_tokens']-work, len(history))
        self.assertEqual(self.sut.snapshot_bytes(candidate), saved)
        self.assertEqual(Store(self.tmp.name).head('upgrade'), head)


if __name__ == '__main__':
    unittest.main(verbosity=2)
