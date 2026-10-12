"""Cold-to-hot materialization preserves canonical input identity and saved work."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from context_time_travel.real import BranchingModel, FreshOracle


class ColdIdentityTests(unittest.TestCase):
    def test_cold_resident_identity_and_no_replay(self):
        profile = json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        backend = BranchingModel.load_backend(profile)
        self.addCleanup(backend.close)
        oracle = FreshOracle(profile)
        self.addCleanup(oracle.close)
        with tempfile.TemporaryDirectory() as directory:
            model = BranchingModel(directory, backend, profile)
            prefix = tuple(profile['fixtures']['prefix'][:9])
            root = model.create(prefix)
            original = model.snapshot_bytes(root)
            paths = model.checkpoint(root)
            envelope = json.loads(paths['manifest_path'].read_bytes())
            manifest = envelope['manifest']
            raw = paths['payload_path'].read_bytes()
            self.assertEqual(len(raw), manifest['state_bytes'] + 4 * backend.vocab_size)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest['payload_sha256'])
            self.assertGreater(manifest['state_bytes'], 0)
            expected = oracle.forward(prefix)
            self.assertEqual(oracle.last['computed_tokens'], len(prefix))
            model.cache.clear()
            backend.release_checkpoints()
            before = backend.stats['consumed_tokens']
            actual = model.next_logits(root)
            self.assertEqual(actual, expected)
            self.assertEqual(backend.stats['consumed_tokens'] - before, 0)
            with Path(os.environ['CTT_SHARING_MEASUREMENTS']).open('a') as stream:
                stream.write(json.dumps({'test': self.id(), 'kind': 'cold-resident',
                    'saved_payload_bytes': len(raw), 'native_state_bytes': manifest['state_bytes'],
                    'max_absolute_error': max(abs(a-b) for a,b in zip(actual, expected)),
                    'oracle_computed_tokens': oracle.last['computed_tokens'],
                    'restoration_computed_tokens': backend.stats['consumed_tokens']-before})+'\n')
            self.assertEqual(model.cache[root].input_identity, model._input_identity(prefix),
                             'Disk-to-native replacement must retain full canonical input identity')
            before = backend.stats['consumed_tokens']
            self.assertEqual(model.next_logits(root), expected)
            self.assertEqual(backend.stats['consumed_tokens'] - before, 0)
            self.assertEqual(model.snapshot_bytes(root), original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
