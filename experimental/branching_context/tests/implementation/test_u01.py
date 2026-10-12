"""Checkpoint layouts are versioned data, never inferred from matching shapes."""
import copy
import hashlib
import unittest

from context_time_travel import checkpoint_cache
from context_time_travel.store import IntegrityError, _json_bytes


class UpgradeTests(unittest.TestCase):
    def test_u01_invariant(self):
        validate = getattr(checkpoint_cache, 'validate_manifest', None)
        self.assertTrue(callable(validate), 'A strict versioned checkpoint manifest validator is required')
        expected = dict(schema=1, snapshot='1'*64, input_identity='2'*64,
                        model_contract_id='3'*64, execution_profile_id='4'*64,
                        consumed=7, positions='absolute-zero-based', logits_count=11)
        manifest = dict(expected, state_bytes=99, payload_sha256='5'*64)

        def envelope(value):
            return dict(manifest=value, checksum=hashlib.sha256(_json_bytes(value)).hexdigest())

        original = copy.deepcopy(manifest)
        self.assertEqual(validate(envelope(manifest), expected), original)
        changes = [('schema', 2), ('schema', True), ('state_bytes', True),
                   ('state_bytes', 0), ('model_contract_id', '6'*64),
                   ('execution_profile_id', '7'*64), ('positions', 'relative'),
                   ('consumed', 8), ('payload_sha256', '../state'),
                   ('tensor_layout', 'unknown-future-layout')]
        for field, value in changes:
            with self.subTest(field=field, value=value), self.assertRaises(IntegrityError):
                validate(envelope(dict(manifest, **{field: value})), expected)
        for field in manifest:
            altered = dict(manifest)
            del altered[field]
            with self.subTest(missing=field), self.assertRaises(IntegrityError):
                validate(envelope(altered), expected)
        bad = envelope(manifest)
        bad['checksum'] = '0'*64
        with self.assertRaises(IntegrityError): validate(bad, expected)
        with self.assertRaises(IntegrityError): validate(dict(envelope(manifest), format=2), expected)
        self.assertEqual(manifest, original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
