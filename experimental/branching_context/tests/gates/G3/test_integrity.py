"""T32: corrupt GPU bytes and an offline persisted payload must be refused."""
import hashlib
import json
from pathlib import Path
import unittest

from context_time_travel.store import IntegrityError
import test_immutability as isolation
import test_offload as offload


class PageIntegrityTests(unittest.TestCase):
    setUpClass = classmethod(isolation.SharedPageTests.setUpClass.__func__)
    setUp = isolation.SharedPageTests.setUp
    record = isolation.SharedPageTests.record
    fresh = isolation.SharedPageTests.fresh
    retain = isolation.SharedPageTests.retain
    unchanged = isolation.SharedPageTests.unchanged
    assert_ssd_store = offload.OffloadTests.assert_ssd_store

    def files(self):
        return {p.relative_to(self.tmp.name).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(self.tmp.name).rglob('*') if p.is_file()}

    def refuse(self, snapshot, phase):
        files = self.files()
        before = self.backend._refresh_stats()
        emitted = []
        with self.assertRaises(IntegrityError) as failed:
            emitted.extend(self.sut.stream(snapshot, 1))
        after = self.backend._refresh_stats()
        self.record(kind='page-integrity-refusal', phase=phase, snapshot=snapshot,
                    before=before, after=after, error=repr(failed.exception), emitted=emitted)
        self.assertEqual(emitted, [])
        self.assertEqual(self.files(), files, 'Refusal cannot publish or alter canonical objects')
        for name in ('forward_tokens', 'captures', 'restores', 'loaded_file_bytes', 'written_file_bytes'):
            self.assertEqual(after[name], before[name], name)
        return before, after

    def test_g3_014_corrupt_page_refusal(self):
        self.assert_ssd_store()
        history = tuple(self.profile['fixtures']['prefix'][:9])
        root = self.sut.create(history)
        kept = [self.retain(root, self.fresh(root, history))]
        tokens = [self.profile['fixtures']['suffix_a'][0], self.profile['fixtures']['suffix_b'][0]]
        self.assertNotEqual(*tokens)
        starts, tips, heads = [], [], []
        for branch, token in enumerate(tokens):
            name = 'integrity-' + str(branch)
            start = self.sut.start_generation(root, dict(allowed=[token]), name)
            starts.append(start)
            result = self.sut.generate(start, 1)
            self.assertEqual(result.accepted_tokens, (token,))
            self.assertEqual(result.boundaries, (result.snapshot_id,))
            tips.append(result.snapshot_id)
            heads.append(self.sut.store.fork(name, result.snapshot_id))
            kept.append(self.retain(result.snapshot_id,
                                    self.fresh(result.snapshot_id, history + (token,))))
        logical = {sid: self.sut.snapshot_bytes(sid) for sid in [root, *starts, *tips]}
        controllers = {sid: self.sut.controller(sid) for sid in tips}
        self.assertEqual([controllers[sid]['next_draw'] for sid in tips], [1, 1])
        root_saved, a_saved, b_saved = kept
        other_pages = {r['allocation_id'] for saved in (root_saved, b_saved)
                       for r in saved['audit']['regions'] if r['pool'] >= 0}
        targets = [r for r in a_saved['audit']['regions']
                   if r['kind'] == 'k' and r['filled_prefix_bytes'] > 0
                   and r['allocation_id'] not in other_pages]
        self.assertTrue(targets, 'Damage must target consumed bytes in A\'s actual private K page')
        target = targets[0]
        before_fault = self.backend._refresh_stats()
        fault = self.backend._request('test-corrupt-page ' + str(target['driver_handle']) + ' 0')
        self.assertEqual(fault['driver_handle'], target['driver_handle'])
        self.assertEqual(fault['offset'], 0)
        self.assertEqual(fault['bytes_written'], 1)
        self.assertEqual(fault['after'], fault['before'] ^ 1)
        damaged = self.backend.audit(a_saved['image'])
        self.assertNotEqual(damaged['sha256'], a_saved['audit']['sha256'])
        actual = next(r for r in damaged['regions'] if r['allocation_id'] == target['allocation_id'])
        self.assertNotEqual(actual['sha256'], target['sha256'])
        self.assertEqual(self.backend._refresh_stats()['forward_tokens'], before_fault['forward_tokens'])
        self.record(kind='actual-gpu-page-fault', snapshot=tips[0], target=target,
                    fault=fault, damaged_audit=damaged)
        before, after = self.refuse(tips[0], 'damaged-hot-K-page')
        self.assertGreater(after['integrity_checked_pages'], before['integrity_checked_pages'])
        self.assertGreater(after['integrity_mismatches'], before['integrity_mismatches'])
        self.assertGreater(after['integrity_device_hashed_bytes'], before['integrity_device_hashed_bytes'])
        self.assertGreater(after['integrity_digest_copy_bytes'], before['integrity_digest_copy_bytes'])
        self.unchanged([root_saved, b_saved])
        self.fresh(tips[1], history + (tokens[1],))

        # Discard every optional cache entry. Only the unchanged canonical
        # inputs/controller may reconstruct the damaged branch.
        self.sut.evict_all()
        before = self.backend._refresh_stats()
        self.fresh(tips[0], history + (tokens[0],))
        after = self.backend._refresh_stats()
        self.assertEqual(after['forward_tokens'] - before['forward_tokens'], len(history) + 1)
        self.record(kind='damaged-page-replay', before=before, after=after)
        result = self.sut.generate(tips[0], 1)
        self.assertEqual(result.accepted_tokens, (tokens[0],))
        self.assertEqual(result.boundaries, (result.snapshot_id,))
        self.assertEqual(self.sut.controller(result.snapshot_id)['next_draw'], 2)
        current = result.snapshot_id
        current_history = history + (tokens[0], tokens[0])
        self.fresh(current, current_history)

        # Remove the real SSD payload while leaving its committed manifest.
        # Missing numerical bytes must be rejected before a native load/forward.
        paths = self.sut.checkpoint(current)
        payload = paths['payload_path']
        manifest_raw = paths['manifest_path'].read_bytes()
        manifest = json.loads(manifest_raw)['manifest']
        measured = dict(payload_bytes=payload.stat().st_size,
                        allocated_file_bytes=payload.stat().st_blocks * 512,
                        payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
                        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest())
        self.assertEqual(measured['payload_sha256'], manifest['payload_sha256'])
        self.assertEqual(measured['payload_bytes'], manifest['state_bytes'] + self.backend.vocab_size * 4)
        self.assertEqual(manifest['model_contract_id'], self.profile['model_contract_id'])
        self.assertEqual(manifest['execution_profile_id'], self.profile['execution_profile_id'])
        self.assertEqual(manifest['consumed'], len(current_history))
        self.sut.cache.pop(current)
        payload.unlink()
        self.assertFalse(payload.exists())
        self.record(kind='offline-persisted-payload', snapshot=current, **measured)
        self.refuse(current, 'missing-SSD-payload')
        self.assertEqual(paths['manifest_path'].read_bytes(), manifest_raw)
        self.sut.evict_all()
        before = self.backend._refresh_stats()
        self.fresh(current, current_history)
        after = self.backend._refresh_stats()
        self.assertEqual(after['forward_tokens'] - before['forward_tokens'], len(current_history))
        self.record(kind='offline-payload-replay', before=before, after=after)
        result = self.sut.generate(current, 1)
        self.assertEqual(result.accepted_tokens, (tokens[0],))
        self.assertEqual(result.boundaries, (result.snapshot_id,))
        self.assertEqual(self.sut.controller(result.snapshot_id)['next_draw'], 3)
        self.fresh(result.snapshot_id, current_history + (tokens[0],))
        self.fresh(tips[1], history + (tokens[1],))
        self.unchanged([root_saved, b_saved])
        for sid, original in logical.items():
            self.assertEqual(self.sut.snapshot_bytes(sid), original)
        for sid, original in controllers.items():
            self.assertEqual(self.sut.controller(sid), original)
        for head in heads:
            self.assertEqual(self.sut.store.head(head.name), head)


if __name__ == '__main__':
    unittest.main()
