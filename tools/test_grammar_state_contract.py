"""G4 application-state tests; candidates are synthetic, not model results."""
import concurrent.futures
from dataclasses import FrozenInstanceError
import threading
import unittest

from tools.grammar_state_contract import Snapshot, StaleDecision, TaskState, derive_grammar, verify_candidate


class StateContract(unittest.TestCase):
    def state(self):
        return TaskState(('task-a', 'task-b'), {'alice': ('task-a', 'task-b'), 'bob': ('task-b',)})

    def test_frozen_contract_has_stable_identity_and_small_source(self):
        app = self.state()
        a, b = app.snapshot('alice'), app.snapshot('bob')
        self.assertEqual(a.fingerprint(), app.snapshot('alice').fingerprint())
        self.assertNotEqual(a.fingerprint(), b.fingerprint())
        self.assertEqual(derive_grammar(a), 'root ::= "WAIT" | "START task-a" | "START task-b"\n')
        with self.assertRaises(FrozenInstanceError):
            a.revision = 8
        self.assertEqual(app.apply('alice', a, 'WAIT'), a)

    def test_stale_output_does_not_apply(self):
        app = self.state()
        frozen = app.snapshot('alice')
        changed = app.apply('alice', frozen, 'START task-a')
        # Still in the frozen language; current revision, not grammar, prevents it.
        verify_candidate(frozen, 'START task-b')
        with self.assertRaises(StaleDecision):
            app.apply('alice', frozen, 'START task-b')
        self.assertEqual(app.snapshot('alice'), changed)

    def test_changed_authorization_and_wrong_principal_fail_before_effects(self):
        app = self.state()
        a = app.snapshot('alice')
        with self.assertRaises(StaleDecision):
            app.apply('bob', a, 'START task-a')
        app.replace_grant('alice', ('task-a',))
        changed = app.snapshot('alice')
        with self.assertRaises(StaleDecision):
            app.apply('alice', a, 'START task-b')
        with self.assertRaises(ValueError):
            app.apply('alice', changed, 'START task-b')
        self.assertEqual(app.snapshot('alice'), changed)

    def test_injection_and_unbounded_names_rejected(self):
        for name in ('a" | "EVIL', 'a\nroot ::= "EVIL"', '../task', 'a\\b', '猫', '', 'a' * 33, 'a\0b'):
            with self.subTest(name=repr(name)), self.assertRaises(ValueError):
                TaskState((name,), {'alice': (name,)})
        for tasks in (['a'], ('a', 'a'), tuple('task-' + str(i) for i in range(33))):
            with self.assertRaises(ValueError):
                TaskState(tasks, {'alice': ()})
        with self.assertRaises(ValueError):
            Snapshot(0, 'alice', ('b', 'a'))

    def test_whole_candidate_is_verified_without_repair(self):
        app = self.state()
        frozen = app.snapshot('alice')
        for candidate in ('START task-a\n', ' START task-a', 'START task-a; START task-b', 'WAIT\0', None):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                app.apply('alice', frozen, candidate)
            self.assertEqual(app.snapshot('alice'), frozen)

    def test_racing_candidates_can_apply_at_most_once(self):
        app = self.state()
        frozen = app.snapshot('alice')
        barrier = threading.Barrier(2)
        def apply(command):
            barrier.wait()
            try:
                app.apply('alice', frozen, command)
                return 'applied'
            except StaleDecision:
                return 'stale'
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            results = list(pool.map(apply, ('START task-a', 'START task-b')))
        self.assertCountEqual(results, ('applied', 'stale'))
        self.assertEqual(app.snapshot('alice').revision, 1)
        self.assertEqual(len(app.snapshot('alice').available), 1)

    def test_revision_exhaustion_precedes_every_effect(self):
        app = self.state()
        app._revision = 2**63 - 1  # deliberate fault injection at this toy's bound
        frozen = app.snapshot('alice')
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            app.apply('alice', frozen, 'START task-a')
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            app.replace_grant('alice', ())
        self.assertEqual(app.snapshot('alice'), frozen)


if __name__ == '__main__':
    unittest.main()
