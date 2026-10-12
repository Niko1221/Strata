"""Executable tests of the reference logical store (no model backend involved)."""

import copy
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest

from context_time_travel import (Conflict, IntegrityError, InvalidOperation,
                                 NotFound, Store)


def _cas_worker(root, operation, queue):
    s = Store(root)
    h = s.head("main")
    try:
        # Simultaneously racing with stale rev=0. All use same old revision.
        result = s.move("main", 0, h.tip, operation)
        queue.put(("win", result.revision))
    except Conflict:
        queue.put(("conflict", 0))
    except Exception as exc:
        queue.put(("error", repr(exc)))


class LogicalStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="context-time-travel-")
        self.addCleanup(self.tmp.cleanup)
        self.s = Store(self.tmp.name)
        self.r = self.s.create_root("test-model@sha256:deadbeef", (1, 3),
                                    {"rng": {"key": "abc", "draw": 0},
                                     "grammar": "idle"})

    def test_01_root_history(self):
        root, history = self.s.lineage(self.r)
        self.assertEqual((root, history), (self.r, [1, 3]))

    def test_02_fork_same_logical_state(self):
        a = self.s.fork("a", self.r)
        b = self.s.fork("b", self.r)
        self.assertEqual((a.tip, b.tip, a.root, b.root), (self.r,) * 4)
        self.assertEqual((a.revision, b.revision), (0, 0))

    def test_03_independent_suffixes(self):
        a = self.s.append(self.r, (5,), {"rng": {"key": "a", "draw": 0}})
        b = self.s.append(self.r, (9,), {"rng": {"key": "b", "draw": 0}})
        self.assertEqual(self.s.lineage(a)[1], [1, 3, 5])
        self.assertEqual(self.s.lineage(b)[1], [1, 3, 9])
        self.assertEqual(self.s.lineage(self.r)[1], [1, 3])

    def test_04_empty_delta_controller_transition(self):
        a = self.s.append(self.r, (), {"rng": {"key": "new", "draw": 37}})
        self.assertEqual(self.s.lineage(a)[1], [1, 3])
        self.assertNotEqual(a, self.r)
        self.assertEqual(self.s.get_snapshot(a)["controller"]["rng"]["draw"], 37)

    def test_05_canonical_content_identity(self):
        s1 = self.s.create_root("m", (7, 2), {"z": 9, "a": 2})
        s2 = self.s.create_root("m", [7, 2], {"a": 2, "z": 9})
        self.assertEqual(s1, s2)

    def test_06_mutable_arguments_do_not_modify_saved_state(self):
        v = {"nested": {"draw": 7, "list": [1, 2]}}
        s = self.s.append(self.r, [8], v)
        v["nested"]["draw"] = 999
        v["nested"]["list"].append(3)
        self.assertEqual(self.s.get_snapshot(s)["controller"],
                         {"nested": {"draw": 7, "list": [1, 2]}})

    def test_07_snapshot_integrity_reject_tampered_blob(self):
        path = Path(self.tmp.name) / "objects" / (self.r + ".json")
        path.write_bytes(path.read_bytes().replace(b'"inputs":[1,3]', b'"inputs":[1,9]'))
        with self.assertRaises(IntegrityError):
            self.s.get_snapshot(self.r)

    def test_08_missing_parent_fails_closed(self):
        a = self.s.append(self.r, [9], {})
        (Path(self.tmp.name) / "objects" / (self.r + ".json")).unlink()
        with self.assertRaises(NotFound):
            self.s.lineage(a)

    def test_09_invalid_inputs_rejected(self):
        for invalid in ([True], [-1], [2**32], [1.3], "token", {"x": 1}):
            with self.subTest(invalid=invalid), self.assertRaises(InvalidOperation):
                self.s.append(self.r, invalid, {})

    def test_10_invalid_controller_rejected(self):
        for value in ({"p": float("nan")}, {"x": {1, 2}}, 5):
            with self.subTest(value=repr(value)), self.assertRaises(InvalidOperation):
                self.s.append(self.r, (1,), value)

    def test_11_fork_duplicate_fails(self):
        self.s.fork("a", self.r)
        with self.assertRaises(Conflict):
            self.s.fork("a", self.r)

    def test_12_missing_head_fails(self):
        with self.assertRaises(NotFound):
            self.s.head("missing")

    def test_13_atomic_CAS_updates_one_revision(self):
        self.s.fork("a", self.r)
        a = self.s.append(self.r, (7,), {})
        new = self.s.move("a", 0, a, "op1")
        self.assertEqual((new.revision, new.tip), (1, a))
        with self.assertRaises(Conflict):
            self.s.move("a", 0, self.r, "op2")
        self.assertEqual(self.s.head("a").tip, a)

    def test_14_successful_retry_is_idempotent(self):
        self.s.fork("a", self.r)
        a = self.s.append(self.r, (7,), {})
        first = self.s.move("a", 0, a, "op1")
        second = self.s.move("a", 0, a, "op1")
        self.assertEqual(first, second)
        self.assertEqual(self.s.head("a").revision, 1)
        self.assertEqual(sum(1 for event in self.s.events() if event["type"] == "move"), 1)

    def test_15_reuse_operation_id_with_different_payload_rejected(self):
        self.s.fork("a", self.r)
        self.s.move("a", 0, self.r, "op1")
        with self.assertRaises(InvalidOperation):
            self.s.move("a", 1, self.r, "op1")

    def test_16_ABA_fails_with_stale_revision(self):
        self.s.fork("a", self.r)
        child = self.s.append(self.r, (9,), {})
        self.s.move("a", 0, child, "first")
        self.s.move("a", 1, self.r, "second")
        with self.assertRaises(Conflict):
            self.s.move("a", 0, child, "stale")
        self.assertEqual((self.s.head("a").revision, self.s.head("a").tip), (2, self.r))

    def test_17_cannot_move_between_unrelated_roots(self):
        self.s.fork("a", self.r)
        another = self.s.create_root("test-model@sha256:deadbeef", (1, 3),
                                     {"different": True})
        with self.assertRaises(InvalidOperation):
            self.s.move("a", 0, another, "wrongroot")

    def test_18_branch_pointer_restart_persists(self):
        self.s.fork("a", self.r)
        new = self.s.append(self.r, (8,), {"draw": 2})
        self.s.move("a", 0, new, "persist")
        reopened = Store(self.tmp.name)
        self.assertEqual(reopened.head("a").tip, new)
        self.assertEqual(reopened.head("a").revision, 1)
        self.assertEqual(reopened.lineage(new)[1], [1, 3, 8])

    def test_19_retry_receipt_restart_persists(self):
        self.s.fork("a", self.r)
        r = self.s.move("a", 0, self.r, "persist")
        reopened = Store(self.tmp.name)
        self.assertEqual(reopened.move("a", 0, self.r, "persist"), r)
        self.assertEqual(reopened.head("a").revision, 1)

    def test_20_crash_before_reference_staging_keeps_old_tip(self):
        self.s.fork("a", self.r)
        child = self.s.append(self.r, (99,), {})
        def fail(step):
            if step == "before_refs_stage":
                raise RuntimeError("injected fault")
        faulty = Store(self.tmp.name, fault_hook=fail)
        with self.assertRaisesRegex(RuntimeError, "injected fault"):
            faulty.move("a", 0, child, "fault1")
        reopened = Store(self.tmp.name)
        self.assertEqual((reopened.head("a").revision, reopened.head("a").tip), (0, self.r))
        self.assertEqual(reopened.lineage(child)[1], [1, 3, 99])

    def test_21_crash_after_stage_before_rename_keeps_old_tip(self):
        self.s.fork("a", self.r)
        child = self.s.append(self.r, (99,), {})
        def fail(step):
            if step == "after_refs_stage_fsync":
                raise RuntimeError("injected fault")
        faulty = Store(self.tmp.name, fault_hook=fail)
        with self.assertRaisesRegex(RuntimeError, "injected fault"):
            faulty.move("a", 0, child, "fault2")
        reopened = Store(self.tmp.name)
        self.assertEqual((reopened.head("a").revision, reopened.head("a").tip), (0, self.r))

    def test_22_failure_after_rename_has_retryable_receipt(self):
        self.s.fork("a", self.r)
        child = self.s.append(self.r, (5,), {})
        def fail(step):
            if step == "after_refs_rename_before_dir_fsync":
                raise RuntimeError("after rename")
        faulty = Store(self.tmp.name, fault_hook=fail)
        with self.assertRaisesRegex(RuntimeError, "after rename"):
            faulty.move("a", 0, child, "fault3")
        reopened = Store(self.tmp.name)
        self.assertEqual((reopened.head("a").revision, reopened.head("a").tip), (1, child))
        self.assertEqual(reopened.move("a", 0, child, "fault3").revision, 1)

    def test_23_evaluation_like_reads_do_not_move_branch(self):
        self.s.fork("a", self.r)
        original = self.s.head("a")
        _, history = self.s.lineage(self.r)
        self.assertEqual(history, [1, 3])
        self.assertEqual(self.s.head("a"), original)

    def test_24_store_ref_index_rejects_corrupt_json(self):
        self.s.fork("a", self.r)
        (Path(self.tmp.name) / "refs.json").write_text('{"schema":1')
        with self.assertRaises(IntegrityError):
            Store(self.tmp.name)

    def test_25_invalid_branch_or_operation_names(self):
        with self.assertRaises(InvalidOperation):
            self.s.fork("../oops", self.r)
        self.s.fork("a", self.r)
        with self.assertRaises(InvalidOperation):
            self.s.move("a", 0, self.r, "../../pwn")

    def test_26_history_can_have_multiple_empty_deltas(self):
        a = self.s.append(self.r, (), {"draw": 1})
        b = self.s.append(a, (), {"draw": 2})
        self.assertEqual(self.s.lineage(b)[1], [1, 3])
        self.assertNotEqual((a, b), (self.r, self.r))

    def test_27_copy_on_read_cannot_modify_store(self):
        child = self.s.append(self.r, (5,), {"rng": {"draw": 3}})
        obj = self.s.get_snapshot(child)
        obj["controller"]["rng"]["draw"] = 999
        self.assertEqual(self.s.get_snapshot(child)["controller"]["rng"]["draw"], 3)

    def test_28_same_history_different_controller_not_same_snapshot(self):
        a = self.s.append(self.r, (), {"draw": 2})
        b = self.s.append(self.r, (), {"draw": 3})
        self.assertNotEqual(a, b)
        self.assertEqual(self.s.lineage(a)[1], self.s.lineage(b)[1])

    def test_29_racing_writers_exactly_one_wins(self):
        self.s.fork("main", self.r)
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        processes = [ctx.Process(target=_cas_worker,
                       args=(self.tmp.name, f"race{i}", queue)) for i in range(6)]
        for p in processes:
            p.start()
        outcomes = [queue.get(timeout=15) for _ in processes]
        for p in processes:
            p.join(timeout=15)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual(sum(outcome[0] == "win" for outcome in outcomes), 1, outcomes)
        self.assertEqual(sum(outcome[0] == "conflict" for outcome in outcomes), 5, outcomes)
        self.assertEqual(self.s.head("main").revision, 1)

    def test_30_unreferenced_snapshot_survives_branch_rewind(self):
        self.s.fork("a", self.r)
        old = self.s.append(self.r, (5, 6), {})
        self.s.move("a", 0, old, "up")
        self.s.move("a", 1, self.r, "back")
        self.assertEqual(self.s.lineage(old)[1], [1, 3, 5, 6])
        self.assertEqual(self.s.lineage(self.s.head("a").tip)[1], [1, 3])

    def test_31_second_future_after_rewind_stays_independent(self):
        self.s.fork("a", self.r)
        a = self.s.append(self.r, (5,), {})
        self.s.move("a", 0, a, "up")
        self.s.move("a", 1, self.r, "rewind")
        b = self.s.append(self.s.head("a").tip, (9,), {})
        self.s.move("a", 2, b, "new")
        self.assertEqual(self.s.lineage(a)[1], [1, 3, 5])
        self.assertEqual(self.s.lineage(b)[1], [1, 3, 9])

    def test_32_external_mutation_after_fork_does_not_propagate(self):
        self.s.fork("a", self.r)
        self.s.fork("b", self.r)
        a = self.s.append(self.s.head("a").tip, (5,), {})
        self.s.move("a", 0, a, "updatea")
        self.assertEqual(self.s.head("b").tip, self.r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
