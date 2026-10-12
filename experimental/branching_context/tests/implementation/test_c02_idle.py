"""Repair C02 without weakening version validation or consumed boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from context_time_travel import controller
from context_time_travel.real import BranchingModel, FreshOracle
from context_time_travel.store import IntegrityError, InvalidOperation


class IdleControllerTests(unittest.TestCase):
    def test_c02_idle_invariant(self):
        idle = {"status": "idle"}
        try:
            advanced = controller.consume(idle, 17, b"x", False)
        except Exception as error:
            self.fail("Forced input at an idle canonical snapshot must remain supported: " + repr(error))
        self.assertEqual(advanced, idle)
        self.assertIsNot(advanced, idle)
        self.assertEqual(idle, {"status": "idle"})
        with self.assertRaises(InvalidOperation):
            controller.choose((0.0,), idle)
        with self.assertRaises(IntegrityError):
            controller.consume({"status": "idle", "rng_algorithm": "unknown"}, 17, b"x", False)


class RealIdleBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Absence is an error, never a silent return or skip of the real checks.
        cls.profile = json.loads(Path(os.environ["CTT_PROFILE"]).read_text(encoding="utf-8"))
        cls.backend = BranchingModel.load_backend(cls.profile)
        cls.addClassCleanup(cls.backend.close)
        cls.oracle = FreshOracle(cls.profile)
        cls.addClassCleanup(cls.oracle.close)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = BranchingModel(self.tmp.name, self.backend, self.profile)
        self.addCleanup(self.model.evict_all)
        self.prefix = tuple(self.profile["fixtures"]["prefix"][:9])
        self.a = self.profile["fixtures"]["suffix_a"][0]
        self.b = self.profile["fixtures"]["suffix_b"][0]
        self.assertNotEqual(self.a, self.b)

    def fresh(self, snapshot, history):
        expected = self.oracle.forward(history)
        actual = self.model.next_logits(snapshot)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), self.backend.vocab_size)
        self.assertEqual(self.model.history(snapshot), history)
        self.assertEqual(self.model.consumed(snapshot), len(history))
        self.assertEqual(self.oracle.last["computed_tokens"], len(history))
        path = Path(os.environ["CTT_SHARING_MEASUREMENTS"])
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"test": self.id(), "kind": "numerics", "history": history,
                                     "max_absolute_error": max(abs(a-b) for a, b in zip(actual, expected)),
                                     "oracle_computed_tokens": self.oracle.last["computed_tokens"]}) + "\n")

    def test_real_idle_forced_futures(self):
        root = self.model.create(self.prefix)
        saved = self.model.snapshot_bytes(root)
        self.fresh(root, self.prefix)
        for suffix in ((self.a, self.a), (self.b, self.b)):
            tip = root
            for i, token in enumerate(suffix):
                before = self.backend.stats["consumed_tokens"]
                tip = self.model.feed(tip, (token,))
                self.assertEqual(self.backend.stats["consumed_tokens"]-before, 1)
                self.assertEqual(self.model.controller(tip), {"status": "idle"})
                self.fresh(tip, self.prefix + suffix[:i+1])
        self.assertEqual(self.model.snapshot_bytes(root), saved)
        self.fresh(root, self.prefix)

    def test_stream_publication_cancel_and_cold_controller_resume(self):
        root = self.model.create(self.prefix)
        policy = {"allowed": [self.a, self.b], "temperature": 1.0,
                  "rejection_threshold": 0.5, "stop_sequences": [[self.a, self.b]],
                  "grammar": {"start": "a", "transitions": {
                      "a": {str(self.a): "b"}, "b": {str(self.b): "a"}}}}
        ready = self.model.start_generation(root, policy, "c02-idle-repair")
        main = self.model.store.fork("main", ready)
        stream = self.model.stream(ready, 4)
        first, token = next(stream)
        self.assertEqual(token, self.a)
        self.assertEqual(self.model.controller(first)["grammar_state"], "b")
        self.assertGreater(self.model.controller(first)["next_draw"], 0)
        self.fresh(first, self.prefix + (self.a,))

        # A fresh OS process observes the consumed canonical snapshot before
        # the iterator advertises its accepted boundary to its caller.
        script = ("import json,sys; sys.path.insert(0,sys.argv[1]); "
                  "from context_time_travel.store import Store; s=Store(sys.argv[2]); "
                  "print(json.dumps(s.get_snapshot(sys.argv[3])))")
        observed = subprocess.check_output([sys.executable, "-c", script,
            str(Path(__file__).resolve().parents[2] / "src"), self.tmp.name, first], text=True)
        self.assertEqual(json.loads(observed), self.model.store.get_snapshot(first))
        self.assertEqual(self.model.store.head("main"), main)

        objects = {p.name for p in self.model.store.objects.iterdir()}
        def cancel(point):
            if point == "before_snapshot_publish":
                raise RuntimeError("cancel after consume before publication")
        self.model.store.fault_hook = cancel
        with self.assertRaisesRegex(RuntimeError, "cancel after consume"):
            next(stream)
        self.model.store.fault_hook = None
        self.assertEqual({p.name for p in self.model.store.objects.iterdir()}, objects)
        self.assertEqual(self.model.store.head("main"), main)
        self.fresh(first, self.prefix + (self.a,))

        saved = self.model.snapshot_bytes(first)
        first_result = self.model.generate(first, 4)
        self.assertEqual(first_result.accepted_tokens, (self.b,))
        self.assertEqual(self.model.controller(first_result.snapshot_id)["status"], "stopped")
        self.model.evict_all()
        second_result = self.model.generate(first, 4)
        self.assertEqual(second_result, first_result)
        self.assertEqual(self.model.snapshot_bytes(first), saved)
        self.fresh(second_result.snapshot_id, self.prefix + (self.a, self.b))


if __name__ == "__main__":
    unittest.main(verbosity=2)
