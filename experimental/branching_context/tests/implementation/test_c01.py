"""Adversarial checks for canonical-history/checkpoint binding."""
import json
import os
import tempfile
import unittest

from context_time_travel.real import BranchingModel, Checkpoint


class CanonicalCheckpointIdentityTests(unittest.TestCase):
    def test_c01_invariant(self):
        prefix = (41, 7, 9, 13)
        other_prefix = (88, 7, 9, 13)
        blob = b"opaque-native-checkpoint"
        frontier = (0.0,)
        expected = BranchingModel._input_identity(prefix)
        fields = dict(
            model_contract_id="model-contract",
            execution_profile_id="execution-profile",
            consumed=len(prefix),
            blob=blob,
            frontier=frontier,
            checksum=Checkpoint.digest(blob, frontier),
        )
        if "input_identity" in Checkpoint.__dataclass_fields__:
            fields["input_identity"] = expected
        checkpoint = Checkpoint(**fields)
        self.assertEqual(
            getattr(checkpoint, "input_identity", None),
            expected,
            "an in-memory numerical checkpoint must identify the exact canonical consumed history",
        )
        self.assertNotEqual(
            expected,
            BranchingModel._input_identity(other_prefix),
            "equal length and equal suffix must not collapse different full histories",
        )

        profile_path = os.environ.get("CTT_PROFILE")
        if profile_path is None:
            return

        profile = json.loads(open(profile_path, encoding="utf-8").read())
        backend = BranchingModel.load_backend(profile)
        try:
            with tempfile.TemporaryDirectory() as directory:
                model = BranchingModel(directory, backend, profile)
                prefix = tuple(profile["fixtures"]["prefix"])
                replacement = (prefix[-1] + 1) % backend.vocab_size
                alternate = prefix[:-1] + (replacement,)
                source = model.create(prefix)
                target = model.create(alternate)
                source_checkpoint = model.cache[source]
                model.cache[target] = source_checkpoint
                before = backend.stats["consumed_tokens"]

                model.next_logits(target)

                self.assertEqual(
                    backend.stats["consumed_tokens"] - before,
                    len(alternate),
                    "a checkpoint bound to another same-length history must miss and replay canonical inputs",
                )
                self.assertEqual(
                    model.cache[target].input_identity,
                    model._input_identity(alternate),
                )
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
