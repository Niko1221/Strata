"""Adversarial checks for versioned controller/RNG restoration."""
import copy
import unittest

from context_time_travel import controller
from context_time_travel.store import IntegrityError


class ControllerVersionTests(unittest.TestCase):
    def test_c02_invariant(self):
        initial = controller.start(
            {"allowed": [0, 1], "temperature": 1.0},
            "versioned-controller-test",
            (),
            2,
        )
        logits = (0.0, 1.0)
        corruptions = (
            ("rng_algorithm", "unknown-rng"),
            ("sampler_order", ["temperature", "categorical"]),
            ("next_draw", -1),
            ("next_draw", 1 << 128),
        )
        rejected = []
        for field, value in corruptions:
            for action in ("choose", "consume"):
                restored = copy.deepcopy(initial)
                restored[field] = value
                try:
                    if action == "choose":
                        controller.choose(logits, restored)
                    else:
                        controller.consume(restored, 0, b"x", False)
                except Exception as error:
                    if not isinstance(error, IntegrityError):
                        rejected.append((field, action, type(error).__name__))
                else:
                    rejected.append((field, action, "accepted"))
        self.assertEqual(
            rejected,
            [],
            "unknown controller versions and invalid RNG cursors must fail as integrity errors before use",
        )

        self.assertEqual(initial["next_draw"], 0)
        self.assertEqual(initial["rng_algorithm"], "hmac-sha256-u53-v1")
        self.assertEqual(
            initial["sampler_order"],
            ["grammar/support", "repetition-penalty", "temperature", "categorical"],
        )


if __name__ == "__main__":
    unittest.main()
