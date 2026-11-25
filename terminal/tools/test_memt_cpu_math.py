"""tools/test_memt_cpu_math.py - MemT memory math: hand cases + C++ vs Python.

    python tools/test_memt_cpu_math.py [-v]

The parity test needs engine/build/cpu-check/memt_cpu_test (the `cpu-check`
preset).  It is skipped, not failed, when that binary is absent.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memt_cpu_oracle as O  # noqa: E402

BINARY = O.locate_binary()


class ScalarMath(unittest.TestCase):
    def test_logsigmoid_and_softplus(self):
        self.assertAlmostEqual(O.logsigmoid(0.0), -math.log(2.0), places=12)
        self.assertAlmostEqual(O.softplus(0.0), math.log(2.0), places=12)
        for x in (-30.0, -2.0, 0.5, 3.0, 25.0):
            self.assertAlmostEqual(
                O.logsigmoid(x), math.log(O.sigmoid(x)), places=9
            )

    def test_l2norm(self):
        x = np.array([3.0, 4.0], dtype=np.float32)
        O.l2norm_in_place(x)
        self.assertAlmostEqual(float(x[0]), 0.6, places=5)
        self.assertAlmostEqual(float(x[1]), 0.8, places=5)

    def test_read_identity(self):
        state = np.eye(2, dtype=np.float32).reshape(1, 2, 2)
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        out = O.memory_read(q, state, heads=1, dim=2)
        self.assertAlmostEqual(float(out[0, 0]), 1.0 / math.sqrt(2.0), places=6)
        self.assertAlmostEqual(float(out[0, 1]), 0.0, places=7)

    def test_read_does_not_move_state(self):
        state = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
        before = state.copy()
        O.memory_read(np.ones((1, 2), dtype=np.float32), state, heads=1, dim=2)
        np.testing.assert_array_equal(state, before)

    def test_update_hand_case(self):
        # S = I; k = [0, 1]; v = [2, 3]; no decay; beta = 1
        # kv = S^T k = [0, 1]; delta = [2, 2]; S += k outer delta = [[1,0],[2,3]]
        state = np.eye(2, dtype=np.float32).reshape(1, 2, 2)
        k = np.array([[0.0, 1.0]], dtype=np.float32)
        v = np.array([[2.0, 3.0]], dtype=np.float32)
        log_decay = np.zeros((1, 2), dtype=np.float32)
        beta = np.ones((1, 2), dtype=np.float32)
        O.memory_update(state, k, v, log_decay, beta, heads=1, dim=2)
        np.testing.assert_allclose(
            state.reshape(2, 2), [[1.0, 0.0], [2.0, 3.0]], atol=1e-6
        )

    def test_update_decays_rows(self):
        state = np.eye(2, dtype=np.float32).reshape(1, 2, 2)
        k = np.zeros((1, 2), dtype=np.float32)
        v = np.zeros((1, 2), dtype=np.float32)
        beta = np.ones((1, 2), dtype=np.float32)
        log_decay = np.log(np.array([[0.5, 0.25]], dtype=np.float32))
        O.memory_update(state, k, v, log_decay, beta, heads=1, dim=2)
        np.testing.assert_allclose(state.reshape(2, 2), [[0.5, 0.0], [0.0, 0.25]], atol=1e-6)

    def test_zero_output_gate_zeroes_read(self):
        x = np.ones((1, 2), dtype=np.float32)
        gate = np.zeros((1, 2), dtype=np.float32)
        weight = np.ones(2, dtype=np.float32)
        out = O.gated_rmsnorm(x, gate, weight, 1e-6, 2.0, heads=1, dim=2)
        np.testing.assert_allclose(out, 0.0, atol=1e-7)


class WriteGate(unittest.TestCase):
    def test_gate_multiplies_the_value_not_the_residual(self):
        # A closed form independent of memory_update's loop:
        #   S' = (I - k k^T) diag(exp(g)) S + k (beta * v)^T
        # while the wrong "beta applied to the residual" form is
        #   S' = diag(exp(g)) S + k beta (v - k^T diag(exp(g)) S).
        H, D = 1, 2
        state = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        k = np.array([[0.5, -0.25]], dtype=np.float32)   # k_eff: already scaled
        v = np.array([[2.0, -1.0]], dtype=np.float32)
        beta = np.array([[0.25, 0.75]], dtype=np.float32)  # != 1 makes the forms differ
        log_decay = np.log(np.array([[0.5, 0.25]], dtype=np.float32))
        decay = np.diag(np.exp(log_decay[0]))
        expected = (np.eye(D) - np.outer(k[0], k[0])) @ decay @ state[0] + np.outer(k[0], beta[0] * v[0])
        wrong = decay @ state[0] + np.outer(k[0], beta[0] * (v[0] - k[0] @ decay @ state[0]))

        got = state.copy()
        O.memory_update(got, k, v, log_decay, beta, heads=H, dim=D)
        np.testing.assert_allclose(got[0], expected, atol=1e-6)
        self.assertFalse(np.allclose(got[0], wrong, atol=1e-6))

        # beta == 1 is the degenerate case where both formulas agree
        projected = (np.eye(D) - np.outer(k[0], k[0])) @ decay @ state[0] + np.outer(k[0], v[0])
        residual = decay @ state[0] + np.outer(k[0], v[0] - k[0] @ decay @ state[0])
        np.testing.assert_allclose(projected, residual, atol=1e-6)


class FrozenSnapshot(unittest.TestCase):
    def test_read_uses_the_request_snapshot_not_the_evolving_state(self):
        values = O.compute()
        trace = O.LAST_TRACE
        # token 2's read must be q2 against the state at request start
        q2 = trace["Q"][2]
        gate = trace["OUTGATE"][2]
        weight = trace["O_NORM"]

        def normed(state):
            return O.gated_rmsnorm(O.memory_read(q2, state), gate, weight, 1e-6, 2.0)

        frozen = normed(trace["STATE0"])
        evolved = normed(trace["STATES"][1])
        got = values["READ"].reshape(O.T, O.H, O.D)[2]
        np.testing.assert_allclose(got, frozen, atol=1e-6)
        self.assertGreater(float(np.max(np.abs(evolved - frozen))), 1e-5)


class Parity(unittest.TestCase):
    @unittest.skipIf(BINARY is None, "memt_cpu_test is not built")
    def test_python_matches_cpp(self):
        import subprocess

        result = subprocess.run([str(BINARY)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = O.compute()
        got = O.parse(result.stdout)
        problems = O.compare(expected, got)
        self.assertEqual(problems, [])

    def test_compare_catches_a_perturbation(self):
        expected = {"X": np.array([1.0, 2.0], dtype=np.float32)}
        got = {"X": np.array([1.0, 2.0 + 1e-3], dtype=np.float32)}
        self.assertTrue(O.compare(expected, got))
        got["X"] = expected["X"].copy()
        self.assertEqual(O.compare(expected, got), [])


if __name__ == "__main__":
    unittest.main()
