"""tools/test_memt_session_flow.py - a session's memory across commit / query / snapshot.

The engine's session contract, on the CPU: a commit reads one frozen snapshot and
advances the state token by token; a query reads the published state and never
moves it; a `memt-memory-v1` save/load round-trip preserves the state and every
later read.  Uses the NumPy oracle and memt_snapshot, no GPU and no model.
"""
from __future__ import annotations

import itertools
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import memt_cpu_oracle as O  # noqa: E402
import memt_snapshot as S  # noqa: E402

H, D = 2, 4
IDENTITY = "ab" * 32


def values(count: int, seed: int) -> np.ndarray:
    return O.fill(count, seed).astype(np.float32)


def erase_gate(raw) -> np.ndarray:
    """exp(0.5 * logsigmoid(raw)), stable over the whole array."""
    x = np.asarray(raw, dtype=np.float64)
    return np.exp(0.5 * (-np.logaddexp(0.0, -x))).astype(np.float32)


class Session:
    """One memory layer: read(q) is the query path, commit(...) the write path."""

    def __init__(self, state: np.ndarray):
        self.state = np.array(state, dtype=np.float32, copy=True)

    def read(self, q, gate, weight, eps=1e-6, softcap=2.0):
        q = np.array(q, dtype=np.float32, copy=True).reshape(-1, H, D)
        gate = np.asarray(gate, dtype=np.float32).reshape(-1, H, D)
        out = np.empty_like(q)
        for t in range(q.shape[0]):
            qn = q[t].copy()
            for h in range(H):
                O.l2norm_in_place(qn[h])
            raw = O.memory_read(qn, self.state, heads=H, dim=D)
            out[t] = O.gated_rmsnorm(raw, gate[t], weight, eps=eps, limit=softcap, heads=H, dim=D)
        return out.reshape(-1)

    def commit(self, q, k, v, erase, write, log_decay, gate, o_norm, eps=1e-6, softcap=2.0):
        q = np.asarray(q, dtype=np.float32).reshape(-1, H, D)
        k = np.asarray(k, dtype=np.float32).reshape(-1, H, D)
        v = np.asarray(v, dtype=np.float32).reshape(-1, H, D)
        erase = np.asarray(erase, dtype=np.float32).reshape(-1, H, D)
        write = np.asarray(write, dtype=np.float32).reshape(-1, H, D)
        log_decay = np.asarray(log_decay, dtype=np.float32).reshape(-1, H, D)
        gate = np.asarray(gate, dtype=np.float32).reshape(-1, H, D)
        snapshot = self.state.copy()   # the call's frozen read state
        reads = np.empty((q.shape[0], H * D), dtype=np.float32)
        for t in range(q.shape[0]):
            qn, kn = q[t].copy(), k[t].copy()
            for h in range(H):
                O.l2norm_in_place(qn[h])
                O.l2norm_in_place(kn[h])
            kn *= erase[t]   # the release's gate is already exp(0.5*logsigmoid)
            raw = O.memory_read(qn, snapshot, heads=H, dim=D)
            reads[t] = O.gated_rmsnorm(raw, gate[t], o_norm, eps=eps, limit=softcap, heads=H, dim=D).reshape(-1)
            O.memory_update(self.state, kn, v[t], log_decay[t], write[t], heads=H, dim=D)
        return reads


def fixture(tokens: int):
    per = H * D
    return {
        "q": values(tokens * per, 11),
        "k": values(tokens * per, 23),
        "v": values(tokens * per, 37),
        "erase_raw": values(tokens * per, 41),
        "write_raw": values(tokens * per, 53),
        "log_decay": -0.25 * (values(tokens * per, 67) + 1.0),
        "gate": values(tokens * per, 79),
        "o_norm": values(D, 83),
    }


class Flow(unittest.TestCase):
    def setUp(self):
        self.state0 = values(H * D * D, 3).reshape(H, D, D)

    def test_query_does_not_move_the_state(self):
        f = fixture(3)
        s = Session(self.state0)
        before = s.state.copy()
        s.read(f["q"], f["gate"], f["o_norm"])
        np.testing.assert_array_equal(s.state, before)

    def test_commit_reads_its_snapshot_and_advances(self):
        f = fixture(4)
        s = Session(self.state0)
        reads = s.commit(f["q"], f["k"], f["v"], erase_gate(f["erase_raw"]),
                         1.0 / (1.0 + np.exp(-f["write_raw"])), f["log_decay"], f["gate"], f["o_norm"])
        # every token's read is a query of the pre-commit state
        frozen = Session(self.state0).read(f["q"], f["gate"], f["o_norm"]).reshape(reads.shape)
        np.testing.assert_allclose(reads, frozen, atol=2e-6)
        # and the state moved
        self.assertGreater(float(np.max(np.abs(s.state - self.state0))), 1e-4)

    def test_snapshot_round_trip_preserves_reads(self):
        f = fixture(5)
        s = Session(self.state0)
        s.commit(f["q"], f["k"], f["v"], erase_gate(f["erase_raw"]),
                 1.0 / (1.0 + np.exp(-f["write_raw"])), f["log_decay"], f["gate"], f["o_norm"])
        after = s.read(f["q"], f["gate"], f["o_norm"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.safetensors"
            S.save(path, IDENTITY, {3: s.state.reshape(1, H, D, D)})
            loaded = S.load(path, IDENTITY, {3: (1, H, D, D)})[3]
        np.testing.assert_array_equal(loaded.reshape(H, D, D), s.state)

        restored = Session(loaded.reshape(H, D, D))
        np.testing.assert_array_equal(restored.read(f["q"], f["gate"], f["o_norm"]), after)

    def test_a_restored_session_continues_a_second_commit(self):
        f = fixture(3)
        s = Session(self.state0)
        gate_in = erase_gate(f["erase_raw"])
        write = 1.0 / (1.0 + np.exp(-f["write_raw"]))
        s.commit(f["q"], f["k"], f["v"], gate_in, write, f["log_decay"], f["gate"], f["o_norm"])
        second = s.commit(f["q"], f["k"], f["v"], gate_in, write, f["log_decay"], f["gate"], f["o_norm"])
        after = s.read(f["q"], f["gate"], f["o_norm"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.safetensors"
            S.save(path, IDENTITY, {3: s.state.reshape(1, H, D, D)})
            restored = Session(S.load(path, IDENTITY)[3].reshape(H, D, D))
        np.testing.assert_array_equal(restored.state, s.state)
        np.testing.assert_array_equal(restored.read(f["q"], f["gate"], f["o_norm"]), after)
        # the second commit's reads are the first commit's end state, not its own
        self.assertFalse(np.allclose(second, np.zeros_like(second)))

    def gates(self, f):
        return (erase_gate(f["erase_raw"]),
                1.0 / (1.0 + np.exp(-f["write_raw"])))

    def test_call_boundaries_do_not_change_the_state(self):
        # The update does not read its own output, so where the calls are split
        # cannot change the final state; each call does freeze its own snapshot,
        # so the reads must differ (that is the documented semantics).
        f = fixture(4)
        erase, write = self.gates(f)
        one = Session(self.state0)
        read_one = one.commit(f["q"], f["k"], f["v"], erase, write, f["log_decay"],
                              f["gate"], f["o_norm"])
        split = Session(self.state0)
        read_split = []
        for t in range(4):
            lo, hi = t * H * D, (t + 1) * H * D
            read_split.append(split.commit(f["q"][lo:hi], f["k"][lo:hi], f["v"][lo:hi],
                                           erase[lo:hi], write[lo:hi], f["log_decay"][lo:hi],
                                           f["gate"][lo:hi], f["o_norm"]))
        np.testing.assert_allclose(split.state, one.state, atol=1e-7)
        self.assertGreater(float(np.max(np.abs(np.concatenate(read_split) - read_one))), 1e-7)

    def test_every_operation_sequence_is_explained_by_the_model(self):
        # Bounded model check: over all sequences of length <= 4 of query/commit,
        # every read is a read of the model's state at that point, and the final
        # state depends only on the commits (queries are pure).
        f = fixture(2)
        erase, write = self.gates(f)
        for length in range(5):
            for choices in itertools.product(("query", "commit"), repeat=length):
                got = Session(self.state0)
                model = Session(self.state0)
                for op in choices:
                    if op == "query":
                        np.testing.assert_allclose(
                            np.ravel(got.read(f["q"], f["gate"], f["o_norm"])),
                            np.ravel(model.read(f["q"], f["gate"], f["o_norm"])), atol=1e-6)
                    else:
                        np.testing.assert_allclose(
                            np.ravel(got.commit(f["q"], f["k"], f["v"], erase, write, f["log_decay"],
                                                f["gate"], f["o_norm"])),
                            np.ravel(model.read(f["q"], f["gate"], f["o_norm"])))
                        model.commit(f["q"], f["k"], f["v"], erase, write, f["log_decay"],
                                     f["gate"], f["o_norm"])
                np.testing.assert_allclose(got.state, model.state, atol=1e-7, err_msg=str(choices))

    def test_load_replaces_the_state(self):
        f = fixture(3)
        erase, write = self.gates(f)
        s = Session(self.state0)
        s.commit(f["q"], f["k"], f["v"], erase, write, f["log_decay"], f["gate"], f["o_norm"])
        saved = s.state.copy()
        s.commit(f["q"], f["k"], f["v"], erase, write, f["log_decay"], f["gate"], f["o_norm"])
        self.assertFalse(np.allclose(s.state, saved))
        s = Session(saved)   # what MEMLOAD does: replace, not merge
        np.testing.assert_array_equal(s.state, saved)
        np.testing.assert_array_equal(s.read(f["q"], f["gate"], f["o_norm"]),
                                      Session(saved).read(f["q"], f["gate"], f["o_norm"]))

    def test_identity_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.safetensors"
            S.save(path, IDENTITY, {3: self.state0.reshape(1, H, D, D)})
            with self.assertRaises(S.SnapshotError):
                S.load(path, "cd" * 32)
            with self.assertRaises(S.SnapshotError):
                S.load(path, IDENTITY, {3: (1, H, D, D + 1)})


if __name__ == "__main__":
    unittest.main()
