from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import hmac
import unittest


@dataclass(frozen=True)
class Snapshot:
    # Tuples and bytes make the contained values immutable as well.
    tokens: tuple[int, ...]
    stream_key: bytes
    next_draw: int = 0

    def __post_init__(self):
        if not self.tokens or any(type(x) is not int for x in self.tokens):
            raise ValueError("Use a nonempty tuple of integer toy values")
        if type(self.tokens) is not tuple or type(self.stream_key) is not bytes:
            raise TypeError("Snapshot storage must be immutable")
        if (not self.stream_key or type(self.next_draw) is not int
                or not 0 <= self.next_draw < 2**128):
            raise ValueError("Invalid stream key or counter")


def attention(s: Snapshot) -> Fraction:
    # Fresh oracle: compute directly from the authoritative input tuple.
    return Fraction(sum(s.tokens), len(s.tokens))


def feed(s: Snapshot, values: tuple[int, ...]) -> Snapshot:
    if type(values) is not tuple or any(type(x) is not int for x in values):
        raise TypeError("Expected a tuple of integer toy inputs")
    # This toy has no grammar or penalties; feeding inputs consumes no RNG.
    return replace(s, tokens=s.tokens + values)


def next_uniform(s: Snapshot) -> Fraction:
    # Exact rational variate in (0, 1); no process-global random state.
    payload = b"context-time-travel-toy-v1:" + s.next_draw.to_bytes(16, "big")
    bits = hmac.new(s.stream_key, payload, hashlib.sha256).digest()
    integer = int.from_bytes(bits, "big")
    return Fraction(2 * integer + 1, 2**257)


def step(s: Snapshot) -> tuple[Snapshot, int]:
    z = attention(s)
    p_positive = Fraction(1, 2) + z / (4 * (1 + abs(z)))
    token = 1 if next_uniform(s) < p_positive else -1
    child = replace(s, tokens=s.tokens + (token,), next_draw=s.next_draw + 1)
    return child, token


def generate(s: Snapshot, count: int) -> tuple[Snapshot, tuple[int, ...]]:
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    output = []
    for _ in range(count):
        s, token = step(s)
        output.append(token)
    return s, tuple(output)


class TimeTravelTests(unittest.TestCase):
    def setUp(self):
        self.p = Snapshot((1, 3), b"fixed-experiment-key")

    def test_worked_attention_outputs(self):
        a = feed(self.p, (5,))
        b = feed(self.p, (9,))
        self.assertEqual(attention(self.p), Fraction(2))
        self.assertEqual(attention(a), Fraction(3))
        self.assertEqual(attention(b), Fraction(13, 3))

    def test_rewind_does_not_delete_old_future(self):
        old_a = feed(self.p, (5,))
        branch_a = old_a
        branch_a = self.p  # Rewind only moves this bookmark.
        branch_a = feed(branch_a, (-1,))
        self.assertEqual(branch_a.tokens, (1, 3, -1))
        self.assertEqual(attention(branch_a), Fraction(1))
        self.assertEqual(old_a.tokens, (1, 3, 5))
        self.assertEqual(self.p.tokens, (1, 3))

    def test_same_snapshot_repeats_sampled_future(self):
        mid, _ = generate(self.p, 7)
        final_a, out_a = generate(mid, 20)
        final_b, out_b = generate(mid, 20)
        self.assertEqual(out_a, out_b)
        self.assertEqual(final_a, final_b)

    def test_chunking_does_not_reset_random_stream(self):
        all_at_once, whole = generate(self.p, 19)
        mid, first = generate(self.p, 7)
        in_chunks, second = generate(mid, 12)
        self.assertEqual(whole, first + second)
        self.assertEqual(all_at_once, in_chunks)

    def test_sibling_scheduling_does_not_change_a(self):
        a = self.p
        b = replace(self.p, stream_key=b"another-recorded-stream")
        a, first = generate(a, 4)
        b, _ = generate(b, 50)  # Sibling activity must not consume A's draws.
        a, second = generate(a, 8)
        isolated, expected = generate(self.p, 12)
        self.assertEqual(first + second, expected)
        self.assertEqual(a, isolated)

    def test_fresh_input_oracle_matches_each_forced_future(self):
        for suffix in ((5,), (9,), (-1,), (5, 9, -1), ()):
            child = feed(self.p, suffix)
            fresh = Snapshot((1, 3) + suffix, self.p.stream_key)
            self.assertEqual(attention(child), attention(fresh))

    def test_eviction_preserves_history_and_rng_position(self):
        mid, _ = generate(self.p, 9)
        # Simulated derived cache; it is not the authoritative history.
        checkpoints = {mid: (sum(mid.tokens), len(mid.tokens))}
        checkpoints.clear()
        reconstructed = Snapshot(mid.tokens, mid.stream_key, mid.next_draw)
        self.assertEqual(generate(mid, 11), generate(reconstructed, 11))

    def test_other_stream_does_not_mutate_parent(self):
        other = replace(self.p, stream_key=b"candidate-B", next_draw=0)
        generate(other, 30)
        self.assertEqual(self.p.tokens, (1, 3))
        self.assertEqual(self.p.next_draw, 0)
        # Deliberately do not assert that different streams produce different text.

    def test_feed_associativity(self):
        left = feed(feed(self.p, (5, 9)), (-1,))
        right = feed(self.p, (5, 9, -1))
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main(verbosity=2)
