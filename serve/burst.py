"""Chunked round-robin decoding for the DeepSeek-V4.1 engine (docs/DSV41_BATCHING.md, Path B2).

The engine runs one request at a time, so a second client waits for the first answer to finish.  A request can
however be split into short generations: GEN with max_new = burst, then a new GEN whose ids are the previous
prompt plus what was generated.  The engine keeps the conversation and reuses the shared prefix, so a
continuation costs only the new tokens - and between two chunks another request can take the engine.

This module is the bookkeeping of that split, with no engine and no HTTP in it: chunk sizes, the growing id list
for the continuation, and the stop conditions (EOS, max_tokens, cancellation).  serve/server.py calls it; the
tests beside it run anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BurstCursor:
    """One request progress through a chunked generation."""

    ids: list[int]                 # the prompt, then prompt + generated
    max_new: int
    burst: int                     # tokens per GEN call
    produced: int = 0
    finished: bool = False

    def next_chunk(self) -> int:
        """How many tokens the next GEN call may ask for (0: nothing left)."""
        if self.finished:
            return 0
        left = self.max_new - self.produced
        if left <= 0:
            self.finished = True
            return 0
        return min(self.burst, left)

    def accept(self, tokens: list[int], eos: int | None = None) -> bool:
        """Record what a chunk produced; True while the request continues."""
        if self.finished:
            return False
        cut = len(tokens)
        if eos is not None:
            for i, t in enumerate(tokens):
                if t == eos:
                    cut = i + 1
                    break
        self.ids.extend(tokens[:cut])
        self.produced += cut
        if eos is not None and cut < len(tokens):
            self.finished = True
        elif self.produced >= self.max_new:
            self.finished = True
        return not self.finished

    def continuation(self) -> list[int]:
        """The ids the next GEN call sends: the prompt and everything generated so far."""
        return list(self.ids)


def split(max_new: int, burst: int) -> list[int]:
    """The chunk sizes one request needs (empty when max_new or burst is not positive)."""
    if max_new <= 0 or burst <= 0:
        return []
    out = []
    left = max_new
    while left > 0:
        n = min(burst, left)
        out.append(n)
        left -= n
    return out


def pass_limit(max_new: int, produced: int, burst: int) -> int:
    """What one GEN pass may ask for: the tokens left, capped by the burst size (0: burst off)."""
    left = max_new - produced
    if left <= 0:
        return 0
    return left if burst <= 0 else min(left, burst)


def ended_by_chunk(max_new: int, produced_in_pass: int, limit: int, burst: int) -> bool:
    """True when a pass stopped at the chunk cap rather than at the model (EOS, cancel, recovery)."""
    return burst > 0 and limit > 0 and produced_in_pass >= limit and produced_in_pass < max_new
