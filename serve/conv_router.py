"""Decide, per request, whether the prompt/conversation cache should be on — and switch the engine when the traffic
shape changes.

Why this exists: on engine 0.1.57 the conversation cache costs ~8 % of a 512-token prefill (measured: pp512 672 with it,
891 without, same engine, toggled live through POST /v1/tune), and it pays that back only when a context is reused
(5.3 s -> 0.29 s on the second pass).  Single-shot traffic therefore pays a tax for a benefit it never collects, and
chat/agent traffic collects a benefit worth 17x the tax.

The decision is deliberately simple and biased to KEEPING the cache:
  - any request with more than one user turn, tools, or a prefix seen recently  -> cache ON (a repeat is coming)
  - a run of single-turn, never-seen prompts (batch_after in a row)             -> cache OFF
  - one conversational request switches it back immediately
A wrong "off" costs one full prefill (5.3 s) on the next repeat; a wrong "on" costs ~120 ms on a fresh request.  The
asymmetry is why the counter has to be a *run* and not a single request.

Nothing here talks to the engine: `decide()` is pure and `Router` keeps only counters and a small prefix history, so the
behaviour is testable without a GPU.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

CHAT = "chat"        # cache on
BATCH = "batch"      # cache off


@dataclass
class Decision:
    mode: str
    reason: str
    changed: bool                  # the engine should be retuned before this request runs
    wanted_cache: int              # the prompt_cache value to apply (0 = off)
    applied_to: str = ""           # the prompt_cache the engine was believed to have before this decision

    def __str__(self) -> str:
        return f"{self.mode} ({self.reason})"


@dataclass
class Router:
    """Hysteresis: `batch_after` consecutive single-shot requests are needed to switch the cache off."""

    cache_on: int = 6               # the prompt_cache value used in chat mode (the shipped setting)
    batch_after: int = 4
    repeat_window: int = 64         # how many recent prefixes count as "seen recently"
    mode: str = CHAT
    consecutive_single: int = 0
    seen: dict = field(default_factory=dict)
    stats: dict = field(default_factory=lambda: {"requests": 0, "chat": 0, "batch": 0,
                                                 "switches_on": 0, "switches_off": 0, "repeats": 0})

    # -- classification ------------------------------------------------------------------------------------------
    @staticmethod
    def user_turns(messages) -> int:
        return sum(1 for m in (messages or []) if isinstance(m, dict) and (m.get("role") or "").lower() == "user")

    @staticmethod
    def prefix_key(messages, chars: int = 1024) -> str:
        """A key for "have we seen this context before": the first user turn, trimmed and hashed.

        Deliberately the *first* turn, not the last: an agent appends to the conversation, so the first turn is what
        stays stable across turns while the tail grows.
        """
        for m in (messages or []):
            if isinstance(m, dict) and (m.get("role") or "").lower() == "user":
                text = m.get("content")
                if not isinstance(text, str):
                    text = " ".join(p.get("text", "") for p in text if isinstance(p, dict)) if isinstance(text, list) else ""
                return hashlib.blake2b(" ".join(text.split())[:chars].encode(), digest_size=8).hexdigest()
        return ""

    # -- the decision --------------------------------------------------------------------------------------------
    def decide(self, messages, tools=None) -> Decision:
        self.stats["requests"] += 1
        turns = self.user_turns(messages)
        key = self.prefix_key(messages)
        repeat = bool(key) and key in self.seen
        if key:
            self.seen[key] = self.seen.get(key, 0) + 1
            if len(self.seen) > self.repeat_window:
                oldest = next(iter(self.seen))
                del self.seen[oldest]
        if repeat:
            self.stats["repeats"] += 1

        conversational = turns > 1 or bool(tools) or repeat
        if conversational:
            self.consecutive_single = 0
            why = "repeat prefix" if repeat else ("tools" if tools else f"{turns} user turns")
            return self._to(CHAT, why)
        # a single, never-seen prompt: batch-shaped, but it has to be a RUN before the cache goes off
        self.consecutive_single += 1
        if self.mode == BATCH:
            self.stats["batch"] += 1
            return Decision(BATCH, "single-shot (already batch)", False, 0)
        if self.consecutive_single >= self.batch_after:
            return self._to(BATCH, f"{self.consecutive_single} single-shot requests in a row")
        self.stats["chat"] += 1
        return Decision(CHAT, f"single-shot {self.consecutive_single}/{self.batch_after}", False, self.cache_on)

    def _to(self, mode: str, reason: str) -> Decision:
        if mode == self.mode:
            self.stats["chat" if mode == CHAT else "batch"] += 1
            return Decision(mode, reason, False, self.cache_on if mode == CHAT else 0)
        self.mode = mode
        if mode == CHAT:
            self.stats["switches_on"] += 1
            self.stats["chat"] += 1
            return Decision(CHAT, f"back to chat: {reason}", True, self.cache_on)
        self.stats["switches_off"] += 1
        self.stats["batch"] += 1
        return Decision(BATCH, f"batch: {reason}", True, 0)

    def note_applied(self, decision: Decision) -> None:
        """The engine took the switch; if it refused, the caller calls fail() and the mode rolls back."""
        decision.applied_to = str(decision.wanted_cache)

    def revert(self) -> None:
        """A refused tune: go back to the cache-on mode so the next decision retries."""
        self.mode = CHAT if self.mode == BATCH else BATCH
        self.consecutive_single = 0

    def report(self) -> dict:
        return {"mode": self.mode, "consecutive_single": self.consecutive_single,
                "batch_after": self.batch_after, "cache_on": self.cache_on, **self.stats}
