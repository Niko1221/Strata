"""Which shared prefixes are actually pinned, and which new ones may be.

A pin keeps a checkpoint in VRAM that the expert cache would otherwise use (docs/BATCHING.md,
"What a slot costs"). The trie may remember every recent prompt; this registry is the budget
on how many of those the server is willing to pin.

The engine never evicts a pin until the process is gone, and this module cannot unpin one.
The default policy is therefore to refuse a new pin once ``max_pins`` are held. ``policy="evict"``
drops the least-used registry entry (fewest accepts, then the one used longest ago — not the
one inserted longest ago) so a caller that really did release the pin can test that path.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time

from serve.prefix_trie import PrefixTrie

# A reused pin may re-read a short overlap (the engine's short-read is 64 tokens). More than
# that past the unpinned suffix means the checkpoint was not there.
MISS_SLACK = 64


def _digest(token_ids, length: int) -> str:
    h = hashlib.sha256()
    h.update(int(length).to_bytes(8, "little", signed=False))
    for tok in token_ids[:length]:
        h.update(int(tok).to_bytes(8, "little", signed=True))
    return h.hexdigest()[:16]


def token_ref(token_ids) -> str:
    return _digest(token_ids, len(token_ids))


class Pin:
    def __init__(self, length: int, digest: str, ref, now: float, order: int):
        self.length = length
        self.digest = digest
        self.ref = ref
        self.hits = 0
        self.last_used = now
        self.order = order
        self.established = False     # the engine has been asked and a later read has not shown the pin missing


class PinRegistry:
    def __init__(self, *, min_tokens: int = 256, max_pins: int = 4, policy: str = "refuse", clock=None):
        if policy not in ("refuse", "evict"):
            raise ValueError("pin policy is 'refuse' or 'evict'")
        self.min_tokens = max(1, int(min_tokens))
        self.max_pins = max(1, int(max_pins))
        self.policy = policy
        self._clock = clock or time.monotonic
        self._pins: dict[tuple[int, str], Pin] = {}
        self._seq = 0

    def __len__(self) -> int:
        return len(self._pins)

    def key_for(self, token_ids, length: int) -> tuple[int, str]:
        length = int(length)
        return (length, _digest(token_ids, length))

    def get(self, key: tuple[int, str]) -> Pin | None:
        return self._pins.get(key)

    def propose(self, token_ids, match_len: int) -> int | None:
        """The length to pin, or None when the match is shorter than ``min_tokens`` or the budget refuses a new one.

        An existing key is reused and does not take a new slot. A new key is reserved (not yet
        established) so two callers cannot both pass the budget check.
        """
        if not isinstance(match_len, int) or isinstance(match_len, bool) or match_len < self.min_tokens:
            return None
        key = self.key_for(token_ids, match_len)
        pin = self._pins.get(key)
        now = self._clock()
        if pin is not None:
            pin.hits += 1
            pin.last_used = now
            return match_len
        if len(self._pins) >= self.max_pins:
            if self.policy != "evict":
                return None
            self.evict_lru()
        self._seq += 1
        self._pins[key] = Pin(match_len, key[1], None, now, self._seq)
        return match_len

    def commit(self, token_ids, length: int, ref, *, force: bool = False) -> bool:
        """Record a pin the server is actually sending. ``force`` is a pin the client (or DeepSeek's own rule) already asked for: it counts even under the minimum and even when the budget is full."""
        if not isinstance(length, int) or isinstance(length, bool) or length < 1:
            return False
        if length < self.min_tokens and not force:
            return False
        key = self.key_for(token_ids, length)
        now = self._clock()
        pin = self._pins.get(key)
        if pin is not None:
            pin.ref = ref
            pin.last_used = now
            return True
        if len(self._pins) >= self.max_pins and not force:
            if self.policy != "evict":
                return False
            self.evict_lru()
        self._seq += 1
        made = Pin(length, key[1], ref, now, self._seq)
        self._pins[key] = made
        return True

    def release(self, token_ids, length: int) -> None:
        """Drop a reservation the engine was never asked to pin (resolve_prefix refused it)."""
        key = self.key_for(token_ids, length)
        pin = self._pins.get(key)
        if pin is not None and not pin.established:
            self._pins.pop(key, None)

    def drop(self, key: tuple[int, str]) -> None:
        self._pins.pop(key, None)

    def evict_lru(self) -> Pin | None:
        """The least-used pin: fewest accepts, then the one whose last accept is oldest."""
        if not self._pins:
            return None
        key = min(self._pins, key=lambda k: (self._pins[k].hits, self._pins[k].last_used, self._pins[k].order))
        return self._pins.pop(key)

    def clear(self) -> None:
        self._pins.clear()


def _env_int(env, name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(env, name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class AutoPrefix:
    """The server-facing pair: match a prompt, stay inside the pin budget, forget the lot when the engine's KV is gone."""

    def __init__(self, trie: PrefixTrie | None = None, registry: PinRegistry | None = None, *, enabled: bool = True):
        self.enabled = bool(enabled)
        self.trie = trie if trie is not None else PrefixTrie()
        self.registry = registry if registry is not None else PinRegistry()
        self._lock = threading.Lock()
        self._epoch = None
        self._armed: dict[str, int] = {}

    @classmethod
    def from_env(cls, env=None) -> AutoPrefix:
        """Off unless STRATA_AUTO_PREFIX is 1/true/on/yes. The default request path does not call it."""
        env = os.environ if env is None else env
        flag = str(env.get("STRATA_AUTO_PREFIX", "0")).strip().lower()
        enabled = flag in {"1", "true", "on", "yes"}
        trie = PrefixTrie(ttl_s=_env_float(env, "STRATA_AUTO_PREFIX_TTL", 600.0),
                          idle_inserts=_env_int(env, "STRATA_AUTO_PREFIX_IDLE", 4096))
        registry = PinRegistry(min_tokens=_env_int(env, "STRATA_AUTO_PREFIX_MIN", 256),
                               max_pins=_env_int(env, "STRATA_AUTO_PREFIX_PINS", 4),
                               policy="refuse")
        return cls(trie, registry, enabled=enabled)

    def sync(self, epoch) -> None:
        """Drop every remembered prompt when the engine process changes. The first epoch is only recorded."""
        with self._lock:
            if self._epoch is None:
                self._epoch = epoch
                return
            if epoch != self._epoch:
                self._wipe()
                self._epoch = epoch

    def invalidate(self, *, keep_armed: bool = False) -> None:
        """The engine restarted, was unloaded, or dropped its KV (a non-finite retry reads from token 0).

        ``keep_armed`` is the one retry that reads the same prompt again: the registry is wrong, but that
        prompt is still in flight and confirm_read must be allowed to record it if the retry finishes.
        """
        with self._lock:
            self._wipe(keep_armed=keep_armed)

    def _wipe(self, *, keep_armed: bool = False) -> None:
        self.trie.invalidate()
        self.registry.clear()
        if not keep_armed:
            self._armed.clear()

    def propose(self, token_ids) -> tuple[int | None, object]:
        """``(pin length, ref)`` to hand to resolve_prefix as ``{"tokens": n}``, or ``(None, ref)``."""
        if not self.enabled:
            return None, None
        with self._lock:
            n, ref = self.trie.longest_match(token_ids)
            if n >= len(token_ids) - 1:
                n = len(token_ids) - 2
            if n < 1:
                return None, ref
            return self.registry.propose(token_ids, n), ref

    def release(self, token_ids, length: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.registry.release(token_ids, length)

    def remember(self, token_ids, pin_tokens: int | None, *, client: bool = False) -> None:
        """The prompt was accepted. It becomes a future match, and a pin that survived resolve_prefix is recorded."""
        if not self.enabled or not token_ids:
            return
        with self._lock:
            ref = token_ref(token_ids)
            self._armed[ref] = self._armed.get(ref, 0) + 1
            self.trie.insert(token_ids, ref)
            if isinstance(pin_tokens, int) and not isinstance(pin_tokens, bool) and pin_tokens > 0:
                self.registry.commit(token_ids, pin_tokens, ref, force=client)

    def confirm_read(self, token_ids, pin_tokens, prompt_read, finish: str) -> None:
        """After the engine read the prompt: keep the pin, or drop it when this read shows the checkpoint was not there.

        The request that first asks for a pin reads the prefix (nothing is cached yet). Only a later
        request can show that the pin is gone, and only when the engine reports ``prompt_read``.
        """
        if not self.enabled or not token_ids:
            return
        with self._lock:
            ref = token_ref(token_ids)
            armed = self._armed.get(ref, 0)
            if armed <= 0:
                return
            if armed == 1:
                self._armed.pop(ref, None)
            else:
                self._armed[ref] = armed - 1
            incomplete = finish in ("cancel", "disconnect", "error") and (
                prompt_read is None or int(prompt_read) < len(token_ids))
            if incomplete:
                self.trie.invalidate(ref)
                if isinstance(pin_tokens, int) and not isinstance(pin_tokens, bool) and pin_tokens > 0:
                    self.registry.drop(self.registry.key_for(token_ids, pin_tokens))
                return
            self.trie.insert(token_ids, ref)
            if not isinstance(pin_tokens, int) or isinstance(pin_tokens, bool) or pin_tokens < 1:
                return
            key = self.registry.key_for(token_ids, pin_tokens)
            pin = self.registry.get(key)
            if pin is None:
                self.registry.commit(token_ids, pin_tokens, ref, force=True)
                pin = self.registry.get(key)
            if pin is None:
                return
            if not pin.established:
                pin.established = True
                pin.hits = max(pin.hits, 1)
                pin.ref = ref
                return
            if prompt_read is None:
                pin.hits += 1
                pin.last_used = time.monotonic()
                pin.ref = ref
                return
            suffix = max(0, len(token_ids) - pin_tokens)
            if int(prompt_read) > suffix + MISS_SLACK:
                self.registry.drop(key)
                self.trie.invalidate(ref)
                return
            pin.hits += 1
            pin.last_used = time.monotonic()
            pin.ref = ref
