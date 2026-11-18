"""A radix trie of token-id sequences for automatic shared-prefix detection.

No Strata imports: the server (and only the server) decides what a match is worth.
Sequences are tuples of token ids, never text. A branch that nobody has touched for
``ttl_s`` seconds, or for ``idle_inserts`` later inserts, is dropped, so the trie
cannot grow for the life of the process.
"""
from __future__ import annotations

import time


class _Node:
    def __init__(self) -> None:
        self.edge: tuple[int, ...] = ()
        self.children: dict[int, _Node] = {}
        self.ref = None
        self.touched_at = 0.0
        self.touched_gen = 0
        self.hits = 0


class PrefixTrie:
    def __init__(self, *, ttl_s: float | None = 600.0, idle_inserts: int | None = 4096, clock=None):
        self.ttl_s = None if ttl_s is None or ttl_s <= 0 else float(ttl_s)
        self.idle_inserts = None if idle_inserts is None or idle_inserts <= 0 else int(idle_inserts)
        self._clock = clock or time.monotonic
        self._root = _Node()
        self._gen = 0
        self._count = 0
        self._last_sweep_gen = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, token_ids, ref) -> None:
        """Register ``token_ids`` as a source of a shared prefix. ``ref`` is whatever the caller uses to tell sources apart."""
        if not token_ids:
            return
        self._gen += 1
        now = self._clock()
        ids = tuple(int(t) for t in token_ids)
        node = self._root
        i = 0
        while i < len(ids):
            child = node.children.get(ids[i])
            if child is not None and self._stale(child, now):
                # Stale: drop it here instead of waiting for the next sweep, and insert fresh in its place.
                del node.children[ids[i]]
                if child.ref is not None:
                    self._count -= 1
                child = None
            if child is None:
                leaf = _Node()
                leaf.edge = ids[i:]
                leaf.ref = ref
                self._touch(leaf, now)
                node.children[ids[i]] = leaf
                self._count += 1
                self._prune(now)
                return
            edge = child.edge
            j = 0
            limit = min(len(edge), len(ids) - i)
            while j < limit and ids[i + j] == edge[j]:
                j += 1
            if j == len(edge):
                self._touch(child, now)
                i += j
                node = child
                continue
            mid = _Node()
            mid.edge = edge[:j]
            self._touch(mid, now)
            child.edge = edge[j:]
            mid.children[child.edge[0]] = child
            node.children[ids[i]] = mid
            if i + j == len(ids):
                mid.ref = ref
                self._count += 1
                self._prune(now)
                return
            leaf = _Node()
            leaf.edge = ids[i + j:]
            leaf.ref = ref
            self._touch(leaf, now)
            mid.children[leaf.edge[0]] = leaf
            self._count += 1
            self._prune(now)
            return
        if node.ref is None:
            self._count += 1
        node.ref = ref
        self._touch(node, now)
        self._prune(now)

    def longest_match(self, token_ids) -> tuple[int, object]:
        """``(length, ref)`` of the longest stored prefix of ``token_ids``. ``(0, None)`` when nothing matches.

        Reads never sweep the whole trie (``insert`` does that, throttled): a stale node found on this call's
        own path is dropped right here, so a lookup costs ``O(len(token_ids))``, not ``O(len(trie))``.
        """
        now = self._clock()
        if not token_ids:
            return 0, None
        node = self._root
        i = 0
        best_len = 0
        best_ref = None
        best_key = (-1, -1.0)
        while i < len(token_ids):
            key = int(token_ids[i])
            child = node.children.get(key)
            if child is None:
                break
            if self._stale(child, now):
                del node.children[key]
                if child.ref is not None:
                    self._count -= 1
                break
            edge = child.edge
            j = 0
            limit = min(len(edge), len(token_ids) - i)
            while j < limit and int(token_ids[i + j]) == edge[j]:
                j += 1
            if j <= 0:
                break
            self._touch(child, now)
            if j < len(edge):
                ref, hits, at = self._cover(child)
                if ref is not None and i + j >= best_len:
                    best_len, best_ref, best_key = i + j, ref, (hits, at)
                break
            i += j
            node = child
            if child.ref is not None and (i > best_len or (i == best_len and (child.hits, child.touched_at) >= best_key)):
                best_len, best_ref, best_key = i, child.ref, (child.hits, child.touched_at)
            elif child.ref is None and i > best_len:
                ref, hits, at = self._cover(child)
                if ref is not None:
                    best_len, best_ref, best_key = i, ref, (hits, at)
        return best_len, best_ref

    def invalidate(self, ref=None) -> None:
        """Drop one source, or every source when ``ref`` is omitted. Shared nodes that another source still needs stay."""
        if ref is None:
            self._root = _Node()
            self._count = 0
            return
        self._drop_ref(self._root, ref)

    def _drop_ref(self, node: _Node, ref) -> bool:
        if node.ref == ref:
            node.ref = None
            self._count = max(0, self._count - 1)
        for key in list(node.children):
            if not self._drop_ref(node.children[key], ref):
                del node.children[key]
        return node is self._root or node.ref is not None or bool(node.children)

    def _cover(self, node: _Node):
        """Some sequence through ``node``: its own ref, else the most-used descendant. A shared fork has no ref of its own."""
        if node.ref is not None:
            return node.ref, node.hits, node.touched_at
        best = (None, -1, -1.0)
        for child in node.children.values():
            ref, hits, at = self._cover(child)
            if ref is not None and (hits, at) >= (best[1], best[2]):
                best = (ref, hits, at)
        return best

    def _touch(self, node: _Node, now: float) -> None:
        node.touched_at = now
        node.touched_gen = self._gen
        node.hits += 1

    def _stale(self, node: _Node, now: float) -> bool:
        old_time = self.ttl_s is not None and (now - node.touched_at) > self.ttl_s
        old_gen = self.idle_inserts is not None and (self._gen - node.touched_gen) >= self.idle_inserts
        return old_time or old_gen

    _SWEEP_EVERY_GENS = 256    # between sweeps, a stale branch still leaves on its own path (see longest_match/insert)

    def _prune(self, now: float) -> None:
        """A full sweep: memory reclamation for branches nobody's path touches, not a correctness mechanism (the
        lazy check in ``insert`` and ``longest_match`` already keeps every answer right between sweeps). Throttled
        so it is not an O(trie size) cost on every call."""
        if self.ttl_s is None and self.idle_inserts is None:
            return
        if self._gen - self._last_sweep_gen < self._SWEEP_EVERY_GENS:
            return
        self._last_sweep_gen = self._gen
        self._prune_node(self._root, now)

    def _prune_node(self, node: _Node, now: float) -> None:
        for key in list(node.children):
            child = node.children[key]
            self._prune_node(child, now)
            if not self._stale(child, now):
                continue
            if child.ref is not None:
                child.ref = None
                self._count -= 1
            if not child.children:
                del node.children[key]
