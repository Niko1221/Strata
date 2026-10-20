"""Data-parallel replicas: N independent engines, each on its own group of cards, behind one server.

A model that fits two cards (IQ3_S on two R9700s) runs better as two 2-card engines than as one 4-card layer split:
the split runs one pipeline whose stages wait on each other, two replicas run two pipelines.  The server keeps its one
HTTP front, its one Service and its one set of endpoints; `ReplicaEngine` stands in for the engine and spreads the
requests over the replicas.

Opt-in, in the run config:

    "replicas": 2                                     the "gpu" list cut into 2 equal groups ([0,1] and [2,3])
    "replicas": [{"gpus": [0, 1]}, {"gpus": [2, 3]}]  the groups named; an entry may add "env" and "args"

or `serve/server.py --replicas 2`.  Nothing changes without it.

Routing.  Each request goes to the replica with the lowest cost, cost = prompt tokens the replica would have to read
again (it holds the start of a conversation in its cache) + LOAD_TOKENS for every request already running there.  A
conversation's next turn therefore goes back to its replica, until that replica is so much busier than another that
reading the prompt again is cheaper than waiting.  What each replica holds is tracked as hashes of 64-token blocks of
the prompts and answers it served (the engine itself checks the real prefix, as it always does).

A conversation is named by the request itself: the standard `prompt_cache_key` when the client sends one (an
explicit name wins), else the hash of the first user message, which every turn resends unchanged (see
`request_conversation`).  A named conversation keeps the replica that first served it for as long as that
replica runs, whatever the load, because a migration reads the prompt again; a new name takes the card holding
the fewest conversations (sessions split across cards even when both look idle) and is bound there; a
conversation whose replica restarted holds nothing there and is placed again.
Clients that name nothing at all (no user text to derive it from) keep the cost rule with no stickiness.

CPU cores.  On Linux the replicas get disjoint physical cores (serve/cpu_affinity.py, from PR #1436 by agorevski), so
the host threads of the engines (the expert workers) do not compete.  "replica_cpus": false turns it off.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import os
import threading
import time

BLOCK = 64                          # tokens per hashed block of a prompt
LOAD_TOKENS = 3000                  # the cost of one request already running on a replica, in prompt tokens
INDEX_MAX = 262144                  # block hashes kept (about 17M tokens of prompts)
CONV_MAX = 4096                     # conversations remembered (each keeps its replica while that one runs)
CONV_MAX_LEN = 128                  # characters kept of a conversation id
NEW_TALK_SLACK = 2 * BLOCK          # tokens a prompt may lose to its own tail (a template boundary) and still be a turn
HEAL_COOLDOWN_S = 60.0              # a replica that died is started again at most this often


class ReplicaConfigError(ValueError):
    pass


def gpu_groups(cfg: dict) -> list[list[int]]:
    """The cards of each replica, from the config's "replicas" (see the module doc); [] when there are none or one."""
    spec = cfg.get("replicas")
    if spec is None or spec is False:
        return []
    cards = _cards(cfg.get("gpu"))
    if isinstance(spec, bool) or (not isinstance(spec, (int, list))):
        raise ReplicaConfigError(f'"replicas" must be a number or a list of {{"gpus": [...]}}, not {spec!r}')
    if isinstance(spec, int):
        if spec <= 1:
            return []
        if not cards:
            raise ReplicaConfigError(f'"replicas": {spec} needs the "gpu" list to cut into groups (e.g. "gpu": [0, 1, 2, 3])')
        if len(cards) % spec:
            raise ReplicaConfigError(f'"replicas": {spec} does not divide the {len(cards)} cards {cards} evenly: name the '
                                     f'groups ("replicas": [{{"gpus": [0, 1]}}, ...]) or change the count')
        n = len(cards) // spec
        return [cards[i * n:(i + 1) * n] for i in range(spec)]
    groups = []
    for e in spec:
        g = e.get("gpus") if isinstance(e, dict) else e
        groups.append(_cards(g))
    if any(not g for g in groups):
        raise ReplicaConfigError('every replica needs its cards: {"gpus": [0, 1]}')
    flat = [c for g in groups for c in g]
    if len(set(flat)) != len(flat):
        raise ReplicaConfigError(f"a card is in two replicas: {groups}")
    if len(groups) < 2:
        return []
    return groups


def _cards(g) -> list[int]:
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    try:
        return [int(str(x).strip()) for x in items if str(x).strip() != ""]
    except ValueError:
        raise ReplicaConfigError(f"cards must be numbers, not {g!r}") from None


def conversation_id(req: dict) -> str | None:
    """The conversation this request belongs to, from the standard `prompt_cache_key` request field: OpenAI, Azure
    and the inference servers that read it define it as "route requests with the same key together so they share a
    cache", which is exactly this router's question.  None when the client sends none: routing then falls back to
    the prefix hashes.  No other field or header is read, so a request that follows the standard behaves the same
    everywhere."""
    v = (req or {}).get("prompt_cache_key")
    return v.strip()[:CONV_MAX_LEN] if isinstance(v, str) and v.strip() else None


def conversation_key(messages) -> str | None:
    """A name for the conversation, taken from the request's own content: the first user message, which a client
    resends unchanged on every turn.  Stable for one conversation, different between two that start differently, so
    no client has to send anything and the protocol keeps its standard shape.  None when the first user message has
    no text (an image, or no user turn at all): routing then falls back to the prefix hashes."""
    for m in messages or ():
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        text = m.get("content")
        if isinstance(text, list):
            text = "".join(p.get("text") or "" for p in text if isinstance(p, dict) and p.get("type") == "text")
        if isinstance(text, str) and text.strip():
            return "first:" + hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:24]
        return None
    return None


def request_conversation(req: dict, messages) -> str | None:
    """The conversation a request belongs to: the standard `prompt_cache_key` when its client sends one (an explicit
    name wins), else the name derived from its first user message (see `conversation_key`)."""
    return conversation_id(req) or conversation_key(messages)


def _suffixed(path, i: int):
    if not isinstance(path, str) or not path:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.r{i}{ext}"


def replica_cfg(cfg: dict, i: int, gpus: list[int]) -> dict:
    """The config of replica i: its own cards, log and learned-profile file; the entry's "env" and "args" added."""
    c = copy.deepcopy(cfg)
    spec = cfg.get("replicas")
    entry = spec[i] if isinstance(spec, list) and isinstance(spec[i], dict) else {}
    c.pop("replicas", None)
    c["gpu"] = gpus if len(gpus) > 1 else gpus[0]
    c.pop("hip_ordinal", None)                 # a one-card config's Windows ordinal: not for a group
    c["log"] = _suffixed(cfg.get("log"), i)
    if cfg.get("expert_profile_save"):
        c["expert_profile_save"] = _suffixed(cfg["expert_profile_save"], i)
    if entry.get("env"):
        c["env"] = {**(c.get("env") or {}), **{str(k): str(v) for k, v in entry["env"].items()}}
    if entry.get("args"):
        c["args"] = list(c["args"]) + [str(x) for x in entry["args"]]
    return c


def cpu_sets(n: int, enabled: bool = True):
    """n disjoint sets of CPUs (whole physical cores, SMT siblings together), or None (off, not Linux, or the topology
    cannot be read: the engines then share the cores as they would without replicas)."""
    if not enabled or n < 2 or not hasattr(os, "sched_getaffinity"):
        return None
    try:
        from serve.cpu_affinity import partition_cpu_sets
        return partition_cpu_sets(n, set(os.sched_getaffinity(0)))
    except (ValueError, OSError, ImportError) as e:
        print(f"[strata] replicas: no separate CPU cores ({e})", flush=True)
        return None


class Router:
    """Which replica holds the start of a prompt: chained hashes of its 64-token blocks, each with the set of replicas
    that served it (a bit mask)."""

    def __init__(self, n: int, cap: int = INDEX_MAX, conv_cap: int = CONV_MAX):
        self.n, self.cap, self.conv_cap = n, cap, conv_cap
        self.blocks: collections.OrderedDict[int, int] = collections.OrderedDict()
        self.convs: collections.OrderedDict[str, int] = collections.OrderedDict()   # conversation id -> replica
        self.longest = [0] * n            # the longest prompt each replica has served: tells a new talk from a turn
        self.conv_balance = [0] * n       # conversations currently bound to each replica: spreads sessions, not steps
        self.lock = threading.Lock()

    @staticmethod
    def chain(ids) -> list[int]:
        out, h = [], 0
        for k in range(len(ids) // BLOCK):
            h = hash((h, tuple(ids[k * BLOCK:(k + 1) * BLOCK])))
            out.append(h)
        return out

    def matched(self, ids) -> list[int]:
        """Per replica: how many leading tokens of `ids` it holds (a multiple of BLOCK)."""
        got = [0] * self.n
        alive = (1 << self.n) - 1
        with self.lock:
            for k, h in enumerate(self.chain(ids)):
                m = self.blocks.get(h, 0) & alive
                if not m:
                    break
                alive = m
                for r in range(self.n):
                    if m >> r & 1:
                        got[r] = (k + 1) * BLOCK
        return got

    def note(self, r: int, ids) -> None:
        with self.lock:
            for h in self.chain(ids):
                self.blocks[h] = self.blocks.pop(h, 0) | (1 << r)
            while len(self.blocks) > self.cap:
                self.blocks.popitem(last=False)
            self.longest[r] = max(self.longest[r], len(ids))

    def held(self, conv: str) -> int | None:
        """The replica that holds this conversation, or None (a return makes it the newest entry again)."""
        with self.lock:
            r = self.convs.get(conv)
            if r is not None:
                self.convs.move_to_end(conv)
            return r

    def bind(self, conv: str, r: int) -> None:
        """Record that replica r serves this conversation from now on (the oldest entry goes past the cap)."""
        with self.lock:
            old = self.convs.get(conv)
            if old is not None and old != r:
                self.conv_balance[old] -= 1
            if old is None:
                self.conv_balance[r] += 1
            self.convs[conv] = r
            self.convs.move_to_end(conv)
            while len(self.convs) > self.conv_cap:
                self.convs.popitem(last=False)
                self.conv_balance[self.convs[next(iter(self.convs))]] -= 1  # popitem removed the oldest

    def forget(self, r: int) -> None:
        """Replica r lost its cache (it restarted), and with it its conversations."""
        with self.lock:
            self.longest[r] = 0
            bit = ~(1 << r)
            for h in [h for h, m in self.blocks.items() if m >> r & 1]:
                m = self.blocks[h] & bit
                if m:
                    self.blocks[h] = m
                else:
                    del self.blocks[h]
            for conv in [c for c, k in self.convs.items() if k == r]:
                del self.convs[conv]
                self.conv_balance[r] -= 1


class ReplicaEngine:
    """Stands in for one engine (the interface Service uses), over several."""

    NOT_HERE = {"session_file", "vram", "ctl", "slot_cv"}       # not offered: one conversation file / VRAM resize / control line
    PROPAGATE = {"silence_s", "gpu_busy"}

    def __init__(self, engines: list, load_tokens: int = LOAD_TOKENS):
        if len(engines) < 2:
            raise ValueError("ReplicaEngine needs two or more engines")
        d = self.__dict__
        d["engines"] = list(engines)
        d["router"] = Router(len(engines))
        d["load_tokens"] = load_tokens
        d["inflight"] = [0] * len(engines)
        d["conv_inflight"] = {}                                 # requests live right now under each conversation name
        d["active_convs"] = {k: set() for k in range(len(engines))}  # conversations with a live request per card
        d["conv_stack"] = {k: [] for k in range(len(engines))}  # the names inflight on each replica, in route order
        d["run_locks"] = [threading.Lock() for _ in engines]    # an engine without batch slots: one request at a time
        d["lock"] = threading.Lock()
        d["tl"] = threading.local()
        d["pin"] = threading.local()          # a dedicated port sets "index": that replica, whatever the cost
        d["healing"] = {}
        d["heal_at"] = [0.0] * len(engines)
        d["routed"] = [0] * len(engines)
        d["gen_seen"] = [getattr(e, "gen", 0) for e in engines]

    # ---- attribute plumbing
    @property
    def current(self):
        return getattr(self.tl, "cur", None) or self.engines[0]

    def __getattr__(self, name):
        if name in self.NOT_HERE or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.current, name)

    def __setattr__(self, name, value):
        if name in self.PROPAGATE:
            for e in self.engines:
                setattr(e, name, value)
        else:
            self.__dict__[name] = value

    @property
    def last(self):
        return self.current.last

    @last.setter
    def last(self, value):
        self.current.last = value

    @property
    def batch(self) -> int:
        """Parallel slots, in total: the Service lets this many requests run at once (it holds no FIFO).  An engine
        without slots counts one."""
        return sum(max(1, int(getattr(e, "batch", 0) or 0)) for e in self.engines)

    @property
    def waiting(self) -> int:
        return sum(int(getattr(e, "waiting", 0) or 0) for e in self.engines)

    @property
    def max_context(self) -> int:
        v = [e.max_context for e in self.engines if getattr(e, "max_context", 0) and e.alive()]
        return min(v) if v else 0

    @property
    def known_ctx(self) -> int:
        v = [getattr(e, "known_ctx", 0) for e in self.engines]
        return min(x for x in v if x) if any(v) else 0

    @property
    def unloaded(self) -> bool:
        return all(getattr(e, "unloaded", False) for e in self.engines)

    @property
    def starting(self) -> bool:
        return not self.alive() and any(getattr(e, "starting", False) for e in self.engines)

    @property
    def info(self) -> dict:
        i = dict(self.engines[0].info)
        i["replicas"] = len(self.engines)
        i["replicas_alive"] = sum(1 for e in self.engines if e.alive())
        return i

    @info.setter
    def info(self, value):
        pass

    @property
    def spawn(self):
        return self.engines[0].spawn

    @property
    def log_path(self):
        return self.engines[0].log_path

    @property
    def progress(self):
        c = getattr(self.tl, "cur", None)
        if c is not None:
            return c.progress
        return next((e.progress for e in self.engines if getattr(e, "progress", None)), None)

    @property
    def prefill_tok_s_mean(self):
        c = getattr(self.tl, "cur", None)
        if c is not None:
            return c.prefill_tok_s_mean
        return next((e.prefill_tok_s_mean for e in self.engines if getattr(e, "prefill_tok_s_mean", None)), None)

    def alive(self) -> bool:
        return any(e.alive() for e in self.engines)

    def exit_code(self):
        return next((c for c in (e.exit_code() for e in self.engines if not e.alive()) if c is not None), None)

    def death_note(self) -> str:
        return next((e.death_note() for e in self.engines if not e.alive()), "")

    def slots_view(self) -> list[dict]:
        out, base = [], 0
        for k, e in enumerate(self.engines):
            n = int(getattr(e, "batch", 0) or 0)
            view = e.slots_view() if n and hasattr(e, "slots_view") else \
                [{"slot": 0, "state": "busy" if self.inflight[k] else "idle", "held_tokens": 0}]
            for s in view:
                out.append({**s, "slot": base + s["slot"], "replica": k})
            base += max(1, n)
        return out

    def replica_view(self) -> list[dict]:
        """For /metrics: each replica's state, requests running, and requests routed to it so far."""
        return [{"replica": k, "alive": e.alive(), "running": self.inflight[k], "routed": self.routed[k],
                 "batch": int(getattr(e, "batch", 0) or 0)} for k, e in enumerate(self.engines)]

    # ---- life cycle
    def restart(self, tries: int = 3):
        dead = [k for k, e in enumerate(self.engines) if not e.alive()]
        errors: dict[int, BaseException] = {}

        def one(k):
            try:
                self.engines[k].restart(tries)
                self.router.forget(k)
            except BaseException as e:           # noqa: BLE001
                errors[k] = e
        ts = [threading.Thread(target=one, args=(k,), daemon=True) for k in dead]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errors and not self.alive():
            raise next(iter(errors.values()))
        for k, e in errors.items():
            print(f"[strata] replica {k} did not start ({e}); the others serve", flush=True)

    def unload(self):
        for e in self.engines:
            e.unload()
        for k in range(len(self.engines)):
            self.router.forget(k)

    def close(self):
        err = None
        for e in self.engines:
            try:
                e.close()
            except BaseException as x:           # noqa: BLE001
                err = err or x
        if err:
            raise err

    # ---- routing
    def route(self, ids, conv: str | None = None) -> int:
        """The replica for this prompt (and its running count raised): lowest cost among the living ones.

        A named `conv` (see `conversation_id`) overrides the cost: it keeps the replica that first served it, and a
        new one goes to the least busy replica, so a warm conversation is never read again on the other card.

        A dedicated port (STRATA_DEDICATED_PORTS in server.py, this module's "pin") names its replica and
        skips the cost: the client asked for that card, and prefix caching must not move it elsewhere."""
        pinned = getattr(self.pin, "index", None)
        got = self.router.matched(ids)
        with self.lock:
            alive = [k for k, e in enumerate(self.engines) if e.alive()]
            if not alive:
                return -1
            if pinned in alive:
                k = pinned
            elif conv:
                k = self._conversation_replica(conv, alive, ids, got)
            else:
                cost = {k: (len(ids) - got[k]) + self.load_tokens * self.inflight[k] for k in alive}
                # the least loaded wins a tie of cost (equal cards, nothing cached): spread, do not pile on replica 0
                k = min(alive, key=lambda k: (cost[k], self.inflight[k], self.routed[k]))
                # A prompt the best replica has not seen up to its own longest one is a new talk, not the next turn of
                # the one it holds: clients that share a long system prompt would otherwise all land on whichever
                # replica served them first, and the other card would idle.  A new talk goes to the least busy card.
                best = max(alive, key=lambda r: got[r])
                longest = self.router.longest[best]
                # the prompt's own last tokens (the assistant header, the turn marker) are not a prefix of the built
                # prompt, so a continuation loses a few dozen tokens: only a real loss of two blocks means a new talk
                if longest and got[best] < 0.98 * longest and longest - got[best] > NEW_TALK_SLACK:
                    # a new talk pays its prefill anyway, so a free card comes before a warm one: a warm card that
                    # keeps taking new conversations while it is busy would leave the other card idle
                    k = min(alive, key=lambda r: (self.inflight[r], cost[r], self.routed[r]))
            self.inflight[k] += 1
            self.routed[k] += 1
            if conv:
                self.conv_inflight[conv] = self.conv_inflight.get(conv, 0) + 1
                self.conv_stack[k].append(conv)
                self.active_convs[k].add(conv)
        if len(alive) < len(self.engines):
            self.heal()
        return k

    def _conversation_replica(self, conv: str, alive: list[int], ids, got) -> int:
        """A named conversation keeps its replica.  A new name is placed by the card holding the fewest
        conversations and bound there; a replica that restarted forgot its conversations (Router.forget), so they
        are new.  Two live requests under one name are two sessions, not one conversation re-asking: a name is
        derived from content, and identical openings collide (measured 2026-10-20: five A/B arms with byte-equal
        prompts merged into one name and ran on a single card).  The second concurrent arrival spreads."""
        held = self.router.held(conv)
        if held in alive and self.conv_inflight.get(conv, 0) < 1:      # the count is taken before this request joins
            return held
        if held in alive:
            alive = [k for k in alive if k != held] or [held]   # one card left: two sessions share it after all
        cost = {k: (len(ids) - got[k]) + self.load_tokens * self.inflight[k] for k in alive}
        # place by conversations, not steps: a session arriving while both cards idle looks free on both, and a
        # warmth-first or step-first rule then piles every hot session onto the card that cached the shared prefix
        # first (live 2026-10-20: two DSH sessions, one card at 100% for 17 steps, the other idle).  A new
        # conversation takes the card holding the fewest conversations, so sessions split and each card works.
        # live conversations per card first: a card that just took one must not take the next while the other
        # idles; inflight breaks a tie of conversations, then warmth, then history
        act = self.active_convs
        k = min(alive, key=lambda k: (len(act[k]), self.inflight[k], cost[k], self.routed[k]))
        self.router.bind(conv, k)
        # the binding table is otherwise invisible from outside: without this line an operator watching the cards
        # cannot tell "both sessions are on one card" from "this card's session is simply idle right now"
        print(f"[strata] replica {k}: bound new conversation {conv[:48]} (live conversations now "
              f"{[len(act[r]) for r in range(len(self.engines))]}, running {self.inflight})", flush=True)
        return k

    def heal(self) -> None:
        """Start a replica that died again, in the background (the others keep serving)."""
        now = time.monotonic()
        for k, e in enumerate(self.engines):
            if e.alive() or getattr(e, "unloaded", False) or getattr(e, "starting", False) or self.healing.get(k):
                continue
            if now < self.heal_at[k]:
                continue
            self.heal_at[k] = now + HEAL_COOLDOWN_S
            self.healing[k] = True

            def run(k=k):
                try:
                    print(f"[strata] replica {k} had stopped; starting it again", flush=True)
                    self.engines[k].restart()
                    self.router.forget(k)
                    print(f"[strata] replica {k} is running again", flush=True)
                except BaseException as x:       # noqa: BLE001
                    print(f"[strata] replica {k} did not start again: {x}", flush=True)
                finally:
                    self.healing[k] = False
            threading.Thread(target=run, daemon=True).start()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        # imported here: serve.server imports this module
        from serve.server import EngineDied, EngineNonFinite
        k = self.route(ids, getattr(cancel, "_strata_conversation", None))
        if k < 0:
            raise EngineDied("no replica is running; this request was not sent")
        e = self.engines[k]
        self.tl.cur = e
        produced: list[int] = []
        held = False
        nonfinite = False
        try:
            selected = getattr(cancel, "_strata_replica_selected", None)
            if selected is not None:
                selected(k)
            self.router.note(k, ids)
            if not int(getattr(e, "batch", 0) or 0):          # no slots: this replica takes one request at a time
                lock = self.run_locks[k]
                while not lock.acquire(timeout=0.25):
                    if cancel is not None and cancel.is_set():
                        return
                held = True
            if embeddings is not None:
                gen = e.generate(ids, max_new, sampling, cancel, embeddings=embeddings)
            else:
                gen = e.generate(ids, max_new, sampling, cancel)
            for t in gen:
                if t is not None:
                    produced.append(t)
                yield t
        except EngineNonFinite:
            nonfinite = True
            raise
        finally:
            if held:
                self.run_locks[k].release()
            with self.lock:
                self.inflight[k] -= 1
                stack = self.conv_stack[k]
                if stack:
                    c = stack.pop()
                    self.conv_inflight[c] -= 1
                    if self.conv_inflight[c] <= 0:
                        del self.conv_inflight[c]
                        self.active_convs[k].discard(c)
            if nonfinite:                       # #879: the engine dropped every cached conversation
                self.router.forget(k)
            elif produced:
                self.router.note(k, list(ids) + produced)
