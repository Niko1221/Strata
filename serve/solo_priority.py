"""Opt-in cooperative scheduling of one native GEN at a time.

The request's parser stays on its original Python thread. STOP is read through DONE,
including every trailing T, before another request may touch the engine. Continuation
uses token IDs, never decoded/reformatted text. No CUDA slots or new native protocol.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import math
import secrets
import threading
import time


@dataclass(eq=False)
class Ticket:
    priority: int = 0
    cancel: threading.Event | None = None
    arrived: float = field(default_factory=time.monotonic)
    queued_at: float = field(default_factory=time.monotonic)
    thread: int = field(default_factory=threading.get_ident)
    active: bool = False
    since: float = 0.0
    pauses: int = 0
    state: dict | None = None


class SoloScheduler:
    """A lock-compatible gate, also used by vision and maintenance operations.

    Maintenance waits for whole requests, including suspended ones. A request that
    waited age_s goes before younger waiters, regardless of priority. A running
    request gets at least quantum_s of work between pauses; queue age does not
    include time spent running.
    """
    def __init__(self, *, quantum_s=5.0, age_s=30.0):
        self.quantum_s, self.age_s = quantum_s, age_s
        self.cv = threading.Condition()
        self.local = threading.local()
        self.owner = None
        self.waiters = []
        self.requests = []
        self.preemptions = 0
        self.resume_read = 0
        self.resume_missed = 0

    @contextlib.contextmanager
    def request(self, ids, max_new, sampling, cancel):
        value = (sampling or {}).get("priority")
        priority = value if type(value) is int else 0
        if value is None and len(ids) <= 4096 and 0 < max_new <= 512:
            priority = 1
        ticket = Ticket(priority=priority, cancel=cancel)
        self.local.ticket = ticket
        with self.cv:
            self.requests.append(ticket)
        try:
            yield
        finally:
            self.release_if_owned()
            with self.cv:
                if ticket in self.waiters:
                    self.waiters.remove(ticket)
                self.requests.remove(ticket)
                self.cv.notify_all()
            del self.local.ticket

    def _key(self, ticket, now):
        if now - ticket.queued_at >= self.age_s:
            return (0, ticket.queued_at, 0)
        return (1, -ticket.priority, ticket.arrived)

    def _first(self):
        # A control operation must not unload/restore over a suspended request.
        active = any(t.active for t in self.requests)
        eligible = [t for t in self.waiters if not (t.cancel and t.cancel.is_set())
                    and (t in self.requests or not active)]
        return min(eligible, key=lambda t: self._key(t, time.monotonic()), default=None)

    def acquire(self, blocking=True, timeout=-1):
        ticket = getattr(self.local, "ticket", None)
        control = ticket is None
        ticket = ticket or Ticket()
        end = time.monotonic() + timeout if timeout >= 0 else math.inf
        with self.cv:
            if ticket not in self.waiters:
                ticket.queued_at = time.monotonic()
                self.waiters.append(ticket)
            try:
                while True:
                    if ticket.cancel and ticket.cancel.is_set():
                        return False
                    if self.owner is None and self._first() is ticket:
                        self.owner = ticket
                        self.waiters.remove(ticket)
                        ticket.active = not control
                        ticket.since = time.monotonic()
                        return True
                    left = end - time.monotonic()
                    if not blocking or left <= 0:
                        return False
                    self.cv.wait(min(left, 0.1))
            finally:
                if control and ticket in self.waiters:
                    self.waiters.remove(ticket)

    def owned(self):
        with self.cv:
            return self.owner is not None and self.owner.thread == threading.get_ident()

    def locked(self):
        with self.cv:
            return self.owner is not None

    def release(self):
        with self.cv:
            if not self.owned():
                raise RuntimeError("cannot release another request's engine turn")
            self.owner = None
            self.cv.notify_all()

    def release_if_owned(self):
        with self.cv:
            if self.owned():
                self.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()

    def should_pause(self):
        with self.cv:
            t, other, now = self.owner, self._first(), time.monotonic()
            return bool(t and t in self.requests and other and other in self.requests
                        and now - t.since >= self.quantum_s
                        and self._key(other, now) < (1, -t.priority, t.arrived))

    def snapshot(self):
        with self.cv:
            return {"enabled": True, "preemptions": self.preemptions,
                    "resume_prompt_read": self.resume_read,
                    "resume_prefix_missed": self.resume_missed,
                    "waiting": len(self.waiters),
                    "paused": sum(t.active and t is not self.owner for t in self.requests)}

    def bind_state(self, state):
        self.local.ticket.state = state

    def current_state(self):
        with self.cv:
            return self.owner.state if self.owner else None

    def generate(self, engine, ids, max_new, sampling, cancel, embeddings):
        from serve.server import request_stats, EngineDied
        ticket = self.local.ticket
        # GENI resumption and checkpoint opt-outs are deliberately not preempted.
        eligible = (not embeddings and engine.can_stop and
                    (sampling or {}).get("strata_checkpoint") is not False)
        sampling = dict(sampling or {})
        if eligible and (sampling.get("temperature") or 0) > 0 and not (sampling.get("seed") or 0) > 0:
            # The native sampler uses Philox(seed, absolute token position).
            sampling["seed"] = secrets.randbelow(2**63 - 1) + 1
        output, segments = [], []
        original_reused = None
        born = engine.gen
        expected_prefix = None
        try:
            while len(output) < max_new and not cancel.is_set():
                requested = False
                leg_generated = 0

                def pause():
                    nonlocal requested
                    # The native engine cannot park a cancelled prefill. It also consumes
                    # a restored checkpoint, so stopping its unfinished suffix would lose
                    # the whole restored prefix. Let a cache hit (and every continuation)
                    # reach decoding before yielding. A wholly cold first read may yield
                    # once, accepting the documented cost of rereading its partial chunk.
                    can_yield = leg_generated or (not segments and not engine.reused)
                    if (eligible and can_yield and
                            max_new - len(output) > 32 and self.should_pause()):
                        requested = True
                    return requested

                before = engine.last
                gen = engine._generate_single(list(ids) + output, max_new - len(output), sampling,
                                              cancel, embeddings, pause=pause if eligible else None)
                try:
                    for token in gen:
                        if token is not None:
                            output.append(token)
                            leg_generated += 1
                        yield token
                finally:
                    gen.close()
                    if engine.last is not before:
                        leg = dict(engine.last)
                        if original_reused is None:
                            original_reused = int(leg.get("reused") or 0)
                        elif segments:
                            with self.cv:
                                self.resume_read += int(leg.get("prompt_read") or 0)
                                self.resume_missed += max(0, expected_prefix - int(leg.get("reused") or 0))
                        segments.append(leg)
                if not requested or engine.last.get("finish") != "cancel" or cancel.is_set():
                    break
                expected_prefix = (len(ids) + len(output) - 1 if output else
                                   int(leg.get("reused") or 0) + int(leg.get("prompt_read") or 0))
                with self.cv:
                    ticket.pauses += 1
                    self.preemptions += 1
                    if ticket.state is not None:
                        ticket.state["solo_paused"] = True
                    ticket.queued_at = time.monotonic()
                    self.waiters.append(ticket)
                    self.release()
                print(f"[strata] solo priority: paused after {len(output)} output tokens", flush=True)
                # The caller's parser/detokenizer stays alive across these pings.
                while not cancel.is_set():
                    if self.acquire(timeout=1.0):
                        break
                    yield None
                if cancel.is_set():
                    break
                if not engine.alive() or engine.gen != born:
                    raise EngineDied("the engine stopped while this request was paused")
                if ticket.state is not None:
                    ticket.state["solo_paused"] = False
        finally:
            if segments:
                combined = request_stats(segments)
                combined["prompt_tokens"] = len(ids)
                combined["reused"] = original_reused
                # API prompt/cache accounting must not charge generated tokens as new input.
                reached = max(min(len(ids), int(s.get("reused") or 0) + int(s.get("prompt_read") or 0))
                              for s in segments)
                combined["prompt_read"] = max(0, reached - original_reused)
                engine.last = combined


def enable(service, config):
    value = config.get("solo_priority", False)
    if type(value) is not bool:
        raise ValueError('"solo_priority" must be true or false')
    if not value:
        return
    engine = service.engine
    args = config.get("args", [])
    def option(name, default=0):
        return int(args[args.index(name) + 1]) if name in args else default
    if (getattr(engine, "batch", 0) or config.get("parallel") or config.get("replicas") or
            "--batch" in args or "--slots" in args or
            not hasattr(engine, "_generate_single")):
        raise ValueError('"solo_priority" needs a single Strata engine without parallel/replicas')
    if option("--conversation-cache-mib") <= 0 or option("--conversation-cache-slots", 4) < 2:
        raise ValueError('"solo_priority" needs a conversation cache with at least two slots')
    if option("--prompt-cache", 1) <= 0:
        raise ValueError('"solo_priority" needs prompt caching')
    service.solo_scheduler = service.fifo = engine.solo_scheduler = SoloScheduler()
    print('[strata] solo priority on: one GEN at a time; at least five seconds between pauses', flush=True)
