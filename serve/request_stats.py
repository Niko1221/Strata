"""Content-free, generation-owned protocol counters.

The engine's global ``last`` is only a compatibility diagnostic. DONE describes
one GEN (or a BGEN's first token), not necessarily an entire API generation.
Bind a collector around *each iterator operation*, not around a suspended yield:
that also isolates two generators interleaved on the same Python thread.
"""
from contextvars import ContextVar
import threading
import time

_current = ContextVar('strata_generation_stats', default=None)


def current_stats(engine):
    value = _current.get()
    return value[1] if value is not None and value[0] is engine else None


class GenerationStats:
    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens
        self.lock = threading.RLock()
        self.parts = []
        self.phase = 'waiting_control'
        self.progress = None
        self.prefill_rate = None
        self.slot = None
        self.slot_generation = None
        self.control_wait_ms = 0.0
        self.slot_wait_ms = 0.0
        self.wait_started = None
        self.wait_kind = None
        self.protocol_events = 0

    def wait(self, phase):
        with self.lock:
            self.phase = phase
            self.wait_started = time.monotonic()
            self.wait_kind = 'control_wait_ms' if phase == 'waiting_control' else 'slot_wait_ms'
            return self.wait_started

    def waited(self, kind, since):
        with self.lock:
            setattr(self, kind, getattr(self, kind) + 1000 * (time.monotonic() - since))
            self.wait_started, self.wait_kind = None, None

    def start(self, mode, prompt_tokens):
        with self.lock:
            self.protocol_events += 1
            self.parts.append({'mode': mode, 'input_tokens': prompt_tokens, 'complete': False})
            self.phase, self.progress, self.prefill_rate = 'prefill', None, None

    def assign_slot(self, slot, generation):
        with self.lock:
            self.slot, self.slot_generation = slot, generation

    def output(self):
        with self.lock:
            if self.phase == 'prefill':
                self.phase = 'admitting' if self.parts and self.parts[-1]['mode'] == 'admit' else 'decode'
                self.progress, self.prefill_rate = None, None

    def prompt_progress(self, progress, rate):
        with self.lock:
            self.progress, self.prefill_rate = progress, rate

    def done(self, values):
        with self.lock:
            self.protocol_events += 1
            if not self.parts or self.parts[-1]['complete']:
                self.start('solo', values['prompt_tokens'])
            part = self.parts[-1]
            part['done'] = dict(values)
            part['complete'] = part['mode'] == 'solo'
            self.phase = 'done' if part['complete'] else 'admitting'

    def admitted(self, continues):
        with self.lock:
            if self.parts:
                part = self.parts[-1]
                part['continues'] = continues
                part['complete'] = not continues and 'done' in part
            self.phase = 'batch' if continues else 'done'
            self.progress, self.prefill_rate = None, None

    def batch_done(self, line, part=None, slot=None):
        f = line.split()
        with self.lock:
            if not self.parts or len(f) < 5 or int(f[1]) != (self.slot if slot is None else slot):
                return
            part = self.parts[-1] if part is None else part
            part['batch'] = {'generated': int(f[2]), 'finish': f[3], 'wall_ms': float(f[4])}
            part['complete'] = 'done' in part
            if part is self.parts[-1]:
                self.phase = 'done'

    def drain_callback(self):
        # A reasoning-budget continuation may start a new part before this drain
        # ends. Capture the old part, never update whichever slot is current later.
        with self.lock:
            part, slot = self.parts[-1], self.slot
        return lambda line: self.batch_done(line, part, slot)

    def view(self):
        with self.lock:
            extra = 1000 * (time.monotonic() - self.wait_started) if self.wait_started is not None else 0.0
            return {'engine_phase': self.phase, 'slot_id': self.slot, 'slot_generation': self.slot_generation,
                    'prompt_progress': self.progress, 'prefill_tok_s_mean': self.prefill_rate,
                    'control_wait_ms': self.control_wait_ms + (extra if self.wait_kind == 'control_wait_ms' else 0),
                    'slot_wait_ms': self.slot_wait_ms + (extra if self.wait_kind == 'slot_wait_ms' else 0),
                    'engine_segments': len(self.parts),
                    'engine_complete': bool(self.parts) and all(p['complete'] for p in self.parts)}

    def result(self):
        """A frozen request result, or {} when a terminal response is still pending.

The first part describes the client's original prompt. Subsequent parts may
re-prefill generated text (solo promotion / reasoning wrap); their work is
summed separately, not added to the client's input or cached-token counts.
BDONE.generated already INCLUDES the admission token. BDONE.wall_ms excludes
that token but INCLUDES pauses caused by other admissions. Expert-cache stats
in admission DONE do NOT measure the following batch decode.
"""
        with self.lock:
            if not self.parts or not all(p['complete'] and 'done' in p for p in self.parts):
                return {}
            first = self.parts[0]['done']
            out = dict(first)
            out['prompt_tokens'] = self.prompt_tokens
            if 'reused' in out:
                out['reused'] = min(self.prompt_tokens, out['reused'])
            out['prompt_ms'] = sum(p['done']['prompt_ms'] for p in self.parts)
            out['generated'] = sum(p.get('batch', p['done'])['generated'] for p in self.parts)
            out['decode_ms'] = sum(p['done']['decode_ms'] + p.get('batch', {}).get('wall_ms', 0)
                                   for p in self.parts)
            out['finish'] = self.parts[-1].get('batch', self.parts[-1]['done'])['finish']
            if all('prompt_read' in p['done'] for p in self.parts):
                out['engine_prompt_read_total'] = sum(p['done']['prompt_read'] for p in self.parts)
            for key in ('drafts_accepted', 'drafts_offered', 'hits', 'lookups', 'ram_blobs', 'file_blobs', 'file_mb'):
                if all(key in p['done'] for p in self.parts):
                    out[key] = sum(p['done'][key] for p in self.parts)
                else:
                    out.pop(key, None)
            batched = any(p.get('continues') for p in self.parts)
            if batched:
                for key in ('hits', 'lookups', 'ram_blobs', 'file_blobs', 'file_mb'):
                    out.pop(key, None)
            out['decode_scope'] = 'engine_segments_including_batch_stalls' if batched else 'engine_decode'
            out['prompt_scope'] = 'admission_including_cache_work'
            out['engine_segments'] = len(self.parts)
            return out


class ObservedGeneration:
    """Wrap an existing engine iterator without changing the engine API or tokens."""
    def __init__(self, engine, iterator, stats):
        self.engine, self.iterator, self.stats = engine, iterator, stats
        self.before = getattr(engine, 'last', None)
        self.events_before = stats.protocol_events
        self.finished = False

    def __iter__(self):
        return self

    def _invoke(self, method):
        token = _current.set((self.engine, self.stats))
        try:
            return method()
        finally:
            _current.reset(token)

    def _finish(self):
        if self.finished:
            return
        self.finished = True
        # Serial third-party/mock engines may only publish `last`, without our
        # parser hook. Never use this compatibility fallback for a batch engine.
        fresh = getattr(self.engine, 'last', None)
        if (not getattr(self.engine, 'batch', 0) and self.stats.protocol_events == self.events_before
                and isinstance(fresh, dict) and fresh is not self.before and fresh):
            self.stats.done(fresh)

    def __next__(self):
        try:
            return self._invoke(lambda: next(self.iterator))
        except BaseException:
            self._finish()
            raise

    def close(self):
        try:
            return self._invoke(self.iterator.close)
        finally:
            self._finish()
