"""Strata's private pipe behind the canonical history/controller API.

Hot images own native immutable handles. Cold images use the existing native
file format; only the canonical checkpoint manifest binds file, frontier,
history and model. The pipe itself never publishes an API snapshot.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import selectors
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import weakref

from .store import IntegrityError, InvalidOperation


def _sha(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _environment():
    return {key: value for key, value in os.environ.items()
            if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_'))}


def _validate_execution(profile):
    if _environment() != profile['execution']['environment']:
        raise IntegrityError('Execution environment differs from pinned Strata profile')
    for name, expected in profile['native_build']['sha256'].items():
        if _sha(name) != expected:
            raise IntegrityError('Native executable/helper fingerprint mismatch: ' + name)
    identity = _digest(dict(build=profile['native_build'], execution=profile['execution'], hardware=profile['hardware']))
    if identity != profile['execution_profile_id']:
        raise IntegrityError('Strata execution profile fingerprint mismatch')
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                        '--format=csv,noheader'], text=True).strip()
    if hardware != profile['hardware']:
        raise IntegrityError('Strata execution hardware differs from pinned profile')


def _validate_model(profile):
    receipt_path = Path(profile['model_hash_receipt'])
    if _sha(receipt_path) != profile['model_hash_receipt_sha256']:
        raise IntegrityError('Executed model hash receipt changed')
    receipt = json.loads(receipt_path.read_text())
    hashed_bytes, reused = 0, 0
    for name, item in profile['model_files'].items():
        stat = Path(name).stat()
        same = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == (item['bytes'], item['mtime_ns'], item['ctime_ns'])
        measured = receipt['model_files'].get(name, {})
        if same and measured.get('sha256') == item['sha256'] and measured.get('bytes') == item['bytes']:
            reused += 1
        else:
            if _sha(name) != item['sha256']:
                raise IntegrityError('Model weight/pack fingerprint mismatch: ' + name)
            hashed_bytes += stat.st_size
    weights = {name: dict(bytes=item['bytes'], sha256=item['sha256']) for name, item in profile['model_files'].items()}
    if _digest(weights) != profile['model_weights_sha256'] or _digest(profile['model_contract']) != profile['model_contract_id']:
        raise IntegrityError('Strata model contract fingerprint mismatch')
    if profile['model_contract']['weights_sha256'] != profile['model_weights_sha256']:
        raise IntegrityError('Strata model contract references different weights')
    return dict(model_bytes_hashed_this_process=hashed_bytes, model_file_hash_receipts_reused=reused)


def _trace(variable, row):
    path = os.environ.get(variable)
    if path:
        with open(path, 'a') as stream:
            stream.write(json.dumps(dict(time=time.time(), **row), allow_nan=False) + '\n')


def _unpack_frontier(raw, vocab):
    if len(raw) != vocab * 4:
        raise IntegrityError('Incomplete full-vocabulary frontier')
    values = struct.unpack('<' + str(vocab) + 'f', raw)
    if not all(map(math.isfinite, values)):
        raise IntegrityError('Nonfinite native frontier')
    return values


@dataclass(frozen=True)
class StrataImage:
    """Process-local ownership, never a persisted history or portable file."""
    identity: bytes
    process_identity: bytes
    handle: int
    consumed: int
    running_bytes: int
    frontier_bytes: int
    pages: tuple

    def __len__(self):
        # Referenced state extent; deduplicated ownership is reported separately.
        return self.running_bytes + self.frontier_bytes + self.consumed * 4 + sum(p[3] for p in self.pages)


@dataclass
class _Lease:
    backend: object
    bound: bool = False
    consumed: int = 0
    image_identity: bytes | None = None
    state: str = 'NEW'
    _owner_thread: int | None = None

    def __enter__(self):
        if self.state != 'NEW':
            raise InvalidOperation('Execution lease cannot be entered twice')
        self.backend._lock.acquire()
        try:
            if self.state != 'NEW' or self.backend._closed:
                raise InvalidOperation('Unavailable execution lease')
            if self.backend._active is not None:
                raise InvalidOperation('Nested private leases are not supported')
            self.backend._flush_drops()
            self._owner_thread = threading.get_ident()
            self.state = 'ACTIVE'
            self.backend._active = self
            self.backend.stats['lease_acquires'] += 1
            return self
        except BaseException:
            self.backend._lock.release()
            raise

    def __exit__(self, exc_type, exc, traceback):
        # Check before changing the active lease or releasing its owner's lock.
        self.backend._check(self)
        self.backend._active = None
        self.state = 'CLOSED'
        self.backend.stats['lease_releases'] += 1
        self.backend._lock.release()
        return False


def _drop_image(backend_ref, handle):
    backend = backend_ref()
    if backend is not None:
        backend._drops.put(handle)


class StrataBackend:
    def __init__(self, profile):
        self.profile = profile
        self.validate_execution_profile()
        self.stats = dict(consumed_tokens=0, checkpoint_export_bytes=0, checkpoint_import_bytes=0,
            contexts_created=0, contexts_freed=0, lease_acquires=0, lease_releases=0,
            frontier_read_bytes=0, **_validate_model(profile))
        helper = Path(profile['gguf_reader'])
        spec = importlib.util.spec_from_file_location('ctt_strata_gguf', helper)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._metadata = module.GGUFFile(Path(profile['model_path'])).metadata
        self._tokens = self._metadata['tokenizer.ggml.tokens']
        self.vocab_size = len(self._tokens)
        if self._metadata['general.architecture'] != 'qwen4exp' or self._metadata['tokenizer.ggml.model'] != 'gpt2':
            raise IntegrityError('Unsupported Strata model/tokenizer architecture')
        # The inspected tools/strata_tokenizer.py uses this exact GPT-2 byte map.
        direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        remaining = [byte for byte in range(256) if byte not in direct]
        self._byte_map = dict(zip(map(chr, direct + list(range(256, 256 + len(remaining)))), direct + remaining))
        self._lock = threading.RLock()
        self._active = None
        self._drops = queue.SimpleQueue()
        self._process_identity = uuid.uuid4().bytes
        self._scratch = tempfile.TemporaryDirectory(prefix='ctt-strata-')
        self._directory = Path(self._scratch.name)
        self._log = tempfile.TemporaryFile()
        self._sequence = 0
        self._closed = False
        self.process = subprocess.Popen(profile['execution']['sut_command'], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._log, text=True, bufsize=1)
        self.stats['contexts_created'] = 1
        try:
            ready = self._read()
            if ready.get('status') != 'ready' or ready.get('vocab') != self.vocab_size or ready.get('capacity') != profile['execution']['capacity']:
                raise IntegrityError('Native geometry differs from pinned profile')
            self._refresh_stats()
        except BaseException:
            self.close()
            raise

    def validate_execution_profile(self):
        _validate_execution(self.profile)

    def metadata(self, key):
        value = self._metadata.get(key)
        return None if value is None else str(value)

    def contract_fields(self):
        return dict(attention=self.profile['model_contract']['attention'],
                    state_kind=self.profile['model_contract']['state_kind'])

    def new_cache(self):
        from .checkpoint_cache import CheckpointCache
        # Conservative: count every image's referenced pages, even when shared.
        # This bounds this cache, not externally held images or the model arena.
        return CheckpointCache(1024 * 1024 * 1024, 256)

    def piece(self, token):
        if type(token) is not int or not 0 <= token < self.vocab_size:
            raise InvalidOperation('Invalid token ID')
        try:
            return bytes(self._byte_map[c] for c in self._tokens[token])
        except KeyError as error:
            raise IntegrityError('Token contains a character outside the pinned byte alphabet') from error

    def is_eog(self, token):
        return token in self.profile['execution']['eos_tokens']

    def _read(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=18000):
                raise TimeoutError('Strata protocol request exceeded execution limit')
        line = self.process.stdout.readline()
        if not line:
            self._log.seek(0)
            raise RuntimeError('Strata process closed: ' + self._log.read().decode(errors='replace')[-4000:])
        return json.loads(line)

    def _request(self, command):
        if self._closed or self.process.poll() is not None:
            raise IntegrityError('Native Strata context is unavailable')
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()
        response = self._read()
        _trace('CTT_NATIVE_TRACE', dict(pid=self.process.pid, command=command, response=response))
        if response.get('status') not in ('ok', 'closed'):
            message = response.get('message', 'Invalid native response')
            if 'bad_alloc' in message:
                raise MemoryError(message)
            raise IntegrityError(message)
        return response

    def _refresh_stats(self):
        raw = self._request('stats')
        self.stats.update({key: value for key, value in raw.items() if key != 'status'})
        self.stats['consumed_tokens'] = raw['forward_tokens']
        self.stats['checkpoint_export_bytes'] = raw['saved_running_bytes'] + raw['written_file_bytes']
        self.stats['checkpoint_import_bytes'] = raw['restored_running_bytes'] + raw['loaded_file_bytes']
        return raw

    def _flush_drops(self):
        while True:
            try:
                handle = self._drops.get_nowait()
            except queue.Empty:
                return
            self._request('drop ' + str(handle))

    def release_checkpoints(self):
        with self._lock:
            self._flush_drops()
            self._request('ram_clear')
            self._refresh_stats()

    def _check(self, lease):
        if self._active is not lease or lease.state != 'ACTIVE' or lease._owner_thread != threading.get_ident():
            raise InvalidOperation('Inactive or foreign private lease')

    def lease(self):
        return _Lease(self)

    def position(self, lease):
        self._check(lease)
        if not lease.bound:
            return -1
        actual = self._refresh_stats()['consumed']
        if actual != lease.consumed:
            raise IntegrityError('Native consumed position differs from private lease')
        return actual - 1

    def consume(self, lease, ids):
        self._check(lease)
        ids = tuple(ids)
        if not ids or any(type(t) is not int or not 0 <= t < self.vocab_size for t in ids):
            raise InvalidOperation('Invalid token IDs')
        if lease.consumed + len(ids) > self.profile['execution']['capacity']:
            raise InvalidOperation('Pinned execution context exceeded')
        if not lease.bound:
            self._request('reset')
            lease.bound = True
        before = self._refresh_stats()['forward_tokens']
        try:
            response = self._request('feed ' + ' '.join(map(str, ids)))
        finally:
            after = self._refresh_stats()['forward_tokens']
        if after - before != len(ids) or response['consumed'] != lease.consumed + len(ids):
            raise IntegrityError('Completed native work/frontier mismatch')
        lease.consumed = response['consumed']
        lease.image_identity = None

    def logits(self, lease):
        self._check(lease)
        self._sequence += 1
        path = self._directory / ('frontier-' + str(self._sequence) + '.f32')
        response = self._request('logits ' + str(path))
        raw = path.read_bytes()
        path.unlink()
        if response['bytes'] != len(raw):
            raise IntegrityError('Native frontier byte counter differs from file')
        self.stats['frontier_read_bytes'] += len(raw)
        _trace('CTT_NATIVE_TRACE', dict(pid=self.process.pid, logits_sha256=hashlib.sha256(raw).hexdigest(), consumed=lease.consumed))
        return _unpack_frontier(raw, self.vocab_size)

    def export(self, lease):
        self._check(lease)
        before = self._refresh_stats()['saved_running_bytes']
        response = self._request('capture')
        after = self._refresh_stats()['saved_running_bytes']
        if response['consumed'] != lease.consumed or after <= before:
            raise IntegrityError('Incomplete consumed native capture')
        identity = self._process_identity + struct.pack('<QQ', response['handle'], response['consumed'])
        pages = tuple(tuple(row) for row in self._request('pages')['pages'])
        image = StrataImage(identity, self._process_identity, response['handle'], response['consumed'],
                            after - before, 4 * self.vocab_size, pages)
        weakref.finalize(image, _drop_image, weakref.ref(self), image.handle)
        lease.image_identity = identity
        return image

    def park(self, lease):
        self._check(lease)
        if not lease.bound or lease.consumed <= 0:
            raise InvalidOperation('RAM parking requires a consumed native lease')
        response = self._request('park')
        if response['consumed'] != lease.consumed:
            raise IntegrityError('Parked native history differs from checked lease')
        return response

    def restore_cached(self, lease, history):
        self._check(lease)
        history = tuple(history)
        if not history or len(history) > self.profile['execution']['capacity'] or any(
                type(t) is not int or not 0 <= t < self.vocab_size for t in history):
            raise InvalidOperation('Invalid canonical RAM-cache history')
        path = self._directory / 'ram-history.ids'
        path.write_text(' '.join(map(str, history)))
        try:
            response = self._request('activate ' + json.dumps(str(path)))
        finally:
            path.unlink(missing_ok=True)
        if not response['hit']:
            return False
        if response['consumed'] != len(history):
            raise IntegrityError('RAM restored history differs from canonical inputs')
        lease.bound, lease.consumed = True, response['consumed']
        lease.image_identity = None
        self._refresh_stats()
        return True

    def clear_cached_histories(self):
        with self._lock:
            return self._request('ram_clear')['removed']

    def restore(self, lease, state, *, history, frontier):
        self._check(lease)
        if isinstance(state, StrataImage):
            if state.process_identity != self._process_identity or state.consumed != len(history):
                raise IntegrityError('Unavailable or incompatible native hot image')
            response = self._request('restore ' + str(state.handle))
            resident = state
        elif isinstance(state, bytes):
            path = self._directory / 'cold.state'
            ids = self._directory / 'cold.ids'
            logits = self._directory / 'cold.f32'
            path.write_bytes(state)
            ids.write_text(' '.join(map(str, history)))
            logits.write_bytes(struct.pack('<' + str(len(frontier)) + 'f', *frontier))
            try:
                response = self._request('load ' + ' '.join(json.dumps(str(p)) for p in (path, ids, logits)))
            finally:
                for p in (path, ids, logits):
                    p.unlink(missing_ok=True)
            resident = None
        else:
            raise IntegrityError('Unsupported derived state representation')
        if response['consumed'] != len(history):
            raise IntegrityError('Native restore did not consume the expected history')
        lease.bound, lease.consumed = True, response['consumed']
        self._refresh_stats()
        if resident is None:
            resident = self.export(lease)
        lease.image_identity = resident.identity
        return resident

    def serialize(self, lease, state):
        self._check(lease)
        if not isinstance(state, StrataImage) or lease.image_identity != state.identity:
            raise IntegrityError('Cold serialization requires the checked active image')
        path = self._directory / 'export.state'
        response = self._request('write ' + json.dumps(str(path)))
        try:
            blob = path.read_bytes()
        finally:
            path.unlink(missing_ok=True)
        if len(blob) != response['bytes']:
            raise IntegrityError('Native saved byte count differs from actual file')
        self._refresh_stats()
        return blob

    def checkpoint_metrics(self, values):
        images = {id(item.blob): item.blob for item in values}
        pages = {}
        for image in images.values():
            if not isinstance(image, StrataImage):
                continue
            for row in image.pages:
                if len(row) != 6 or row[4] not in ('k', 'v', 'k_scale', 'v_scale', 'indexer_pool') or row[5] < 0:
                    raise IntegrityError('Native ownership omits the allocation pool kind')
                allocation = (row[3], row[4], row[5])
                previous = pages.setdefault(row[0], allocation)
                if previous != allocation:
                    raise IntegrityError('One physical allocation has conflicting pool metadata')
        kv = [row[0] for row in pages.values() if row[1] != 'indexer_pool']
        index = [row[0] for row in pages.values() if row[1] == 'indexer_pool']
        host_bytes = sum(image.running_bytes if isinstance(image, StrataImage) else len(image) for image in images.values())
        return dict(owned_checkpoint_bytes=host_bytes, checkpoint_unique_kv_bytes=sum(kv),
                    checkpoint_unique_kv_pages=len(kv), checkpoint_unique_index_bytes=sum(index),
                    checkpoint_unique_index_pages=len(index),
                    checkpoint_unique_physical_bytes=sum(row[0] for row in pages.values()),
                    checkpoint_unique_physical_pages=len(pages), native_counters=dict(self.stats))

    def audit(self, state):
        """Hash bytes read from a retained native image without activating it."""
        if not isinstance(state, StrataImage) or state.process_identity != self._process_identity:
            raise IntegrityError('Audit requires an image owned by this native process')
        with self._lock:
            path = self._directory / 'image-audit.bin'
            try:
                report = self._request('audit ' + str(state.handle) + ' ' + json.dumps(str(path)))
                if report['consumed'] != state.consumed or report['bytes'] != path.stat().st_size:
                    raise IntegrityError('Audit consumed count or file extent differs')
                whole = hashlib.sha256()
                offset = 0
                with path.open('rb') as stream:
                    for region in report['regions']:
                        if region['offset'] != offset or type(region['bytes']) is not int or region['bytes'] <= 0:
                            raise IntegrityError('Invalid audit region coverage')
                        if not 0 <= region['filled_prefix_bytes'] <= region['bytes']:
                            raise IntegrityError('Invalid consumed-prefix audit coverage')
                        section = hashlib.sha256()
                        remaining = region['bytes']
                        while remaining:
                            chunk = stream.read(min(remaining, 1024 * 1024))
                            if not chunk:
                                raise IntegrityError('Incomplete physical audit bytes')
                            whole.update(chunk)
                            section.update(chunk)
                            remaining -= len(chunk)
                        region['sha256'] = section.hexdigest()
                        offset += region['bytes']
                    if offset != report['bytes'] or stream.read(1):
                        raise IntegrityError('Unaccounted physical audit bytes')
                report['sha256'] = whole.hexdigest()
                self._refresh_stats()
                _trace('CTT_NATIVE_TRACE', dict(pid=self.process.pid, retained_image_audit=report))
                return report
            finally:
                path.unlink(missing_ok=True)

    def close(self):
        if self._closed:
            return
        with self._lock:
            try:
                if self.process.poll() is None:
                    self._flush_drops()
                    self._refresh_stats()
                    self._request('quit')
                    self.process.wait(timeout=30)
            finally:
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait()
                self.stats['contexts_freed'] = 1
                self._closed = True
                self.process.stdin.close()
                self.process.stdout.close()
                trace = os.environ.get('CTT_NATIVE_TRACE')
                if trace:
                    self._log.seek(0)
                    Path(trace + '.stderr').write_bytes(self._log.read())
                _trace('CTT_NATIVE_TRACE', dict(pid=self.process.pid, closed=True, counters=dict(self.stats),
                    native_counter_phase='last measurement before native teardown; process exit observed separately'))
                self._log.close()
                self._scratch.cleanup()


class StrataFreshOracle:
    """One new native process/empty session per full-history request."""
    def __init__(self, profile):
        _validate_execution(profile)
        self.model_verification = _validate_model(profile)
        self.profile = profile
        self.requests = 0
        self.tokens_computed = 0
        self._scratch = tempfile.TemporaryDirectory(prefix='ctt-fresh-')
        self._directory = Path(self._scratch.name)

    def forward(self, tokens, *, chunk=1):
        ids = tuple(tokens)
        if chunk != 1:
            raise InvalidOperation('Pinned Strata oracle consumes one token per operator call')
        if not ids or len(ids) > self.profile['execution']['capacity'] or any(type(t) is not int or t < 0 or t >= 248320 for t in ids):
            raise InvalidOperation('Invalid fresh oracle history')
        _validate_execution(self.profile)
        self.model_verification = _validate_model(self.profile)
        prefix = self._directory / ('forward-' + str(self.requests))
        token_path = prefix.with_suffix('.tokens')
        logits_path = prefix.with_suffix('.f32')
        token_path.write_text(','.join(map(str, ids)))
        command = self.profile['execution']['oracle_command'] + ['--tokens-file', str(token_path)]
        environment = dict(os.environ, STRATA_DUMP_FIRST_LOGITS=str(logits_path))
        start = time.monotonic()
        result = subprocess.run(command, env=environment, capture_output=True, timeout=18000)
        trace = os.environ.get('CTT_ORACLE_TRACE')
        if trace:
            Path(trace + '.' + str(self.requests) + '.stderr').write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError('Independent fresh process failed: ' + result.stderr.decode(errors='replace')[-4000:])
        work = json.loads(result.stdout)
        if work['forward_tokens'] != len(ids) or work['consumed'] != len(ids) or work['position_max'] != len(ids) - 1:
            raise IntegrityError('Fresh oracle did not compute every input')
        raw = logits_path.read_bytes()
        if work['logits_bytes'] != len(raw):
            raise IntegrityError('Fresh oracle output size differs from measured bytes')
        values = _unpack_frontier(raw, 248320)
        self.requests += 1
        self.tokens_computed += work['forward_tokens']
        self.last = dict(tokens=ids, computed_tokens=work['forward_tokens'], position_max=work['position_max'],
                         logits_sha256=hashlib.sha256(raw).hexdigest())
        _trace('CTT_ORACLE_TRACE', dict(command=command, work=work, result=self.last,
                                      seconds=time.monotonic() - start, returncode=result.returncode))
        token_path.unlink()
        logits_path.unlink()
        return values

    def close(self):
        self._scratch.cleanup()
