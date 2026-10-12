"""Canonical immutable API with checked native backend materializations."""
from contextlib import contextmanager
import ctypes as C
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import copy
from types import MappingProxyType

from .oracle import FreshOracle
from .store import Store, IntegrityError, InvalidOperation, _stage, _sync_dir, _json_bytes, _decode, _private_directory
from . import controller


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class NativeBackend:
    def __init__(self, profile):
        self.profile = profile
        self.validate_execution_profile()
        if digest_file(profile['model_path']) != profile['model_weights_sha256']:
            raise IntegrityError('Model weight fingerprint mismatch')
        self.lib = C.CDLL(profile['bridge_library'])
        signatures = {
            'error': (C.c_char_p, []), 'load': (C.c_void_p, [C.c_char_p]),
            'unload': (None, [C.c_void_p]), 'vocab': (C.c_int, [C.c_void_p]),
            'meta': (C.c_int, [C.c_void_p,C.c_char_p,C.c_void_p,C.c_size_t]),
            'piece': (C.c_int,[C.c_void_p,C.c_int,C.c_void_p,C.c_int]),
            'eog': (C.c_bool,[C.c_void_p,C.c_int]),
            'new': (C.c_void_p, [C.c_void_p]), 'free': (None, [C.c_void_p]),
            'position': (C.c_int, [C.c_void_p]),
            'consume': (C.c_int, [C.c_void_p,C.POINTER(C.c_int32),C.c_int]),
            'logits': (C.POINTER(C.c_float), [C.c_void_p]),
            'work': (C.c_uint64,[C.c_void_p]), 'state_size': (C.c_size_t,[C.c_void_p]),
            'export': (C.c_size_t,[C.c_void_p,C.c_void_p,C.c_size_t]),
            'import': (C.c_size_t,[C.c_void_p,C.c_void_p,C.c_size_t]),
        }
        for name, (result,args) in signatures.items():
            fn = getattr(self.lib,'ctt_'+name)
            fn.restype = result
            fn.argtypes = args
        self.model = self.lib.ctt_load(os.fsencode(profile['model_path']))
        if not self.model:
            raise RuntimeError(self.lib.ctt_error().decode())
        self.vocab_size = self.lib.ctt_vocab(self.model)
        self.stats = dict(consumed_tokens=0, checkpoint_export_bytes=0,
                          checkpoint_import_bytes=0, contexts_created=0,
                          contexts_freed=0, shared_prefix_pages=None)

    def validate_execution_profile(self):
        execution = self.profile['execution']
        actual_math = os.environ.get('GGML_CUDA_CUBLAS_COMPUTE_TYPE','auto')
        actual_tf32 = os.environ.get('NVIDIA_TF32_OVERRIDE','unset')
        if actual_math != execution['cuda_cublas_compute_type'] or actual_tf32 != execution['nvidia_tf32_override']:
            raise IntegrityError('Execution math profile mismatch')
        for path,expected in self.profile['native_build']['sha256'].items():
            if digest_file(path) != expected:
                raise IntegrityError('Native binary/header fingerprint mismatch: '+path)
        expected_id = hashlib.sha256(json.dumps(dict(build=self.profile['native_build'],
            execution=execution,hardware=self.profile['hardware']),sort_keys=True).encode()).hexdigest()
        if expected_id != self.profile['execution_profile_id']:
            raise IntegrityError('Execution profile fingerprint mismatch')

    def metadata(self, key):
        data = C.create_string_buffer(8192)
        n = self.lib.ctt_meta(self.model,key.encode(),data,len(data))
        return data.value.decode() if n>=0 else None

    def contract_fields(self):
        return dict(attention='causal-full',state_kind='full-attention-transformer')

    def new_cache(self):
        return {}

    def park(self, lease):
        # This adapter has no independent native history cache.
        return None

    def restore_cached(self, lease, history):
        return False

    def clear_cached_histories(self):
        return 0

    def release_checkpoints(self):
        # Owned byte images are released by dropping their Python references.
        pass

    def position(self, lease):
        return self.lib.ctt_position(lease)

    def is_eog(self, token):
        return self.lib.ctt_eog(self.model,token)

    @contextmanager
    def lease(self):
        p = self.lib.ctt_new(self.model)
        if not p:
            raise RuntimeError(self.lib.ctt_error().decode())
        self.stats['contexts_created'] += 1
        try:
            yield p
        finally:
            self.lib.ctt_free(p)
            self.stats['contexts_freed'] += 1

    def consume(self, lease, ids):
        if not ids or any(type(t) is not int or not 0<=t<self.vocab_size for t in ids):
            raise InvalidOperation('Invalid token IDs')
        if self.lib.ctt_position(lease)+1+len(ids)>2048:
            raise InvalidOperation('Certified execution context exceeded')
        native_ids = (C.c_int32*len(ids))(*ids)
        before = self.lib.ctt_work(lease)
        if self.lib.ctt_consume(lease,native_ids,len(ids)):
            raise RuntimeError(self.lib.ctt_error().decode())
        delta = self.lib.ctt_work(lease)-before
        if delta != len(ids):
            raise AssertionError('Native consumed-work counter mismatch')
        self.stats['consumed_tokens'] += delta

    def logits(self, lease):
        data = self.lib.ctt_logits(lease)
        if not data:
            raise IntegrityError('Missing consumed frontier')
        return tuple(data[:self.vocab_size])

    def export(self, lease):
        size = self.lib.ctt_state_size(lease)
        data = C.create_string_buffer(size)
        if self.lib.ctt_export(lease,data,size) != size:
            raise IntegrityError('Incomplete state export')
        self.stats['checkpoint_export_bytes'] += size
        return data.raw

    def piece(self, token):
        data = C.create_string_buffer(4096)
        n = self.lib.ctt_piece(self.model,token,data,len(data))
        if n < 0:
            data = C.create_string_buffer(-n)
            n = self.lib.ctt_piece(self.model,token,data,len(data))
        if n < 0:
            raise IntegrityError('Token output conversion failed')
        return data.raw[:n]

    def restore(self, lease, blob, *, history=None, frontier=None):
        data = C.create_string_buffer(blob,len(blob))
        if getattr(self.lib,'ctt_import')(lease,data,len(blob)) != len(blob):
            raise IntegrityError('Incomplete state import')
        self.stats['checkpoint_import_bytes'] += len(blob)
        return blob

    def serialize(self, lease, state):
        return state

    def checkpoint_metrics(self, values):
        unique = {id(item.blob):len(item.blob) for item in values}
        return dict(native_counters=dict(self.stats),owned_checkpoint_bytes=sum(unique.values()))

    def close(self):
        if self.model:
            self.lib.ctt_unload(self.model)
            self.model = None


@dataclass(frozen=True)
class Checkpoint:
    model_contract_id: str
    execution_profile_id: str
    consumed: int
    blob: object
    frontier: tuple
    checksum: str
    input_identity: str = ''

    @staticmethod
    def digest(blob, frontier):
        state_identity = blob if isinstance(blob,bytes) else blob.identity
        return hashlib.sha256(state_identity+struct.pack('<'+str(len(frontier))+'f',*frontier)).hexdigest()


@dataclass(frozen=True)
class Generation:
    snapshot_id: str
    accepted_tokens: tuple
    boundaries: tuple


class BranchingModel:
    def slot_queue(self, count):
        from .slots import BranchSlots
        return BranchSlots(self, count)

    @staticmethod
    def load_backend(profile):
        kind = profile.get('backend','llama.cpp')
        if kind == 'strata':
            from .strata_backend import StrataBackend
            return StrataBackend(profile)
        if kind != 'llama.cpp':
            raise IntegrityError('Unsupported backend profile')
        return NativeBackend(profile)
    durability_scope = MappingProxyType(dict(platform='local-posix',
        process_crash_tested=True,power_loss_tested=False,
        publication='fsync-file; atomic-link-or-rename; fsync-directory'))

    def __init__(self, directory, backend, profile):
        if profile['model_contract_id'] != backend.profile['model_contract_id'] or profile['execution_profile_id'] != backend.profile['execution_profile_id']:
            raise IntegrityError('Backend profile does not match requested contract')
        self.store = Store(directory)
        self.backend = backend
        self.profile = profile
        self.cache = backend.new_cache()
        self.checkpoint_directory = self.store.root/'checkpoints'
        _private_directory(self.checkpoint_directory)
        self._checkpoint_fault = lambda point: None
        self._lease_fault = lambda point: None
        self._allocation_fault = lambda point, lease: None
        architecture = backend.metadata('general.architecture')
        self.contract = MappingProxyType(dict(weights_sha256=profile['model_weights_sha256'],
            architecture=architecture,rope_base=backend.metadata(architecture+'.rope.freq_base'),
            yarn='model-default',tokenizer_identity=profile['model_weights_sha256'],
            template_identity='exact-token-ids/no-template',
            positions='absolute-zero-based',input_kind='token_ids',adapters=(),
            **backend.contract_fields()))

    def history(self, snapshot):
        return tuple(self.store.lineage(snapshot)[1])

    def for_principal(self, principal):
        bound = copy.copy(self)
        bound.store = self.store.for_principal(principal)
        return bound

    def snapshot_bytes(self, snapshot):
        self.store.authorize(snapshot, 'read')
        return (self.store.objects/(snapshot+'.json')).read_bytes()

    def _save(self, lease, history):
        self._allocation_fault('before_native_export',lease)
        blob = self.backend.export(lease)
        frontier = self.backend.logits(lease)
        self.backend.park(lease)
        return Checkpoint(self.profile['model_contract_id'],self.profile['execution_profile_id'],
            self.backend.position(lease)+1,blob,frontier,Checkpoint.digest(blob,frontier),
            self._input_identity(history))

    @contextmanager
    def materialize(self, snapshot):
        self.store.authorize(snapshot, 'materialize')
        with self.store.read_lease(snapshot):
            with self._materialize(snapshot) as state:
                yield state

    @contextmanager
    def _materialize(self, snapshot):
        record = self.store.get_snapshot(snapshot)
        if record['model_contract_id'] != self.profile['model_contract_id']:
            raise IntegrityError('Snapshot model contract mismatch')
        history = self.history(snapshot)
        input_identity = self._input_identity(history)
        with self.backend.lease() as lease:
            saved = self.cache.get(snapshot)
            if saved is None and (self.checkpoint_directory/(snapshot+'.json')).exists():
                saved = self._load_checkpoint(snapshot,history)
                self.cache[snapshot] = saved
            if saved is not None and saved.input_identity != input_identity:
                # The cache is only an accelerator. Never restore a same-sized
                # image for another canonical input history; rebuild from the
                # immutable snapshot inputs instead.
                self.cache.pop(snapshot, None)
                self.backend.clear_cached_histories()
                saved = None
            if saved:
                if saved.model_contract_id != self.profile['model_contract_id'] or saved.execution_profile_id != self.profile['execution_profile_id']:
                    raise IntegrityError('Checkpoint contract/profile mismatch')
                if saved.consumed != len(history) or saved.checksum != Checkpoint.digest(saved.blob,saved.frontier):
                    raise IntegrityError('Checkpoint integrity failure')
                resident = self.backend.restore(lease,saved.blob,history=history,frontier=saved.frontier)
                if resident is not saved.blob:
                    saved = Checkpoint(saved.model_contract_id,saved.execution_profile_id,saved.consumed,
                        resident,saved.frontier,Checkpoint.digest(resident,saved.frontier),saved.input_identity)
                    self.cache[snapshot] = saved
            else:
                if not self.backend.restore_cached(lease,history):
                    self.backend.consume(lease,history)
                saved = self._save(lease,history)
                self.cache[snapshot] = saved
            if self.backend.position(lease) != len(history)-1:
                raise IntegrityError('Restored consumed position mismatch')
            yield lease,saved.frontier

    def create(self, ids, *, contract=None):
        if self.store._principal is not None:
            raise PermissionError('Only the local owner can create model histories')
        if contract is not None and dict(contract) != dict(self.contract):
            raise IntegrityError('Unsupported or incompatible model/input semantics')
        ids = tuple(ids)
        with self.backend.lease() as lease:
            self.backend.consume(lease,ids)
            saved = self._save(lease,ids)
        sid = self.store.create_root(self.profile['model_contract_id'],ids,{'status':'idle'})
        self.cache[sid] = saved
        return sid

    def feed(self, snapshot, ids):
        return self._feed(snapshot,ids,self.controller(snapshot))

    def _feed(self, snapshot, ids, decoder):
        self.store.authorize(snapshot, 'append')
        if not ids:
            return snapshot
        history = self.history(snapshot) + tuple(ids)
        next_decoder = copy.deepcopy(decoder)
        for token in ids:
            if type(token) is not int or not 0<=token<self.backend.vocab_size:
                raise InvalidOperation('Invalid token ID')
            next_decoder = controller.consume(next_decoder,token,self.backend.piece(token),
                self.backend.is_eog(token))
        with self.materialize(snapshot) as (lease,_):
            for token in ids:
                self.backend.consume(lease,(token,))
                self._lease_fault('after_token_consume')
            saved = self._save(lease,history)
        sid = self.store.append(snapshot,ids,next_decoder)
        self.cache[sid] = saved
        return sid

    def controller(self, snapshot):
        return self.store.get_snapshot(snapshot)['controller']

    def _controller_transition(self, snapshot, decoder):
        sid = self.store.append(snapshot,(),decoder)
        saved = self.cache.get(snapshot)
        if saved is not None:
            self.cache[sid] = saved
        return sid

    def start_generation(self, snapshot, policy, stream):
        decoder = controller.start(policy,stream,self.history(snapshot),self.backend.vocab_size)
        return self._controller_transition(snapshot,decoder)

    def set_random_stream(self, snapshot, stream):
        decoder = self.controller(snapshot)
        if decoder['status'] == 'idle':
            raise InvalidOperation('No generation scope')
        decoder['stream_key'] = hashlib.sha256(stream.encode()).hexdigest()
        decoder['next_draw'] = 0
        return self._controller_transition(snapshot,decoder)

    def stream(self, snapshot, max_steps):
        if type(max_steps) is not int or max_steps<0:
            raise InvalidOperation('Invalid generation work bound')
        for _ in range(max_steps):
            decoder = self.controller(snapshot)
            if decoder['status'] == 'stopped':
                break
            token,next_decoder = controller.choose(self.next_logits(snapshot),decoder)
            snapshot = self._feed(snapshot,(token,),next_decoder)
            # Store.append has fsynced and published the canonical object.
            yield snapshot,token

    def generate(self, snapshot, max_steps):
        boundaries,tokens = [],[]
        for snapshot,token in self.stream(snapshot,max_steps):
            boundaries.append(snapshot)
            tokens.append(token)
        return Generation(snapshot,tuple(tokens),tuple(boundaries))

    def next_logits(self, snapshot):
        with self.materialize(snapshot) as (_,frontier):
            return frontier

    def consumed(self, snapshot):
        with self.materialize(snapshot) as (lease,_):
            return self.backend.position(lease)+1

    def position_max(self, snapshot):
        return self.consumed(snapshot)-1

    @staticmethod
    def _input_identity(history):
        return hashlib.sha256(_json_bytes(list(history))).hexdigest()

    def _publish_checkpoint_file(self, destination, data):
        stage = _stage(self.checkpoint_directory,data)
        try:
            os.replace(stage,destination)
            _sync_dir(self.checkpoint_directory)
        finally:
            stage.unlink(missing_ok=True)

    def checkpoint(self, snapshot):
        self.store.authorize(snapshot, 'checkpoint')
        with self.materialize(snapshot) as (lease,_):
            saved = self.cache[snapshot]
            blob = self.backend.serialize(lease,saved.blob)
        payload = blob+struct.pack('<'+str(len(saved.frontier))+'f',*saved.frontier)
        payload_hash = hashlib.sha256(payload).hexdigest()
        payload_path = self.checkpoint_directory/(payload_hash+'.state')
        self._publish_checkpoint_file(payload_path,payload)
        manifest = dict(schema=1,snapshot=snapshot,input_identity=self._input_identity(self.history(snapshot)),
            model_contract_id=saved.model_contract_id,execution_profile_id=saved.execution_profile_id,
            consumed=saved.consumed,positions='absolute-zero-based',state_bytes=len(blob),
            logits_count=len(saved.frontier),payload_sha256=payload_hash)
        envelope = dict(manifest=manifest,checksum=hashlib.sha256(_json_bytes(manifest)).hexdigest())
        self._checkpoint_fault('before_manifest_publish')
        manifest_path = self.checkpoint_directory/(snapshot+'.json')
        self._publish_checkpoint_file(manifest_path,_json_bytes(envelope))
        return dict(manifest_path=manifest_path,payload_path=payload_path)

    def _load_checkpoint(self, snapshot, history):
        from .checkpoint_cache import validate_manifest
        try:
            envelope = _decode((self.checkpoint_directory/(snapshot+'.json')).read_bytes())
            required = dict(schema=1,snapshot=snapshot,input_identity=self._input_identity(history),
                model_contract_id=self.profile['model_contract_id'],execution_profile_id=self.profile['execution_profile_id'],
                consumed=len(history),positions='absolute-zero-based',logits_count=self.backend.vocab_size)
            manifest = validate_manifest(envelope, required)
            sha = manifest['payload_sha256']
            payload = (self.checkpoint_directory/(sha+'.state')).read_bytes()
            size = manifest['state_bytes']
            if type(size) is not int or size<=0 or len(payload)!=size+4*self.backend.vocab_size or hashlib.sha256(payload).hexdigest()!=sha:
                raise IntegrityError('Checkpoint payload corruption')
            logits = struct.unpack('<'+str(self.backend.vocab_size)+'f',payload[size:])
            return Checkpoint(required['model_contract_id'],required['execution_profile_id'],len(history),
                payload[:size],logits,sha,required['input_identity'])
        except (OSError,KeyError,TypeError,ValueError) as error:
            raise IntegrityError('Invalid or incomplete checkpoint') from error

    def evict_all(self):
        if self.store._principal is not None:
            raise PermissionError('Global eviction requires local owner access')
        self.cache.clear()
        self.backend.release_checkpoints()
        for path in self.checkpoint_directory.iterdir():
            if path.is_file():
                path.unlink()
        _sync_dir(self.checkpoint_directory)

    def measure(self, report, snapshot, history):
        path = os.environ.get('CTT_MEASUREMENTS')
        if path:
            with open(path,'a') as output:
                output.write(json.dumps(dict(snapshot=snapshot,history=history,numerics=asdict(report),
                    **self.backend.checkpoint_metrics(self.cache.values())))+'\n')
