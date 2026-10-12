"""Bounded optional numerical images; authoritative history is never stored here."""
from collections import OrderedDict
from collections.abc import MutableMapping
import threading
import hashlib

from .store import IntegrityError, _json_bytes


def validate_manifest(envelope, expected):
    """Reject unknown layouts before any numerical payload is deserialized."""
    if not isinstance(envelope, dict) or set(envelope) != {'manifest', 'checksum'}:
        raise IntegrityError('Invalid checkpoint envelope')
    manifest = envelope['manifest']
    fields = {'schema', 'snapshot', 'input_identity', 'model_contract_id',
              'execution_profile_id', 'consumed', 'positions', 'logits_count',
              'state_bytes', 'payload_sha256'}
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise IntegrityError('Unknown checkpoint manifest layout')
    if envelope['checksum'] != hashlib.sha256(_json_bytes(manifest)).hexdigest():
        raise IntegrityError('Checkpoint manifest checksum mismatch')
    if type(manifest['schema']) is not int or manifest['schema'] != 1:
        raise IntegrityError('Unsupported checkpoint version')
    for name in ('consumed', 'logits_count', 'state_bytes'):
        if type(manifest[name]) is not int or manifest[name] < (0 if name == 'consumed' else 1):
            raise IntegrityError('Invalid checkpoint size')
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise IntegrityError('Checkpoint manifest incompatible')
    digest = manifest['payload_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest):
        raise IntegrityError('Invalid payload identity')
    return manifest


class CheckpointCache(MutableMapping):
    def __init__(self, byte_limit, entry_limit):
        self.byte_limit = byte_limit
        self.entry_limit = entry_limit
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self.entry_evictions = 0

    def _referenced_bytes(self):
        images = {id(value.blob): value.blob for value in self._entries.values()}
        return sum(map(len, images.values()))

    def __getitem__(self, key):
        with self._lock:
            value = self._entries[key]
            self._entries.move_to_end(key)
            return value

    def __setitem__(self, key, value):
        with self._lock:
            if len(value.blob) > self.byte_limit:
                raise MemoryError('Derived image exceeds the resident checkpoint budget')
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.entry_limit or self._referenced_bytes() > self.byte_limit:
                self._entries.popitem(last=False)
                self.entry_evictions += 1

    def __delitem__(self, key):
        with self._lock:
            del self._entries[key]

    def __iter__(self):
        with self._lock:
            return iter(tuple(self._entries))

    def __len__(self):
        with self._lock:
            return len(self._entries)

    def values(self):
        with self._lock:
            return tuple(self._entries.values())

    def clear(self):
        with self._lock:
            self._entries.clear()
