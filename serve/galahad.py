"""Optional Galahad 1.31 storage for complete, named Strata session images.

The caller holds Service.fifo throughout the engine and storage operations.
Galahad owns encryption and persistence; the engine validates its own image on restore.
"""
from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
import sys
import tempfile
import time
from pathlib import Path


class GalahadError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def validate_config(directory, fingerprint, max_session_mib):
    if not isinstance(directory, str) or not directory.strip() or any(ord(c) < 32 for c in directory):
        raise ValueError("galahad cache_dir must be a directory path")
    if isinstance(fingerprint, bool) or not isinstance(fingerprint, int) or not 0 < fingerprint < 2**64:
        raise ValueError("galahad model_fingerprint must be a positive 64-bit integer identifying model, tokenizer and quantisation")
    if isinstance(max_session_mib, bool) or not isinstance(max_session_mib, int) or max_session_mib <= 0:
        raise ValueError("galahad max_session_mib must be a positive integer")


class GalahadSessions:
    def __init__(self, directory, fingerprint, max_session_mib=2048):
        validate_config(directory, fingerprint, max_session_mib)
        if sys.platform != "linux":
            raise ValueError("Galahad currently requires Linux x86-64 and a licensed NVIDIA GPU")
        # Import only when explicitly enabled. No Galahad dependency on the default path.
        try:
            import galahad
            from merlin import MerlinLookupResult
        except ImportError as e:
            raise ValueError("install the optional dependency: python -m pip install galahad-kv==1.31.5") from e
        self.directory = str(Path(directory).expanduser().resolve())
        Path(self.directory).mkdir(mode=0o700, parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.max_bytes = max_session_mib * 1024**2
        self.result_type = MerlinLookupResult
        self.handle = galahad.init(
            fingerprint, block_dir=os.path.join(self.directory, "blocks"),
            ledger_path=os.path.join(self.directory, "ledger.bin"), tenancy="single",
            fsync_payloads=True, rehydrate_on_init=True)
        self.lib = self.handle.lib

    def close(self):
        self.handle.close()

    def keys(self, filename):
        # Domain-separated names: these opaque complete images must never be offered
        # as token-prefix blocks to another host. Model isolation is also set at init.
        data = f"strata-session-v1\0{self.fingerprint}\0{filename}".encode("utf-8")
        key = int.from_bytes(hashlib.sha256(data).digest()[:8], "little")
        confirm = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")
        return key, confirm

    def check(self, status, operation):
        if status:
            detail = self.lib.merlin_last_error()
            detail = detail.decode("utf-8", "replace") if detail else f"status {status}"
            raise GalahadError(507 if status == 21 else 500, f"Galahad {operation}: {detail}")

    def lookup(self, key, confirm):
        result = self.result_type()
        result.struct_size = ctypes.sizeof(result)
        status = self.lib.merlin_lookup_confirmed(key, confirm, 0, ctypes.byref(result))
        if status == 9:  # MERLIN_ERR_NOT_FOUND
            return None
        self.check(status, "lookup")
        return result if result.tier else None

    def session_file(self, engine, action, filename):
        started = time.monotonic()
        key, confirm = self.keys(filename)
        found = self.lookup(key, confirm)
        if action == "save" and found:
            raise GalahadError(409, "Galahad session names are immutable; save under a new filename")
        if action == "restore" and not found:
            raise GalahadError(404, f"no Galahad session named {filename}")
        # Plaintext is needed by the engine's file protocol. Never retain it in the
        # durable store; a private temporary directory is removed even on refusal.
        with tempfile.TemporaryDirectory(prefix="strata-galahad-") as scratch:
            path = os.path.join(scratch, "session.bin")
            if action == "save":
                result = engine.session_file("save", path)
                size = os.stat(path).st_size
                if not 0 < size <= self.max_bytes:
                    raise GalahadError(413, "session exceeds Galahad max_session_mib (or is empty)")
                if not 0 < result["tokens"] < 2**31 or size != result["bytes"]:
                    raise GalahadError(500, "engine returned inconsistent session size or token count")
                with open(path, "rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY) as image:
                    buffer = (ctypes.c_uint8 * size).from_buffer(image)
                    try:
                        self.check(self.lib.merlin_deposit_bytes(key, 0, buffer, size, result["tokens"], confirm), "save")
                    finally:
                        del buffer
                count = ctypes.c_size_t()
                self.check(self.lib.merlin_checkpoint(ctypes.byref(count)), "checkpoint")
                saved = self.lookup(key, confirm)
                if not saved or saved.block_size != size or saved.token_count != result["tokens"]:
                    raise GalahadError(500, "Galahad did not retain the session; check licence with galahad doctor")
            else:
                size = found.block_size
                if not 0 < size <= self.max_bytes:
                    raise GalahadError(413, "stored session exceeds Galahad max_session_mib (or is empty)")
                with open(path, "xb") as stream:
                    os.chmod(path, 0o600)
                    stream.truncate(size)
                with open(path, "r+b") as stream, mmap.mmap(stream.fileno(), size) as image:
                    buffer = (ctypes.c_uint8 * size).from_buffer(image)
                    loaded = ctypes.c_size_t()
                    try:
                        self.check(self.lib.merlin_load_block(key, 0, buffer, size, ctypes.byref(loaded)), "restore")
                    finally:
                        del buffer
                    if loaded.value != size:
                        raise GalahadError(500, "Galahad returned an incomplete session")
                    image.flush()
                result = engine.session_file("restore", path)
        return dict(result, ms=(time.monotonic() - started) * 1000)
