"""Private journals and conservative admission for one suspended request.

These helpers do not own an engine, consume stdout, or resume requests after a
server restart. The active request owner must drain its accepted token stream,
commit a journal, and then unload. No external source or dependency was copied.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat

GIB, MIB = 2**30, 2**20
MAX_PREFIX_TOKENS = 2**20
MAX_JOURNAL_BYTES = 64 * MIB
MAX_EMBEDDING_BYTES = GIB


class ParkingError(RuntimeError):
    """The request must not claim durable suspension or safe reload."""


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value, minimum=0, maximum=2**63 - 1):
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise ParkingError("parking metadata must contain finite JSON values") from exc


def _json_copy(value):
    return json.loads(_canonical(value))


def effective_sampling(sampling):
    """Select one positive effective seed BEFORE the first native GEN.

    Retaining the seed and token positions preserves random-draw identity; it
    does not claim bit-exact logits after full prompt recomputation.
    """
    if not isinstance(sampling, dict):
        raise ParkingError("sampling must be an object")
    result = _json_copy(sampling)
    seed = result.get("seed")
    if seed is None or seed == 0 and not isinstance(seed, bool):
        result["seed"] = secrets.randbelow(2**63 - 1) + 1
    elif not _integer(seed, 1, 2**64 - 1):
        raise ParkingError("parking requires a positive integer seed, zero, or no seed")
    return result


def _reject_link(path):
    entry = path.lstat()
    if stat.S_ISLNK(entry.st_mode) or getattr(entry, "st_file_attributes", 0) & 0x400:
        raise ParkingError("parking paths cannot be symlinks or reparse points")
    return entry


def _windows_volume_flags(path):
    """Read the containing volume, including mounted-volume paths.

    CreateDirectoryW silently ignores the security descriptor on filesystems
    without FILE_PERSISTENT_ACLS; private journals must reject those volumes.
    """
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetVolumePathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    kernel.GetVolumeInformationW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
        wintypes.LPWSTR, wintypes.DWORD]
    volume = ctypes.create_unicode_buffer(32768)
    if not kernel.GetVolumePathNameW(str(path), volume, len(volume)):
        raise ctypes.WinError(ctypes.get_last_error())
    flags = wintypes.DWORD()
    if not kernel.GetVolumeInformationW(volume.value, None, 0, None, None, ctypes.byref(flags), None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return flags.value


def _private_directory(path):
    if os.name != "nt":
        path.mkdir(mode=0o700)
        os.chmod(path, 0o700)
        return
    if not _windows_volume_flags(path.parent) & 0x00000008:  # FILE_PERSISTENT_ACLS
        raise ParkingError("private parking journals require a filesystem with persistent ACLs")
    # Windows ignores POSIX mode bits in supported Python versions. Create the
    # directory with a protected DACL from its first observable instant. Only
    # this user and SYSTEM receive access; children inherit these permissions.
    import ctypes
    from ctypes import wintypes
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    adv.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    adv.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                       wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    adv.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    adv.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                                                         ctypes.POINTER(ctypes.c_void_p),
                                                                         ctypes.POINTER(wintypes.DWORD)]
    class SecurityAttributes(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("descriptor", ctypes.c_void_p), ("inherit", wintypes.BOOL)]
    kernel.CreateDirectoryW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(SecurityAttributes)]
    token, sid_text, descriptor = wintypes.HANDLE(), wintypes.LPWSTR(), ctypes.c_void_p()
    try:
        if not adv.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
            raise ctypes.WinError(ctypes.get_last_error())
        needed = wintypes.DWORD()
        adv.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
        if not 0 < needed.value <= 65536:
            raise ParkingError("cannot read current user SID")
        buffer = ctypes.create_string_buffer(needed.value)
        if not adv.GetTokenInformation(token, 1, buffer, needed, ctypes.byref(needed)):
            raise ctypes.WinError(ctypes.get_last_error())
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        if not adv.ConvertSidToStringSidW(sid, ctypes.byref(sid_text)):
            raise ctypes.WinError(ctypes.get_last_error())
        sddl = "D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;" + sid_text.value + ")"
        if not adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ctypes.byref(descriptor), None):
            raise ctypes.WinError(ctypes.get_last_error())
        attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
        if not kernel.CreateDirectoryW(str(path), ctypes.byref(attributes)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        if descriptor.value:
            kernel.LocalFree(descriptor)
        if sid_text:
            kernel.LocalFree(ctypes.cast(sid_text, ctypes.c_void_p))
        if token.value:
            kernel.CloseHandle(token)


class ParkingJournal:
    """An owned private directory; remove() never recursively deletes a tree."""
    def __init__(self, directory, identity, request_id=None):
        base = Path(directory)
        if not base.is_absolute():
            raise ParkingError("journal directory must be absolute")
        base.mkdir(parents=True, exist_ok=True)
        _reject_link(base)
        self.base = base.resolve(strict=True)
        self.identity = _json_copy(identity)
        if not identity or len(_canonical(identity)) > 16384:
            raise ParkingError("missing or oversized model/build/config identity")
        self.request_id = secrets.token_hex(16) if request_id is None else request_id
        if not isinstance(self.request_id, str) or not 1 <= len(self.request_id) <= 256:
            raise ParkingError("invalid request identity")
        self.journal_id = secrets.token_hex(16)
        self.directory = self.base / ("request-" + self.journal_id)
        _private_directory(self.directory)
        owner = _reject_link(self.directory)
        self._directory_identity = (owner.st_dev, owner.st_ino)
        self.path = self.directory / "request.json"
        self._owned = set()
        self.removed = False

    def _check_directory(self):
        if self.removed:
            raise ParkingError("journal was removed")
        entry = _reject_link(self.directory)
        if (entry.st_dev, entry.st_ino) != self._directory_identity or self.directory.parent != self.base:
            raise ParkingError("owned journal directory changed")

    def _atomic_bytes(self, destination, data):
        self._check_directory()
        temporary = self.directory / (".tmp-" + secrets.token_hex(16))
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            self._check_directory()
            if destination.exists():
                _reject_link(destination)
            os.replace(temporary, destination)
            self._owned.add(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _copy_embedding(self, source):
        source = Path(source)
        info = _reject_link(source)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_EMBEDDING_BYTES:
            raise ParkingError("invalid or oversized embedding file")
        temporary = self.directory / (".embedding-" + secrets.token_hex(16))
        self._check_directory()
        digest, length = hashlib.sha256(), 0
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with source.open("rb") as src, os.fdopen(descriptor, "wb") as dst:
                while block := src.read(MIB):
                    length += len(block)
                    if length > MAX_EMBEDDING_BYTES:
                        raise ParkingError("embedding changed beyond its permitted size")
                    digest.update(block)
                    dst.write(block)
                dst.flush()
                os.fsync(dst.fileno())
            after = _reject_link(source)
            if (length != info.st_size or after.st_size != info.st_size
                    or after.st_mtime_ns != info.st_mtime_ns or after.st_ino != info.st_ino):
                raise ParkingError("embedding changed while being journaled")
            filename = "embedding-" + digest.hexdigest() + ".bin"
            destination = self.directory / filename
            os.replace(temporary, destination)
            self._owned.add(destination)
            return {"file": filename, "bytes": length, "sha256": digest.hexdigest()}
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _validate_record(record):
        required = {"journal_id", "request_id", "identity", "prefix_ids", "sampling", "initial_budget",
                    "consumed_output", "remaining_budget", "sequence", "embedding"}
        if not isinstance(record, dict) or set(record) != required:
            raise ParkingError("invalid journal fields")
        ids = record["prefix_ids"]
        if (not isinstance(ids, list) or not 1 <= len(ids) <= MAX_PREFIX_TOKENS
                or not all(_integer(token, 0, 2**31 - 1) for token in ids)):
            raise ParkingError("invalid exact token prefix")
        initial, consumed = record["initial_budget"], record["consumed_output"]
        if (not _integer(initial, 1, MAX_PREFIX_TOKENS) or not _integer(consumed, 0, initial)
                or record["remaining_budget"] != initial - consumed
                or not _integer(record["remaining_budget"], 0, initial)
                or not _integer(record["sequence"])):
            raise ParkingError("invalid remaining output budget")
        if (not isinstance(record["sampling"], dict)
                or not _integer(record["sampling"].get("seed"), 1, 2**64 - 1)):
            raise ParkingError("journal must retain the previously selected positive seed")
        embedding = record["embedding"]
        if embedding is not None:
            if (not isinstance(embedding, dict) or set(embedding) != {"file", "bytes", "sha256"}
                    or not isinstance(embedding["sha256"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", embedding["sha256"]) is None
                    or embedding["file"] != "embedding-" + embedding["sha256"] + ".bin"
                    or not _integer(embedding["bytes"], 1, MAX_EMBEDDING_BYTES)):
                raise ParkingError("invalid journal embedding identity")
        _canonical(record)

    def write(self, prefix_ids, sampling, initial_budget, consumed_output, sequence=0, embedding=None):
        """Commit only after all accepted output through native DONE is drained."""
        self._check_directory()
        if (not _integer(initial_budget, 1, MAX_PREFIX_TOKENS)
                or not _integer(consumed_output, 0, initial_budget)):
            raise ParkingError("invalid output budget")
        try:
            prefix_ids = list(prefix_ids)
        except TypeError as exc:
            raise ParkingError("invalid exact token prefix") from exc
        record = {"journal_id": self.journal_id, "request_id": self.request_id, "identity": self.identity,
                  "prefix_ids": prefix_ids, "sampling": _json_copy(sampling), "initial_budget": initial_budget,
                  "consumed_output": consumed_output, "remaining_budget": initial_budget - consumed_output,
                  "sequence": sequence, "embedding": None}
        self._validate_record(record)  # Reject bad metadata before copying large embeddings.
        if embedding is not None:
            record["embedding"] = self._copy_embedding(embedding)
        self._validate_record(record)
        body = _canonical(record)
        envelope = _canonical({"version": 1, "sha256": hashlib.sha256(body).hexdigest(), "record": record})
        if len(envelope) > MAX_JOURNAL_BYTES:
            raise ParkingError("journal exceeds its size limit")
        self._atomic_bytes(self.path, envelope)
        return self.load(self.identity)

    def load(self, expected_identity):
        self._check_directory()
        try:
            entry = _reject_link(self.path)
            if not stat.S_ISREG(entry.st_mode) or not 0 < entry.st_size <= MAX_JOURNAL_BYTES:
                raise ParkingError("invalid journal size or type")
            with self.path.open("rb") as stream:
                raw = stream.read(MAX_JOURNAL_BYTES + 1)
            if len(raw) > MAX_JOURNAL_BYTES:
                raise ParkingError("journal grew beyond its size limit")
            envelope = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise ParkingError("cannot read a valid journal") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"version", "sha256", "record"} or envelope["version"] != 1:
            raise ParkingError("invalid journal envelope")
        record = envelope["record"]
        if envelope["sha256"] != hashlib.sha256(_canonical(record)).hexdigest():
            raise ParkingError("journal checksum mismatch")
        self._validate_record(record)
        if record["journal_id"] != self.journal_id or record["request_id"] != self.request_id:
            raise ParkingError("journal does not belong to this active request")
        if _canonical(record["identity"]) != _canonical(expected_identity):
            raise ParkingError("model/build/config identity changed")
        result = _json_copy(record)
        result["embedding_path"] = None
        embedding = record["embedding"]
        if embedding is not None:
            path = self.directory / embedding["file"]
            try:
                info = _reject_link(path)
                if not stat.S_ISREG(info.st_mode) or info.st_size != embedding["bytes"]:
                    raise ParkingError("embedding size or type changed")
                digest = hashlib.sha256()
                length = 0
                with path.open("rb") as stream:
                    while block := stream.read(MIB):
                        length += len(block)
                        if length > embedding["bytes"]:
                            raise ParkingError("embedding grew while being read")
                        digest.update(block)
                if length != embedding["bytes"] or digest.hexdigest() != embedding["sha256"]:
                    raise ParkingError("embedding checksum mismatch")
            except OSError as exc:
                raise ParkingError("journal embedding is unavailable") from exc
            result["embedding_path"] = str(path)
        return result

    def remove(self):
        """Delete only files this object created, then its empty private directory."""
        if self.removed:
            return
        self._check_directory()
        for path in tuple(self._owned):
            if path.parent != self.directory:
                raise ParkingError("refusing removal outside the owned journal")
            if path.exists():
                _reject_link(path)
                path.unlink()
            self._owned.discard(path)
        self.directory.rmdir()  # Unknown files are deliberately never removed.
        self.removed = True


class ResumeAdmission:
    """Fresh global capacity must admit the full reload footprint, not 250 MiB."""
    def __init__(self, footprint, *, ram_headroom_gib=3, vram_headroom_mib=320,
                 recovery_seconds=10, retry_seconds=30, max_age=5, require_commit=None):
        if (not isinstance(footprint, dict) or set(footprint) != {"ram_bytes", "commit_bytes", "gpu_bytes"}
                or not all(_number(v) and v > 0 for v in footprint.values())):
            raise ParkingError("a measured full RAM, commit and GPU reload footprint is required")
        for name, value, minimum, maximum in (
                ("ram headroom", ram_headroom_gib, 1, 128), ("VRAM headroom", vram_headroom_mib, 250, 65536),
                ("recovery dwell", recovery_seconds, 5, 300), ("retry interval", retry_seconds, 5, 300),
                ("sample age", max_age, 1, 10)):
            if not _number(value) or not minimum <= value <= maximum:
                raise ParkingError("invalid reload " + name)
        if require_commit is not None and not isinstance(require_commit, bool):
            raise ParkingError("require_commit must be boolean")
        self.required = {"ram_bytes": footprint["ram_bytes"] + ram_headroom_gib * GIB,
                         "commit_bytes": footprint["commit_bytes"] + ram_headroom_gib * GIB,
                         "gpu_bytes": footprint["gpu_bytes"] + vram_headroom_mib * MIB}
        self.recovery_seconds, self.retry_seconds, self.max_age = recovery_seconds, retry_seconds, max_age
        self.require_commit = os.name == "nt" if require_commit is None else require_commit
        self.since = self.last_stamp = None
        self.retry_at = 0
        self.failures = 0

    def reset(self):
        self.since = self.last_stamp = None

    def failed(self, now):
        if not _number(now):
            raise ParkingError("invalid retry timestamp")
        self.failures += 1
        self.retry_at = now + min(300, self.retry_seconds * 2 ** min(4, self.failures - 1))
        self.reset()

    def observe(self, snapshot, now):
        result = {"ready": False, "reason": "telemetry_unavailable", "required": dict(self.required),
                  "retry_at": self.retry_at, "stable_seconds": 0}
        if not isinstance(snapshot, dict) or not _number(now):
            self.reset()
            return result
        stamp = snapshot.get("sampled_at")
        keys = ("ram_total", "ram_used", "gpu_mem_total", "gpu_mem_used")
        if (not _number(stamp) or not 0 <= now - stamp <= self.max_age
                or not all(_number(snapshot.get(k)) for k in keys)
                or snapshot.get("native_capacity_required")
                or not 0 <= snapshot["ram_used"] <= snapshot["ram_total"] or snapshot["ram_total"] <= 0
                or not 0 <= snapshot["gpu_mem_used"] <= snapshot["gpu_mem_total"] or snapshot["gpu_mem_total"] <= 0):
            self.reset()
            return result
        commit = snapshot.get("ram_commit_available")
        if (self.require_commit or snapshot.get("ram_commit_required") or "ram_commit_available" in snapshot) and (
                not _number(commit) or commit < 0):
            self.reset()
            result["reason"] = "commit_unavailable"
            return result
        if self.last_stamp is not None and stamp <= self.last_stamp:
            self.since = None
            result["reason"] = "telemetry_not_advancing"
            return result
        if self.last_stamp is not None and stamp - self.last_stamp > self.max_age:
            self.since = None
        self.last_stamp = stamp
        free_ram = snapshot["ram_total"] - snapshot["ram_used"]
        free_gpu = snapshot["gpu_mem_total"] - snapshot["gpu_mem_used"]
        result["available"] = {"ram_bytes": free_ram, "commit_bytes": commit, "gpu_bytes": free_gpu}
        if now < self.retry_at:
            self.since = None
            result["reason"] = "reload_backoff"
        elif (free_ram < self.required["ram_bytes"] or free_gpu < self.required["gpu_bytes"]
              or commit is not None and commit < self.required["commit_bytes"]):
            self.since = None
            result["reason"] = "reload_footprint_unavailable"
        else:
            self.since = stamp if self.since is None else self.since
            result["stable_seconds"] = stamp - self.since
            result["ready"] = result["stable_seconds"] >= self.recovery_seconds
            result["reason"] = "reload_admitted" if result["ready"] else "recovery_dwell"
        return result


class RequestParking:
    """Explicit opt-in configuration and the persistent-pressure timer."""
    def __init__(self, config=None):
        config = {} if config is None else config
        allowed = {"enabled", "directory", "pressure_seconds", "recovery_seconds", "retry_seconds",
                   "min_ram_headroom_gib", "min_vram_headroom_mib", "max_sample_age_seconds"}
        if not isinstance(config, dict) or set(config) - allowed or not isinstance(config.get("enabled", False), bool):
            raise ParkingError("invalid request_parking configuration")
        self.enabled = config.get("enabled", False)
        self.directory = config.get("directory")
        if self.enabled and (not isinstance(self.directory, str) or not Path(self.directory).is_absolute()):
            raise ParkingError("enabled request parking requires an absolute private journal directory")
        limits = {"pressure_seconds": (10, 2, 300), "recovery_seconds": (10, 5, 300),
                  "retry_seconds": (30, 5, 300), "min_ram_headroom_gib": (3, 1, 128),
                  "min_vram_headroom_mib": (320, 250, 65536), "max_sample_age_seconds": (5, 1, 10)}
        for key, (default, minimum, maximum) in limits.items():
            value = config.get(key, default)
            if not _number(value) or not minimum <= value <= maximum:
                raise ParkingError("invalid request_parking setting: " + key)
            setattr(self, key, value)
        self.since = self.last_observed = None

    def should_park(self, decision, now):
        reasons = {"ram_floor", "vram_floor", "memory_floors", "non_evictable_ram_floor", "non_evictable_vram_floor"}
        critical = (self.enabled and isinstance(decision, dict) and decision.get("action") == "wait"
                    and decision.get("reason") in reasons and _number(now))
        if (not critical or self.last_observed is not None and (
                now <= self.last_observed or now - self.last_observed > self.max_sample_age_seconds)):
            self.since = None
        self.last_observed = now if _number(now) else None
        if not critical:
            return False
        self.since = now if self.since is None else self.since
        return now - self.since >= self.pressure_seconds

    effective_sampling = staticmethod(effective_sampling)

    def create_journal(self, identity, request_id=None):
        if not self.enabled:
            raise ParkingError("request parking is disabled")
        return ParkingJournal(self.directory, identity, request_id)

    def admission(self, footprint):
        return ResumeAdmission(footprint, ram_headroom_gib=self.min_ram_headroom_gib,
                               vram_headroom_mib=self.min_vram_headroom_mib,
                               recovery_seconds=self.recovery_seconds, retry_seconds=self.retry_seconds,
                               max_age=self.max_sample_age_seconds)
