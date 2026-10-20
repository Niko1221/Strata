#!/usr/bin/env python3
"""Reviewed, private model preparation and owned lazy serving over SSH.

The transport supplies the fixed source bundle. Existing API processes are never
adopted, stopped or reconfigured. All mutable state belongs to one connection.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import importlib
import itertools
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid

spec = importlib.util.spec_from_file_location("_strata_remote_files", Path(__file__).with_name("remote_model_manager.py"))
files = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = files
spec.loader.exec_module(files)
ApiError = files.ApiError
HEX = re.compile(r"[a-f0-9]{64}\Z")
ID = re.compile(r"[a-zA-Z0-9_-]{1,160}\Z")
MAX_STATE = 4 * 1024 * 1024
ACTIVE = ("queued", "running")
INPUT_FLAGS = frozenset(("--native", "--pack", "--mtp", "--expert-profile", "--native-dense-gguf", "--ple-gguf", "--embd-gguf", "--mtp-draft-vocab", "--control-vector"))
SCALAR_FLAGS = frozenset(("--max-context", "--expert-cache", "--prefill", "--spec", "--spec-min-p", "--kv", "--kv-resident", "--rope-scaling", "--rope-scale", "--ple-io", "--resident-budget-gib", "--vram-reserve-mib", "--pool-workers", "--threads", "--cpu-threads", "--kv-threads", "--prefill-threads", "--batch", "--batch-groups", "--layer-split", "--pcie-mode", "--pcie-frac", "--adapt-every", "--adapt-decay", "--adapt-min-gain", "--adapt-async", "--adapt-swaps", "--mtp-hnorm", "--mtp-q4", "--cvec-mode", "--cvec-dir", "--prompt-cache", "--pipeline-windows", "--mtp-max-t", "--lookup-chain", "--lookup-chain-min"))
BOOLEAN_FLAGS = frozenset(("--mmap-experts", "--resident-experts", "--resident-cpu-experts", "--expert-cache-cpu-order", "--expert-cache-per-layer", "--spec-split", "--no-spec-split", "--stop-eos", "--prefill-head-alias", "--no-hit-poke", "--vision", "--remote-expert-opt"))
GUARD = '''import json,subprocess,sys
try:
 cards=json.loads(sys.argv[1]); minimum=int(sys.argv[2])
 result=subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10,check=True)
 free={int(a.strip()):int(b.strip()) for a,b in (line.split(",") for line in result.stdout.splitlines())}
 if not cards or any(free.get(card,-1)<minimum for card in cards): raise ValueError("insufficient free GPU memory")
except Exception:
 print("Strata kept the model unloaded: GPU memory could not be safely reserved. Existing models were left running.",file=sys.stderr);sys.exit(1)
'''


def checked_json(path, maximum=MAX_STATE):
    path = Path(path)
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or details.st_size > maximum or details.st_uid != os.getuid():
        raise ApiError(403, "The private lifecycle records are unavailable.")
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum: raise ApiError(413, "The lifecycle records are too large.")
    value = json.loads(raw)
    if not isinstance(value, dict): raise ApiError(400, "The lifecycle records need repair.")
    return value


def hash_file(path):
    path = Path(path)
    before = files.identity(path)
    digest = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
        if files.file_identity(path, os.fstat(stream.fileno())) != before:
            raise ApiError(409, "A reviewed file changed. Review the model again.")
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    if files.identity(path) != before: raise ApiError(409, "A reviewed file changed. Review the model again.")
    return {**before, "sha256": digest.hexdigest()}


def source_signature(source):
    paths = [source / "setup.py", Path(__file__).resolve(), Path(__file__).with_name("remote_model_manager.py").resolve()]
    paths.extend(sorted((source / "data").rglob("*")))
    paths.extend(sorted(source.glob("requirements*.txt")))
    for folder in ("strata_control", "strata_cli", "serve", "tools"):
        paths.extend(sorted((source / folder).rglob("*.py")))
    return files.digest([(str(path), hash_file(path)) for path in paths if path.is_file()])


def located(raw, cwd):
    if not isinstance(raw, str) or not raw or "\0" in raw: raise ApiError(409, "The existing profile contains an invalid model-data path.")
    path = Path(raw).expanduser()
    if not path.is_absolute(): path = Path(cwd) / path
    return path.resolve(strict=True)


def reuse_arguments(cfg, config_path):
    """Close every engine path channel before reusing a third-party run config."""
    cwd = cfg.get("cwd") or str(Path(config_path).parent)
    if not isinstance(cwd, str) or not Path(cwd).is_absolute(): raise ApiError(409, "The existing profile needs an absolute working folder before it can be reused.")
    raw, result, index = cfg["args"], [], 0
    while index < len(raw):
        flag = raw[index]; index += 1
        if flag in BOOLEAN_FLAGS: result.append({"flag": flag, "values": []}); continue
        arity = 2 if flag == "--control-vector-layer-range" else 1
        if flag not in INPUT_FLAGS | SCALAR_FLAGS | {"--control-vector-scaled", "--control-vector-layer-range"}:
            raise ApiError(409, "This existing profile contains engine options that cannot be safely isolated. Prepare it with standard Strata settings first.")
        values = raw[index:index + arity]; index += arity
        if len(values) != arity: raise ApiError(409, "The existing profile has incomplete engine options.")
        if flag in INPUT_FLAGS:
            result.append({"flag": flag, "paths": [str(located(values[0], cwd))]})
        elif flag == "--control-vector-scaled":
            vectors = []
            for value in values[0].split(","):
                path, separator, scale = value.rpartition(":")
                if not separator or not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", scale): raise ApiError(409, "The existing profile has an invalid control-vector option.")
                vectors.append({"path": str(located(path, cwd)), "scale": scale})
            result.append({"flag": flag, "vectors": vectors})
        else:
            if any(not re.fullmatch(r"[A-Za-z0-9_.:+,-]{1,128}", value) for value in values):
                raise ApiError(409, "The existing profile contains an unqualified engine path or option.")
            result.append({"flag": flag, "values": values})
    return result, str(located(cfg["tokenizer"], cwd))


def dependency_identities(paths):
    result = {}
    for raw in paths:
        path = Path(raw)
        if path.is_symlink(): raise ApiError(409, "Prepared dependencies must not be symbolic links.")
        members = itertools.chain((path,), path.rglob("*")) if path.is_dir() else (path,)
        for member in members:
            if member.is_symlink(): raise ApiError(409, "Prepared dependencies must not be symbolic links.")
            details = member.lstat()
            if member.is_dir():
                result[str(member)] = {"path": str(member), "directory": True, "device": details.st_dev, "inode": details.st_ino, "mtime": details.st_mtime_ns, "ctime": details.st_ctime_ns}
            else: result[str(member)] = files.identity(member)
            if len(result) > 10000: raise ApiError(409, "The prepared dependency folder is too large to inspect safely.")
    return [result[key] for key in sorted(result)]


def gpu_indices(value):
    values = value.split(",") if isinstance(value, str) else value if isinstance(value, list) else [value]
    if not values or len(values) > 64: raise ApiError(409, "The selected Strata configuration needs explicit NVIDIA device indices.")
    result = []
    for item in values:
        if isinstance(item, str) and re.fullmatch(r"\s*\d{1,4}\s*", item): item = int(item)
        if type(item) is not int or not 0 <= item <= 1023 or item in result: raise ApiError(409, "The selected Strata configuration needs distinct NVIDIA device indices.")
        result.append(item)
    return result


class Lifecycle:
    def __init__(self, source, namespace, destination, catalog_source=None, home=None, now=time.time):
        if not isinstance(namespace, str) or not HEX.fullmatch(namespace) or not isinstance(destination, str) or not HEX.fullmatch(destination):
            raise ApiError(400, "Choose a saved Strata connection.")
        self.source = Path(source).resolve(strict=True)
        if not files.strata_source(self.source): raise ApiError(503, "This Strata build has no preparation source bundle.")
        self.pin_source_namespaces()
        self.manager = files.Manager(home=home, source=catalog_source)
        self.namespace, self.destination, self.now = namespace, destination, now
        self.root = files.owned_directory(self.manager.home, "lifecycle/" + namespace)
        self.state_file = self.root / "state.json"
        self.runtime_name = "mac-" + namespace[:24]

    @contextlib.contextmanager
    def locked(self):
        # All file-manager and lifecycle mutations share one filesystem lock.
        with self.manager.locked(): yield

    def load(self):
        if not self.state_file.exists(): return {"plans": {}, "jobs": {}, "profiles": {}, "runtime": {}, "receipts": {}, "intent": {}}
        value = checked_json(self.state_file)
        if any(not isinstance(value.get(key), dict) for key in ("plans", "jobs", "profiles", "runtime", "receipts")):
            raise ApiError(500, "The private lifecycle records need repair.")
        value.setdefault("intent", {})
        if not isinstance(value["intent"], dict): raise ApiError(500, "The private serving intent needs repair.")
        return value

    def save(self, state):
        if len(json.dumps(state).encode()) > MAX_STATE: raise ApiError(429, "The private lifecycle records are full.")
        files.atomic_json(self.state_file, state)

    def prune(self, state):
        state["plans"] = {key: value for key, value in state["plans"].items() if value["expiresAt"] > self.now() * 1000}
        for job in state["jobs"].values():
            if job["state"] in ACTIVE and self.now() - job.get("heartbeat", job["created"]) > 60:
                observed = files.process_identity(job.get("pid", 0)) if job.get("pid") else None
                if not observed or observed[0] != job.get("processIdentity") or job["token"] not in observed[1]:
                    job.update(state="error", summary="The operation stopped unexpectedly. Existing APIs were left unchanged; review before retrying.")
        finished = [key for key, job in state["jobs"].items() if job["state"] not in ACTIVE]
        for key in finished[:-50]: del state["jobs"][key]

    @contextlib.contextmanager
    def environment(self, install=None, acquire=False):
        old = dict(os.environ)
        old_path = list(sys.path)
        # Isolate discovery as well as config and runtime writes. Never inherit a
        # user's active install/settings/component or runtime storage locations.
        for key in list(os.environ):
            if key.startswith("STRATA_"): del os.environ[key]
        os.environ.update(self.env(install=install, acquire=acquire))
        sys.path.insert(0, str(self.source))
        try: yield
        finally:
            os.environ.clear(); os.environ.update(old); sys.path[:] = old_path

    def env(self, install=None, acquire=False):
        install = Path(install or self.root / "runtime-install")
        result = {"STRATA_INSTALL_DIR": str(install), "STRATA_SETTINGS_FILE": str(install / "settings.json"),
                "STRATA_COMPONENTS_DIR": str(self.root / "components"), "STRATA_PROFILE_DIR": str(self.root / "runtime-profiles"),
                "XDG_STATE_HOME": str(self.root / "runtime-state"), "XDG_CONFIG_HOME": str(self.root / "config"),
                "STRATA_CONTROL_INSTALL": "1", "STRATA_ALLOW_NO_ENGINE": "1",
                "STRATA_KEEP_LEFTOVERS": "1", "STRATA_NO_PRESET": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(self.source)}
        if not acquire: result["STRATA_NO_ACQUIRE"] = "1"
        return result

    def source_modules(self):
        self.pin_source_namespaces()
        from strata_control import models, profiles, runtime
        return models, profiles, runtime

    def pin_source_namespaces(self):
        # Reading an older setup.py executes its top-level catalog definitions.
        # Pin package imports first, so that older checkout cannot seed cached
        # control/runtime modules that bypass this build's ownership guards.
        old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
        try:
            sys.path.insert(0, str(self.source)); sys.dont_write_bytecode = True
            for prefix in ("strata_control", "strata_cli", "serve"):
                names = [name for name in sys.modules if name == prefix or name.startswith(prefix + ".")]
                foreign = any(getattr(sys.modules[name], "__file__", None) and not Path(sys.modules[name].__file__).resolve().is_relative_to(self.source) for name in names)
                if foreign:
                    for name in names: del sys.modules[name]
                package = importlib.import_module(prefix)
                if not getattr(package, "__file__", None) or not Path(package.__file__).resolve().is_relative_to(self.source):
                    raise ApiError(503, "The private Strata source bundle could not be qualified.")
            importlib.import_module("strata_control.acquisition")
        finally:
            sys.path[:], sys.dont_write_bytecode = old_path, old_bytecode

    @contextlib.contextmanager
    def interpreter(self, path):
        previous = sys.executable
        sys.executable = str(path)
        try: yield
        finally: sys.executable = previous

    def check_dependencies(self, python):
        try:
            result = subprocess.run([python, "-c", "import sys,jinja2,numpy; sys.exit(0 if sys.version_info >= (3,10) else 1)"],
                                    env={**os.environ, **self.env()}, cwd=self.source, stdin=subprocess.DEVNULL,
                                    capture_output=True, timeout=20, check=False)
        except (OSError, subprocess.SubprocessError): result = None
        if result is None or result.returncode:
            raise ApiError(503, "The selected Strata Python runtime needs Python 3.10 or newer, Jinja2 and NumPy. Update that Strata runtime before preparing or serving; existing APIs were left unchanged.")

    def input_model(self, model_id):
        if not isinstance(model_id, str) or not ID.fullmatch(model_id): raise ApiError(400, "Choose a model in this server's library.")
        with self.manager.locked():
            state = self.manager.load()
            imported = copy.deepcopy(state["imports"].get(model_id))
            jobs = copy.deepcopy(list(state["jobs"].values()))
        row = None
        if imported:
            current = files.inspect_files(imported["path"])
            if current != imported: raise ApiError(409, "The imported files changed. Import them again before preparing.")
            paths = [Path(entry["path"]) for entry in current["files"]]
            with self.environment():
                models, _, _ = self.source_modules()
                inspection = models.plan_import(current["path"])
            candidate = inspection.get("candidate_selection")
            if inspection.get("file_set_status") != "plausible" or inspection.get("issues") or not isinstance(candidate, dict):
                raise ApiError(409, "These imported files are not a supported complete Strata model. Their reference is kept; choose supported model files.")
            selection = {key: candidate[key] for key in ("family", "model")}
            name, kind = current["name"], "imported"
            identities = [{key: entry[key] for key in ("path", "device", "inode", "bytes", "mtime", "ctime")} for entry in current["files"]]
        else:
            row = self.manager.current_catalog().get(model_id)
            job = next((job for job in reversed(jobs) if row and job["modelId"] == model_id and self.manager.row_ready(job, row)), None)
            if not row or not job: raise ApiError(409, "Download and verify this model before preparing it.")
            if not all(entry["path"].lower().endswith(".gguf") for entry in row["files"]):
                raise ApiError(409, "This preparation workflow currently supports GGUF models. The downloaded files are kept.")
            folder = self.manager.assets / files.digest(job["manifest"]["files"])
            paths = [folder / entry["path"] for entry in row["files"]]
            identities = [files.identity(path) for path in paths]
            selection = {key: row[key] for key in ("family", "model")}
            name, kind = row["name"], "downloaded"
        return {"id": model_id, "name": name, "kind": kind, "selection": selection, "files": identities,
                "bytes": sum(entry["bytes"] for entry in identities), "catalog": files.digest(row) if row else None}

    def engine_candidates(self):
        folders = []
        if self.manager.source: folders.append(self.manager.source)
        folders.append(Path.home() / "Strata")
        try:
            with self.environment():
                models, _, _ = self.source_modules()
                folders.extend(Path(path) for path in models.list_installed().get("install_dirs", []))
        except Exception: pass
        result, seen = [], set()
        for folder in folders:
            if not folder.is_dir(): continue
            for path in sorted(folder.glob("strata-*.json"))[:128]:
                try:
                    path = path.absolute()
                    if str(path) in seen: continue
                    seen.add(str(path)); cfg = checked_json(path, 1024 * 1024)
                    exe = Path(cfg.get("exe", ""))
                    if not exe.is_absolute() or exe.name.lower() not in ("strata", "strata.exe") or not exe.is_file() or exe.is_symlink(): continue
                    if "maya" in str(exe).lower() or "maya" in str(cfg.get("model_name", "")).lower() or cfg.get("before_load"):
                        continue
                    if not isinstance(cfg.get("args"), list) or not all(isinstance(arg, str) for arg in cfg["args"]): continue
                    if "--native" not in cfg["args"] or not isinstance(cfg.get("tokenizer"), str): continue
                    config_identity = files.identity(path); engine_identity = files.identity(exe)
                    python_candidates = [Path(cfg.get("cwd", str(folder))) / ".venv/bin/python", folder / ".venv/bin/python", Path(sys.executable)]
                    python = next((candidate.absolute() for candidate in python_candidates if candidate.is_file() and os.access(candidate, os.X_OK)), None)
                    if python is None: continue
                    resolved_libraries = [str(located(value, cfg.get("cwd") or folder)) for value in cfg.get("lib_dirs", []) if isinstance(value, str)]
                    libraries = []
                    for library in [exe.parent, *(Path(value) for value in resolved_libraries)]:
                        if not library.is_dir(): continue
                        for member in library.iterdir():
                            if member.is_file() and not member.is_symlink() and (".so" in member.name or member.suffix.lower() in (".dll", ".dylib") or member.name == "BUILD.json"):
                                libraries.append(files.identity(member))
                                if len(libraries) > 2048: raise ApiError(409, "The engine library list is too large to inspect safely.")
                    cards = gpu_indices(cfg.get("gpu", 0))
                    identifier = "engine-" + files.digest([config_identity, engine_identity])[:24]
                    result.append({"id": identifier, "name": files.clean(cfg.get("model_name") or path.stem), "backend": cfg.get("backend", "cuda"),
                                   "configPath": str(path), "configIdentity": config_identity, "engineIdentity": engine_identity, "config": cfg,
                                   "python": str(python), "pythonIdentity": files.identity(python.resolve()), "libraries": sorted(libraries, key=lambda row: row["path"])})
                    result[-1]["resolvedLibDirs"] = resolved_libraries
                    result[-1]["gpuIndices"] = cards
                except (OSError, ValueError, TypeError, ApiError): continue
        return result

    def engine(self, engine_id):
        candidates = self.engine_candidates()
        if engine_id is None and len(candidates) == 1: return candidates[0]
        result = next((row for row in candidates if row["id"] == engine_id), None)
        if not result: raise ApiError(409, "Choose an existing Strata engine configuration on this server. Maya configurations are excluded.")
        if result["backend"] not in (None, "cuda"):
            raise ApiError(409, "Serving needs a supported NVIDIA memory guard. This engine is kept unchanged.")
        return result

    def runtime_snapshot(self, state):
        try:
            with self.environment():
                _, _, runtime = self.source_modules()
                observed = runtime.inspect(self.runtime_name)
            record, intent = state["runtime"], state.get("intent", {})
            # This private owner is always inspected, including after a worker
            # died before publishing its endpoint. Never mistake an unrecorded
            # live/starting owner for an available port.
            if observed.get("state") == "stopped" and not observed.get("owner_host_alive"):
                return {"state": "stopped", **({key: record.get(key) for key in ("profileId", "modelId", "host", "port")} if record else {})}
            candidates = [candidate for candidate in (intent, record) if candidate.get("profileId")]
            matched = next((candidate for candidate in candidates if self.matches_owned_source(candidate, observed)), None)
            if matched is None:
                return {"state": observed.get("state", "unavailable") if observed.get("state") != "ready" else "stale",
                        "profileId": intent.get("profileId") or record.get("profileId"), "busy": None}
            if observed.get("state") == "ready" and matched is intent:
                self.publish_runtime(state, intent, observed)
                record = state["runtime"]; matched = record
            if observed.get("child_alive") is False and observed.get("owner_host_alive") is True:
                return {"state": "stopped", **{key: matched.get(key) for key in ("profileId", "modelId", "host", "port")},
                        "loaded": False, "busy": False, "reapNeeded": True, "ownerId": observed["owner_id"], "sourceSha256": observed["source_sha256"]}
            if observed.get("state") == "ready" and matched is record and record.get("instanceId") != observed.get("instance_id"):
                # A successful rollback has a new instance with the exact old
                # immutable config. Fresh attestation proves that recovery.
                with self.locked():
                    latest = self.load()
                    if latest["runtime"].get("sourceHash") == record.get("sourceHash"):
                        latest["runtime"].update(instanceId=observed["instance_id"], ownerId=observed["owner_id"])
                        latest["receipts"] = {}  # older receipts are instance-bound
                        self.save(latest); state.clear(); state.update(latest); record = latest["runtime"]
                matched = record
            loaded = observed.get("loaded")
            if type(loaded) is not bool: loaded = None
            return {"state": observed.get("state", "unknown"), **{key: matched.get(key) for key in ("profileId", "modelId", "host", "port")},
                    "loaded": loaded, "busy": observed.get("busy") if not observed.get("draining") else True,
                    "ownerId": observed.get("owner_id"), "instanceId": observed.get("instance_id"),
                    "sourceSha256": observed.get("source_sha256")}
        except Exception: return {"state": "unavailable", "profileId": state.get("intent", {}).get("profileId") or state["runtime"].get("profileId"), "busy": None}

    def matches_owned_source(self, record, observed):
        if not record.get("sourceHash"):
            # Compatibility only for pre-intent records already bound to an
            # exact live owner+instance. New writes always carry source custody.
            return bool(record.get("ownerId") and record.get("instanceId") and record["ownerId"] == observed.get("owner_id") and record["instanceId"] == observed.get("instance_id"))
        if observed.get("source_sha256") != record["sourceHash"]: return False
        expected = Path(record["configPath"])
        if not expected.is_relative_to(self.root / "runtime-configs") or hashlib.sha256(expected.read_bytes()).hexdigest() != record["sourceHash"]:
            return False
        if observed.get("source_config") not in (None, str(expected)): return False
        if observed.get("state") == "ready":
            if observed.get("source_config") != str(expected) or observed.get("model_id") != record["modelId"] or not observed.get("launch_attested") or not observed.get("api_identity_verified"):
                return False
            self.profile(self.load(), record["profileId"])
        return bool(re.fullmatch(r"[a-f0-9]{32}", str(observed.get("owner_id", ""))))

    def publish_runtime(self, state, intent, observed):
        record = {key: intent[key] for key in ("profileId", "modelId", "host", "port", "apiKey", "destination", "receiptId", "configPath", "sourceHash")}
        record.update(instanceId=observed["instance_id"], ownerId=observed["owner_id"])
        with self.locked():
            latest = self.load()
            if latest.get("intent", {}).get("jobId") != intent.get("jobId"): return
            latest["runtime"] = record; latest["receipts"] = {record["receiptId"]: record}; latest["intent"] = {}
            job = latest["jobs"].get(intent.get("jobId"))
            if job is not None:
                job.update(state="done", summary="The separate Strata API is ready with its model unloaded. Connect it to the Mac app.",
                           profileId=record["profileId"], receiptId=record["receiptId"], heartbeat=self.now())
            self.save(latest); state.clear(); state.update(latest)

    def job_view(self, job):
        return {key: job[key] for key in ("id", "kind", "state", "requestId", "summary", "percent", "profileId", "receiptId") if key in job} | {"canCancel": job["state"] in ACTIVE and job["kind"] == "prepare"}

    def snapshot(self):
        with self.locked():
            state = self.load(); self.prune(state); self.save(state)
        runtime_view = self.runtime_snapshot(state)
        library = self.manager.dispatch("desktop", "", "GET", None)
        models = []
        for item in library["models"]:
            row = {key: item[key] for key in ("id", "name", "state")}
            row["preparable"] = item["state"] in ("downloaded", "imported")
            if not row["preparable"]: row["reason"] = "Download the model files first."
            elif item["state"] in ("downloaded", "imported"):
                try: self.input_model(item["id"])
                except ApiError as error: row.update(preparable=False, reason=error.message)
            profile = next((value for value in reversed(list(state["profiles"].values())) if value["modelId"] == item["id"]), None)
            if profile: row["profileId"] = profile["id"]
            models.append(row)
        engines = [{key: row[key] for key in ("id", "name", "backend")} for row in self.engine_candidates()]
        return {"supported": True, "capabilities": {"prepare": bool(engines), "serve": bool(state["profiles"])}, "models": models,
                "engines": engines, "profiles": [{key: row[key] for key in ("id", "name", "modelId", "context", "state", "executionCompatibility")} for row in state["profiles"].values()],
                "jobs": [self.job_view(job) for job in state["jobs"].values()], "runtime": runtime_view,
                **({"reason": "Choose an existing Strata engine configuration. Maya is kept separate."} if not engines else {})}

    def remember(self, kind, value):
        with self.locked():
            state = self.load(); self.prune(state)
            if len(state["plans"]) >= 100: raise ApiError(429, "Close an old model review and try again shortly.")
            identifier, expires = str(uuid.uuid4()), (self.now() + 300) * 1000
            state["plans"][identifier] = {"kind": kind, "destination": self.destination, "expiresAt": expires, "value": value}
            self.save(state)
        return {"planId": identifier, "expiresAt": expires}

    def consume(self, kind, plan_id):
        if not isinstance(plan_id, str) or not ID.fullmatch(plan_id): raise ApiError(400, "Review the model before continuing.")
        with self.locked():
            state = self.load(); self.prune(state); plan = state["plans"].get(plan_id)
            if not plan or plan["kind"] != kind or plan["destination"] != self.destination or plan["expiresAt"] <= self.now() * 1000:
                raise ApiError(409, "This model review expired or was already used. Review it again.")
            del state["plans"][plan_id]; self.save(state)
        return plan["value"]

    def prepare_plan(self, body):
        files.exact(body, ("modelId", "engineId", "context"), ("modelId",))
        context = body.get("context", 32768)
        if type(context) is not int or context not in (8192, 16384, 32768, 65536, 131072): raise ApiError(400, "Choose a supported context size.")
        model = self.input_model(body["modelId"]); engine = self.engine(body.get("engineId"))
        engine["engineHash"] = hash_file(engine["config"]["exe"])
        # A private staged copy costs the full input size even on filesystems
        # where a reflink may later make it cheaper. Helpers/derived packs need
        # another full input-size allowance plus 12 GiB.
        needed = model["bytes"] * 2 + 12 * 1024 ** 3
        free = shutil.disk_usage(self.root).free
        reuse = self.reuse_plan(model, engine)
        value = {"model": model, "engine": engine, "context": context, "sourceSignature": source_signature(self.source), "diskNeeded": needed, "reuse": reuse}
        return {**self.remember("prepare", value), "modelName": model["name"], "engineName": engine["name"], "context": context,
                "diskNeededGb": needed / 1e9, "diskFreeGb": free / 1e9, "diskShort": free < needed,
                "warnings": ["Prepare an isolated copy on this Strata server. Existing model files, engines, APIs and Maya stay unchanged.",
                             "The new prepared profile is text-only. Existing picture-reading settings on the original server stay unchanged.",
                             "Preparation may download pinned tokenizer and preparation tools and create derived data. It does not load a model.",
                             "The existing Strata engine is reused without updating it. Execution compatibility remains unverified until loaded.",
                             "Serving starts a separate API with its model unloaded. Loading refuses when selected NVIDIA GPUs lack free memory; other models are never unloaded."]}

    def check_prepare(self, value, *, disk=True):
        if source_signature(self.source) != value["sourceSignature"] or self.input_model(value["model"]["id"]) != value["model"]:
            raise ApiError(409, "The source or model files changed. Review preparation again.")
        current = self.engine(value["engine"]["id"])
        if current != {key: entry for key, entry in value["engine"].items() if key != "engineHash"} or hash_file(current["config"]["exe"]) != value["engine"]["engineHash"]:
            raise ApiError(409, "The selected engine changed. Review preparation again.")
        if self.reuse_plan(value["model"], current) != value.get("reuse"):
            raise ApiError(409, "Existing prepared model data changed. Review preparation again.")
        if disk and shutil.disk_usage(self.root).free < value["diskNeeded"]: raise ApiError(409, "Free disk space on this server and review preparation again.")

    def reuse_plan(self, model, engine):
        cfg = engine["config"]
        cwd = cfg.get("cwd") or str(Path(engine["configPath"]).parent)
        try: native = located(cfg["args"][cfg["args"].index("--native") + 1], cwd)
        except (ValueError, OSError, IndexError): return None
        if native != Path(model["files"][0]["path"]).resolve(): return None
        arguments, tokenizer = reuse_arguments(cfg, engine["configPath"])
        paths = [tokenizer]
        for argument in arguments:
            paths.extend(argument.get("paths", []))
            paths.extend(vector["path"] for vector in argument.get("vectors", []))
        return {"arguments": arguments, "tokenizer": tokenizer, "dependencies": dependency_identities(paths)}

    def start_job(self, kind, plan_id):
        value = self.consume(kind, plan_id)
        if kind == "prepare": self.check_prepare(value)
        else: self.check_serve(value)
        with self.locked():
            state = self.load(); self.prune(state)
            if any(job["state"] in ACTIVE for job in state["jobs"].values()): raise ApiError(409, "Finish the current preparation or serving operation first.")
            job_id = str(uuid.uuid4())
            job = {"id": job_id, "kind": kind, "state": "queued", "requestId": plan_id, "summary": "Waiting to prepare model files…" if kind == "prepare" else "Starting the separate Strata API…",
                   "created": self.now(), "token": secrets.token_hex(32), "value": value, "destination": self.destination,
                   "sourcePath": str(self.source), "catalogSourcePath": str(self.manager.source) if self.manager.source else None, "namespace": self.namespace}
            state["jobs"][job_id] = job; self.save(state)
            try:
                child = self.spawn_worker(job)
                if child:
                    observed = files.process_identity(child.pid)
                    job.update(pid=child.pid, processIdentity=observed[0] if observed else None)
                self.save(state)
            except Exception:
                job.update(state="error", summary="The private model worker could not start."); self.save(state)
                raise ApiError(503, "The private model worker could not start.") from None
        return {"job": self.job_view(job)}

    def spawn_worker(self, job):
        log = self.root / ("job-" + job["id"] + ".log")
        fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "ab") as output:
            return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(self.manager.home), self.namespace, job["id"], job["token"]],
                                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)

    def profile(self, state, profile_id):
        row = state["profiles"].get(profile_id)
        if not row: raise ApiError(404, "Choose a prepared model profile on this server.")
        if hash_file(row["configPath"]) != row["configIdentity"] or hash_file(row["enginePath"]) != row["engineIdentity"]:
            raise ApiError(409, "This prepared profile or engine changed. Prepare the model again.")
        for item in row["inputs"]:
            if files.identity(Path(item["path"])) != item: raise ApiError(409, "Prepared model files changed. Prepare them again.")
        for item in row.get("libraries", []):
            if files.identity(Path(item["path"])) != item: raise ApiError(409, "The engine libraries changed. Prepare the model again.")
        if files.identity(Path(row.get("python", sys.executable)).resolve()) != row.get("pythonIdentity"):
            raise ApiError(409, "The prepared Python runtime changed. Prepare the model again.")
        return row

    def serve_plan(self, body):
        files.exact(body, ("profileId", "host", "port"), ("profileId",))
        with self.locked(): state = self.load()
        profile = self.profile(state, body["profileId"]); current = self.runtime_snapshot(state)
        if current["state"] not in ("stopped", "ready"): raise ApiError(409, "This managed Strata API is not ready. Review its status before serving.")
        existing = state["runtime"] or state.get("intent", {})
        host = body.get("host", existing.get("host", "0.0.0.0")); port = body.get("port", existing.get("port"))
        if host not in ("0.0.0.0", "127.0.0.1"): raise ApiError(400, "Choose local-only or network serving.")
        if current["state"] == "ready":
            if host != existing.get("host") or port != existing.get("port"): raise ApiError(409, "A running managed API keeps its current address and port.")
            if current.get("busy") is not False: raise ApiError(409, "Wait for requests on this managed API to finish before switching.")
        else:
            if port is None:
                with socket.socket() as listener: listener.bind((host, 0)); port = listener.getsockname()[1]
            if type(port) is not int or not 1024 <= port <= 65535: raise ApiError(400, "Choose a port between 1024 and 65535.")
            try:
                with socket.socket() as listener: listener.bind((host, port))
            except OSError: raise ApiError(409, "That port is in use. Choose a separate port; existing APIs are left running.") from None
        value = {"profileId": profile["id"], "profileIdentity": profile["configIdentity"], "host": host, "port": port,
                 "current": current, "sourceSignature": source_signature(self.source)}
        return {**self.remember("serve", value), "modelName": profile["name"], "profileId": profile["id"], "host": host, "port": port,
                "action": "switch" if current["state"] == "ready" else "start", "warnings": [
                    "Start or switch only this app's separate Strata API. Original Strata APIs and Maya stay running.",
                    "The selected model starts unloaded. Loading later requires enough free NVIDIA GPU memory; no other model is unloaded.",
                    "Switching refuses new or queued work on this managed API. If startup fails, Strata attempts to restore its previous prepared profile.",
                    "A private API key protects the separate listener. Connect it to the Mac app after serving succeeds.",
                    *(["The previous private API child has exited. Its exact owned control host will be retired before retrying; live processes are never stopped by this recovery."] if current.get("reapNeeded") else [])]}

    def check_serve(self, value):
        with self.locked(): state = self.load()
        profile = self.profile(state, value["profileId"])
        if profile["configIdentity"] != value["profileIdentity"] or source_signature(self.source) != value["sourceSignature"] or self.runtime_snapshot(state) != value["current"]:
            raise ApiError(409, "The prepared profile or managed API changed. Review serving again.")
        if value["current"]["state"] == "stopped":
            try:
                with socket.socket() as listener: listener.bind((value["host"], value["port"]))
            except OSError: raise ApiError(409, "That port became busy. Existing APIs were left running; review a new port.") from None

    def cancel(self, job_id):
        with self.locked():
            state = self.load(); job = state["jobs"].get(job_id)
            if not job or job["state"] not in ACTIVE or job["kind"] != "prepare": raise ApiError(409, "This preparation can no longer be stopped.")
            job["cancelRequested"] = True; self.save(state)
            return {"job": self.job_view(job)}

    def connect(self, body):
        files.exact(body, ("receiptId",), ("receiptId",))
        with self.locked(): state = self.load()
        receipt = state["receipts"].get(body["receiptId"])
        if not receipt or receipt["destination"] != self.destination:
            raise ApiError(409, "Serve this model before connecting its API.")
        self.profile(state, receipt["profileId"])
        observed = self.runtime_snapshot(state)
        if observed["state"] != "ready" or observed.get("instanceId") != receipt["instanceId"] or observed.get("profileId") != receipt["profileId"]:
            raise ApiError(409, "This serving receipt is stale. Refresh the managed API before connecting.")
        return {"endpoint": {key: receipt[key] for key in ("host", "port", "apiKey", "modelId", "receiptId")} | {"basePath": "/v1"}}

    def dispatch(self, owner, action, body=None):
        if owner != "desktop": raise ApiError(403, "Prepare and serve models from the Strata Mac app.")
        if action == "snapshot":
            if body not in (None, {}): raise ApiError(400, "Choose a supported model lifecycle action.")
            return self.snapshot()
        if action == "prepare-plan": return self.prepare_plan(body)
        if action == "serve-plan": return self.serve_plan(body)
        if action in ("prepare", "serve"):
            files.exact(body, ("planId",), ("planId",))
            return self.start_job(action, body["planId"])
        if action == "connect": return self.connect(body)
        match = re.fullmatch(r"jobs/([a-zA-Z0-9_-]{1,160})/cancel", action)
        if match:
            files.exact(body, ()); return self.cancel(match[1])
        raise ApiError(404, "Unknown model lifecycle action.")


def handle(envelope):
    try:
        files.exact(envelope, ("owner", "namespace", "destination", "sourcePath", "catalogSourcePath", "action", "method", "body", "requestId"),
                    ("owner", "namespace", "destination", "sourcePath", "action"))
        if envelope["owner"] != "desktop": raise ApiError(403, "Prepare and serve models from the Strata Mac app.")
        if envelope.get("method", "GET" if envelope["action"] == "snapshot" else "POST") != ("GET" if envelope["action"] == "snapshot" else "POST"):
            raise ApiError(400, "Choose a supported model lifecycle method.")
        lifecycle = Lifecycle(envelope["sourcePath"], envelope["namespace"], envelope["destination"], envelope.get("catalogSourcePath"))
        return {"status": 200, "body": lifecycle.dispatch(envelope["owner"], envelope["action"], envelope.get("body"))}
    except ApiError as error: return {"status": error.status, "body": {"error": error.message}}
    except Exception: return {"status": 400, "body": {"error": "The private model operation could not finish. Existing APIs were left unchanged; review its status and try again."}}


class Cancelled(Exception): pass


def worker(home, namespace, job_id, token):
    if not HEX.fullmatch(namespace) or not ID.fullmatch(job_id) or not HEX.fullmatch(token): return
    root = Path(home) / "lifecycle" / namespace
    try:
        initial = checked_json(root / "state.json")["jobs"].get(job_id)
        if not initial or initial["token"] != token or initial["state"] not in ACTIVE: return
        lifecycle = Lifecycle(initial["sourcePath"], namespace, initial["destination"], initial["catalogSourcePath"], home=home)
    except Exception: return

    def update(**changes):
        with lifecycle.locked():
            state = lifecycle.load(); job = state["jobs"].get(job_id)
            if not job or job["token"] != token or job["state"] not in ACTIVE: raise Cancelled()
            if job.get("cancelRequested"): raise Cancelled()
            job.update(heartbeat=time.time(), **changes); lifecycle.save(state)
            return copy.deepcopy(job)

    def finish(state_name, summary, **extra):
        with lifecycle.locked():
            state = lifecycle.load(); job = state["jobs"].get(job_id)
            if job and job["token"] == token and job["state"] in ACTIVE:
                job.update(state=state_name, summary=summary, heartbeat=time.time(), **extra); lifecycle.save(state)

    def cleanup_unpublished():
        # A cancelled/failed preparation must not consume another full model's
        # disk space on every retry. Never remove a published profile or anything
        # outside this exact job's privately created staging directory.
        if initial["kind"] != "prepare": return
        with lifecycle.locked():
            state = lifecycle.load()
            if "profile-" + job_id in state["profiles"]: return
            folder = lifecycle.root / "profiles" / job_id
            if folder.is_symlink(): folder.unlink(); return
            if folder.exists(): shutil.rmtree(folder)

    def reconcile_serving_failure():
        if initial["kind"] != "serve": return
        state = lifecycle.load()
        observed = lifecycle.runtime_snapshot(state)
        pending = state.get("intent", {})
        if (pending.get("jobId") == job_id and observed.get("state") == "ready" and
                observed.get("profileId") == state["runtime"].get("profileId") and
                pending.get("sourceHash") != state["runtime"].get("sourceHash")):
            with lifecycle.locked():
                latest = lifecycle.load()
                if latest.get("intent", {}).get("jobId") == job_id:
                    latest["intent"] = {}; lifecycle.save(latest)

    def copy_file(source, destination):
        source, destination = Path(source), Path(destination)
        before = files.identity(source)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            with os.fdopen(fd, "wb") as output, os.fdopen(os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as incoming:
                if files.file_identity(source, os.fstat(incoming.fileno())) != before: raise ApiError(409, "The model files changed while staging.")
                cloned = False
                if sys.platform.startswith("linux"):
                    try:
                        import fcntl
                        fcntl.ioctl(output.fileno(), 0x40049409, incoming.fileno()); cloned = True
                    except OSError: pass
                if not cloned:
                    for block in iter(lambda: incoming.read(4 * 1024 * 1024), b""):
                        output.write(block); update()
                output.flush(); os.fsync(output.fileno())
            if files.identity(source) != before: raise ApiError(409, "The model files changed while staging.")
        except BaseException:
            destination.unlink(missing_ok=True); raise

    def copy_tree(source, destination):
        source, destination = Path(source), Path(destination)
        if source.is_symlink(): raise ApiError(409, "Prepared dependencies must not be symbolic links.")
        if source.is_file(): copy_file(source, destination); return
        if not source.is_dir(): raise ApiError(409, "A required prepared dependency is missing.")
        files.private_directory(destination)
        count = 0
        for folder, directories, names in os.walk(source, followlinks=False):
            update()
            for name in directories:
                child = Path(folder) / name
                if child.is_symlink(): raise ApiError(409, "Prepared dependencies must not be symbolic links.")
                files.private_directory(destination / child.relative_to(source))
            for name in names:
                count += 1
                if count > 10000: raise ApiError(409, "The prepared dependency folder is too large to inspect safely.")
                child = Path(folder) / name
                copy_file(child, destination / child.relative_to(source))

    def subprocess_wait(command, *, env, cwd, timeout=7200):
        # The Popen handle is created here. Cancellation never signals a stored
        # PID or scans another model's process tree.
        descriptor = os.open(root / ("prepare-" + job_id + ".log"), os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     close_fds=True, start_new_session=True)
            deadline = time.monotonic() + timeout
            try:
                while child.poll() is None:
                    update()
                    if time.monotonic() >= deadline: raise ApiError(504, "Model preparation timed out. Its isolated files were kept for review.")
                    time.sleep(0.3)
                if child.returncode: raise ApiError(409, "Model preparation failed. Review the private server preparation log; existing APIs were left unchanged.")
            except BaseException:
                if child.poll() is None:
                    if os.name == "posix": os.killpg(child.pid, signal.SIGTERM)
                    else: child.terminate()
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix": os.killpg(child.pid, signal.SIGKILL)
                        else: child.kill()
                        child.wait(timeout=10)
                raise

    try:
        observed = files.process_identity(os.getpid())
        job = update(state="running", pid=os.getpid(), processIdentity=observed[0] if observed else None)
        value = job["value"]
        if job["kind"] == "prepare":
            lifecycle.check_prepare(value)
            folder = files.owned_directory(root, "profiles/" + job_id)
            inputs = files.owned_directory(folder, "inputs")
            update(summary="Staging private copies of the selected model files…", percent=5)
            for entry in value["model"]["files"]:
                copy_file(entry["path"], inputs / Path(entry["path"]).name)
            cfg = copy.deepcopy(value["engine"]["config"])
            args = cfg["args"]
            # Reusing an already prepared configuration avoids helper downloads.
            # All engine-read assets are copied/reflinked into the private tree.
            if value.get("reuse"):
                update(summary="Copying existing prepared data into the private profile…", percent=25)
                replacements = {str(Path(entry["path"]).resolve()): str(inputs / Path(entry["path"]).name) for entry in value["model"]["files"]}
                def private_dependency(raw):
                    if raw not in replacements:
                        original = Path(raw)
                        destination = folder / "dependencies" / (files.digest(raw)[:16] + "-" + original.name)
                        copy_tree(original, destination); replacements[raw] = str(destination)
                    return replacements[raw]
                args = []
                for argument in value["reuse"]["arguments"]:
                    if argument["flag"] == "--vision": continue  # new profiles deliberately prepare text-only
                    args.append(argument["flag"])
                    if "paths" in argument: args.extend(private_dependency(raw) for raw in argument["paths"])
                    elif "vectors" in argument: args.append(",".join(private_dependency(vector["path"]) + ":" + vector["scale"] for vector in argument["vectors"]))
                    else: args.extend(argument["values"])
                cfg["args"] = args
                tokenizer = folder / "tokenizer"
                copy_tree(value["reuse"]["tokenizer"], tokenizer); cfg["tokenizer"] = str(tokenizer)
                if "--max-context" in args: args[args.index("--max-context") + 1] = str(value["context"])
                else: args += ["--max-context", str(value["context"])]
                for flag in ("--rope-scaling", "--rope-scale"):
                    if flag in args:
                        index = args.index(flag); del args[index:index+2]
            else:
                update(summary="Preparing tokenizer and model data with pinned tools…", percent=25)
                install = files.owned_directory(folder, "install")
                environment = dict(os.environ)
                for key in list(environment):
                    if key.startswith("STRATA_"): del environment[key]
                environment.update(lifecycle.env(install, acquire=True))
                selection = value["model"]["selection"]
                lifecycle.check_dependencies(value["engine"]["python"])
                command = [value["engine"]["python"], str(lifecycle.source / "setup.py"), "--yes", "--no-start", "--no-browser", "--family", selection["family"],
                           "--model", selection["model"], "--gguf-dir", str(inputs), "--data-dir", str(folder / "data"), "--models-dir", str(inputs),
                           "--context", str(value["context"]), "--vision", "no", "--parallel", "1", "--host", "127.0.0.1"]
                cards = value["engine"]["gpuIndices"]
                command += ["--gpu", str(cards[0])] if len(cards) == 1 else ["--gpus", ",".join(map(str, cards))]
                subprocess_wait(command, env=environment, cwd=lifecycle.source)
                candidates = list(install.glob("strata-*.json"))
                if len(candidates) != 1: raise ApiError(409, "Preparation did not produce one complete model configuration.")
                cfg = checked_json(candidates[0], 1024 * 1024)
            update(summary="Checking and saving the isolated prepared profile…", percent=90)
            lifecycle.check_prepare(value, disk=False)
            engine = value["engine"]["config"]
            # Never carry a user's unload scripts, network services, image
            # generation, or external hooks into the private model profile.
            keep = ("args", "tokenizer", "model_name", "backend", "cuda", "gpu", "gpus_asked", "layer_split", "lib_dirs")
            cfg = {key: cfg[key] for key in keep if key in cfg}
            cfg.update(exe=engine["exe"], lib_dirs=value["engine"]["resolvedLibDirs"], cwd=str(lifecycle.source), lazy_load=True, open_browser=False,
                       host="127.0.0.1", port=0, parallel=1, log=str(folder / "engine.log"))
            cards = value["engine"]["gpuIndices"]
            cfg["gpu"] = cards[0] if len(cards) == 1 else cards
            guard = folder / "check-free-memory.py"
            guard.write_text(GUARD, encoding="utf-8"); guard.chmod(0o600)
            minimum = engine.get("min_free_vram_mib")
            minimum = max(8192, minimum) if type(minimum) is int and minimum > 0 else 8192
            cfg.update(before_load=[value["engine"]["python"], str(guard), json.dumps(cards), str(minimum)], before_load_strict=True,
                       before_load_ignore_failure=False, before_load_timeout_s=15)
            config = folder / "prepared.json"; files.atomic_json(config, cfg)
            identities = [files.identity(path) for path in folder.rglob("*") if path.is_file() and path not in (config,)]
            profile = {"id": "profile-" + job_id, "name": value["model"]["name"], "modelId": value["model"]["id"], "context": value["context"],
                       "state": "prepared", "executionCompatibility": "unverified", "configPath": str(config), "configIdentity": hash_file(config),
                       "enginePath": engine["exe"], "engineIdentity": value["engine"]["engineHash"], "inputs": identities, "sourceSignature": value["sourceSignature"],
                       "python": value["engine"]["python"], "pythonIdentity": value["engine"]["pythonIdentity"], "libraries": value["engine"]["libraries"]}
            update()
            with lifecycle.locked():
                state = lifecycle.load()
                if state["jobs"][job_id].get("cancelRequested"): raise Cancelled()
                state["profiles"][profile["id"]] = profile; lifecycle.save(state)
            finish("done", "Model prepared in its private server profile. Start its separate API when ready; loading remains a separate action.", profileId=profile["id"], percent=100)
        else:
            lifecycle.check_serve(value)
            with lifecycle.locked():
                state = lifecycle.load(); profile = lifecycle.profile(state, value["profileId"]); previous = copy.deepcopy(state["runtime"] or state.get("intent", {}))
            lifecycle.check_dependencies(profile["python"])
            if value["current"].get("reapNeeded"):
                with lifecycle.environment(), lifecycle.interpreter(profile["python"]):
                    _, _, runtime = lifecycle.source_modules()
                    if not hasattr(runtime, "reap_if_exited"): raise ApiError(503, "Update this Mac app before retrying the failed private API.")
                    reaped = runtime.reap_if_exited(lifecycle.runtime_name, expected_owner_id=value["current"]["ownerId"],
                                                   expected_source_sha256=value["current"]["sourceSha256"], timeout=10)
                    if not reaped.get("ok"): raise ApiError(409, "The private control host changed or still has a live child. Refresh its status; no live process was stopped.")
            cfg = checked_json(profile["configPath"])
            api_key = previous.get("apiKey") or secrets.token_urlsafe(36)
            cfg.update(host=value["host"], port=value["port"], api_key=api_key, lazy_load=True, open_browser=False)
            config_path = files.owned_directory(root, "runtime-configs") / (job_id + ".json")
            files.atomic_json(config_path, cfg)
            intent = {"jobId": job_id, "profileId": profile["id"], "modelId": cfg.get("model_name", profile["modelId"]),
                      "host": value["host"], "port": value["port"], "apiKey": api_key, "destination": lifecycle.destination,
                      "receiptId": str(uuid.uuid4()), "configPath": str(config_path), "sourceHash": hashlib.sha256(config_path.read_bytes()).hexdigest()}
            with lifecycle.locked():
                state = lifecycle.load(); state["intent"] = intent; lifecycle.save(state)
            update(summary="Starting the owned API with its model unloaded…")
            with lifecycle.environment(), lifecycle.interpreter(profile["python"]):
                _, profiles, runtime = lifecycle.source_modules()
                if not hasattr(runtime, "inspect") or not hasattr(runtime, "restart_if_idle"):
                    raise ApiError(503, "Update the Mac app's serving support before switching models.")
                if value["current"]["state"] == "stopped":
                    try:
                        saved, revision = profiles.load(lifecycle.runtime_name)
                        proposed, expected = profiles.propose_source(lifecycle.runtime_name, str(config_path))
                        saved, revision = profiles.save(lifecycle.runtime_name, proposed, expected_revision=expected)
                    except profiles.ProfileError as error:
                        if "is not saved" not in str(error): raise
                        saved, revision = profiles.create(lifecycle.runtime_name, str(config_path))
                    result = runtime.start(lifecycle.runtime_name, wait=120, expected_profile_revision=revision,
                                           expected_source_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
                else:
                    old_saved, old_revision = profiles.load(lifecycle.runtime_name)
                    proposed, revision = profiles.propose_source(lifecycle.runtime_name, str(config_path))
                    saved, revision = profiles.save(lifecycle.runtime_name, proposed, expected_revision=revision)
                    try:
                        result = runtime.restart_if_idle(lifecycle.runtime_name, timeout=360, startup_wait=120,
                            expected_profile_revision=revision, expected_source_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                            expected_owner_id=value["current"]["ownerId"], expected_instance_id=value["current"]["instanceId"], allow_source_switch=True)
                    except BaseException:
                        profiles.save(lifecycle.runtime_name, old_saved, expected_revision=revision); raise
                    if not result.get("ok"):
                        profiles.save(lifecycle.runtime_name, old_saved, expected_revision=revision)
                if not result.get("ok") or result.get("state") != "ready":
                    if result.get("rolled_back"):
                        raise ApiError(409, "The new model API could not start. The previous prepared profile was restored.")
                    raise ApiError(409, "The managed API could not start. Existing external APIs and Maya were left unchanged; inspect its status before retrying.")
                observed = runtime.inspect(lifecycle.runtime_name)
                if observed.get("state") != "ready" or not observed.get("instance_id") or not lifecycle.matches_owned_source(intent, observed):
                    raise ApiError(409, "The new API identity could not be verified. Refresh its serving status.")
            lifecycle.publish_runtime(lifecycle.load(), intent, observed)
            finish("done", "The separate Strata API is ready with its model unloaded. Connect it to the Mac app; load the model when resources are available.",
                   profileId=profile["id"], receiptId=intent["receiptId"])
    except Cancelled:
        cleanup_unpublished()
        finish("cancelled", "Preparation stopped. Temporary preparation copies were removed; original model files and existing APIs were left unchanged.")
    except ApiError as error:
        cleanup_unpublished(); reconcile_serving_failure(); finish("error", error.message)
    except Exception:
        cleanup_unpublished()
        reconcile_serving_failure()
        finish("error", "The private model operation failed. Temporary preparation copies were removed; existing external APIs were left unchanged. Review the server's private operation log.")


def main():
    if len(sys.argv) == 6 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:]); return
    raw = sys.stdin.buffer.read(16385)
    if len(raw) > 16384: result = {"status": 413, "body": {"error": "The model lifecycle request is too large."}}
    else:
        try: result = handle(json.loads(raw))
        except Exception: result = {"status": 400, "body": {"error": "Choose a supported model lifecycle action."}}
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__": main()
