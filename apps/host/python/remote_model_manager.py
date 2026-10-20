#!/usr/bin/env python3
"""Private model files over SSH. No installer, engine, runtime or profile actions.

One JSON envelope on stdin, one {status, body} response on stdout. The same
standalone file runs detached download workers; no package installation is needed.
"""
from __future__ import annotations

import contextlib
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid

MAX_JSON = 4 * 1024 * 1024
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,160}\Z")
SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf\Z", re.I)


class ApiError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


def exact(value, keys, required=()):
    if not isinstance(value, dict) or set(value) - set(keys) or set(required) - set(value):
        raise ApiError(400, "Choose a supported model action.")


def clean(value, maximum=300):
    return "".join(c if c.isprintable() else " " for c in str(value or ""))[:maximum]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def private_directory(path):
    path = Path(path).absolute()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode) or (hasattr(os, "getuid") and details.st_uid != os.getuid()):
        raise ApiError(403, "The private model folder is unavailable.")
    path.chmod(0o700)
    return path.resolve()


def owned_directory(root, relative):
    """Never follow an existing symlink anywhere inside our output tree."""
    current=private_directory(root)
    for part in Path(relative).parts:
        if part in ("", ".", "..") or Path(part).is_absolute(): raise ApiError(400,"The catalog file path is invalid.")
        current=private_directory(current/part)
    return current


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + "." + secrets.token_hex(8))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, separators=(",", ":")); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def default_home():
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "Strata/remote-models"
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/Strata/remote-models"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "strata/remote-models"


def strata_source(path, raw=None):
    """Reject an unrelated setuptools setup.py before importing any local code."""
    try:
        setup=path/"setup.py"
        if not (path/"strata.py").is_file() or not (path/"strata_control/acquisition.py").is_file() or setup.stat().st_size>8*1024*1024:
            return False
        tree=ast.parse(setup.read_bytes() if raw is None else raw,filename=str(setup))
        names=set()
        for node in tree.body:
            if isinstance(node,ast.Assign): names.update(target.id for target in node.targets if isinstance(target,ast.Name))
            elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name): names.add(node.target.id)
        return {"HF_REVISIONS","MODELS","FAMILIES","REPO_PINS"}<=names
    except (OSError,ValueError,SyntaxError,RecursionError): return False


def find_source(explicit=None):
    if explicit is not None:
        if not isinstance(explicit, (str, Path)) or not str(explicit) or len(str(explicit)) > 4096 or not Path(explicit).expanduser().is_absolute():
            raise ApiError(400, "Choose the full Strata source folder on this computer.")
        candidates = [Path(explicit).expanduser()]
    else:
        candidates = [Path.cwd(), *list(Path.cwd().parents)[:3]]
        for key in ("STRATA_SOURCE_PATH", "STRATA_INSTALL_DIR", "STRATA_HOME"):
            if os.environ.get(key): candidates.append(Path(os.environ[key]).expanduser())
        executable = shutil.which("strata")
        if executable:
            location = Path(executable).resolve()
            candidates.extend(list(location.parents)[:4])
        candidates += [Path.home() / name for name in ("Strata", ".local/share/strata/current", ".local/share/strata/terminal/current", ".local/share/strata/terminal/current/python", ".strata")]
        settings = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "strata/settings.json"
        try:
            if settings.stat().st_size <= 1024 * 1024:
                value = json.loads(settings.read_text())
                candidates += [Path(p) for p in value.get("installs", [])[:20] if isinstance(p, str)]
        except (OSError, ValueError, TypeError): pass
    for candidate in candidates:
        for location in (candidate,candidate/"python"):
            if strata_source(location): return location.resolve()
    if explicit is not None:
        raise ApiError(404, "This folder has no usable Strata model catalog.")
    return None


def catalog(source):
    """Read the locally installed source's pinned tables, never setup.main()."""
    if source is None: return {}
    setup = source / "setup.py"
    before = setup.read_bytes()
    if not strata_source(source,before): raise ApiError(503,"Choose a Strata source folder with a pinned model catalog.")
    spec = importlib.util.spec_from_file_location("strata_remote_catalog_" + secrets.token_hex(6), setup)
    module = importlib.util.module_from_spec(spec)
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        sys.modules[spec.name]=module
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(before,str(setup),"exec"),module.__dict__)
    except (Exception, SystemExit):
        raise ApiError(503, "The Strata source catalog could not be read. Update it or choose its source folder.") from None
    finally:
        sys.path[:], sys.dont_write_bytecode = old_path, old_bytecode
        sys.modules.pop(spec.name,None)
    if setup.read_bytes() != before:
        raise ApiError(409, "The Strata catalog changed. Review the model again.")
    source_signature = hashlib.sha256(before).hexdigest()
    models, families = getattr(module, "MODELS", {}), getattr(module, "FAMILIES", {})
    pins = getattr(module, "REPO_PINS", {})
    own = getattr(module, "OWN_PATH_FAMILIES", {})
    if any(not isinstance(value, dict) for value in (models, families, pins, own)):
        raise ApiError(503, "The Strata source has no usable pinned model catalog.")
    result = {}

    def file_entry(url, relative, pin, role="model"):
        parsed, relative = urllib.parse.urlsplit(url), str(relative)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or not re.search(r"/resolve/[a-f0-9]{40}/", parsed.path):
            raise ValueError("unversioned URL")
        if not isinstance(pin, (list, tuple)) or len(pin) != 2 or type(pin[0]) is not int or pin[0] <= 0 or not re.fullmatch(r"[a-fA-F0-9]{64}", str(pin[1])):
            raise ValueError("missing pin")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative or not path.parts:
            raise ValueError("unsafe catalog path")
        return {"url": url, "path": relative, "bytes": pin[0], "sha256": pin[1].lower(), "role": role}

    def url_pin(url):
        match = re.match(r"https://[^/]+/([^/]+/[^/]+)/resolve/[^/]+/(.+)", url)
        return pins.get(match[1], {}).get(match[2].rsplit("/", 1)[-1]) if match else None

    for family, family_row in {**families, **own}.items():
        sizes = list(family_row.get("models", {})) if family in own else [m for m, row in models.items() if family in row.get("families", ("qwen", "swift"))]
        for model in sizes:
            model_row = family_row["models"][model] if family in own else models[model]
            identifier = "native-" + (str(family_row.get("tag", "")) + model).lower()
            if not SAFE_ID.fullmatch(identifier): continue
            try:
                files, vision = [], []
                if family in own:
                    filename = family_row["file"].format(q=model)
                    files = [file_entry(family_row["hf"] + filename, filename, family_row["sha256"][filename])]
                elif model_row.get("repo"):
                    ready = getattr(module, "NVFP4_REPOS", {})[model_row["repo"]]
                    for filename, (role, size, sha) in ready["files"].items():
                        url = "https://huggingface.co/" + model_row["repo"] + "/resolve/" + ready["revision"] + "/" + filename
                        files.append(file_entry(url, filename, (size, sha), role))
                else:
                    count = module.model_shards(family_row, model) if hasattr(module, "model_shards") else family_row.get("shards", 2)
                    if type(count) is not int or not 1 <= count <= 256: raise ValueError("shards")
                    for index in range(1, count + 1):
                        filename = module.model_file(family_row, model, index) if hasattr(module, "model_file") else family_row["file"].format(q=model, i=index)
                        url = family_row["hf"].format(q=model) + filename
                        files.append(file_entry(url, filename, family_row.get("sha256", {}).get(filename) or url_pin(url)))
                if model_row.get("vision", family_row.get("vision", True)) is not False and family_row.get("mmproj"):
                    try:
                        filename = family_row["mmproj"]
                        url = family_row["mmproj_hf"] + filename
                        pin = family_row.get("mmproj_sha256") or family_row.get("sha256", {}).get(filename) or url_pin(url)
                        vision = [file_entry(url, filename, pin, "vision")]
                    except (KeyError, ValueError, TypeError): pass
                if len({f["path"] for f in files + vision}) != len(files + vision): raise ValueError("duplicate paths")
                result[identifier] = {"id": identifier, "name": clean(family_row.get("title", family) + " " + model), "detail": clean(model_row.get("about", "Catalog model")), "family": family, "model": model,
                                      "source": str(source), "sourceSignature": source_signature, "files": files, "visionFiles": vision}
            except (KeyError, ValueError, TypeError, AttributeError):
                continue  # An unknown hash/size is never an unverified download.
    return result


def file_identity(path, details):
    if not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
        raise ApiError(400, "Choose regular GGUF files, without symbolic links.")
    return {"path": str(path.resolve()), "device": details.st_dev, "inode": details.st_ino, "bytes": details.st_size, "mtime": details.st_mtime_ns, "ctime": details.st_ctime_ns}


def identity(path):
    return file_identity(path, path.lstat())


def inspect_gguf(path):
    """Bounded GGUF v2/v3 header and complete tensor extents; no payload loading."""
    before = identity(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        if file_identity(path, os.fstat(stream.fileno())) != before:
            raise ApiError(409, "The GGUF changed during inspection. Inspect it again.")
        scanned, header = 0, hashlib.sha256()
        def take(count):
            nonlocal scanned
            if count < 0 or scanned + count > min(before["bytes"], 64 * 1024 * 1024): raise ApiError(400, "GGUF metadata exceeds the inspection limit.")
            value = stream.read(count)
            if len(value) != count: raise ApiError(400, "GGUF metadata is incomplete.")
            scanned += count; header.update(value); return value
        def integer(size=8): return int.from_bytes(take(size), "little")
        def string(capture=False):
            length = integer()
            if length > 1024 * 1024: raise ApiError(400, "GGUF text exceeds the inspection limit.")
            raw = take(length)
            if capture:
                try: return raw.decode("utf-8")
                except UnicodeError: raise ApiError(400, "GGUF metadata is invalid.") from None
            return None
        scalar = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        def value(kind, capture=False):
            if kind == 8: return string(capture)
            if kind == 9:
                element, count = integer(4), integer()
                if count > 500_000 or element == 9: raise ApiError(400, "GGUF arrays exceed the inspection limit.")
                for _ in range(count): value(element)
                return count
            if kind not in scalar: raise ApiError(400, "GGUF metadata has an unsupported encoding.")
            raw = take(scalar[kind]); return int.from_bytes(raw,"little") if capture else None
        if take(4) != b"GGUF" or integer(4) not in (2,3): raise ApiError(400, "Choose a GGUF version 2 or 3 model.")
        tensors, keys = integer(), integer()
        if not 1 <= tensors <= 100_000 or keys > 4096: raise ApiError(400, "GGUF counts exceed the inspection limit.")
        metadata = {}
        for _ in range(keys):
            key = string(True)
            if len(key) > 256 or key in metadata: raise ApiError(400, "GGUF metadata keys are invalid.")
            kind = integer(4)
            capture = key in ("general.architecture", "general.name", "general.type", "general.alignment", "split.count", "split.no", "split.tensors.count", "tokenizer.ggml.model")
            metadata[key] = value(kind, capture)
        if not isinstance(metadata.get("general.architecture"),str) or not metadata["general.architecture"] or metadata.get("general.type") not in (None,"model"): raise ApiError(400, "Choose a complete model GGUF with architecture metadata.")
        if path.name.lower().startswith("mmproj"): raise ApiError(400, "Choose the model GGUF, rather than its picture reader.")
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or alignment < 1 or alignment > 4096 or alignment & (alignment-1): raise ApiError(400, "GGUF alignment is invalid.")
        # Common GGML encodings: block elements and bytes. Unknown encodings fail closed.
        geometry = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,36),10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292),16:(256,66),17:(256,74),18:(256,98),19:(256,50),20:(32,18),21:(256,110),22:(256,82),23:(256,136),24:(1,1),25:(1,2),26:(1,4),27:(1,8),28:(1,8),29:(256,56),30:(1,2),39:(32,17),40:(64,36),42:(64,18),143:(128,28),144:(128,34),145:(128,26),146:(128,30)}
        extents = []
        for _ in range(tensors):
            string(); dimensions = integer(4)
            if not 1 <= dimensions <= 8: raise ApiError(400, "GGUF tensor dimensions are invalid.")
            elements = 1
            for _ in range(dimensions):
                n = integer()
                if not 1 <= n <= 2**32: raise ApiError(400, "GGUF tensor shape is invalid.")
                elements *= n
                if elements > 2**63: raise ApiError(400, "GGUF tensor shape is too large.")
            encoding, offset = integer(4), integer()
            if encoding not in geometry or elements % geometry[encoding][0] or offset % alignment: raise ApiError(400, "GGUF tensor encoding or offset is unsupported.")
            length = elements // geometry[encoding][0] * geometry[encoding][1]
            extents.append((offset,offset+length))
        data_start = (scanned + alignment - 1) // alignment * alignment
        extents.sort()
        if data_start + max(end for _,end in extents) > before["bytes"] or any(start < end for (_,end),(start,_) in zip(extents,extents[1:])):
            raise ApiError(400, "GGUF tensors are incomplete or overlap.")
        if identity(path) != before or file_identity(path, os.fstat(stream.fileno())) != before: raise ApiError(409, "The GGUF changed during inspection. Inspect it again.")
        return {**before, "headerSha256": header.hexdigest(), "architecture": metadata["general.architecture"], "name": clean(metadata.get("general.name") or path.stem), "splitCount": metadata.get("split.count",1), "splitNo": metadata.get("split.no",0), "tensorCount": tensors, "splitTensors": metadata.get("split.tensors.count")}


def inspect_files(raw):
    if not isinstance(raw,str) or len(raw) > 4096 or "\0" in raw or not Path(raw).expanduser().is_absolute(): raise ApiError(400,"Choose a full GGUF file path on this computer.")
    path = Path(raw).expanduser()
    if path.is_symlink(): raise ApiError(400,"Choose regular GGUF files, without symbolic links.")
    if path.suffix.lower() != ".gguf": raise ApiError(400,"Choose a GGUF model file.")
    path = path.resolve(strict=True)
    match = SHARD.fullmatch(path.name)
    count = int(match[3]) if match else 1
    if not 1 <= count <= 128: raise ApiError(400,"The model has too many GGUF shards.")
    paths = [path.with_name(f"{match[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(1,count+1)] if match else [path]
    files = [inspect_gguf(p) for p in paths]
    if any(f["splitCount"] != count or (count > 1 and f["splitNo"] != i) or f["architecture"] != files[0]["architecture"] for i,f in enumerate(files)):
        raise ApiError(400,"Keep a matching, complete set of GGUF shards together.")
    if count > 1 and (files[0]["splitTensors"] is None or sum(f["tensorCount"] for f in files) != files[0]["splitTensors"] or any(f["splitTensors"] != files[0]["splitTensors"] for f in files)):
        raise ApiError(400,"The GGUF shard tensor counts do not match.")
    if any(identity(p) != {k:v for k,v in f.items() if k in ("path","device","inode","bytes","mtime","ctime")} for p,f in zip(paths,files)):
        raise ApiError(409,"The GGUF files changed. Inspect them again.")
    return {"id":"local-"+digest(str(paths[0]))[:24],"name":files[0]["name"],"files":files,"bytes":sum(f["bytes"] for f in files),"path":str(paths[0])}


def process_identity(pid):
    try:
        if sys.platform.startswith("linux"):
            raw = Path(f"/proc/{pid}/stat").read_text(); fields = raw[raw.rfind(")")+2:].split()
            if fields[0] == "Z": return None
            command = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\0"," ")
            return fields[19], command
        output = subprocess.run(["ps","-ww","-p",str(pid),"-o","lstart=","-o","command="],capture_output=True,text=True,timeout=3,check=False).stdout.strip()
        return (output[:24],output[24:]) if output else None
    except (OSError, subprocess.SubprocessError, IndexError): return None


class Manager:
    def __init__(self, home=None, source=None, now=time.time):
        self.home = private_directory(home if home is not None else default_home())
        self.source_error = None
        try: self.source = find_source(source)
        except ApiError as error:
            # Saved jobs depend on their pinned manifest, not on the checkout
            # still existing after a Terminal update. Stop remains available.
            self.source, self.source_error = None, error
        self.now = now
        self.state_file = self.home / "state.json"
        self.assets = private_directory(self.home / "downloads")

    def current_catalog(self):
        if self.source_error: raise self.source_error
        try: return catalog(self.source)
        except ApiError: raise
        except (OSError,ValueError,TypeError,KeyError,AttributeError):
            raise ApiError(503,"The Strata source catalog is unavailable. Choose its current source folder.") from None

    @contextlib.contextmanager
    def locked(self):
        lock = self.home / "lock"
        fd = os.open(lock,os.O_RDWR|os.O_CREAT|getattr(os,"O_NOFOLLOW",0),0o600)
        try:
            if os.name == "nt":
                import msvcrt
                if os.fstat(fd).st_size == 0: os.write(fd,b"0")
                os.lseek(fd,0,0); msvcrt.locking(fd,msvcrt.LK_LOCK,1)
            else:
                import fcntl
                fcntl.flock(fd,fcntl.LOCK_EX)
            yield
        finally: os.close(fd)

    def load(self):
        if not self.state_file.exists(): return {"plans":{},"jobs":{},"imports":{}}
        details = self.state_file.lstat()
        if not stat.S_ISREG(details.st_mode) or details.st_size > MAX_JSON or (hasattr(os,"getuid") and details.st_uid != os.getuid()): raise ApiError(500,"The private model records need repair.")
        value = json.loads(self.state_file.read_text())
        if not isinstance(value,dict) or any(not isinstance(value.get(k),dict) for k in ("plans","jobs","imports")): raise ApiError(500,"The private model records need repair.")
        return value

    def save(self,state):
        if len(json.dumps(state,separators=(",", ":")).encode()) > MAX_JSON:
            raise ApiError(429,"The private model catalog is full. Remove old references on this computer first.")
        atomic_json(self.state_file,state)

    def prune(self,state):
        state["plans"] = {k:v for k,v in state["plans"].items() if v["expires"] > self.now()}
        for job in state["jobs"].values():
            if job["state"] in ("running","queued") and self.now()-job.get("heartbeat",job["created"])>30 and not self.worker_alive(job):
                job["state"] = "cancelled" if job.get("cancelRequested") else "error"
        if len(state["jobs"]) > 100:
            for key in list(state["jobs"]):
                if len(state["jobs"]) <= 100: break
                if state["jobs"][key]["state"] not in ("running","queued"): del state["jobs"][key]

    def worker_alive(self,job):
        observed = process_identity(job.get("pid",0)) if job.get("pid") else None
        return bool(observed and observed[0] == job.get("identity") and job.get("workerToken", "missing") in observed[1] and "--worker" in observed[1] and job["id"] in observed[1])

    def job_view(self,job,owner):
        state = job["state"]
        return {"id":job["id"],"modelId":job["modelId"],"name":job["name"],"kind":"download","state":state,"requestId":job["requestId"],
                "canCancel":owner == "desktop" or owner == job["owner"],"percent":min(100,round(100*job.get("doneBytes",0)/max(1,job["totalBytes"]),1)),"doneGb":job.get("doneBytes",0)/1e9,"totalGb":job["totalBytes"]/1e9,
                "summary":"Model files are ready. Prepare and load them on this computer to use them; engine compatibility is unverified." if state == "done" else "Stopped. Completed files are kept." if state == "cancelled" else "Download could not finish. Review and retry; completed files are kept." if state == "error" else "Stopping the download…" if job.get("cancelRequested") else "Downloading and verifying model files…"}

    def remaining(self,manifest):
        folder=self.assets/digest(manifest["files"])
        remaining=0
        for entry in manifest["files"]:
            final=folder/entry["path"]; partial=final.with_name(final.name+".part")
            present=next((p.stat().st_size for p in (final,partial) if p.is_file() and not p.is_symlink()),0)
            remaining+=max(0,entry["bytes"]-min(present,entry["bytes"]))
        return remaining+64*1024*1024

    def row_ready(self,job,row):
        """Downloaded describes these current pinned bytes, never merely a reused model ID."""
        manifest=job["manifest"]
        if job["state"]!="done" or manifest.get("source")!=row["source"] or manifest.get("sourceSignature")!=row["sourceSignature"] or manifest.get("files",[])[:len(row["files"])]!=row["files"]:
            return False
        verified=job.get("verifiedFiles",{})
        try:
            for entry in row["files"]:
                path=self.assets/job["assetKey"]/entry["path"]
                if identity(path)!=verified.get(entry["path"]): return False
        except (OSError,ApiError): return False
        return True

    def remember(self,state,owner,kind,value):
        self.prune(state)
        if len(state["plans"]) >= 100: raise ApiError(429,"Close an old preview and try again shortly.")
        plan_id = str(uuid.uuid4()); expires = self.now()+300
        state["plans"][plan_id] = {"owner":owner,"kind":kind,"value":value,"expires":expires}
        return {"planId":plan_id,"expiresAt":int(expires*1000)}

    def consume(self,state,owner,plan_id,kind):
        if not isinstance(plan_id,str) or not SAFE_ID.fullmatch(plan_id): raise ApiError(400,"Review the model again.")
        plan = state["plans"].get(plan_id)
        if not plan or plan["owner"] != owner or plan["kind"] != kind or plan["expires"] <= self.now(): raise ApiError(409,"This preview expired or was already used. Review it again.")
        del state["plans"][plan_id]; self.save(state)
        return plan["value"]

    def spawn_worker(self,job):
        command = [sys.executable,str(Path(__file__).resolve()),"--worker",job["id"],"--home",str(self.home),"--token",job["workerToken"]]
        environment = dict(os.environ); environment.pop("STRATA_NO_ACQUIRE",None)
        options = {"creationflags":subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session":True}
        return subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=environment,close_fds=True,**options)

    def dispatch(self,owner,action,method="GET",body=None,request_id=None):
        if not isinstance(owner,str) or not 1 <= len(owner) <= 300 or any(ord(c)<32 for c in owner): raise ApiError(400,"A valid management owner is required.")
        if request_id is not None and (not isinstance(request_id,str) or not SAFE_ID.fullmatch(request_id)): raise ApiError(400,"The model request receipt is invalid.")
        body = {} if body is None else body
        target_name = clean(socket.gethostname()) or "Strata computer"
        if action == "" and method == "GET":
            exact(body,())
            reason=None
            try: rows=self.current_catalog()
            except ApiError as error: rows={}; reason=error.message
            with self.locked():
                state = self.load(); self.prune(state); self.save(state)
                ready = {j["modelId"] for j in state["jobs"].values() if j["modelId"] in rows and self.row_ready(j,rows[j["modelId"]])}
                return {"supported":True,"targetName":target_name,"storageMode":"files","downloadDirectory":str(self.assets),"capabilities":{"downloads":bool(rows),"imports":True},"reason":None if rows else reason or "Choose a Strata source folder with a catalog of pinned model files.",
                        "models":[{"id":r["id"],"name":r["name"],"detail":r["detail"],"state":"downloaded" if r["id"] in ready else "available","sizeGb":sum(f["bytes"] for f in r["files"])/1e9,"readPictures":bool(r["visionFiles"])} for r in rows.values()] + [{"id":r["id"],"name":r["name"],"sizeGb":r["bytes"]/1e9,"state":"imported","readPictures":False,"detail":"Existing GGUF reference. Engine compatibility is unverified."} for r in state["imports"].values()],"jobs":[self.job_view(j,owner) for j in state["jobs"].values()]}
        if method != "POST": raise ApiError(404,"Unknown model management action.")
        if action == "/plan":
            exact(body,("modelId","readPictures"),("modelId","readPictures"))
            if not isinstance(body["modelId"],str) or not SAFE_ID.fullmatch(body["modelId"]): raise ApiError(400,"Choose a catalog model.")
            if type(body["readPictures"]) is not bool: raise ApiError(400,"Choose the picture-reading option.")
            row = self.current_catalog().get(body["modelId"])
            if not row: raise ApiError(404,"Choose a fully pinned catalog model.")
            if body["readPictures"] and not row["visionFiles"]: raise ApiError(400,"Pinned picture-reading files are unavailable for this model.")
            manifest = {**row,"files":row["files"]+(row["visionFiles"] if body["readPictures"] else []),"readPictures":body["readPictures"]}
            total = sum(f["bytes"] for f in manifest["files"]); free = shutil.disk_usage(self.home).free; needed=self.remaining(manifest)
            with self.locked():
                state=self.load(); receipt=self.remember(state,owner,"download",manifest); self.save(state)
            return {**receipt,"targetName":target_name,"storageMode":"files","modelName":row["name"],"downloadGb":total/1e9,"diskFreeGb":free/1e9,"diskNeededGb":needed/1e9,"diskShort":"Free space on this computer and review again." if free < needed else None,"readPictures":body["readPictures"],"warnings":["Only model files are downloaded to this computer. The running model, engine and API settings stay unchanged.","Preparation and loading are separate steps on this computer. Engine compatibility is unverified."]}
        if action == "/downloads":
            exact(body,("planId",),("planId",))
            if request_id is not None and request_id != body["planId"]: raise ApiError(400,"The download receipt must match its preview.")
            with self.locked():
                state=self.load(); manifest=self.consume(state,owner,body["planId"],"download"); self.prune(state)
                if any(j["state"] in ("running","queued") for j in state["jobs"].values()): raise ApiError(409,"Finish or stop the current download first.")
                row=self.current_catalog().get(manifest["id"])
                current={**row,"files":row["files"]+(row["visionFiles"] if manifest["readPictures"] else []),"readPictures":manifest["readPictures"]} if row else None
                if digest(current) != digest(manifest): raise ApiError(409,"The catalog changed. Review the model again.")
                total=sum(f["bytes"] for f in manifest["files"])
                if shutil.disk_usage(self.home).free < self.remaining(manifest): raise ApiError(409,"The model no longer fits on disk. Free space and review again.")
                job_id=str(uuid.uuid4()); job={"id":job_id,"owner":owner,"requestId":body["planId"],"modelId":manifest["id"],"name":manifest["name"],"manifest":manifest,"assetKey":digest(manifest["files"]),"state":"queued","totalBytes":total,"doneBytes":0,"workerToken":secrets.token_hex(32),"created":self.now()}
                state["jobs"][job_id]=job; self.save(state)
                try:
                    child=self.spawn_worker(job)
                    if child is not None:
                        job["pid"]=child.pid
                        observed=process_identity(child.pid)
                        job["identity"]=observed[0] if observed else None
                    self.save(state)
                except Exception:
                    job["state"]="error"; self.save(state); raise ApiError(503,"The model download worker could not start.") from None
                return {"job":self.job_view(job,owner)}
        cancel=re.fullmatch(r"/downloads/([A-Za-z0-9_-]{1,160})/cancel",action)
        if cancel:
            exact(body,())
            with self.locked():
                state=self.load(); job=state["jobs"].get(cancel[1])
                if not job or job["state"] not in ("running","queued"): raise ApiError(409,"This download is no longer running.")
                if owner != "desktop" and owner != job["owner"]: raise ApiError(403,"Stop this download from the device that started it.")
                job["cancelRequested"]=True; self.save(state)
                # The worker observes its private flag and ends itself. No stored PID is ever signalled.
                if not job.get("pid"):
                    job["state"]="cancelled"; self.save(state)
            return {"job":self.job_view(job,owner)}
        if action == "/imports/browse":
            exact(body,("directory",))
            raw=body.get("directory",str(self.assets))
            if not isinstance(raw,str) or len(raw)>4096 or not Path(raw).expanduser().is_absolute(): raise ApiError(400,"Choose a full folder path on this computer.")
            path=Path(raw).expanduser().resolve(strict=True)
            if not path.is_dir(): raise ApiError(400,"Choose a folder.")
            entries=[]
            with os.scandir(path) as listing:
                for entry in listing:
                    if entry.name.startswith(".") or entry.is_symlink(): continue
                    if entry.is_dir(follow_symlinks=False): kind="folder"
                    elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".gguf"): kind="model"
                    else: continue
                    entries.append({"name":clean(entry.name),"path":str(path/entry.name),"kind":kind,"bytes":entry.stat(follow_symlinks=False).st_size if kind=="model" else None})
                    if len(entries)>=1000: break
            return {"directory":str(path),"parent":str(path.parent) if path.parent!=path else None,"entries":sorted(entries,key=lambda e:(e["kind"]!="folder",e["name"].lower()))}
        if action == "/imports/inspect":
            exact(body,("path",),("path",)); preview=inspect_files(body["path"])
            with self.locked():
                state=self.load(); receipt=self.remember(state,owner,"import",preview); self.save(state)
            return {**receipt,"targetName":target_name,"modelName":preview["name"],"sizeGb":preview["bytes"]/1e9,"files":[Path(f["path"]).name for f in preview["files"]],"warnings":["Importing stores a reference to these existing files; it does not copy, prepare or load them.","GGUF metadata and tensor extents were inspected. File provenance and engine compatibility remain unverified."]}
        if action == "/imports":
            exact(body,("planId",),("planId",))
            with self.locked():
                state=self.load(); saved=self.consume(state,owner,body["planId"],"import"); preview=inspect_files(saved["path"])
                if preview!=saved: raise ApiError(409,"The model files changed. Inspect them again.")
                if len(state["imports"])>=500 and preview["id"] not in state["imports"]: raise ApiError(429,"The reference catalog is full.")
                state["imports"][preview["id"]]=preview; self.save(state)
            return {"model":{"id":preview["id"],"name":preview["name"]},"storageMode":"files","message":"GGUF reference saved. Preparation and loading remain separate on this server."}
        raise ApiError(404,"Unknown model management action.")


def worker_exit(code):
    os._exit(code)


def worker(home,job_id,token):
    manager=Manager(home=home,source=None)
    cancelled=threading.Event()
    finished=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:cancelled.set())
    def check():
        if cancelled.is_set(): raise InterruptedError("cancelled")
    def update(**patch):
        with manager.locked():
            state=manager.load(); job=state["jobs"].get(job_id)
            if not job or job.get("workerToken")!=token: raise InterruptedError("cancelled")
            if job.get("cancelRequested"): cancelled.set()
            check(); job.update(patch); job["heartbeat"]=time.time(); manager.save(state)
    def hash_file(path):
        value=hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                check(); chunk=stream.read(1024*1024)
                if not chunk: break
                value.update(chunk)
        return value.hexdigest()
    def watch_cancel():
        heartbeat=0
        while not finished.wait(0.2):
            try:
                with manager.locked():
                    state=manager.load(); current=state["jobs"].get(job_id)
                    if not current or current.get("workerToken")!=token or current.get("cancelRequested"):
                        cancelled.set(); break
                    if time.monotonic()-heartbeat>2:
                        current["heartbeat"]=time.time(); manager.save(state); heartbeat=time.monotonic()
            except Exception:
                cancelled.set(); break
        if cancelled.is_set() and not finished.wait(2):
            # Slow-drip HTTP can keep a read alive forever. Only this owned worker
            # exits, under its lease, so another retry cannot overlap the old writer.
            with manager.locked():
                state=manager.load(); current=state["jobs"].get(job_id)
                if current and current.get("workerToken")==token:
                    current["state"]="cancelled"; manager.save(state)
                worker_exit(130)
    try:
        with manager.locked():
            state=manager.load(); job=state["jobs"].get(job_id)
            if not job or job.get("workerToken")!=token or job["state"] not in ("queued","running"): return
            if job.get("cancelRequested"): raise InterruptedError("cancelled")
            observed=process_identity(os.getpid()); job.update(pid=os.getpid(),identity=observed[0] if observed else None,state="running",heartbeat=time.time()); manager.save(state)
            manifest=job["manifest"]
        threading.Thread(target=watch_cancel,daemon=True).start()
        folder=owned_directory(manager.assets,job["assetKey"]); done=0; last=0; verified={}
        for entry in manifest["files"]:
            check(); destination=folder/entry["path"]
            if destination.parent != folder: owned_directory(folder,Path(entry["path"]).parent)
            if destination.exists():
                if destination.is_symlink() or destination.stat().st_size!=entry["bytes"] or hash_file(destination)!=entry["sha256"]: raise ValueError("existing file changed")
                done+=entry["bytes"]; verified[entry["path"]]=identity(destination); update(doneBytes=done,verifiedFiles=verified); continue
            partial=destination.with_name(destination.name+".part")
            if partial.is_symlink(): raise ValueError("partial symlink")
            have=partial.stat().st_size if partial.exists() else 0
            if have>entry["bytes"]: raise ValueError("partial oversized")
            if have<entry["bytes"]:
                request=urllib.request.Request(entry["url"],headers={"User-Agent":"strata-remote-models/1","Range":f"bytes={have}-"} if have else {"User-Agent":"strata-remote-models/1"})
                with urllib.request.urlopen(request,timeout=10) as response:
                    if urllib.parse.urlsplit(response.url).scheme!="https": raise ValueError("insecure redirect")
                    if have and response.status!=206: have=0
                    if response.status==206 and not re.fullmatch(rf"bytes {have}-\d+/{entry['bytes']}",response.headers.get("Content-Range","")): raise ValueError("invalid range")
                    flags=os.O_WRONLY|os.O_CREAT|getattr(os,"O_NOFOLLOW",0)|(os.O_APPEND if have else os.O_TRUNC)
                    fd=os.open(partial,flags,0o600)
                    with os.fdopen(fd,"ab" if have else "wb") as output:
                        while True:
                            check(); block=response.read(1024*1024)
                            if not block: break
                            if have+len(block)>entry["bytes"]: raise ValueError("oversized response")
                            output.write(block); have+=len(block)
                            if time.monotonic()-last>=0.5: update(doneBytes=done+have); last=time.monotonic()
                        output.flush(); os.fsync(output.fileno())
            check()
            if partial.stat().st_size!=entry["bytes"]: raise ValueError("incomplete file")
            if hash_file(partial)!=entry["sha256"]:
                partial.unlink()  # Corrupt complete bytes cannot be resumed; verified files remain untouched.
                raise ValueError("file hash mismatch")
            check()
            # The private content folder is never an active model folder. Never overwrite even here.
            os.link(partial,destination); partial.unlink(); done+=entry["bytes"]; verified[entry["path"]]=identity(destination); update(doneBytes=done,verifiedFiles=verified)
        update(state="done",doneBytes=done)
    except BaseException as exc:
        with manager.locked():
            state=manager.load(); job=state["jobs"].get(job_id)
            if job and job.get("workerToken")==token:
                job["state"]="cancelled" if isinstance(exc,InterruptedError) or cancelled.is_set() or job.get("cancelRequested") else "error"
                manager.save(state)
    finally: finished.set()


def handle(envelope):
    try:
        exact(envelope,("owner","sourcePath","action","method","body","requestId"),("owner","action","method"))
        if not isinstance(envelope["action"],str) or not isinstance(envelope["method"],str): raise ApiError(400,"Choose a supported model action.")
        manager=Manager(source=envelope.get("sourcePath"))
        result=manager.dispatch(envelope["owner"],envelope["action"],envelope["method"],envelope.get("body"),envelope.get("requestId"))
        return {"status":200,"body":result}
    except ApiError as exc: return {"status":exc.status,"body":{"error":exc.message}}
    except Exception: return {"status":400,"body":{"error":"This model action could not finish. Check the source folder and selected files, then try again."}}


def main():
    if len(sys.argv)==7 and sys.argv[1]=="--worker" and sys.argv[3]=="--home" and sys.argv[5]=="--token":
        if SAFE_ID.fullmatch(sys.argv[2]) and re.fullmatch(r"[a-f0-9]{64}",sys.argv[6]): worker(Path(sys.argv[4]),sys.argv[2],sys.argv[6])
        return
    if len(sys.argv)!=1: return
    try:
        raw=sys.stdin.buffer.read(16*1024+1)
        result=handle(json.loads(raw)) if len(raw)<=16*1024 else {"status":413,"body":{"error":"The model request is too large."}}
    except (ValueError,UnicodeError): result={"status":400,"body":{"error":"Send one JSON model request."}}
    print(json.dumps(result,separators=(",",":")))


if __name__=="__main__": main()
