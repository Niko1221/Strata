"""Serial native-session bridge for the experimental branch/checkpoint policy."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

from serve.branch_store import BranchStore, digest
from serve.checkpoint_store import CheckpointStore
from serve.session_metadata import session_prefixes


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def execution_identity(svc, cfg):
    """Hash actual artifacts, including every GGUF shard. No model-name aliases."""
    spawn = getattr(svc.engine, "spawn", None)
    if spawn is None:
        return {"mock": type(svc.engine).__name__, "template": svc.template.source}
    exe, args, cwd, _, _ = spawn
    root = Path(cwd or os.getcwd())
    paths = [root / exe]
    native = Path(svc.engine.model_path)
    if not native.is_absolute():
        native = root / native
    if native.is_file():
        pattern = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "-*-of-*.gguf", native.name)
        paths += sorted(native.parent.glob(pattern))
    else:
        paths += sorted(p for p in native.rglob("*") if p.is_file())
    # All externally supplied engine artifacts (MTP, embedding, control vector, etc.).
    for index, arg in enumerate(args):
        p = root / arg
        if not arg.startswith("--") and p.is_file():
            paths.append(p)
        elif index and args[index-1] in ("--mtp", "--pack") and p.is_dir():
            paths += sorted(x for x in p.rglob("*") if x.is_file())
    execution_args = []
    ignored = {"--session-min-free-mib", "--conversation-cache-mib", "--conversation-cache-disk-mib",
               "--conversation-cache-dir", "--conversation-cache-min-free-mib"}
    i = 0
    while i < len(args):
        if args[i] in ignored:
            i += 2
        else:
            execution_args.append(args[i])
            i += 1
    identity = {"artifacts": sorted({file_hash(p) for p in set(paths)}), "engine_args": execution_args,
                "template": svc.template.source,
                "frontend": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")
                             if not p.name.startswith("test_")},
                "environment": {k:v for k,v in {**os.environ, **(spawn[4] or {})}.items() if k.startswith("STRATA_")}}
    # Tokenizer assets are often beside the template rather than engine args.
    token_dir = cfg.get("tokenizer") or cfg.get("tokenizer_dir")
    if token_dir:
        identity["tokenizer"] = {p.name: file_hash(p) for p in (root / token_dir).iterdir() if p.is_file()}
    return identity


class CheckpointRuntime:
    def __init__(self, catalog, fingerprint, max_snapshot_bytes):
        self.catalog, self.fingerprint = catalog, fingerprint
        self.max_snapshot_bytes = max_snapshot_bytes
        self.seconds_per_token = .01  # conservative cold prior, replaced by uncached measurements
        self.live = []
        self.events = []
        self.current_fingerprint = fingerprint
        self.engine_last = None

    def before(self, engine, ids, sampling=None):
        current = digest([self.fingerprint, (sampling or {}).get("experimental_speed_projection", True)])
        if self.engine_last is not getattr(engine, "last", None) or current != self.current_fingerprint:
            self.live = []
        self.current_fingerprint = current
        event = {"action": "replay", "restore_s": 0, "checkpoint": None}
        row, _ = self.catalog.best(current, ids, len(ids)*self.seconds_per_token)
        # Do not replace a better already-live prefix with a disk copy.
        live = 0
        for a, b in zip(self.live, ids):
            if a != b:
                break
            live += 1
        if row and len(row["tokens"]) > live:
            start = time.perf_counter()
            try:
                with self.catalog.materialize(row["id"]) as path:
                    if getattr(engine,"spawn",None):
                        prefixes = session_prefixes(path,engine.max_context)
                        if row["tokens"] not in prefixes:
                            raise ValueError("catalog prefix is absent from native session")
                        live_tokens = len(prefixes[0])
                    else:
                        live_tokens = len(row["tokens"])
                    result = engine.session_file("restore", str(path))
                if result["tokens"] != live_tokens:
                    raise ValueError("restored checkpoint token count differs from its catalog")
                elapsed = time.perf_counter()-start
                self.catalog.restored(row["id"], elapsed)
                event.update(action="restore", restore_s=elapsed, checkpoint=row["id"], tier=row["tier"])
                self.live = row["tokens"]
            except (OSError, ValueError) as exc:
                self.live = []
                event.update(action="replay", error=str(exc))
        self.events.append(event)
        self.events[:] = self.events[-100:]
        return event

    def after(self, engine, ids, executed_ids, successful, response_id=None):
        self.live = list(executed_ids) if successful else []
        if not successful:
            return
        last = getattr(engine, "last", {}) or {}
        self.engine_last = getattr(engine, "last", None)
        read = last.get("prompt_tokens", len(ids)) - (last.get("reused") or 0)
        prefill_ms = last.get("prompt_ms")
        if prefill_ms and read > 0:
            self.seconds_per_token = prefill_ms/1000/read
        self.catalog.observe(self.current_fingerprint, ids, len(ids)*self.seconds_per_token)
        temporary = self.catalog.root / (uuid.uuid4().hex + ".save.tmp")
        try:
            # Account for both native staging and the admitted copy before writing either.
            self.catalog.enforce(incoming=2*self.max_snapshot_bytes)
            result = engine.session_file("save", str(temporary))
            n = result["tokens"]
            if getattr(engine,"spawn",None):
                prefixes = session_prefixes(temporary,engine.max_context)
                known = min(n,len(executed_ids))
                if n!=len(prefixes[0]) or prefixes[0][:known]!=executed_ids[:known]:
                    raise ValueError("saved native state does not match recorded execution prefix")
            else:
                if not 0 < n <= len(executed_ids):
                    raise ValueError("saved session does not match recorded execution length")
                prefixes = [executed_ids[:n]]
            if temporary.stat().st_size > self.max_snapshot_bytes:
                raise ValueError("native session exceeded configured snapshot admission bound")
            live_sid = self.catalog.admit(temporary, self.current_fingerprint, prefixes[0],
                                         restore_s=max(.001, temporary.stat().st_size/(1024**3)))
            for prefix in prefixes:
                sid = live_sid if prefix == prefixes[0] else self.catalog.add_prefix(live_sid,prefix)
                if response_id:
                    self.catalog.attach(sid, response_id)
        except (OSError, ValueError) as exc:
            self.events.append({"action": "admission_skipped", "error": str(exc)})
        finally:
            temporary.unlink(missing_ok=True)


def enable(svc, cfg, history_path, *, store_class=BranchStore, store_kwargs=None):
    if getattr(svc.engine, "batch", 0):
        raise ValueError("experimental branch checkpoints require serial execution")
    spawn = getattr(svc.engine, "spawn", None)
    if spawn:
        args = spawn[1]
        def number(flag, default):
            return int(args[args.index(flag)+1]) if flag in args else default
        if number("--session-min-free-mib", 4096) < cfg.get("history_reserve_mib", 4096):
            raise ValueError("native --session-min-free-mib must cover history_reserve_mib")
        if number("--conversation-cache-disk-mib", 0):
            raise ValueError("disable the native FIFO disk tier when enabling experimental branch checkpoints")
    identity = execution_identity(svc, cfg)
    from serve.server import Vision
    def asset_loader(url):
        if not url.startswith(("https://", "http://")):
            raise ValueError("durable image references require data: or HTTP(S) URLs")
        return "data:application/octet-stream;base64," + base64.b64encode(Vision.download(url)).decode()
    store = store_class(history_path, execution_identity=identity, asset_loader=asset_loader,
                        max_bytes=cfg.get("responses_store_max_mib", 1024)*1024**2, **(store_kwargs or {}))
    try:
        catalog = CheckpointStore(Path(history_path)/"execution-cache",
            budget_bytes=cfg.get("checkpoint_budget_mib", 32768)*1024**2,
            reserve_bytes=cfg.get("history_reserve_mib", 4096)*1024**2,
            archive=cfg.get("checkpoint_archive_path"),
            archive_budget_bytes=cfg.get("checkpoint_archive_budget_mib", 0)*1024**2,
            chunk_bytes=cfg.get("checkpoint_chunk_mib", 0)*1024**2,
            policy=cfg.get("checkpoint_policy", "utility"))
    except BaseException:
        store.close()
        raise
    svc.response_store = store
    store.on_expire = catalog.expire_owners
    catalog.restart_cleanup(store.protected_checkpoint_owners())
    store.on_shutdown = catalog.restart_cleanup
    svc.branch_checkpoints = CheckpointRuntime(catalog, store.fingerprint,
                                             cfg.get("checkpoint_max_snapshot_mib", 4096)*1024**2)
    return store
