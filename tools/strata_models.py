#!/usr/bin/env python3
"""Opt-in localhost browser launcher for setup.py's model catalog.

The normal installer, server and engine are unchanged. This process keeps the
Models page available while the official installer or model server is running.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import secrets
import threading
import time
import webbrowser
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

try:
    from . import strata_mcp as M
    from .gguf_reader import GGUFFile
except ImportError:
    import strata_mcp as M
    from gguf_reader import GGUFFile

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "serve" / "web"
STANDARD_IDS = {"q2_0", "iq2_xs", "iq3_xxs", "iq3_s", "coder-iq1_m"}


def arg_value(args, flag):
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def config_path(cfg, value, root):
    cwd = Path(cfg.get("cwd") or root)
    if not cwd.is_absolute():
        cwd = root / cwd
    p = Path(value)
    return p if p.is_absolute() else cwd / p


class ModelManager:
    def __init__(self, strata=None, model_port=8080):
        self.s = strata or M.Strata(ROOT)
        self.s.state_dir = self.s.root / ".strata-models"
        self.tools = M.Tools(self.s)
        self.model_port = model_port
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.operation = None
        self.plans = {}
        self.load_timeout = 600
        # Import before concurrent HTTP requests. Never authorize browser installs
        # from the MCP helper's fallback table when setup.py cannot be imported.
        self.s.setup_module()
        pending = self.s.state("web-install")
        job = self.s.state("install")
        if pending and pending.get("job") == job.get("job"):
            self.begin("install", pending["model"], lambda: self.finish_install(pending))

    def catalog(self):
        if self.s.setup_module() is None:
            raise M.ToolError("setup.py could not be loaded. Repair this checkout before installing models.")
        hw = self.s.hardware()
        rows = []
        for family in self.s.model_table(hw):
            if not family.get("experimental"):
                for size in family["sizes"]:
                    if not size.get("experimental"):
                        rows.append({**size, "family": family["family"], "title": family["title"]})
        return rows, hw

    def select(self, model_id):
        if isinstance(model_id, str):
            rows, _ = self.catalog()
            row = next((r for r in rows if r["id"] == model_id), None)
            if row:
                return row
        raise M.ToolError("Choose a regular model from Strata's official installer catalog.")

    def installed(self, row):
        p = self.s.root / f"strata-{row['id']}.json"
        cfg = self.s.read_config(p)
        if not isinstance(cfg.get("exe"), str) or not isinstance(cfg.get("args"), list):
            return None
        paths = [cfg["exe"], cfg.get("tokenizer"), arg_value(cfg["args"], "--pack")]
        paths += [a for a in cfg["args"] if isinstance(a, str) and a.endswith(".gguf")]
        missing = [str(v) for v in paths if not v or not config_path(cfg, v, self.s.root).exists()]
        return {"id": row["id"], "model_name": cfg.get("model_name"), "ready": not missing, "missing": missing[:3],
                "context": arg_value(cfg["args"], "--max-context"), "images": bool(cfg.get("vision")),
                "lookup": arg_value(cfg["args"], "--ple-gguf")}

    def live(self):
        tracked = self.s.tracked_server()
        running = []
        for port in dict.fromkeys([self.model_port, *self.s.ports(None)]):
            info = self.s.probe(port)
            if info and not info.get("other"):
                own = bool(tracked and not tracked.get("ended") and tracked.get("port") == port)
                running.append({**info, "managed": own})
        return running, tracked

    def preferred_lookup(self):
        candidates = []
        for config in self.s.configs():
            if config.stem.removeprefix("strata-") not in STANDARD_IDS:
                continue
            cfg = self.s.read_config(config)
            args = cfg.get("args", [])
            value = arg_value(args, "--ple-gguf") if isinstance(args, list) else None
            if value:
                p = config_path(cfg, value, self.s.root)
                if (p.is_file() and p.name.startswith("Qwen3.8-Flash-Next-GSQ-RCO-")
                        and p.name.endswith("-00002-of-00002.gguf")):
                    candidates.append(p)
        # Prefer a table moved out of the normal data folder, such as an SSD.
        return next((p for p in candidates if not M.is_inside(p.resolve(), self.s.data_dir().resolve())),
                    candidates[0] if candidates else None)

    def status(self):
        with self.lock:
            rows, hw = self.catalog()
            running, tracked = self.live()
            for row in rows:
                row["installation"] = self.installed(row)
            lookup = self.preferred_lookup()
            logs = []
            if self.operation:
                source = "setup.log" if self.operation["kind"] == "install" else "server.log"
                logs = M.read_tail(self.s.state_dir / source, 12)
            return {"token": self.token, "models": rows, "hardware": hw,
                    "data_folder": str(self.s.data_dir()), "lookup": str(lookup) if lookup else None,
                    "running": running, "loading": bool(tracked and not tracked.get("ended") and not running),
                    "operation": dict(self.operation) if self.operation else None,
                    "install": self.s.install_job(), "logs": logs,
                    "model_url": f"http://127.0.0.1:{self.model_port}/"}

    def idle(self):
        if self.operation and self.operation["state"] == "running":
            raise M.ToolError("Another operation is running. Wait for it to finish.")
        for job in (self.s.install_job(), M.Strata(self.s.root).install_job()):
            if job and job["running"]:
                raise M.ToolError("An installer is already running. Wait for it to finish.")

    def plan(self, model_id):
        with self.lock:
            self.idle()
            row = self.select(model_id)
            plan, args = self.tools.install_plan(row["family"], row["model"], None, None, None,
                                                None, None, None, None, self.model_port, None)
            # Disk estimates/data location from the official management helper;
            # all inference settings are selected by setup.py itself.
            for flag in ("--context", "--vision", "--experimental-speed-projection"):
                if flag in args:
                    i = args.index(flag)
                    del args[i:i + 2]
            args.insert(0, "--setup")
            args += ["--data-dir", plan["data_folder"]]
            plan.update(context="chosen by setup", images="setup's default (off)",
                        setup_command="setup.py " + " ".join(args), id=model_id,
                        settings="GPU, backend, context, KV, RAM mode and cache use setup.py's defaults.")
            lookup = self.preferred_lookup() if model_id in STANDARD_IDS else None
            plan["lookup"] = str(lookup) if lookup else "model's own table, placed by setup"
            plan["note"] = "Setup may download engine files or compile its supported backend on this PC."
            pid = secrets.token_urlsafe(24)
            self.plans = {pid: (time.monotonic(), plan, args, lookup)}
            return {"plan_id": pid, "plan": plan}

    def require_owned(self):
        running, tracked = self.live()
        if any(not r["managed"] for r in running):
            raise M.ToolError("Close Strata's existing server window first, then start models from this table. "
                              "This launcher does not stop servers started elsewhere.")
        return tracked if tracked and not tracked.get("ended") else None

    def stop_owned(self):
        tracked = self.require_owned()
        if tracked:
            port = tracked["port"]
            status, body = M.http_json(f"http://127.0.0.1:{port}/status",
                                       api_key=self.s.api_key_for(port), timeout=3)
            if status == 200 and isinstance(body, dict):
                if body.get("busy"):
                    raise M.ToolError("A request is running. Wait for it to finish before switching models.")
            elif not M.port_free(port):
                raise M.ToolError("Cannot verify whether the model is idle. Check its API key or stop it from its own window.")
            result = self.tools.strata_stop(force=False)
            remaining = self.s.tracked_server()
            if remaining and not remaining.get("ended"):
                raise M.ToolError(result["summary"])
        return tracked

    def begin(self, kind, model_id, task):
        self.operation = {"kind": kind, "model": model_id, "state": "running", "message": "Starting..."}
        def work():
            try:
                message, state = task(), "done"
            except Exception as exc:
                message, state = str(exc), "failed"
            with self.lock:
                self.operation.update(state=state, message=message)
        threading.Thread(target=work, daemon=True).start()
        return {"operation": dict(self.operation)}

    def install(self, plan_id):
        with self.lock:
            self.idle()
            saved = self.plans.pop(plan_id, None) if isinstance(plan_id, str) else None
            if not saved or time.monotonic() - saved[0] > 900:
                raise M.ToolError("This download plan expired. Review the download again.")
            _, plan, args, lookup = saved
            self.select(plan["id"])
            if plan.get("disk_short"):
                raise M.ToolError(plan["disk_short"])
            free = self.s.disk_free(Path(plan["data_folder"]))
            if free is not None and free < plan["disk_needed_gb"]:
                raise M.ToolError("Free disk space changed. Review the download again.")
            self.require_owned()
            def run():
                self.stop_owned()
                with self.s.lock:
                    self.tools.launch_install(plan, args)
                pending = {"job": self.s.state("install")["job"], "model": plan["id"],
                           "lookup": str(lookup) if lookup else None}
                self.s.save_state("web-install", pending)
                return self.finish_install(pending)
            return self.begin("install", plan["id"], run)

    def finish_install(self, pending):
        while True:
            if self.s.state("install").get("job") != pending["job"]:
                raise M.ToolError("The installer job changed. Refresh and review its output.")
            job = self.s.install_job()
            if job and not job["running"]:
                try:
                    if self.s.state("web-install").get("cancelled"):
                        return "Download stopped. Review the download again to resume it."
                    if job.get("exit_code") != 0:
                        raise M.ToolError(job.get("error") or "Setup failed. Retry resumes downloads; see the output below.")
                    if pending.get("lookup"):
                        self.reuse_lookup(pending["model"], Path(pending["lookup"]))
                    return "Installed. Select Start to load this model."
                finally:
                    self.s.save_state("web-install", None)
            time.sleep(1)

    def reuse_lookup(self, model_id, lookup):
        """Keep an existing SSD location, changing paths only after setup succeeds."""
        if model_id not in STANDARD_IDS:
            return
        p = self.s.root / f"strata-{model_id}.json"
        cfg = self.s.read_config(p)
        args = cfg.get("args", [])
        old, native = arg_value(args, "--ple-gguf"), arg_value(args, "--native")
        if not old or not native:
            raise M.ToolError("Setup finished, but its lookup path is missing. Review the model configuration.")
        old_path = config_path(cfg, old, self.s.root)
        if old_path.resolve() == lookup.resolve():
            return
        def signature(path):
            g = GGUFFile(path)
            if not any(t.name == "per_layer_token_embd.weight" for t in g.tensors):
                raise M.ToolError("The shared file has no lookup table. Keep the model's own shard.")
            return (path.stat().st_size, g.metadata.get("general.architecture"),
                    sorted((t.name, t.shape, t.type_id) for t in g.tensors))
        if signature(old_path) != signature(lookup):
            raise M.ToolError("The lookup shards do not match. The model remains installed with setup's original paths.")
        updated, i = [], 0
        while i < len(args):
            if args[i] == "--native-dense-gguf":
                i += 2
                continue
            updated.append(str(lookup) if i and args[i - 1] == "--ple-gguf" else args[i])
            i += 1
        updated += ["--native-dense-gguf", native, "--native-dense-gguf", str(lookup)]
        cfg["args"] = updated
        p.with_suffix(p.suffix + ".bak").write_bytes(p.read_bytes())
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, p)

    def preflight(self, model_id):
        row = self.select(model_id)
        desc = self.installed(row)
        if not desc or not desc["ready"]:
            raise M.ToolError("This model's files are missing. Download or repair it before starting it.")
        cfg = self.s.read_config(self.s.root / f"strata-{model_id}.json")
        if cfg.get("host", "127.0.0.1") not in ("127.0.0.1", "localhost"):
            raise M.ToolError("This launcher starts local servers only. Use the start script for its network configuration.")
        if not Path(self.s.run_python()).exists():
            raise M.ToolError("The Python environment is missing. Run setup first.")
        return cfg

    def wait_model(self, cfg):
        end = time.monotonic() + self.load_timeout
        while time.monotonic() < end:
            tracked = self.s.tracked_server()
            if not tracked or tracked.get("ended"):
                raise M.ToolError("The model stopped while loading. See the server output below.")
            info = self.s.probe(self.model_port, deep=False)
            if info and not info.get("other") and (info.get("loaded") or cfg.get("lazy_load") is True):
                return
            time.sleep(1)
        raise M.ToolError("Model loading timed out. See the server output below.")

    def start(self, model_id):
        with self.lock:
            self.idle()
            cfg = self.preflight(model_id)
            tracked = self.require_owned()
            if tracked and tracked.get("model") == model_id:
                return {"message": "This model is already running or loading."}
            if not tracked and not M.port_free(self.model_port):
                raise M.ToolError("The model port is already used. Close the other server before starting a model.")
            def run():
                previous = self.stop_owned()
                try:
                    self.tools.strata_start(model=model_id, port=self.model_port, wait_seconds=0)
                    self.wait_model(cfg)
                except Exception as exc:
                    self.stop_owned()
                    if previous:
                        old = previous["model"]
                        old_cfg = self.preflight(old)
                        self.tools.strata_start(model=old, port=self.model_port, wait_seconds=0)
                        self.wait_model(old_cfg)
                        raise M.ToolError(f"{exc} The previous model was restored.") from exc
                    raise
                return "Model ready. Open Chat to use it."
            return self.begin("start", model_id, run)

    def stop(self):
        with self.lock:
            self.idle()
            self.require_owned()
            def run():
                self.stop_owned()
                return "Model stopped; its RAM and VRAM are free."
            return self.begin("stop", None, run)

    def cancel(self):
        with self.lock:
            if not self.operation or self.operation["state"] != "running" or self.operation["kind"] != "install":
                raise M.ToolError("There is no active download to stop.")
            job = self.s.install_job()
            if not job or not job["running"]:
                raise M.ToolError("Setup is still starting. Wait a moment, then stop the download.")
            pending = self.s.state("web-install")
            self.s.save_state("web-install", {**pending, "cancelled": True})
            return self.tools.strata_install(cancel=True)


def make_handler(manager):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, data, content_type="application/json"):
            if isinstance(data, dict):
                data = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                             "font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(data)

        def local(self, mutation=False):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host", "") not in allowed or self.client_address[0] != "127.0.0.1":
                self.send(403, {"error": "Open this page on localhost."})
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + h for h in allowed}:
                self.send(403, {"error": "Only Strata's own Models page may manage models."})
                return False
            if mutation and (not origin or not secrets.compare_digest(self.headers.get("X-Strata-Token", "").encode(), manager.token.encode())):
                self.send(403, {"error": "Reload the Models page before continuing."})
                return False
            return True

        def do_GET(self):
            if not self.local():
                return
            path = urlsplit(self.path).path
            if path == "/api/models/status":
                try:
                    self.send(200, manager.status())
                except M.ToolError as exc:
                    self.send(503, {"error": str(exc)})
                return
            files = {"/": "models.html", "/models": "models.html"}
            files.update({"/web/" + name: name for name in
                          ("models.js", "models.css", "tokens.css", "components.css", "sprite.svg", "favicon.svg")})
            files.update({"/fonts/" + name: "fonts/" + name for name in
                          ("outfit-latin-wght.woff2", "outfit-latin-ext-wght.woff2")})
            if path not in files:
                self.send(404, {"error": "Not found"})
                return
            p = WEB / files[path]
            self.send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")

        def do_POST(self):
            if not self.local(mutation=True):
                return
            try:
                if self.headers.get("Transfer-Encoding") or self.headers.get_content_type() != "application/json":
                    raise ValueError("Send a JSON request.")
                length = int(self.headers.get("Content-Length", "0"))
                if length < 2 or length > 4096:
                    raise ValueError("Invalid request length.")
                body = json.loads(self.rfile.read(length))
                actions = {"/api/models/plan": ("model", manager.plan),
                           "/api/models/install": ("plan_id", manager.install),
                           "/api/models/start": ("model", manager.start),
                           "/api/models/stop": (None, manager.stop),
                           "/api/models/cancel": (None, manager.cancel)}
                action = actions.get(urlsplit(self.path).path)
                if action is None:
                    self.send(404, {"error": "Not found"})
                    return
                key, fn = action
                if not isinstance(body, dict) or set(body) != ({key} if key else set()):
                    raise ValueError("Unexpected request fields.")
                self.send(200, fn(body[key]) if key else fn())
            except (ValueError, TypeError, M.ToolError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception:
                self.send(500, {"error": "The operation failed. Check the launcher's console and retry."})
    return Handler



@contextmanager
def launcher_lock(root):
    """One browser launcher per checkout, including during loading/recovery."""
    state = root / ".strata-models"
    state.mkdir(parents=True, exist_ok=True)
    with (state / "launcher.lock").open("a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise M.ToolError("A Models launcher already runs for this checkout. Open its existing page.") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--port", type=int, default=8090, help="Models page port (local only)")
    ap.add_argument("--model-port", type=int, default=8080, help="chat and API port")
    ap.add_argument("--open", action="store_true", help="open the Models page")
    a = ap.parse_args(argv)
    if not all(1 <= p <= 65535 for p in (a.port, a.model_port)) or a.port == a.model_port:
        ap.error("Choose two different ports between 1 and 65535.")
    if not (a.root / "setup.py").is_file():
        ap.error("--root must be a Strata checkout")
    try:
        with launcher_lock(a.root):
            manager = ModelManager(M.Strata(a.root), a.model_port)
            with ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(manager)) as server:
                url = f"http://127.0.0.1:{a.port}/"
                print(f"Strata Models: {url}", flush=True)
                if a.open:
                    webbrowser.open(url)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
    except (M.ToolError, OSError) as exc:
        ap.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())