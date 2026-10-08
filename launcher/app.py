"""launcher/app.py - the small local server that serves the launcher page and its API.

    python -m launcher                     # http://127.0.0.1:8090/ in your browser
    python -m launcher --port 8091 --no-browser

It is stdlib-only, like the MCP server it shares its controller with, so it runs before setup has made a .venv and
after: the page is the front-end, `launcher/api.py` is the logic, and setup.py / serve/server.py stay the ones that
actually install and run a model.  It listens on 127.0.0.1 only - the same rule the model server follows: nothing of
this PC leaves it unless you set up a network address with an API key.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .api import Launcher

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web"
SHARED = ROOT / "serve" / "web"          # the chat page's design tokens and its font (SIL OFL)
DEFAULT_PORT = 8090
TYPES = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml",
         ".html": "text/html; charset=utf-8", ".woff2": "font/woff2", ".png": "image/png", ".ico": "image/x-icon"}


def static(folder: Path, name: str) -> tuple[bytes, str] | None:
    """A file from a folder, never above it (no ../ in a URL)."""
    if "/" in name or "\\" in name or "\x00" in name:
        return None
    f = folder / name
    ext = f.suffix.lower()
    if ext not in TYPES or not f.is_file():
        return None
    try:
        return f.read_bytes(), TYPES[ext]
    except OSError:
        return None


class ClientGone(Exception):
    """The browser went away while the launcher was answering - the page reloads every 3 seconds and a reload or a
    closed tab cuts the connection.  Normal here: no 500, no stack on the console."""


class Handler(BaseHTTPRequestHandler):
    server_version = "StrataLauncher"
    api: Launcher = None                      # set on the class before the server starts

    # ---- plumbing
    def log_message(self, fmt, *args):        # the console stays readable: only the API's problems are worth a line
        if self.path.startswith("/api/") and args and str(args[1] if len(args) > 1 else "")[:1] == "5":
            sys.stderr.write(f"launcher: {self.path} {fmt % args}\n")

    def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-cache") -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError) as e:
            raise ClientGone(str(e)) from None

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if not n:
            return {}
        try:
            raw = json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ValueError("the request body is not JSON")
        return raw if isinstance(raw, dict) else {}

    # ---- routing
    def do_GET(self):
        self._route(body=None)

    def do_HEAD(self):
        self._route(body=None)

    def do_POST(self):
        try:
            self._route(body=self._body())
        except ClientGone:
            return
        except ValueError as e:
            self._json(400, {"error": {"message": str(e)}})

    def _route(self, body):
        url = urllib.parse.urlsplit(self.path)
        path = url.path.rstrip("/") or "/"
        q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        try:
            if path == "/":
                page = static(WEB, "index.html")
                if not page:
                    return self._send(404, b"launcher/web/index.html is missing", "text/plain; charset=utf-8")
                return self._send(200, *page)
            if path.startswith("/web/"):
                found = static(WEB, path[len("/web/"):])
            elif path.startswith("/ui/"):
                found = static(SHARED, path[len("/ui/"):])
            elif path.startswith("/fonts/"):
                # the same font the chat page uses (Outfit, SIL Open Font License 1.1)
                found = static(SHARED / "fonts", path[len("/fonts/"):])
            else:
                return self._api(path.lstrip("/"), q, body)
            if not found:
                return self._json(404, {"error": {"message": f"no {path}"}})
            data, ctype = found
            return self._send(200, data, ctype, "no-cache" if ctype.endswith("utf-8") else "max-age=86400")
        except ClientGone:
            return                                            # nothing to answer, nothing to print
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return                                            # the same, from a read or a probe instead of a write
        except Exception as e:                                    # noqa: BLE001 - a page must never hang on a stack
            tool_error = getattr(getattr(self.api, "m", None), "ToolError", ())
            if tool_error and isinstance(e, tool_error):
                return self._json(400, {"error": {"message": str(e)}})
            if isinstance(e, ValueError):
                return self._json(400, {"error": {"message": str(e)}})
            traceback.print_exc()
            return self._json(500, {"error": {"message": f"the launcher hit a problem: {e}"}})

    def _api(self, path: str, q: dict, body) -> None:
        a = self.api
        if path == "api/state":
            return self._json(200, a.state(hardware=q.get("hardware") == "1"))
        if path == "api/schema":
            return self._json(200, a.schema())
        if path == "api/models":
            return self._json(200, a.ui_catalog())
        if path == "api/plan":
            return self._json(200, a.plan(q.get("id") or ""))
        if path == "api/model":
            return self._json(200, a.model_for_preset(q.get("id") or ""))
        if path == "api/jobs":
            return self._json(200, dict(a.install_status(), **a.calibrate_status()))
        if path == "api/logs":
            return self._json(200, a.logs(q.get("source", ""), q.get("model", ""), int(q.get("lines", 80))))
        if path == "api/connect":
            return self._json(200, a.connect())
        if path == "api/preset/new":
            return self._json(200, {"preset": a.new_preset()})
        if path == "api/preset/save":
            return self._json(200, {"preset": a.save_preset(body)})
        if path == "api/preset/delete":
            return self._json(200, a.delete_preset(body.get("id", "")))
        if path == "api/preset/from-model":
            return self._json(200, {"preset": a.preset_from_model(body.get("model", ""), body.get("name", ""))})
        if path == "api/preset/diff":
            return self._json(200, a.preset_diff(body))
        if path == "api/install":
            return self._json(200, a.install(body.get("id") or ""))
        if path == "api/apply":
            return self._json(200, a.apply_by_id(body.get("id") or ""))
        if path == "api/install/cancel":
            return self._json(200, a.install_cancel())
        if path == "api/start":
            return self._json(200, a.start(body.get("id") or "", int(body.get("wait_seconds", 10)),
                                          bool(body.get("apply", True))))
        if path == "api/stop":
            return self._json(200, a.stop(bool(body.get("force"))))
        if path == "api/calibrate":
            return self._json(200, a.calibrate(body.get("model") or "", bool(body.get("then_start"))))
        if path == "api/calibrate/cancel":
            return self._json(200, a.calibrate_cancel())
        return self._json(404, {"error": {"message": f"no /{path}"}})


class Server(ThreadingHTTPServer):
    """A launcher per port.  With the default SO_REUSEADDR, Windows lets a second launcher bind a port that is
    already listening and the first one keeps answering - so `--port` would silently do nothing."""
    allow_reuse_address = False

    def handle_error(self, request, client_address):
        """socketserver prints a stack for every thread that dies.  A browser cutting the connection halfway
        through an answer is ordinary for a page that polls, so those stay off the console."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (ClientGone, ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def serve(root=None, port: int = DEFAULT_PORT, open_browser: bool = True, quiet: bool = False) -> int:
    api = Launcher(root)
    if not (api.root / "setup.py").is_file() or not (api.root / "serve" / "server.py").is_file():
        print(f"{api.root} is not a Strata folder (no setup.py / serve/server.py)")
        return 2
    Handler.api = api
    httpd, chosen = None, None
    for p in range(port, port + 20):
        try:
            httpd = Server(("127.0.0.1", p), Handler)
            chosen = p
            break
        except OSError:
            continue
    if httpd is None:
        print(f"no free port from {port} to {port + 19}: close something and try again")
        return 2
    url = f"http://127.0.0.1:{chosen}/"
    if not quiet:
        print(f"Strata launcher: {url} (this PC only; close this window or press Ctrl+C to stop it)", flush=True)
        print(f"  Strata folder: {api.root}", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:                                           # noqa: BLE001 - no browser: the URL is printed
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        if not quiet:
            print("\nlauncher stopped (a model it started keeps running: stop it on the page or close its window)")
    finally:
        httpd.server_close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Strata's launcher: pick a model, a preset, and start it.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="the Strata folder (default: the one this file is in)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"the page's port (default {DEFAULT_PORT})")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    ap.add_argument("--quiet", action="store_true", help="print nothing")
    a = ap.parse_args(argv)
    return serve(a.root, a.port, not a.no_browser, a.quiet)


if __name__ == "__main__":
    sys.exit(main())
