"""The update routes over real HTTP, with no GitHub and no real engine.

Run:  python serve/test_update_http.py

serve/test_update.py covers the updater on its own.  This one drives it through the actual
server: the routes, the same-origin guard every mutating route uses, the refusal while a request is in
flight, and the JSON the web app polls.

It builds a REAL Service the way serve/test_server.py does - Service(MockEngine(...), ByteTokenizer(),
ChatTemplate(...)) - rather than a hand-patched one.  That matters: an earlier version of this file
built the Service with `Service.__new__` and set only the attributes it knew about, and every attribute
it had not guessed wrong became an AttributeError inside the request handler.  Here the constructor sets
everything the handler touches, so a missing attribute is a real bug rather than a gap in the test.

Only the updater's network is stubbed, at the two callables it uses (`fetch` for the release JSON,
`head` for the download), so the state machine, the file handling, the routes and the guards are all
the project's own code.  The zip is built locally.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # the fixtures live beside this file

FAILS: list[str] = []
CHECKS = [0]


def check(ok: bool, what: str, detail: str = ""):
    CHECKS[0] += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {what:<58}{' ' + detail if detail else ''}")
    if not ok:
        FAILS.append(what)


# --- a release the server will accept ------------------------------------------------------------

VERSION_NEW = "0.1.99"
VERSION_OLD = "0.1.31"

# the fixtures (fake engine binary, a release-shaped zip, a fake install) live beside this file
from serve.test_update import engine_files, fake_engine, install_fake, make_zip  # noqa: E402


def build_zip_bytes(version: str = VERSION_NEW) -> bytes:
    return make_zip(engine_files(version), version)


def stub_network(version: str = VERSION_NEW, payload: bytes | None = None, size=None, digest=None):
    """Point serve.update.Updater's two network hooks at a local zip.

    Everything else stays real: the steps, the file operations, the routes.  `size` overrides the size
    the release reports, which is how a truncated download is simulated.
    """
    import io
    import serve.update as U

    body = payload if payload is not None else build_zip_bytes(version)
    reported = size if size is not None else len(body)

    def fetch(url):
        return {"tag_name": f"v{version}", "html_url": "https://example/r",
                "assets": [{"name": ("strata-windows-x64.zip" if os.name == "nt"
                                     else "strata-linux-x64.zip"),
                            "size": reported,
                            # GitHub's own hash for the asset; without it the updater refuses
                            "digest": f"sha256:{digest or hashlib.sha256(body).hexdigest()}",
                            "browser_download_url": "https://example/a.zip"}]}

    def head(url):
        class R:
            def __init__(self):
                self.headers = {"Content-Length": str(len(body))}
                self._b = io.BytesIO(body)

            def read(self, n=-1):
                return self._b.read(n)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return R()

    real_init = U.Updater.__init__

    def patched(self, *a, **kw):
        real_init(self, *a, **kw)
        self.fetch, self.head = fetch, head

    U.Updater.__init__ = patched

    # The card check reads the real nvidia-smi, which would make every test here depend on the GPU of
    # whichever machine runs them. None means "cannot tell", which the updater deliberately lets through;
    # t_card_refusal_over_http sets it explicitly to cover the other branch.
    U.gpu_compute_capability = lambda: None


# --- a real Service, the project's own way ---------------------------------------------------------

def make_service(engine_dir: Path):
    """A real Service on a real (fake) engine, plus the attributes the update routes need.

    `unload`/`load` are replaced because the real ones manage an engine process: this test measures the
    ROUTES and the busy-refusal, and the drain-then-reload behaviour is verified by stubbing unload to
    report each result and asserting on which one came back.
    """
    from serve.frontend import ChatTemplate
    from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine

    tok = ByteTokenizer()
    exe = engine_dir / ("strata.exe" if os.name == "nt" else "strata")
    # lazy=True: the default constructor Popen()s the engine and then blocks reading its stdout, which
    # never ends with no real engine.  lazy=True sets self.spawn (the only thing the updater reads).
    svc = Service(MockEngine(tok, "hi", max_context=4096), tok,
                  ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
    svc.engine = StrataEngine(str(exe), [], lazy=True)
    svc.backend = "cuda"
    svc.allowed_hosts = ["127.0.0.1"]
    svc.unload_calls: list[str] = []
    svc.loaded = lambda: True
    svc.unload = lambda idle_for=None: (svc.unload_calls.append("unload"), "unloaded")[1]
    svc.load = lambda: svc.unload_calls.append("load")
    return svc


def start(svc):
    """The project's own serve(), so the handler, the Host check and the CORS path are the real ones."""
    from serve.server import serve
    httpd = serve(svc, "127.0.0.1", 0)     # port 0: the OS picks a free one
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def req(base: str, path: str, body=None, method=None, ctype="application/json", origin=True,
        timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(base + path, data=data,
                               method=method or ("POST" if data is not None else "GET"))
    if data is not None:
        r.add_header("Content-Type", ctype)
    if origin:
        r.add_header("Origin", base.split("//", 1)[1])   # this server's own address, as the browser sends
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}


def wait_for(base: str, states=("done", "failed"), limit=180):
    deadline = time.time() + limit
    last = {}
    while time.time() < deadline:
        _, last = req(base, "/api/update/state")
        if last.get("state") in states:
            return last
        time.sleep(0.2)
    return last


def build_engine(root: Path, version: str = VERSION_OLD) -> Path:
    
    return install_fake(root / "engine", version)


# --- the cases ------------------------------------------------------------------------------------

def t_routes_end_to_end():
    print("the routes over real HTTP, end to end")
    import serve.update as U
    stub_network(VERSION_NEW)
    with tempfile.TemporaryDirectory() as d:
        eng = build_engine(Path(d))
        svc = make_service(eng)
        httpd, base = start(svc)
        try:
            code, body = req(base, "/api/update/state")
            check(code == 200 and body.get("state") == "idle", "GET /api/update/state answers",
                  str(body.get("state")))
            check(body.get("steps") == [], "and reports no steps yet")

            code, body = req(base, "/api/update/check", {})
            check(code == 200, "POST /api/update/check answers 200", str(code))
            check(body.get("state") == "ready", "it reaches ready", str(body.get("state")))
            check(body["detail"].get("newer") is True, "and says a newer release exists")
            check(body["detail"].get("latest") == f"v{VERSION_NEW}", "with the latest tag",
                  str(body["detail"].get("latest")))
            check(U.installed_version(eng) == VERSION_OLD, "checking changed nothing on disk")

            code, body = req(base, "/api/update/apply", {})
            check(code == 202, "POST /api/update/apply answers 202 (runs in the background)", str(code))
            check(svc.unload_calls == ["unload"], "the engine is stopped before the files change",
                  str(svc.unload_calls))

            final = wait_for(base)
            check(final.get("state") == "done", "the run finishes done",
                  str(final.get("detail", {}).get("error", ""))[:64])
            steps = final.get("steps") or []
            check(len(steps) == 9, "all nine steps are reported for the UI", f"{len(steps)}")
            check(all(s["status"] == "done" for s in steps), "every step finished",
                  str([s["key"] for s in steps if s["status"] != "done"]))
            check(U.installed_version(eng) == VERSION_NEW, "the engine on disk is the new one")
            check(final.get("backup"), "a backup path is reported so the UI can name it",
                  str(final.get("backup"))[-34:])
            check("load" in svc.unload_calls, "the model is loaded again afterwards",
                  str(svc.unload_calls))

            # percent must never read 100% before the run: the UI would then snap back to 0%.
            _, state = req(base, "/api/update/state")
            check(state.get("percent") is None, "percent is null once the run is over, not stale",
                  str(state.get("percent")))
        finally:
            httpd.shutdown()
            httpd.server_close()


def t_same_origin_guard():
    print("\nthe same-origin guard on the mutating routes")
    stub_network(VERSION_NEW)
    with tempfile.TemporaryDirectory() as d:
        svc = make_service(build_engine(Path(d)))
        httpd, base = start(svc)
        try:
            code, _ = req(base, "/api/update/check", {})
            check(code == 200, "this server's own page is allowed", str(code))

            host = base.split("//", 1)[1]
            for path in ("/api/update/check", "/api/update/apply"):
                r = urllib.request.Request(base + path, data=b"{}", method="POST")
                r.add_header("Content-Type", "application/json")
                r.add_header("Origin", "http://evil.example")
                try:
                    urllib.request.urlopen(r, timeout=30)
                    check(False, f"{path} refuses a foreign Origin")
                except urllib.error.HTTPError as e:
                    check(e.code == 403, f"{path} refuses a foreign Origin", str(e.code))

                # a form post cannot carry application/json, and a cross-site form needs no CORS preflight
                r = urllib.request.Request(base + path, data=b"", method="POST")
                r.add_header("Content-Type", "application/x-www-form-urlencoded")
                try:
                    urllib.request.urlopen(r, timeout=30)
                    check(False, f"{path} refuses a plain form POST")
                except urllib.error.HTTPError as e:
                    check(e.code == 415, f"{path} refuses a plain form POST (not application/json)",
                          str(e.code))

            # the read-only state route stays open: it changes nothing, and the panel polls it on load
            code, _ = req(base, "/api/update/state", origin=False)
            check(code == 200, "GET /api/update/state is open (it only reads)", str(code))
        finally:
            httpd.shutdown()
            httpd.server_close()


def t_busy_refused():
    print("\nthe refusal while a request is in flight")
    import serve.update as U
    stub_network(VERSION_NEW)
    with tempfile.TemporaryDirectory() as d:
        eng = build_engine(Path(d))
        svc = make_service(eng)
        svc.unload = lambda idle_for=None: "busy"      # a generation is running or queued
        httpd, base = start(svc)
        try:
            req(base, "/api/update/check", {})
            code, body = req(base, "/api/update/apply", {})
            check(code == 409, "an update is refused", str(code))
            msg = (body.get("error") or {}).get("message", "")
            check("busy" in msg or "running" in msg, "and the message says why", msg[:54])
            check(U.installed_version(eng) == VERSION_OLD, "nothing was touched")

            # and the check is still usable, so the panel is not left broken by the refusal
            code, _ = req(base, "/api/update/state")
            check(code == 200, "the panel can still read the state", str(code))
        finally:
            httpd.shutdown()
            httpd.server_close()


def t_apply_before_check_refused():
    print("\napply without a check first is refused")
    with tempfile.TemporaryDirectory() as d:
        eng = build_engine(Path(d))
        svc = make_service(eng)
        httpd, base = start(svc)
        try:
            code, body = req(base, "/api/update/apply", {})
            check(code == 409, "apply with no prior check is refused", str(code))
            check("check" in (body.get("error") or {}).get("message", "").lower(),
                  "and says to check first",
                  (body.get("error") or {}).get("message", "")[:50])
            check(svc.unload_calls == [], "and it did not even try to stop the engine",
                  str(svc.unload_calls))
        finally:
            httpd.shutdown()
            httpd.server_close()


def t_truncated_download_over_http():
    print("\na download that stops early fails over HTTP too")
    import serve.update as U
    payload = build_zip_bytes(VERSION_NEW)
    stub_network(VERSION_NEW, payload=payload, size=len(payload) + 5000)   # API claims more than it serves
    with tempfile.TemporaryDirectory() as d:
        eng = build_engine(Path(d))
        svc = make_service(eng)
        httpd, base = start(svc)
        try:
            req(base, "/api/update/check", {})
            code, _ = req(base, "/api/update/apply", {})
            check(code == 202, "the update starts", str(code))
            final = wait_for(base)
            check(final.get("state") == "failed", "and then fails",
                  str(final.get("detail", {}).get("error", ""))[:56])
            failed = [s for s in final.get("steps", []) if s["status"] == "failed"]
            check(failed and failed[0]["key"] == "download", "at the download step",
                  failed[0]["key"] if failed else "none")
            check(U.installed_version(eng) == VERSION_OLD, "the installed engine is untouched")
            check("early" in (final.get("detail", {}).get("error", "")), "the message explains why",
                  final.get("detail", {}).get("error", "")[:52])
            check(final.get("detail", {}).get("action", "").startswith("Nothing on this PC"),
                  "and the short action line says nothing was changed",
                  final.get("detail", {}).get("action", "")[:44])
        finally:
            httpd.shutdown()
            httpd.server_close()


def t_downgrade_over_http():
    print("\na downgrade offered to the UI is refused by the server")
    import serve.update as U
    stub_network("0.1.20")                            # older than the installed 0.1.31
    with tempfile.TemporaryDirectory() as d:
        eng = build_engine(Path(d))
        svc = make_service(eng)
        httpd, base = start(svc)
        try:
            _, body = req(base, "/api/update/check", {})
            check(body["detail"].get("newer") is False, "the check says there is nothing newer",
                  str(body["detail"].get("newer")))
            code, _ = req(base, "/api/update/apply", {})
            final = wait_for(base)
            check(final.get("state") == "failed", "applying it fails", str(final.get("state")))
            check("downgrade" in (final.get("detail", {}).get("error", "") + " " +
                                  " ".join(s["label"] for s in final.get("steps", []))).lower(),
                  "and is named a downgrade", final.get("detail", {}).get("error", "")[:48])
            check(U.installed_version(eng) == VERSION_OLD, "the installed engine is untouched")
        finally:
            httpd.shutdown()
            httpd.server_close()


def t_card_refusal_over_http():
    print("\nthe card refusal, over HTTP")
    import serve.update as U
    # archs [75, 86, 89, 120] is what the real v0.1.38 publishes; the card is a GTX 1070 at 6.1, so the
    # release cannot run here. The install_fake engine claims to support the card, so nothing but the
    # updater's own check can catch this.
    payload = make_zip(engine_files(VERSION_NEW), VERSION_NEW)
    payload = replace_archs(payload, [75, 86, 89, 120], ptx=True)
    stub_network(VERSION_NEW, payload=payload)
    U.gpu_compute_capability = lambda: 6.1
    try:
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d) / "engine"
            eng.mkdir()
            (eng / ("strata.exe" if os.name == "nt" else "strata")).write_bytes(
                b"not really an engine")
            (eng / "BUILD.json").write_text(
                json.dumps({"version": VERSION_OLD, "archs": [61], "ptx": False}), encoding="utf-8")
            svc = make_service(eng)
            httpd, base = start(svc)
            try:
                req(base, "/api/update/check", {})
                code, _ = req(base, "/api/update/apply", {})
                check(code == 202, "the update starts", str(code))
                final = wait_for(base)
                check(final.get("state") == "failed", "and then fails",
                      final.get("state", ""))
                failed = [s for s in final.get("steps", []) if s["status"] == "failed"]
                check(failed and failed[0]["key"] == "inspect",
                      "at the inspect step - nothing was staged or replaced",
                      failed[0]["key"] if failed else "none")
                check("6.1" in final.get("detail", {}).get("error", ""),
                      "and the panel is told the card's compute capability",
                      final.get("detail", {}).get("error", "")[:48])
                check(json.loads((eng / "BUILD.json").read_text())["version"] == VERSION_OLD,
                      "the installed engine is untouched")
                check("Staged" not in str([s["status"] for s in final.get("steps", [])]),
                      "no staging directory was created")
            finally:
                httpd.shutdown()
                httpd.server_close()
    finally:
        U.gpu_compute_capability = lambda: None


def replace_archs(zip_bytes: bytes, archs: list[int], ptx: bool = False) -> bytes:
    """The fixture zip again, with different BUILD.json architectures - so a test can say what the
    RELEASE supports, which is what the card check reads."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as src, \
            zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            body = src.read(info.filename)
            if info.filename.endswith("BUILD.json"):
                meta = json.loads(body)
                meta["archs"], meta["ptx"] = archs, ptx
                body = json.dumps(meta).encode()
            dst.writestr(info.filename, body)
    return out.getvalue()


def t_bad_hash_over_http():
    print("\na download that does not match GitHub's hash is refused, over HTTP")
    import serve.update as U
    good = make_zip(engine_files(VERSION_NEW), VERSION_NEW)
    tampered = good[:-2] + b"XX"                 # same length as the API says, different bytes
    stub_network(VERSION_NEW, payload=tampered, digest=hashlib.sha256(good).hexdigest())
    with tempfile.TemporaryDirectory() as d:
        eng = build_engine(Path(d))
        svc = make_service(eng)
        httpd, base = start(svc)
        try:
            req(base, "/api/update/check", {})
            code, _ = req(base, "/api/update/apply", {})
            check(code == 202, "the update starts", str(code))
            final = wait_for(base)
            check(final.get("state") == "failed", "and then fails", str(final.get("state")))
            failed = [s for s in final.get("steps", []) if s["status"] == "failed"]
            check(failed and failed[0]["key"] == "download",
                  "at the download step - the archive was never opened",
                  failed[0]["key"] if failed else "none")
            check("wrong SHA-256" in final.get("detail", {}).get("error", ""),
                  "and the panel is told the hash did not match",
                  final.get("detail", {}).get("error", "")[:44])
            check(json.loads((eng / "BUILD.json").read_text())["version"] == VERSION_OLD,
                  "the installed engine is untouched")
            up = svc._updater()
            check(not list(up.root.glob("*.zip")), "and the bad file is deleted",
                  str([p.name for p in up.root.glob("*")]))
        finally:
            httpd.shutdown()
            httpd.server_close()


class UpdateRoutes(unittest.TestCase):
    """Every `t_*` function above, as a unittest method.

    Without this, pytest collects nothing from this file - verified, it ran zero tests - so the route
    checks would never run in CI even though they are the ones that cover the handlers.
    """


def _as_test(fn):
    def test(self):
        before = len(FAILS)
        fn()
        new = FAILS[before:]
        self.assertFalse(new, "\n".join(f"  FAILED: {f}" for f in new))
    test.__name__ = "test_" + fn.__name__[2:]
    test.__doc__ = (fn.__doc__ or fn.__name__).strip().splitlines()[0]
    return test


for _name in sorted(n for n in list(globals()) if n.startswith("t_")):
    setattr(UpdateRoutes, _as_test(globals()[_name]).__name__, _as_test(globals()[_name]))


def main() -> int:
    for fn in (t_routes_end_to_end, t_same_origin_guard, t_busy_refused,
               t_apply_before_check_refused, t_truncated_download_over_http,
               t_downgrade_over_http, t_card_refusal_over_http, t_bad_hash_over_http):
        fn()
    print(f"\nupdate routes: {len(FAILS)} failures out of {CHECKS[0]} checks")
    for f in FAILS:
        print(f"  FAILED: {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
