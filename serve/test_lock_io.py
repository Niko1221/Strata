"""#2432: nothing in serve/ waits on the console while it holds a lock (no GPU or model pack).

    python -m unittest serve.test_lock_io -v

On Windows a console paused by a QuickEdit selection blocks every write to it.  The request-finish path printed its
"[strata] done:" line inside `with self.status_lock:`, so a paused console kept status_lock held and every later
request parked in _progress.  The first test runs a real request with a stdout that blocks like a paused console;
the second keeps the rule structural: no console or log output inside any `with <lock>:` block in serve/.
"""
import ast
import json
import os
import sys
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

SERVE = Path(__file__).parent


class PausedConsole:
    """A stdout that, from the first "[strata] done:" line on, blocks every write until `resume` is set."""

    def __init__(self):
        self.lines, self.paused, self.resume = [], threading.Event(), threading.Event()
        self.lock = threading.Lock()

    def write(self, text):
        if "[strata] done:" in text:
            self.paused.set()
        if self.paused.is_set():
            self.resume.wait(30)
        with self.lock:
            self.lines.append(text)
        return len(text)

    def flush(self):
        if self.paused.is_set():
            self.resume.wait(30)


class PausedConsoleDoesNotHoldStatusLock(unittest.TestCase):
    def test_status_lock_is_free_while_the_done_line_blocks(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                      ChatTemplate(SERVE / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        console = PausedConsole()
        result = {}

        def chat():
            req = urllib.request.Request(
                f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                 "max_tokens": 32}).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                result["code"], result["body"] = r.status, json.loads(r.read())

        try:
            with mock.patch.object(sys, "stdout", console):
                t = threading.Thread(target=chat, daemon=True)
                t.start()
                self.assertTrue(console.paused.wait(10), "the request never reached its done line")
                # another status_lock user (_progress, /v1/status, the next request) must get the lock now
                got = svc.status_lock.acquire(timeout=2)
                if got:
                    svc.status_lock.release()
                got_turn = svc.fifo.acquire(timeout=2)
                if got_turn:
                    svc.fifo.release()
                console.resume.set()
                t.join(15)
            self.assertTrue(got, "status_lock stayed held while the done line waited on the console (#2432)")
            self.assertTrue(got_turn, "the completed request kept the engine turn while its console write blocked")
            self.assertFalse(t.is_alive())
            self.assertEqual(result.get("code"), 200)
            out = "".join(console.lines)
            self.assertIn("[strata] done:", out)                  # the line is still printed, once the console resumes
            self.assertFalse(svc.status["busy"])
        finally:
            console.resume.set()
            httpd.shutdown()
            httpd.server_close()


OUTPUT = {"print"}
LOG_ROOTS = {"logging", "LOGGER", "logger", "log"}
LOG_CALLS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}


def _dotted(node):
    if isinstance(node, ast.Attribute):
        return _dotted(node.value) + "." + node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _dotted(node.func) + "()"
    if isinstance(node, ast.Subscript):
        return _dotted(node.value) + "[]"
    return "?"


def _is_lock(expr) -> bool:
    last = _dotted(expr).rsplit(".", 1)[-1].rstrip("()[]").lower()
    return last.endswith(("lock", "_cv", "cond"))


def _outputs(call) -> bool:
    """A call that writes to the console or a logger."""
    name = _dotted(call.func)
    parts = name.split(".")
    if name in OUTPUT:
        return True
    if name.startswith(("sys.stdout.", "sys.stderr.")):
        return True
    return len(parts) == 2 and parts[0] in LOG_ROOTS and parts[1] in LOG_CALLS


class NoOutputUnderALock(unittest.TestCase):
    """Every `with <...lock / ..._cv / ...cond>:` in serve/ has no print, sys.stdout/stderr write or logging call in
    it, nor a call to a function or method of the same module that does one directly."""

    def test_no_console_or_log_output_inside_a_lock(self):
        found = []
        for path in sorted(SERVE.glob("*.py")):
            if path.name.startswith("test_") or path.name == "mcp_fake_server.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            speaks = {f.name for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and any(isinstance(c, ast.Call) and _outputs(c) for c in ast.walk(f))}
            for w in ast.walk(tree):
                if not isinstance(w, ast.With) or not any(_is_lock(i.context_expr) for i in w.items):
                    continue
                for stmt in w.body:
                    for c in ast.walk(stmt):
                        if not isinstance(c, ast.Call):
                            continue
                        callee = _dotted(c.func).rsplit(".", 1)[-1]
                        if _outputs(c) or (callee in speaks and _dotted(c.func).split(".")[0] in ("self", callee)):
                            found.append(f"{path.name}:{c.lineno} {_dotted(c.func)} inside the lock at line {w.lineno}")
        self.assertEqual(found, [], "console/log output while a lock is held (#2432):\n" + "\n".join(found))


class DoneLineSurvivesAnException(unittest.TestCase):
    """An exception inside the status_lock block after the done line is built still prints that line - after the lock
    is released - and still propagates (before #2432 the line was printed at once, so it must not be lost now)."""

    def test_done_line_printed_after_release_when_the_block_raises(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                      ChatTemplate(SERVE / "chat_template.jinja"))
        real_decode, written, raised = tok.decode, [], []

        def decode(ids, *a, **k):
            # with STRATA_DEBUG set, the status_lock block decodes the raw ids right after the done line
            # (server.py, `said.append(f"[strata] raw: ...")`); raise there, while the lock is held
            if svc.status_lock.locked():
                raised.append(1)
                raise RuntimeError("injected")
            return real_decode(ids, *a, **k)

        class Console:
            def write(self, text):
                if "[strata] done:" in text:
                    free = svc.status_lock.acquire(blocking=False)
                    if free:
                        svc.status_lock.release()
                    written.append((text, free))
                return len(text)

            def flush(self):
                pass

        httpd = serve(svc, port=0)
        try:
            with mock.patch.object(sys, "stdout", Console()), mock.patch.dict(os.environ, {"STRATA_DEBUG": "1"}), \
                    mock.patch.object(tok, "decode", side_effect=decode):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                    data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                     "max_tokens": 32}).encode(), headers={"Content-Type": "application/json"})
                try:
                    urllib.request.urlopen(req, timeout=30).read()
                except Exception:
                    pass                                  # the injected error propagates; the request may fail
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertTrue(raised, "the injection point inside status_lock was not reached")
        self.assertEqual(len(written), 1, "the done line was lost when the status_lock block raised")
        self.assertTrue(written[0][1], "the done line was written while status_lock was held")


class McpRestartLineSurvivesAnException(unittest.TestCase):
    def test_restart_line_printed_when_start_raises(self):
        from serve.mcp import McpServer
        srv = McpServer.__new__(McpServer)
        srv.lock, srv.transport, srv.status, srv.name, srv.error, srv.last_start = threading.Lock(), None, "ready", "x", None, 0.0
        out = []

        def boom(said):
            raise RuntimeError("injected")

        srv._start = boom
        with mock.patch("builtins.print", lambda *a, **k: out.append((a[0], srv.lock.locked()))):
            with self.assertRaises(RuntimeError):
                srv.call("t", {}, 1.0)
        self.assertEqual(len(out), 1)
        self.assertIn("had stopped", out[0][0])
        self.assertFalse(out[0][1], "printed while the MCP lock was held")


if __name__ == "__main__":
    unittest.main()
