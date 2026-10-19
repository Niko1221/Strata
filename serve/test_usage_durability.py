"""Persistence and serving integration, without a GPU. Also run on Windows."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from serve.usage import Usage

ROOT = Path(__file__).resolve().parents[1]


class Durability(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "usage.json"

    def store(self):
        u = Usage(self.path)
        self.addCleanup(u.close)
        return u

    def test_abrupt_exit_after_two_quick_requests_loses_neither(self):
        script = """from serve.usage import Usage
import os,sys
u=Usage(sys.argv[1]);u.record(output_tokens=7);u.record(output_tokens=11);os._exit(0)
"""
        subprocess.run([sys.executable, "-c", script, str(self.path)], cwd=ROOT, check=True, timeout=10)
        u = self.store()
        self.assertEqual(u.lifetime["output_tokens"], 18)
        self.assertEqual(u.lifetime["requests"], 2)

    def test_concurrent_requests_sum_once(self):
        u = self.store()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda n: u.record(output_tokens=n), range(1, 65)))
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["lifetime"]["requests"], 64)
        self.assertEqual(saved["lifetime"]["output_tokens"], sum(range(1, 65)))

    def test_second_process_cannot_overwrite_the_owner(self):
        u = self.store()
        u.record(output_tokens=23)
        script = """from serve.usage import Usage
import sys
u=Usage(sys.argv[1]);assert not u.enabled;assert u.persistence_error;u.record(output_tokens=999);u.close()
"""
        subprocess.run([sys.executable, "-c", script, str(self.path)], cwd=ROOT, check=True, timeout=10,
                       stdout=subprocess.DEVNULL)
        self.assertEqual(json.loads(self.path.read_text())["lifetime"]["output_tokens"], 23)
        u.close()
        again = self.store()
        self.assertTrue(again.enabled)
        again.record(output_tokens=5)
        self.assertEqual(again.lifetime["output_tokens"], 28)

    def test_failed_replace_keeps_disk_and_retry_does_not_double_count(self):
        u = self.store()
        u.record(output_tokens=7)
        before = self.path.read_bytes()
        with mock.patch("serve.usage.os.replace", side_effect=OSError("disk unavailable")):
            u.record(output_tokens=11)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(u.view()["persistence_error"])
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])
        u.flush()
        self.assertEqual(json.loads(self.path.read_text())["lifetime"]["output_tokens"], 18)
        self.assertIsNone(u.view()["persistence_error"])

    def test_restart_keeps_fractional_timings(self):
        u = self.store()
        u.record(prompt_ms=0.125, decode_ms=1.375)
        u.close()
        u = self.store()
        self.assertEqual(u.lifetime["prompt_ms"], 0.125)
        self.assertEqual(u.lifetime["decode_ms"], 1.375)

    def test_corrupt_original_is_preserved(self):
        self.path.write_bytes(b'{"lifetime":')
        u = self.store()
        u.record(output_tokens=4)
        backups = list(self.path.parent.glob("usage.json.corrupt-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b'{"lifetime":')
        self.assertEqual(u.lifetime["output_tokens"], 4)

    def test_undated_imports_survive_without_polluting_rates_or_ranges(self):
        self.path.write_text(json.dumps({"imported": {"requests": 80, "output_tokens": 1000000}}))
        u = self.store()
        u.record(output_tokens=100, decode_ms=1000)
        u.close()
        u = self.store()
        view = u.view()
        self.assertEqual(view["imported"]["output_tokens"], 1000000)
        self.assertEqual(view["lifetime"]["output_tokens"], 100)
        self.assertEqual(view["views"]["1d"]["total"]["output_tokens"], 100)

    def test_wrong_json_shape_is_backed_up_before_replacement(self):
        self.path.write_text('[123, 456]')
        u = self.store()
        u.record(output_tokens=4)
        backups = list(self.path.parent.glob("usage.json.corrupt-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), '[123, 456]')

    def test_nonfinite_custom_ranges_do_not_break_metrics(self):
        u = self.store()
        for value in (float("inf"), float("-inf"), float("nan")):
            self.assertNotIn("custom", u.view(custom=(value, value))["views"])


class Serving(unittest.TestCase):
    def setUp(self):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "usage.json"
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, "A thought.</think>\nDone.", max_context=4096)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.usage = Usage(self.path)
        self.addCleanup(self.svc.usage.close)

    def run_request(self, cancel=None):
        return list(self.svc.run([1, 2, 3], True, [], 128, {}, cancel or threading.Event()))

    def test_settled_usage_matches_session_totals_after_multisession_work(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: self.run_request(), range(9)))
        disk = json.loads(self.path.read_text())
        for key in ("requests", "prompt_tokens", "reused", "output_tokens"):
            self.assertEqual(disk["lifetime"][key], self.svc.totals[key])
        self.assertEqual(disk["lifetime"]["requests"], 9)

    def test_disconnected_generation_is_settled_once_then_can_resume(self):
        gen = self.svc.run([1, 2, 3], True, [], 128, {}, threading.Event())
        for kind, _ in gen:
            if kind == "event":
                break
        gen.close()
        self.assertEqual(self.svc.totals["requests"], 1)
        self.assertGreater(self.svc.totals["output_tokens"], 0)
        self.run_request()
        saved = json.loads(self.path.read_text())["lifetime"]
        self.assertEqual(saved["requests"], 2)
        self.assertEqual(saved["output_tokens"], self.svc.totals["output_tokens"])

    def test_tool_call_tokens_are_counted_without_saving_the_arguments(self):
        from serve.server import MockEngine
        secret = "private-argument-value"
        self.svc.engine = MockEngine(self.tok, '</think>\n<tool_call><function=f><parameter=value>'+secret+'</parameter></function></tool_call>', max_context=4096)
        events = list(self.svc.run([1, 2, 3], True, [{"name": "f", "parameters": {"type": "object"}}],
                                   256, {}, threading.Event()))
        self.assertTrue(any(kind == "event" and event.kind == "tool_call" for kind, event in events))
        raw = self.path.read_text()
        self.assertNotIn(secret, raw)
        self.assertEqual(json.loads(raw)["lifetime"]["output_tokens"], self.svc.totals["output_tokens"])

    def test_disabled_path_does_not_add_a_metrics_field(self):
        self.svc.usage.close()
        self.svc.usage = Usage(None)
        self.assertNotIn("usage", self.svc.metrics())
        self.run_request()
        self.assertEqual(self.svc.usage.lifetime["requests"], 0)


class ServerOwnership(unittest.TestCase):
    """Embedded beta server lifetimes must release the ledger without process exit."""

    def test_main_releases_store_after_stop_and_startup_failure(self):
        from serve import server as S
        from contextlib import ExitStack
        import io
        for failure in ("none", "ready", "bind"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                cfg = Path(folder) / "config.json"
                ledger = Path(folder) / "usage.json"
                cfg.write_text(json.dumps({"usage_stats": True, "usage_stats_path": str(ledger)}))
                seen = []
                httpd = mock.Mock(server_address=("127.0.0.1", 12345))

                def start(svc, **kwargs):
                    seen.append(svc.usage)
                    stack.callback(svc.usage.close)
                    svc.usage.record(output_tokens=7)
                    if failure == "bind":
                        raise OSError("test bind failure")
                    return httpd

                stack.enter_context(mock.patch.object(S, "serve", side_effect=start))
                stack.enter_context(mock.patch.object(S.lifeline, "install_stop_handlers"))
                stack.enter_context(mock.patch.object(S.usage, "install_atexit"))
                stack.enter_context(mock.patch.object(S.time, "sleep", side_effect=KeyboardInterrupt))
                stack.enter_context(mock.patch.object(sys, "stdout", io.StringIO()))
                stack.enter_context(mock.patch.object(sys, "stderr", io.StringIO()))
                if failure == "ready":
                    stack.enter_context(mock.patch.object(S.ready_file, "publish", side_effect=OSError("test ready failure")))
                args = ["--engine", "mock", "--port", "0", "--config", str(cfg)]
                if failure == "ready":
                    args += ["--ready-file", str(Path(folder) / "ready.json")]
                if failure == "bind":
                    with self.assertRaisesRegex(OSError, "test bind failure"):
                        S.main(args)
                else:
                    self.assertEqual(S.main(args), 6 if failure == "ready" else 0)
                again = Usage(ledger)
                stack.callback(again.close)
                self.assertTrue(again.enabled, failure)
                self.assertEqual(again.lifetime["output_tokens"], 7)


if __name__ == "__main__":
    unittest.main()
