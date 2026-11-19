"""Real constructor/lifecycle and HTTP metrics with controlled child pipes, no GPU."""

import contextlib
import io
import json
from pathlib import Path
import queue
import threading
import time
import unittest
import urllib.request
from unittest import mock

from serve import server
from serve import test_q36_memory_contract as contract


class Lines:
    def __init__(self) -> None:
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self) -> str:
        line = self.queue.get(timeout=10)
        if line is None:
            raise StopIteration
        return line

    def send(self, text: str) -> None:
        for line in text.splitlines():
            self.queue.put(line + "\n")

    def close(self) -> None:
        self.queue.put(None)


class Child:
    def __init__(self, pid: int, text: str = "", defer_quit: bool = False) -> None:
        self.pid, self.returncode = pid, None
        self.quit_requested, self.release_quit = threading.Event(), threading.Event()
        self.stdout = Lines()
        self.stdout.send(text)
        child = self

        class Input(io.StringIO):
            def write(self, text: str) -> int:
                count = super().write(text)
                if "QUIT" in text:
                    child.quit_requested.set()
                    if not defer_quit:
                        child.exit(0)
                return count

        self.stdin = Input()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if (
            self.returncode is None
            and self.quit_requested.is_set()
            and self.release_quit.wait(timeout)
        ):
            self.exit(0)
        if self.returncode is None:
            raise AssertionError("test child must have been stopped before wait")
        return self.returncode

    def exit(self, code: int = 0) -> None:
        self.returncode = code
        self.stdout.close()

    def kill(self) -> None:
        self.exit(137)

    def terminate(self) -> None:
        self.exit(143)


def receipt(cpu: int = 18512896) -> str:
    return (
        contract.RECEIPT.replace("18512896", str(cpu))
        + "INFO engine=fixture q36_private_cache=2\n"
    )


class MemoryObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.children = queue.Queue()
        self.stack.enter_context(mock.patch.object(server, "end_leftover_engines"))
        self.stack.enter_context(
            mock.patch.object(server, "engine_nice", return_value=0)
        )
        self.stack.enter_context(
            mock.patch.object(server, "q36_binary_sha256", return_value=contract.SHA)
        )
        self.stack.enter_context(
            mock.patch.object(
                server,
                "q36_probe_capabilities",
                return_value={
                    "kvCapability": contract.VERIFIED,
                    "memoryCapability": contract.BOUND,
                },
            )
        )
        self.stack.enter_context(
            mock.patch.object(
                server, "popen", side_effect=lambda *a, **k: self.children.get_nowait()
            )
        )
        self.args = [
            *contract.ARGS,
            "--prompt-cache",
            "0",
            "--conversation-cache-mib",
            "0",
            "--private-conversation-cache",
        ]
        self.env = {
            **contract.ENV,
            "STRATA_REQUIRE_Q36_PRIVATE_CACHE": "2",
            "STRATA_Q36_LAUNCH_GENERATION": "host-launch",
        }
        self.engine = server.StrataEngine("fake", self.args, env=self.env, lazy=True)
        self.addCleanup(self.engine.close)
        self.service = server.Service(
            self.engine,
            server.ByteTokenizer(),
            server.ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"),
        )
        self.service.api_key = "memory-observation-test"
        self.service.start_telemetry = mock.Mock()  # no hardware/GPU probes
        self.http = server.serve(self.service, port=0)
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"

    def request(self, path: str, body=None) -> dict:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": "Bearer memory-observation-test",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            self.assertEqual(response.status, 200)
            return json.load(response)

    def start(self, child: Child) -> None:
        self.children.put(child)
        self.engine.restart(tries=1)

    def assert_unknown(self) -> dict:
        info = self.request("/metrics")["engine"]
        self.assertFalse(any(k.startswith("q36_memory_") for k in info), info)
        self.assertFalse(
            any(k.startswith("q36_memory_") for k in self.engine.info), self.engine.info
        )
        return info

    def assert_bound(self, pid: int, cpu: int = 18512896) -> dict:
        info = self.request("/metrics")["engine"]
        child = info["native_child"]
        self.assertTrue(child["admitted"])
        self.assertTrue(child["alive"])
        self.assertEqual(child["process_id"], pid)
        self.assertRegex(child["generation"], r"^[a-f0-9]{32}$")
        self.assertEqual(info["q36_memory_process_id"], pid)
        self.assertEqual(info["q36_memory_generation"], child["generation"])
        self.assertTrue(info["q36_memory_admitted"])
        self.assertEqual(info["q36_memory_cpu_stage_bytes"], cpu)
        return info

    def test_http_unload_keeps_service_and_static_info_but_drops_memory(self) -> None:
        self.start(Child(101, receipt() + "READY 256 stop\n"))
        self.assertEqual(
            self.request("/metrics")["engine"]["q36_memory_mode"], "bounded-experts-v1"
        )
        self.request("/unload", {})
        info = self.assert_unknown()
        self.assertEqual(info["engine"], "fixture")
        self.assertEqual(self.engine.info["engine"], "fixture")
        self.assertEqual(self.request("/metrics")["live"]["state"], "unloaded")

    def test_pre_ready_info_is_private_until_all_admission_succeeds(self) -> None:
        child = Child(102, receipt())
        self.children.put(child)
        errors = []

        def start() -> None:
            try:
                self.engine.restart(tries=1)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=start)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while (
                self.engine.info.get("engine") != "fixture"
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            self.assertEqual(self.engine.info.get("engine"), "fixture")
            self.assert_unknown()
        finally:
            child.stdout.send("READY 256 stop\n")
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assert_bound(102)

    def test_cancel_recovery_failed_restart_then_pid_reuse_gets_only_new_receipt(
        self,
    ) -> None:
        self.start(Child(103, receipt() + "READY 256 stop\n"))
        first = self.request("/metrics")["engine"]
        self.assertTrue(self.service.end_engine_if_unchanged(self.engine.gen))
        self.assert_unknown()
        with self.assertRaisesRegex(server.EngineDied, "memory"):
            self.start(
                Child(104, "INFO engine=fixture q36_private_cache=2\nREADY 256 stop\n")
            )
        self.assert_unknown()
        self.start(Child(103, receipt(12345) + "READY 256 stop\n"))
        current = self.assert_bound(103, 12345)
        self.assertNotEqual(
            first["q36_memory_generation"], current["q36_memory_generation"]
        )
        self.assertEqual(self.engine.info["engine"], "fixture")

    def test_memory_is_not_published_when_private_or_kv_admission_fails(self) -> None:
        for gate in ("private", "kv"):
            with self.subTest(gate=gate):
                text = (
                    receipt()
                    if gate == "kv"
                    else receipt().replace("q36_private_cache=2", "q36_private_cache=1")
                )
                self.engine.spawn = (
                    "fake",
                    self.args,
                    None,
                    None,
                    {
                        **self.env,
                        **(
                            {"STRATA_Q36_KV_HOST_SHA256": contract.SHA}
                            if gate == "kv"
                            else {}
                        ),
                    },
                    False,
                    None,
                )
                with self.assertRaises(server.EngineDied):
                    self.start(Child(105, text + "READY 256 stop\n"))
                self.assert_unknown()

    def test_eof_and_unadmitted_process_replacement_cannot_reuse_memory(self) -> None:
        child = Child(106, receipt() + "READY 256 stop\n")
        self.start(child)
        self.assert_bound(106)
        child.exit()
        self.assert_unknown()
        self.start(Child(107, receipt() + "READY 256 stop\n"))
        admitted = self.engine.proc
        replacement = Child(107)
        try:
            self.engine.proc = replacement
            self.assert_unknown()  # PID equality alone is not child ownership.
        finally:
            self.engine.proc = admitted
            replacement.exit()

    def test_stop_during_ready_cannot_publish_the_cancelled_attempt_late(self) -> None:
        child = Child(108, receipt(), defer_quit=True)
        self.children.put(child)
        errors = []

        def run(operation) -> None:
            try:
                operation()
            except Exception as exc:
                errors.append(exc)

        starter = threading.Thread(
            target=run, args=(lambda: self.engine.restart(tries=1),)
        )
        closer = threading.Thread(target=run, args=(self.engine.close,))
        starter.start()
        try:
            deadline = time.monotonic() + 5
            while (
                self.engine.info.get("engine") != "fixture"
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            self.assertEqual(self.engine.info.get("engine"), "fixture")
            closer.start()
            self.assertTrue(child.quit_requested.wait(5))
            child.stdout.send("READY 256 stop\n")
            starter.join(timeout=5)
            self.assertFalse(starter.is_alive())
            self.assert_unknown()
        finally:
            child.stdout.send("READY 256 stop\n")
            child.release_quit.set()
            starter.join(timeout=5)
            if closer.ident is not None:
                closer.join(timeout=5)
        self.assertEqual(errors, [])
