"""All request exit paths finish while diagnostic output remains blocked."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve import console as console_module
from serve.test_console_support import blocked_console
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, EngineSilent, MockEngine, Service

SERVE = Path(__file__).parent


class PausedConsole:
    """A stdout that, from the first "[strata] done:" line on, blocks every write until `resume` is set."""

    def __init__(self, trigger="[strata] done:"):
        self.trigger = trigger
        self.lines, self.paused, self.resume = [], threading.Event(), threading.Event()
        self.lock = threading.Lock()

    def write(self, text):
        if self.trigger in text:
            self.paused.set()
        if self.paused.is_set():
            self.resume.wait(30)
        with self.lock:
            self.lines.append(text)
        return len(text)


class PausedCompletionHandover(unittest.TestCase):
    def test_next_request_enters_engine_after_each_exit(self):
        for exit_mode in ("complete", "cancel", "disconnect", "error", "value-error", "engine-died", "close-error"):
            with self.subTest(exit_mode=exit_mode):
                tok = ByteTokenizer()
                entered, closed = threading.Event(), threading.Event()
                cancel = threading.Event()
                errors = []
                output = []
                temporary = tempfile.TemporaryDirectory()
                embedding = Path(temporary.name) / "request.bin"
                embedding.write_bytes(b"fixture")

                class Engine(MockEngine):
                    calls = 0

                    def generate(self, *args, **kwargs):
                        self.calls += 1
                        first = self.calls == 1
                        if not first:
                            entered.set()
                        inner = super().generate(*args, **kwargs)

                        class Stream:
                            count = 0

                            def __iter__(self):
                                return self

                            def __next__(self):
                                self.count += 1
                                if first and self.count == 9:
                                    if exit_mode in ("error", "value-error", "engine-died"):
                                        kind = {"error": RuntimeError, "value-error": ValueError,
                                                "engine-died": EngineDied}[exit_mode]
                                        raise kind("injected engine failure")
                                    if exit_mode == "cancel":
                                        cancel.set()
                                        raise StopIteration
                                return next(inner)

                            def close(self):
                                inner.close()
                                if first:
                                    closed.set()
                                    if exit_mode == "close-error":
                                        raise EngineSilent("injected engine failure")

                        return Stream()

                svc = Service(Engine(tok, "</think>\n\n" + "answer " * 8, max_context=4096), tok,
                              ChatTemplate(SERVE / "chat_template.jinja"))
                console = PausedConsole("injected engine failure" if exit_mode in
                                        ("value-error", "engine-died", "close-error") else "[strata] done:")

                def first():
                    try:
                        svc.embeddings.path = str(embedding)
                        stream = svc.run(tok.encode("first"), True, None, 32, {}, cancel)
                        if exit_mode == "disconnect":
                            for _ in range(5):
                                next(stream)
                            stream.close()
                        else:
                            list(stream)
                    except Exception as error:
                        errors.append(error)

                def second():
                    try:
                        output.extend(svc.run(tok.encode("second"), True, None, 32, {}, threading.Event()))
                    except Exception as error:
                        errors.append(error)

                one = threading.Thread(target=first, daemon=True)
                two = threading.Thread(target=second, daemon=True)
                output_writer = console_module.Console()
                with blocked_console(output_writer, console.write):
                    one.start()
                    try:
                        self.assertTrue(console.paused.wait(3), "first request did not reach completion output")
                        one.join(2)
                        self.assertFalse(one.is_alive(), "diagnostics blocked request cleanup")
                        self.assertTrue(closed.is_set(), "diagnostic output delayed engine close")
                        self.assertFalse(embedding.exists(), "diagnostic output delayed embedding cleanup")
                        self.assertFalse(svc.terminal_requests)
                        two.start()
                        self.assertTrue(entered.wait(2), "console output prevented the next engine turn")
                        one.join(2)
                        two.join(2)
                        self.assertFalse(one.is_alive(), "diagnostics blocked first request completion")
                        self.assertFalse(two.is_alive(), "diagnostics blocked second request completion")
                    finally:
                        console.resume.set()
                        one.join(3)
                        if two.ident is not None:
                            two.join(3)
                        output_writer.close()
                        temporary.cleanup()
                self.assertFalse(one.is_alive())
                self.assertFalse(two.is_alive())
                self.assertEqual([str(e) for e in errors],
                                 ["injected engine failure"] if exit_mode in
                                 ("error", "value-error", "engine-died", "close-error") else [])
                self.assertEqual(output[-1][0], "done")
                self.assertEqual(svc.status["queued"], 0)
                self.assertFalse(svc.status["busy"])
                self.assertFalse(svc.terminal_requests)
