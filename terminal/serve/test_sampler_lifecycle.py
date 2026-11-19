import threading
import time
import unittest

from serve.server import Service, serve, ByteTokenizer
from serve.telemetry import Telemetry


class MockEngine:
    def __init__(self, tok, script, max_context=16384):
        self.tok = tok
        self.script = script
        self.max_context = max_context

    def close(self):
        pass


class TestSamplerLifecycle(unittest.TestCase):
    def test_own_sampler_thread_lifecycle(self):
        """Test that the sampler owns its thread reference and joins on close."""
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, "template")

        # Create an unrelated thread with name ending in (_loop)
        dummy_stop = threading.Event()
        dummy_thread = threading.Thread(target=lambda: dummy_stop.wait(5.0), daemon=True, name="unrelated(_loop)")
        dummy_thread.start()

        try:
            httpd = serve(svc, port=0)
            tel = getattr(svc, "telemetry", None)
            self.assertIsNotNone(tel)
            self.assertTrue(hasattr(tel, "_thread"), "Telemetry must store reference to its sampler thread")
            self.assertTrue(tel._thread.is_alive(), "Sampler thread must be running after init")

            httpd.shutdown()
            httpd.server_close()

            # Verify that the server's own sampler thread is joined and not alive
            self.assertFalse(tel._thread.is_alive(), "Sampler thread must be terminated after server_close")
            # Unrelated thread is still alive, proving test isolation
            self.assertTrue(dummy_thread.is_alive(), "Unrelated thread must not affect sampler isolation check")
        finally:
            dummy_stop.set()
            dummy_thread.join(timeout=1.0)

    def test_close_twice_is_idempotent(self):
        """Verify calling close multiple times does not raise or deadlock."""
        tel = Telemetry()
        self.assertTrue(hasattr(tel, "_thread"))
        self.assertTrue(tel._thread.is_alive())
        tel.close()
        self.assertFalse(tel._thread.is_alive())
        # Second close
        tel.close()
        self.assertFalse(tel._thread.is_alive())

    def test_concurrent_services_independent(self):
        """Verify that two independent services manage their own sampler threads independently."""
        tel1 = Telemetry()
        tel2 = Telemetry()
        self.assertNotEqual(tel1._thread.ident, tel2._thread.ident)
        self.assertTrue(tel1._thread.is_alive())
        self.assertTrue(tel2._thread.is_alive())

        tel1.close()
        self.assertFalse(tel1._thread.is_alive())
        self.assertTrue(tel2._thread.is_alive())

        tel2.close()
        self.assertFalse(tel2._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
