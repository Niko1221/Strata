"""Error reporting is opt-in and inert without a DSN, and it survives a missing package.
What is reported is the propagating exception at the handler boundary; that boundary is the
whole coverage, so these tests hold it: no DSN does nothing, a missing package says one line
and the server runs, and a client hanging up is not a failure."""

from __future__ import annotations

import io
import sys
import types
import unittest
from unittest import mock

from serve import error_reporting


class FakeSdk(types.ModuleType):
    def __init__(self):
        super().__init__("sentry_sdk")
        self.inits = []
        self.captured = []

    def init(self, **kwargs):
        self.inits.append(kwargs)

    def capture_exception(self, exc):
        self.captured.append(exc)


class ErrorReportingTests(unittest.TestCase):
    def reset(self):
        error_reporting._sdk = None
        error_reporting.active = False

    def test_inert_without_a_dsn(self):
        self.reset()
        with mock.patch.dict(sys.modules, {"sentry_sdk": None}), \
                mock.patch.dict(error_reporting.os.environ, {}, clear=True):
            self.assertFalse(error_reporting.init())
            self.assertFalse(error_reporting.active)

    def test_the_dsn_comes_from_the_config_too(self):
        self.reset()
        with mock.patch.dict(error_reporting.os.environ, {}, clear=True):
            self.assertEqual(error_reporting.dsn({"error_reporting": {"dsn": "https://k@host/1"}}),
                             "https://k@host/1")

    def test_a_missing_package_says_one_line_and_does_not_stop_a_start(self):
        self.reset()
        with mock.patch.dict(sys.modules, {"sentry_sdk": None}), \
                mock.patch.dict(error_reporting.os.environ, {"STRATA_GLITCHTIP_DSN": "https://k@host/1"}):
            err = io.StringIO()
            with mock.patch.object(error_reporting.sys, "stderr", err):
                self.assertFalse(error_reporting.init())
            self.assertIn("sentry-sdk is not installed", err.getvalue())

    def test_active_when_the_package_is_there_and_the_environment_and_release_are_taken(self):
        self.reset()
        fake = FakeSdk()
        with mock.patch.dict(sys.modules, {"sentry_sdk": fake}), mock.patch.dict(
                error_reporting.os.environ, {"STRATA_GLITCHTIP_DSN": "https://k@host/1",
                                             "STRATA_GLITCHTIP_ENV": "strata-general",
                                             "STRATA_RELEASE": "v0.2.3.1-engine-0.1.48.1"}):
            self.assertTrue(error_reporting.init())
        self.assertEqual(fake.inits[0]["environment"], "strata-general")
        self.assertEqual(fake.inits[0]["release"], "v0.2.3.1-engine-0.1.48.1")
        self.assertFalse(fake.inits[0]["send_default_pii"])

    def test_capture_needs_an_active_client(self):
        self.reset()
        self.assertFalse(error_reporting.capture(RuntimeError("x")))

    def test_a_client_hanging_up_is_not_reported(self):
        self.reset()
        fake = FakeSdk()
        error_reporting._sdk = fake
        error_reporting.active = True
        self.assertFalse(error_reporting.capture(ConnectionResetError("the client went away")))
        self.assertEqual(fake.captured, [])
        self.assertTrue(error_reporting.capture(RuntimeError("a real failure")))
        self.assertEqual(len(fake.captured), 1)


if __name__ == "__main__":
    unittest.main()
