"""Remote recovery and ownership checks use fake services, never a real public tunnel."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve.remote_access import RemoteMonitor, RuntimeChecks, funnel_ownership

HOST = "test-device.test-tailnet.ts.net:443"
OWNED = {"Web": {HOST: {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8787"}}}},
         "TCP": {"443": {"HTTPS": True}}}
OWNERSHIP_CASES = [
    ({}, "empty"), (None, "empty"), (OWNED, "owned"),
    ({"Web": {HOST: {"Handlers": {"/": {"Proxy": "http://127.0.0.1:9000"}}}}}, "other"),
    ({"Web": {HOST: {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8787"},
                                     "/admin": {"Proxy": "http://127.0.0.1:9000"}}}}}, "other"),
    ({"Web": {HOST: {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8787"},
                                     "/other": {"Proxy": "http://127.0.0.1:8787"}}}}}, "other"),
    ({"TCP": {"443": {"TCPForward": "127.0.0.1:9000"}}}, "other"),
    ({"Web": "invalid"}, "other"), ({"Web": []}, "other"), ({"TCP": []}, "other"),
    ({"TCP": {"443": {}}}, "other"),
    ({"Web": {HOST: {}}}, "other"),
    ({"Web": {HOST.replace(":443", ":8443"): {"Handlers": {"/": {"Proxy": "http://127.0.0.1:9000"}}}}}, "empty"),
    ({"Foreground": {"session": OWNED}}, "other"),
    ({"Foreground": []}, "other"),
]


class FunnelOwnership(unittest.TestCase):
    def test_python_classifier_rejects_foreign_additional_and_malformed_bindings(self):
        for configuration, expected in OWNERSHIP_CASES:
            with self.subTest(configuration=configuration):
                self.assertEqual(funnel_ownership(configuration, 8787), expected)

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell ownership checks require pwsh")
    def test_powershell_classifier_matches_the_same_security_contract(self):
        script = Path(__file__).resolve().parents[1] / "remote-access" / "RemoteCommon.ps1"
        command = "$ErrorActionPreference='Stop'; . '" + str(script).replace("'", "''") + "'; " + \
            "$cases=$env:STRATA_TEST_CASES | ConvertFrom-Json; " + \
            "@($cases | ForEach-Object { Get-StrataFunnelOwnership -Configuration $_.configuration -Port 8787 }) | ConvertTo-Json"
        environment = {**os.environ, "STRATA_TEST_CASES": json.dumps([
            {"configuration": config} for config, _ in OWNERSHIP_CASES])}
        result = subprocess.run([shutil.which("pwsh"), "-NoProfile", "-NonInteractive", "-Command", command],
                                env=environment, capture_output=True, text=True, timeout=20, check=True)
        self.assertEqual(json.loads(result.stdout), [expected for _, expected in OWNERSHIP_CASES])


class RemoteSafety(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def test_additional_tailnet_private_handler_is_not_accepted_as_owned(self):
        (self.root / "gateway.json").write_text(json.dumps({"apiKey": "test-key", "port": 8787}))
        host = "test-device.test-tailnet.ts.net"
        funnel = {"AllowFunnel": {host + ":443": True}, "TCP": {"443": {"HTTPS": True}},
                  "Web": {host + ":443": {"Handlers": {
                      "/": {"Proxy": "http://127.0.0.1:8787"},
                      "/admin": {"Proxy": "http://127.0.0.1:9000"}}}}}
        checks = RuntimeChecks(self.root, self.root)
        checks.tailscale_json = lambda command, *args: {
            "BackendState": "Running", "Self": {"DNSName": host}} if command == "status" else funnel
        with mock.patch("serve.remote_access.http.client.HTTPConnection") as connection:
            connection.return_value.getresponse.return_value = SimpleNamespace(status=405)
            healthy, _ = checks.health()
        self.assertFalse(healthy, "Publishing this port would expose /admin outside the authenticated gateway")

    def monitor(self, health=(True, "ready"), public=(True, "ready")):
        self.at = 0
        self.repairs = []
        self.messages = []
        return RemoteMonitor(self.root, lambda: health, lambda: public,
                             lambda refresh: self.repairs.append(refresh),
                             clock=lambda: self.at, say=self.messages.append)

    def test_disabled_marker_prevents_any_repair(self):
        monitor = self.monitor(health=(False, "unavailable"))
        (self.root / "disabled").touch()
        monitor.tick()
        self.assertEqual(self.repairs, [])

    def test_missing_gateway_repairs_without_resetting_other_tunnels(self):
        self.monitor(health=(False, "unavailable")).tick()
        self.assertEqual(self.repairs, [False])

    def test_two_public_failures_are_required_before_refresh(self):
        monitor = self.monitor(public=(False, "unavailable"))
        monitor.tick()
        self.assertEqual(self.repairs, [])
        self.at = 59
        monitor.tick()
        self.assertEqual(self.repairs, [])
        self.at = 60
        monitor.tick()
        self.assertEqual(self.repairs, [True])

    def test_unknown_dns_does_not_reset_a_configured_tunnel(self):
        monitor = self.monitor(public=(None, "resolver unavailable"))
        for self.at in (0, 60, 120):
            monitor.tick()
        self.assertEqual(self.repairs, [])

    def test_check_exception_is_reported_and_can_be_retried(self):
        monitor = self.monitor()
        monitor.health = mock.Mock(side_effect=OSError("offline"))
        monitor.tick()
        self.assertEqual(self.repairs, [])
        self.assertIn("will retry", self.messages[0])
        monitor.health = lambda: (False, "unavailable")
        monitor.tick()
        self.assertEqual(self.repairs, [False])
