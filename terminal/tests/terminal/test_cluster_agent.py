"""Authenticated control-plane agent (groundwork, loopback only)."""
from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request

from strata_control.cluster import Agent


class ClusterAgent(unittest.TestCase):
    def test_ping_requires_the_shared_token(self):
        agent = Agent("secret-token")
        port = agent.start()
        try:
            with self.assertRaises(urllib.error.HTTPError) as denied:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/cluster/ping", timeout=2)
            self.assertEqual(denied.exception.code, 401)
            request = urllib.request.Request(f"http://127.0.0.1:{port}/cluster/ping",
                                             headers={"X-Cluster-Token": "secret-token"})
            with urllib.request.urlopen(request, timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(json.loads(response.read())["ok"])
        finally:
            agent.stop()

    def test_info_reports_real_local_resources(self):
        agent = Agent("token")
        port = agent.start()
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{port}/cluster/info",
                                             headers={"X-Cluster-Token": "token"})
            with urllib.request.urlopen(request, timeout=2) as response:
                info = json.loads(response.read())
            self.assertTrue(info["ok"])
            self.assertTrue(info["hostname"])
            self.assertTrue(info["os"])
            self.assertGreaterEqual(info["ram_gib"], 0)
            self.assertGreaterEqual(info["cpu_cores"], 1)
            self.assertIn(info["status"], ("healthy", "degraded"))
        finally:
            agent.stop()

    def test_unknown_path_is_not_found(self):
        agent = Agent("token")
        port = agent.start()
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{port}/other",
                                             headers={"X-Cluster-Token": "token"})
            with self.assertRaises(urllib.error.HTTPError) as missing:
                urllib.request.urlopen(request, timeout=2)
            self.assertEqual(missing.exception.code, 404)
        finally:
            agent.stop()


if __name__ == "__main__":
    unittest.main()
