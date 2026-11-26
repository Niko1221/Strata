"""CLI surface for the opt-in LAN cluster registry (not wired into `strata` yet)."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from strata_cli.cluster_cmd import main, parser
from strata_control.cluster import Registry


class ClusterCommand(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="strata-cluster-cli-"))
        env = patch.dict("os.environ", {"STRATA_HOME": str(self.root)}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.registry = Registry()

    def call(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv), registry=self.registry)
        return code, out.getvalue(), err.getvalue()

    def test_parser_help_lists_all_actions(self):
        text = parser().format_help()
        for action in ("list", "create", "show", "add", "remove", "delete"):
            self.assertIn(action, text)

    def test_create_add_show_remove_delete_flow(self):
        self.assertEqual(self.call("create", "home-lab", "--model", "hy4/SQ-64")[0], 0)
        code, out, _ = self.call("add", "home-lab", "--id", "desk", "--address", "192.168.1.10",
                                 "--vram-gib", "24", "--ram-gib", "64", "--expert-slots", "8192",
                                 "--role", "head")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["nodes"]["desk"]["role"], "head")
        code, out, _ = self.call("show", "home-lab")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["capacities"]["vram_gib"], 24.0)
        self.assertEqual(self.call("remove", "home-lab", "desk")[0], 0)
        self.assertEqual(self.call("delete", "home-lab")[0], 0)
        self.assertEqual(self.registry.names(), [])

    def test_errors_are_reported_without_a_traceback(self):
        code, _, err = self.call("create", "Bad Name")
        self.assertEqual(code, 2)
        self.assertIn("cluster name", err)
        code, _, err = self.call("show", "missing")
        self.assertEqual(code, 2)
        self.assertIn("does not exist", err)

    def test_secret_is_never_printed(self):
        with patch.dict("os.environ", {"STRATA_CLUSTER_TOKEN": "super-secret"}, clear=False):
            self.call("create", "lab", "--token-env", "STRATA_CLUSTER_TOKEN")
            code, out, _ = self.call("show", "lab")
        self.assertEqual(code, 0)
        self.assertNotIn("super-secret", out)


if __name__ == "__main__":
    unittest.main()
