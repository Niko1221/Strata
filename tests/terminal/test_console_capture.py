"""Planning notes are available before the CLI returns, without a diagnostic drain."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest import mock

from strata_cli import cli


class PlanningConsoleCapture(unittest.TestCase):
    def test_real_engine_args_notes_are_captured_synchronously(self):
        cfg = {"exe": "strata", "args": ["--mtp"], "mtp": False,
               "vision": {"gpu": True}, "parallel": 99, "pool_cores": "invalid"}
        outside = io.StringIO()
        with mock.patch.object(cli, "_source", return_value=(Path("run.json"), None, None, cfg)), \
                mock.patch("serve.server.drafting_card_class", return_value="other"), \
                mock.patch.dict("os.environ", {"STRATA_BATCH_MTP": "0"}), \
                contextlib.redirect_stdout(outside):
            plan, _ = cli._launch_plan("fixture")
        notes = "\n".join(plan["notes"])
        for expected in ('"mtp": false', 'added --vision', '"parallel": 99', '"pool_cores"'):
            self.assertIn(expected, notes)
        self.assertEqual(outside.getvalue(), "")
        self.assertNotIn("--mtp", plan["engine_argv"])
        self.assertIn("--vision", plan["engine_argv"])
