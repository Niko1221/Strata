"""The container and permissions section of `strata doctor` (#4309), without a container or a GPU."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from strata_cli.cli import main
from strata_control import doctor


class ContainerReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.marker = self.root / "dockerenv"
        self.cgroup = self.root / "cgroup"

    def report(self, **changes):
        options = {"environ": {}, "markers": (), "cgroup_path": str(self.cgroup)}
        options.update(changes)
        return doctor.container_report(self.root, **options)

    def add_runtime(self):
        path = self.root / "python" / "bin" / "python3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")
        return path

    @staticmethod
    def runs(stdout="Python 3.12.15\n", returncode=0):
        return mock.Mock(return_value=mock.Mock(returncode=returncode, stdout=stdout, stderr=""))

    def test_container_hints_come_from_files_environment_and_cgroup(self):
        self.marker.write_text("", encoding="utf-8")
        self.cgroup.write_text("0::/docker/abc\n", encoding="utf-8")
        report = self.report(markers=(str(self.marker),),
                             environ={"container": "podman", "STRATA_COMPONENTS_DIR": "/opt/strata/components"})
        self.assertTrue(report["detected"])
        evidence = " ".join(report["evidence"])
        for hint in (str(self.marker), "container=podman", "cgroup/docker", "STRATA_COMPONENTS_DIR"):
            self.assertIn(hint, evidence)
        self.assertEqual(doctor.container_lines(report), ["Container: yes (" + ", ".join(report["evidence"]) + ")"])

    def test_access_denied_while_the_runtime_starts_names_the_old_docker_profile(self):
        path = self.add_runtime()
        self.marker.write_text("", encoding="utf-8")
        before = sorted(p.relative_to(self.root) for p in self.root.rglob("*"))
        run = self.runs()
        report = self.report(markers=(str(self.marker),), access=lambda path, mode: False, run=run)
        self.assertEqual(sorted(p.relative_to(self.root) for p in self.root.rglob("*")), before)   # read-only
        entry = report["runtimes"][0]
        self.assertEqual((entry["path"], entry["runs"], entry["access_x_ok"], entry["permission_mismatch"]),
                         (str(path), True, False, True))
        self.assertEqual(entry["version"], "Python 3.12.15")
        self.assertEqual(run.call_args.args[0], [str(path), "-V"])
        lines = doctor.container_lines(report)
        self.assertTrue(lines[0].startswith("Container: yes"))
        self.assertEqual(len(lines), 2)                                # the container line and one sentence
        self.assertIn("older than 20.10", lines[1])
        self.assertIn("--security-opt seccomp=unconfined", lines[1])

    def test_a_healthy_runtime_and_one_that_cannot_start_add_no_warning(self):
        self.add_runtime()
        healthy = self.report(access=lambda path, mode: True, run=self.runs())
        self.assertFalse(healthy["detected"])                          # a source checkout is a plain host
        self.assertEqual(doctor.container_lines(healthy), ["Container: no"])
        broken = self.report(access=lambda path, mode: False, run=self.runs("", returncode=1))
        self.assertFalse(broken["runtimes"][0]["runs"])
        self.assertFalse(broken["runtimes"][0]["permission_mismatch"])
        self.assertEqual(doctor.container_lines(broken), ["Container: no"])

    def test_doctor_prints_the_section_and_json_carries_the_report(self):
        runtime = {"path": "/opt/strata/terminal/python/bin/python3", "access_x_ok": False, "runs": True,
                   "version": "Python 3.12.15", "permission_mismatch": True}
        found = {"detected": True, "evidence": ["/.dockerenv"], "runtimes": [runtime]}
        output = io.StringIO()
        with mock.patch.object(doctor, "container_report", return_value=found), redirect_stdout(output):
            self.assertEqual(main(["doctor"]), 0)
        self.assertIn("Container: yes (/.dockerenv)", output.getvalue())
        self.assertIn("--security-opt seccomp=unconfined", output.getvalue())
        output = io.StringIO()
        with mock.patch.object(doctor, "container_report", return_value=found), redirect_stdout(output):
            self.assertEqual(main(["doctor", "--output", "json"]), 0)
        self.assertEqual(json.loads(output.getvalue())["report"]["container"], found)
