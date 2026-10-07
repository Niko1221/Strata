from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from serve.cpu_affinity import partition_cpu_sets


REPOSITORY = Path(__file__).resolve().parents[1]


class CpuAffinity(unittest.TestCase):
    def topology(self, root: Path, packages: int, cores: int) -> set[int]:
        total = packages * cores
        for cpu in range(total * 2):
            physical = cpu % total
            directory = root / "devices/system/cpu" / f"cpu{cpu}" / "topology"
            directory.mkdir(parents=True)
            (directory / "physical_package_id").write_text(str(physical // cores))
            (directory / "core_id").write_text(str(physical % cores))
        return set(range(total * 2))

    def test_four_instances_on_eighteen_smt_cores(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY, prefix=".cpu-affinity-test-") as directory:
            root = Path(directory)
            allowed = self.topology(root, 1, 18)
            groups = partition_cpu_sets(4, allowed, sys_root=root)
        self.assertEqual([len(group) for group in groups], [10, 10, 8, 8])
        self.assertEqual(set().union(*map(set, groups)), allowed)
        for left, group in enumerate(groups):
            for cpu in group:
                self.assertIn((cpu + 18) % 36, group)
            for other in groups[left + 1:]:
                self.assertTrue(set(group).isdisjoint(other))

    def test_restricted_mask_never_adds_a_sibling(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY, prefix=".cpu-affinity-test-") as directory:
            root = Path(directory)
            self.topology(root, 1, 4)
            allowed = {1, 2, 5, 7}
            groups = partition_cpu_sets(2, allowed, sys_root=root)
        self.assertEqual(groups, [[1, 2, 5], [7]])
        self.assertEqual(set().union(*map(set, groups)), allowed)

    def test_socket_ids_distinguish_identical_core_ids(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY, prefix=".cpu-affinity-test-") as directory:
            root = Path(directory)
            allowed = self.topology(root, 2, 2)
            groups = partition_cpu_sets(2, allowed, sys_root=root)
        self.assertEqual(groups, [[0, 1, 4, 5], [2, 3, 6, 7]])

    def test_reserved_sibling_excludes_the_whole_physical_core(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY, prefix=".cpu-affinity-test-") as directory:
            root = Path(directory)
            allowed = self.topology(root, 1, 4)
            groups = partition_cpu_sets(2, allowed, {0}, sys_root=root)
        self.assertEqual(groups, [[1, 2, 5, 6], [3, 7]])

    def test_single_instance_keeps_full_existing_affinity(self):
        self.assertEqual(partition_cpu_sets(1, {0, 2, 9}), [[0, 2, 9]])

    def test_not_enough_physical_cores_and_missing_topology_fail(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY, prefix=".cpu-affinity-test-") as directory:
            root = Path(directory)
            allowed = self.topology(root, 1, 2)
            with self.assertRaisesRegex(ValueError, "only 2"):
                partition_cpu_sets(3, allowed, sys_root=root)
            (root / "devices/system/cpu/cpu0/topology/core_id").unlink()
            with self.assertRaisesRegex(ValueError, "topology for CPU 0"):
                partition_cpu_sets(2, allowed, sys_root=root)
        with self.assertRaises(ValueError):
            partition_cpu_sets(0, {0})
        with self.assertRaises(ValueError):
            partition_cpu_sets(1, set())


@unittest.skipUnless(hasattr(os, "sched_getaffinity") and shutil.which("taskset") and shutil.which("flock"),
                     "Linux affinity tools are required")
class LauncherAffinity(unittest.TestCase):
    def stop_fixture_server(self, pid: int):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                status = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            except FileNotFoundError:
                return
            if status == "Z":
                return
            time.sleep(0.05)
        os.kill(pid, signal.SIGKILL)
        self.fail(f"fixture server PID {pid} did not stop after SIGTERM")

    def run_launcher(self, gpu_list: str, locked: bool = False,
                     reserved: set[int] | None = None) -> dict[int, set[int]]:
        with tempfile.TemporaryDirectory(dir=REPOSITORY, prefix=".cpu-affinity-test-") as directory:
            root = Path(directory)
            (root / ".venv/bin").mkdir(parents=True)
            (root / ".venv/bin/python").symlink_to(sys.executable)
            (root / "serve").mkdir()
            shutil.copy2(REPOSITORY / "setup_multigpu.sh", root)
            shutil.copy2(REPOSITORY / "serve/cpu_affinity.py", root / "serve")
            (root / "serve/server.py").write_text(
                "import argparse, json, os, pathlib, time\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('--config')\n"
                "a, _ = p.parse_known_args()\n"
                "pathlib.Path(a.config + '.affinity').write_text(json.dumps(sorted(os.sched_getaffinity(0))))\n"
                "time.sleep(60)\n"
            )
            binary = root / "bin"
            binary.mkdir()
            for name, body in {
                "nvidia-smi": "exit 0",
                "sudo": 'exec "$@"',
                "ss": "exit 0",
                "curl": "exit 0",
                "tail": "exit 0",
            }.items():
                path = binary / name
                path.write_text("#!/bin/sh\n" + body + "\n")
                path.chmod(0o755)
            gpus = [int(gpu) for gpu in gpu_list.split(",")]
            for gpu in gpus:
                (root / f"strata-iq3_s_gpu{gpu}.json").write_text("{}")
            environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                               STRATA_API_KEY="test-key", STRATA_API_KEY_FILE=str(root / "api-key"))
            lock = None
            existing = None
            try:
                if reserved:
                    config = root / "strata-iq3_s_gpu0.json"
                    config.write_text("{}")
                    existing = subprocess.Popen(
                        ["taskset", "--cpu-list", ",".join(map(str, sorted(reserved))), sys.executable,
                         str(root / "serve/server.py"), "--config", str(config)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    (root / "strata-iq3_s_gpu0.pid").write_text(str(existing.pid))
                    deadline = time.monotonic() + 5
                    while not Path(str(config) + ".affinity").exists() and time.monotonic() < deadline:
                        time.sleep(0.05)
                if locked:
                    import fcntl
                    lock = (root / ".strata-multigpu.lock").open("w")
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if existing:
                    self.assertIsNone(existing.poll())
                    self.assertTrue(Path(str(config) + ".affinity").exists())
                result = subprocess.run(
                    ["bash", str(root / "setup_multigpu.sh"), "--gpus", gpu_list, "--host", "127.0.0.1"],
                    env=environment, capture_output=True, text=True, timeout=20,
                )
                if locked:
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn("Another Strata launcher", result.stderr)
                    self.assertFalse(any(root.glob("*.pid")))
                    self.assertFalse((root / "api-key").exists())
                    return {}
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if existing:
                    self.assertIsNone(existing.poll(), "unselected managed server was stopped")
                paths = {gpu: root / f"strata-iq3_s_gpu{gpu}.json.affinity" for gpu in gpus}
                deadline = time.monotonic() + 5
                while not all(path.exists() for path in paths.values()) and time.monotonic() < deadline:
                    time.sleep(0.05)
                return {gpu: set(json.loads(path.read_text())) for gpu, path in paths.items()}
            finally:
                if lock:
                    lock.close()
                if existing:
                    existing.terminate()
                    existing.wait(timeout=5)
                for gpu in gpus:
                    pid_file = root / f"strata-iq3_s_gpu{gpu}.pid"
                    if pid_file.exists():
                        self.stop_fixture_server(int(pid_file.read_text()))

    def test_multiple_servers_inherit_disjoint_cpu_sets(self):
        if len(os.sched_getaffinity(0)) < 4:
            self.skipTest("not enough allowed CPUs for two physical cores")
        groups = self.run_launcher("0,3")
        self.assertTrue(groups[0].isdisjoint(groups[3]))
        self.assertEqual(groups[0] | groups[3], os.sched_getaffinity(0))

    def test_single_server_keeps_the_full_cpu_set(self):
        self.assertEqual(self.run_launcher("3")[3], os.sched_getaffinity(0))

    def test_concurrent_launcher_fails_before_changing_servers(self):
        self.run_launcher("3", locked=True)

    def test_unselected_managed_server_keeps_its_physical_cores(self):
        allowed = os.sched_getaffinity(0)
        try:
            occupied, remaining = partition_cpu_sets(2, allowed)
        except ValueError as error:
            self.skipTest(str(error))
        groups = self.run_launcher("3", reserved=set(occupied))
        self.assertEqual(groups[3], set(remaining))


if __name__ == "__main__":
    unittest.main()
