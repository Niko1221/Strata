"""Actual host processes/limits/syscalls; these tests do not certify a GPU gate."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))


class AllocationProbeTests(unittest.TestCase):
    def exercise(self, limited):
        from allocation_probe import HostAllocationProbe

        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / 'host-allocation-child'
            subprocess.run(['c++', '-std=c++17', '-O2',
                            str(ROOT / 'tests/native/host_allocation_child.cpp'),
                            '-o', str(binary)], capture_output=True, check=True)
            probe = HostAllocationProbe(temporary, [str(binary)])
            with tempfile.TemporaryFile(mode='w+t') as stderr:
                child = subprocess.Popen(probe.command, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)

                def request(command):
                    child.stdin.write(command + '\n')
                    child.stdin.flush()
                    return json.loads(child.stdout.readline())

                try:
                    ready = json.loads(child.stdout.readline())
                    probe.bind(child)
                    self.assertEqual(probe.native_pid, ready['pid'])
                    self.assertNotEqual(probe.native_pid, os.getpid())
                    self.assertNotEqual(probe.native_pid, child.pid)
                    before = resource.prlimit(probe.native_pid, resource.RLIMIT_AS)
                    parent_before = resource.getrlimit(resource.RLIMIT_AS)
                    if limited:
                        probe.arm()
                        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS), parent_before)
                        self.assertEqual(resource.prlimit(probe.native_pid, resource.RLIMIT_AS),
                                         tuple(probe.limits['applied']))
                    first = request('allocate')
                    probe.restore_limit()
                    self.assertEqual(resource.prlimit(probe.native_pid, resource.RLIMIT_AS), before)
                    failures = probe.allocation_failures()
                    if limited:
                        self.assertEqual(first['status'], 'bad_alloc')
                        self.assertTrue(failures, 'A raised exception alone is not allocation evidence')
                        self.assertTrue(all(row['errno'] == 'ENOMEM' for row in failures))
                        self.assertTrue(any(row['requested_bytes'] >= 64 * 1024 * 1024 for row in failures))
                    else:
                        self.assertEqual(first['status'], 'allocated')
                        self.assertEqual(failures, [], 'Success must never become an OOM receipt')
                    recovered = request('allocate')
                    self.assertEqual(recovered['status'], 'allocated')
                    self.assertEqual(recovered['bytes'], 64 * 1024 * 1024)
                    self.assertEqual(recovered['last'], 81)
                    child.stdin.write('quit\n')
                    child.stdin.flush()
                    self.assertEqual(child.wait(timeout=10), 0)
                    receipt = probe.receipt()
                    self.assertTrue(receipt['native_process_exited'])
                    self.assertEqual(receipt['tracer_returncode'], 0)
                    self.assertEqual(receipt['allocation_failures'], failures)
                    destination = os.environ.get('CTT_HOST_ALLOCATION_OBSERVATIONS')
                    if destination:
                        with Path(destination).open('a') as output:
                            output.write(json.dumps(dict(test=self.id(), limited=limited,
                                first=first, recovered=recovered, receipt=receipt)) + '\n')
                finally:
                    probe.restore_limit()
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=10)
                    child.stdin.close()
                    child.stdout.close()

    def test_real_child_limit_fails_and_recovers(self):
        self.exercise(True)

    def test_unlimited_child_has_no_allocation_failure(self):
        self.exercise(False)


if __name__ == '__main__':
    unittest.main()
