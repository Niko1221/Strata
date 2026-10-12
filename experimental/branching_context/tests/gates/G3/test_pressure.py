"""T32: actual device exhaustion must leave committed futures recoverable."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from context_time_travel.store import IntegrityError
import test_immutability as isolation


class PressureTests(unittest.TestCase):
    setUpClass = classmethod(isolation.SharedPageTests.setUpClass.__func__)
    setUp = isolation.SharedPageTests.setUp
    record = isolation.SharedPageTests.record
    fresh = isolation.SharedPageTests.fresh
    retain = isolation.SharedPageTests.retain
    unchanged = isolation.SharedPageTests.unchanged

    def files(self):
        return {p.relative_to(self.tmp.name).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(self.tmp.name).rglob('*') if p.is_file()}

    def gpu_processes(self):
        raw = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True)
        rows = [line.split(',', 1) for line in raw.splitlines() if line.strip()]
        return {int(pid.strip()): memory.strip() for pid, memory in rows}

    def test_g3_009_memory_pressure_recovery(self):
        counters = self.backend._refresh_stats()
        required = {'vmm_driver_failures', 'vmm_out_of_memory_failures',
                    'vmm_last_driver_error', 'vmm_last_failed_operation'}
        self.assertTrue(required <= counters.keys(),
                        'Memory pressure requires direct CUDA driver failure instrumentation')
        helper = Path(self.profile['native_build']['pressure_helper'])
        self.assertEqual(hashlib.sha256(helper.read_bytes()).hexdigest(),
                         self.profile['native_build']['sha256'][str(helper)])
        self.assertEqual(set(self.gpu_processes()), {self.backend.process.pid},
                         'Exhaustion must run with an exclusive device, after other GPU jobs exit')

        history = tuple(self.profile['fixtures']['prefix'][:9])
        root = self.sut.create(history)
        retained = [self.retain(root, self.fresh(root, history))]
        tokens = [self.profile['fixtures']['suffix_a'][0], self.profile['fixtures']['suffix_b'][0]]
        self.assertNotEqual(*tokens)
        starts, tips, heads = [], [], []
        for branch, token in enumerate(tokens):
            name = 'pressure-' + str(branch)
            start = self.sut.start_generation(root, dict(allowed=[token]), name)
            starts.append(start)
            head = self.sut.store.fork(name, start)
            result = self.sut.generate(start, 1)
            self.assertEqual(result.accepted_tokens, (token,))
            self.assertEqual(result.boundaries, (result.snapshot_id,))
            tip = result.snapshot_id
            tips.append(tip)
            heads.append(self.sut.store.move(name, head.revision, tip, name + '-publish'))
            retained.append(self.retain(tip, self.fresh(tip, history + (token,))))
        controllers = {tip: self.sut.controller(tip) for tip in tips}
        self.assertEqual([controllers[tip]['next_draw'] for tip in tips], [1, 1])
        logical = {sid: self.sut.snapshot_bytes(sid) for sid in [root, *starts, *tips]}
        files = self.files()
        emitted = []

        # Exhaust real physical allocations in an independent native process.
        # No production allocation override or simulated GPU is involved.
        with tempfile.TemporaryFile(mode='w+t') as errors:
            pressure = subprocess.Popen([str(helper)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=errors, text=True, bufsize=1)

            def read():
                line = pressure.stdout.readline()
                self.assertTrue(line, 'The actual CUDA pressure helper must remain alive')
                row = json.loads(line)
                self.record(kind='pressure-helper', pid=pressure.pid, response=row)
                return row

            def request(command):
                pressure.stdin.write(command + '\n')
                pressure.stdin.flush()
                return read()

            try:
                ready = read()
                self.assertEqual(ready['status'], 'ready')
                processes = self.gpu_processes()
                self.record(kind='exclusive-pressure-processes', processes=processes)
                self.assertEqual(set(processes), {self.backend.process.pid, pressure.pid})
                full = request('fill')
                self.assertEqual(full['status'], 'full')
                self.assertEqual(full['last_error'], 2)  # CUDA_ERROR_OUT_OF_MEMORY in the pinned headers.
                self.assertEqual(full['last_operation'], 'cuMemCreate')
                self.assertGreater(full['out_of_memory_failures'], 0)
                self.assertGreater(full['held_bytes'], 0)
                self.assertEqual(full['physical_allocations'] * ready['granularity'], full['held_bytes'])
                before = self.backend._refresh_stats()
                with self.assertRaises((MemoryError, IntegrityError)) as failed:
                    emitted.extend(self.sut.stream(tips[0], 1))
                after = self.backend._refresh_stats()
                self.record(kind='allocation-refusal', before=before, after=after,
                            error=repr(failed.exception), emitted=emitted)
                self.assertGreater(after['vmm_out_of_memory_failures'], before['vmm_out_of_memory_failures'],
                                   'The SUT itself must observe a real CUDA out-of-memory response')
                self.assertGreater(after['vmm_driver_failures'], before['vmm_driver_failures'])
                self.assertEqual(after['vmm_last_driver_error'], 2)
                self.assertTrue(after['vmm_last_failed_operation'].startswith('cuMem'))
                self.assertEqual(after['forward_tokens'], before['forward_tokens'])
                self.assertEqual(after['captures'], before['captures'])
                self.assertEqual(emitted, [], 'An unconsumed token cannot escape as committed output')
                self.assertEqual(self.files(), files, 'Allocation refusal cannot publish a canonical object')
                for tip in tips:
                    self.assertEqual(self.sut.controller(tip), controllers[tip])
                for head in heads:
                    self.assertEqual(self.sut.store.head(head.name), head)
                released = request('release')
                self.assertEqual(released['status'], 'released')
                self.assertEqual(released['released_bytes'], full['held_bytes'])
                self.assertEqual(pressure.wait(), 0)
            finally:
                # EOF releases the helper's handles even when an assertion or
                # native request fails. Do not leave pressure on a shared host.
                pressure.stdin.close()
                pressure.wait()
                pressure.stdout.close()
                errors.seek(0)
                self.record(kind='pressure-helper-exit', pid=pressure.pid,
                            exit_code=pressure.returncode, stderr=errors.read())

        self.unchanged(retained)
        self.fresh(root, history)
        for branch, tip in enumerate(tips):
            self.fresh(tip, history + (tokens[branch],))
            before = self.backend._refresh_stats()
            result = self.sut.generate(tip, 1)
            after = self.backend._refresh_stats()
            self.record(kind='post-pressure-retry', branch=branch, before=before, after=after)
            self.assertEqual(result.accepted_tokens, (tokens[branch],))
            self.assertEqual(result.boundaries, (result.snapshot_id,))
            self.assertEqual(self.sut.controller(result.snapshot_id)['next_draw'], 2)
            self.assertEqual(after['forward_tokens'] - before['forward_tokens'], 1)
            self.fresh(result.snapshot_id, history + (tokens[branch], tokens[branch]))
            self.unchanged(retained)
        for sid, original in logical.items():
            self.assertEqual(self.sut.snapshot_bytes(sid), original)
        for head in heads:
            self.assertEqual(self.sut.store.head(head.name), head)


if __name__ == '__main__':
    unittest.main()
