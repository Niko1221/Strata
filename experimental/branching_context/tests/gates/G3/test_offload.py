"""T32: actual host bytes and an SSD checkpoint must preserve continuation."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import test_immutability as isolation


class OffloadTests(unittest.TestCase):
    setUpClass = classmethod(isolation.SharedPageTests.setUpClass.__func__)
    setUp = isolation.SharedPageTests.setUp
    record = isolation.SharedPageTests.record
    fresh = isolation.SharedPageTests.fresh
    retain = isolation.SharedPageTests.retain
    unchanged = isolation.SharedPageTests.unchanged

    def assert_ssd_store(self):
        mount = json.loads(subprocess.check_output(
            ['findmnt', '--json', '--target', self.tmp.name,
             '--output', 'TARGET,SOURCE,FSTYPE,MAJ:MIN'], text=True))
        devices = json.loads(subprocess.check_output(
            ['lsblk', '--json', '--paths', '--output', 'NAME,TYPE,ROTA,MAJ:MIN'], text=True))
        self.assertEqual(len(mount['filesystems']), 1)
        volume = mount['filesystems'][0]
        paths = []

        def visit(device, parents):
            current = parents + [device]
            if device['maj:min'] == volume['maj:min']:
                paths.append(current)
            for child in device.get('children', []):
                visit(child, current)

        for device in devices['blockdevices']:
            visit(device, [])
        disks = [device for path in paths for device in path if device['type'] == 'disk']
        self.record(kind='offload-filesystem', store=self.tmp.name, mount=mount, block_devices=devices)
        self.assertTrue(disks, 'An SSD test must resolve to a real block device, not tmpfs')
        self.assertTrue(all(device['rota'] is False for device in disks),
                        'The offload store must be backed by nonrotating storage')

    def compare(self, actual, history, phase):
        expected = self.oracle.forward(history)
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(len(actual), self.backend.vocab_size)
        self.record(kind='offload-numerics', phase=phase, history=history,
                    max_absolute_error=max(abs(a - b) for a, b in zip(actual, expected)),
                    actual_sha256=hashlib.sha256(struct.pack('<' + str(len(actual)) + 'f', *actual)).hexdigest(),
                    expected_sha256=self.oracle.last['logits_sha256'],
                    oracle_computed_tokens=self.oracle.last['computed_tokens'])
        self.assertEqual(tuple(actual), expected, 'Cold-restored continuation must match the fresh oracle')
        self.assertEqual(self.oracle.last['computed_tokens'], len(history))
        self.assertEqual(self.oracle.last['position_max'], len(history) - 1)

    def test_g3_008_cpu_ram_ssd_roundtrip(self):
        self.assert_ssd_store()
        self.assertTrue(callable(getattr(self.backend, 'audit', None)))
        history = tuple(self.profile['fixtures']['prefix'][:9])
        root = self.sut.create(history)
        kept = [self.retain(root, self.fresh(root, history))]
        tokens = [self.profile['fixtures']['suffix_a'][0],
                  self.profile['fixtures']['suffix_b'][0]]
        self.assertNotEqual(*tokens)
        tips, histories, frontiers, heads = [], [], [], []
        for branch, token in enumerate(tokens):
            name = 'offload-' + str(branch)
            start = self.sut.start_generation(root, dict(allowed=[token]), name)
            result = self.sut.generate(start, 1)
            self.assertEqual(result.accepted_tokens, (token,))
            self.assertEqual(result.boundaries, (result.snapshot_id,))
            tip = result.snapshot_id
            tips.append(tip)
            histories.append(history + (token,))
            frontiers.append(self.fresh(tip, histories[-1]))
            kept.append(self.retain(tip, frontiers[-1]))
            heads.append(self.sut.store.fork(name, tip))
        controllers = [self.sut.controller(tip) for tip in tips]
        self.assertEqual([value['next_draw'] for value in controllers], [1, 1])
        evidence = Path(os.environ.get('CTT_SHARING_MEASUREMENTS',
                                       'evidence/runs/sharing-measurements.jsonl')).parent
        workers = Path(tempfile.mkdtemp(prefix='g3-008-workers-', dir=evidence))
        worker = Path(__file__).resolve().parents[3] / 'scripts/recovery_worker.py'

        for branch, tip in enumerate(tips):
            before = self.backend._refresh_stats()
            with self.sut.materialize(tip) as (lease, frontier):
                host_bytes = self.backend.serialize(lease, self.sut.cache[tip].blob)
            after_host = self.backend._refresh_stats()
            self.assertIsInstance(host_bytes, bytes)
            self.assertGreater(len(host_bytes), 0)
            self.assertEqual(after_host['written_file_bytes'] - before['written_file_bytes'], len(host_bytes))
            self.assertEqual(after_host['forward_tokens'], before['forward_tokens'])
            files = self.sut.checkpoint(tip)
            after_disk = self.backend._refresh_stats()
            manifest_raw = files['manifest_path'].read_bytes()
            manifest = json.loads(manifest_raw)['manifest']
            payload = files['payload_path'].read_bytes()
            self.assertEqual(manifest['state_bytes'], len(host_bytes))
            self.assertEqual(payload[:len(host_bytes)], host_bytes)
            self.assertEqual(payload[len(host_bytes):], struct.pack('<' + str(len(frontier)) + 'f', *frontier))
            self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest['payload_sha256'])
            self.assertEqual(files['payload_path'].stat().st_size, len(payload))
            self.assertEqual(manifest['model_contract_id'], self.profile['model_contract_id'])
            self.assertEqual(manifest['execution_profile_id'], self.profile['execution_profile_id'])
            self.assertEqual(manifest['consumed'], len(histories[branch]))
            self.assertEqual(after_disk['written_file_bytes'] - after_host['written_file_bytes'], len(host_bytes))
            self.assertEqual(after_disk['forward_tokens'], before['forward_tokens'])
            self.record(kind='offload-persisted-bytes', snapshot=tip, before=before,
                        after_host=after_host, after_disk=after_disk,
                        host_bytes=len(host_bytes), host_sha256=hashlib.sha256(host_bytes).hexdigest(),
                        payload_bytes=len(payload), payload_sha256=hashlib.sha256(payload).hexdigest(),
                        allocated_file_bytes=files['payload_path'].stat().st_blocks * 512,
                        manifest_bytes=len(manifest_raw), manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                        manifest=manifest,
                        transport='Owned native-file bytes; the adapter uses scratch files for native write/load')

            before = self.backend._refresh_stats()
            with self.backend.lease() as lease:
                resident = self.backend.restore(lease, host_bytes, history=histories[branch],
                                                frontier=frontiers[branch])
                restored = self.backend._refresh_stats()
                self.assertEqual(restored['forward_tokens'], before['forward_tokens'])
                self.assertEqual(restored['loaded_file_bytes'] - before['loaded_file_bytes'], len(host_bytes))
                self.assertEqual(self.backend.position(lease), len(histories[branch]) - 1)
                self.assertEqual(self.backend.logits(lease), frontiers[branch])
                self.backend.consume(lease, (tokens[branch],))
                continued = self.backend._refresh_stats()
                self.assertEqual(continued['forward_tokens'] - restored['forward_tokens'], 1)
                self.compare(self.backend.logits(lease), histories[branch] + (tokens[branch],), 'host-bytes')
            del resident
            self.backend.release_checkpoints()
            self.record(kind='offload-host-restore', snapshot=tip, before=before,
                        restored=restored, continued=continued)
            self.unchanged(kept)

            # This process starts with no live checkpoint handles. It must read
            # the durable manifest/payload and compute only the new token.
            before = self.backend._refresh_stats()
            stdout = workers / (str(branch) + '.json')
            stderr = workers / (str(branch) + '.stderr')
            observations = workers / (str(branch) + '.observations.jsonl')
            command = [sys.executable, str(worker), self.tmp.name, tip, '1']
            env = dict(os.environ, CTT_RECOVERY_OBSERVATIONS=str(observations))
            with stdout.open('x') as output, stderr.open('x') as errors:
                result = subprocess.run(command, env=env, stdout=output, stderr=errors)
            self.record(kind='offload-worker', snapshot=tip, command=command, exit_code=result.returncode,
                        stdout=str(stdout), stdout_sha256=hashlib.sha256(stdout.read_bytes()).hexdigest(),
                        stderr=str(stderr), stderr_sha256=hashlib.sha256(stderr.read_bytes()).hexdigest(),
                        observations=str(observations))
            self.assertEqual(result.returncode, 0, stderr.read_text()[-4000:])
            child = json.loads(stdout.read_text())
            self.assertNotEqual(child['pid'], os.getpid())
            self.assertEqual(tuple(child['history']), histories[branch])
            self.assertEqual(child['controller'], controllers[branch])
            self.assertEqual(child['generated'], [tokens[branch]])
            self.assertEqual(tuple(child['final_history']), histories[branch] + (tokens[branch],))
            self.assertEqual(child['final_controller']['next_draw'], 2)
            self.assertEqual(child['counters']['forward_tokens'], 1)
            self.assertEqual(child['counters']['loaded_file_bytes'], len(host_bytes))
            self.assertEqual(child['counters']['written_file_bytes'], 0)
            self.compare(child['logits'], histories[branch], 'disk-restored-frontier')
            self.compare(child['final_logits'], histories[branch] + (tokens[branch],), 'disk-continuation')
            after_worker = self.backend._refresh_stats()
            self.assertEqual(after_worker['forward_tokens'], before['forward_tokens'])
            self.assertEqual(files['manifest_path'].read_bytes(), manifest_raw)
            self.assertEqual(files['payload_path'].read_bytes(), payload)
            self.unchanged(kept)
            for index, snapshot in enumerate(tips):
                self.assertEqual(self.sut.controller(snapshot), controllers[index])
            for head in heads:
                self.assertEqual(self.sut.store.head(head.name), head)
            del host_bytes, payload, child


if __name__ == '__main__':
    unittest.main()
