"""T33: measure real work/memory through the pinned execution's full capacity."""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import unittest

from context_time_travel.numerics import compare_logits
from context_time_travel.store import IntegrityError, InvalidOperation
import test_immutability as isolation
import test_lifecycle as lifecycle
import test_partial_cow as partial


class ResourceTests(unittest.TestCase):
    setUpClass = classmethod(isolation.SharedPageTests.setUpClass.__func__)
    setUp = isolation.SharedPageTests.setUp
    record = isolation.SharedPageTests.record
    unchanged = isolation.SharedPageTests.unchanged
    pages = staticmethod(partial.PartialTailTests.pages)
    tail_pages = partial.PartialTailTests.tail_pages
    require_instrumentation = lifecycle.LifecycleTests.require_instrumentation
    balanced = lifecycle.LifecycleTests.balanced

    def retain(self, snapshot, frontier):
        saved = isolation.SharedPageTests.retain(self, snapshot, frontier)
        # Keep the actual checkpoint object as well as its native image. The
        # bounded cache may evict an entry while this test retains its owner.
        saved['checkpoint'] = self.sut.cache[snapshot]
        return saved

    def resources(self, phase):
        counters = self.backend._refresh_stats()
        self.require_instrumentation(counters)
        pid = self.backend.process.pid
        status = (Path('/proc') / str(pid) / 'status').read_text()
        host = {}
        for line in status.splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM', 'VmSize'):
                number, unit = value.split()
                self.assertEqual(unit, 'kB')
                host[key] = int(number) * 1024
        self.assertEqual(set(host), {'VmRSS', 'VmHWM', 'VmSize'})
        self.assertGreater(host['VmRSS'], 0)
        self.assertGreaterEqual(host['VmHWM'], host['VmRSS'])
        command = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                   '--format=csv,noheader,nounits']
        output = subprocess.check_output(command, text=True)
        devices = []
        for line in output.splitlines():
            uuid, owner, memory = [part.strip() for part in line.split(',')]
            if owner == str(pid):
                self.assertTrue(memory.isdigit(), 'Real GPU process-memory instrumentation is required')
                devices.append(dict(uuid=uuid, bytes=int(memory) * 1024 * 1024))
        self.assertEqual(len(devices), 1, 'The selected single-GPU native process must be observed')
        self.assertIn(devices[0]['uuid'], self.profile['hardware'])
        self.assertGreaterEqual(devices[0]['bytes'], counters['live_bytes'])
        measured = dict(phase=phase, pid=pid, counters=counters,
                        native_process_status_bytes=host, gpu_process_memory=devices,
                        checkpoint_metrics=self.backend.checkpoint_metrics(self.sut.cache.values()))
        self.record(kind='context-resource-sample', **measured,
                    scope='NVML process memory is sampled, VmHWM is the process high-water mark; VMM and image counters have their own narrower scopes')
        return measured

    def compare(self, actual, snapshot, history, phase):
        expected = self.oracle.forward(history)
        self.assertEqual(len(actual), self.backend.vocab_size)
        self.assertEqual(self.oracle.last['computed_tokens'], len(history))
        self.assertEqual(self.oracle.last['position_max'], len(history) - 1)
        report = compare_logits(actual, expected, atol=self.profile['atol'], rtol=self.profile['rtol'])
        self.record(kind='resource-numerics', phase=phase, snapshot=snapshot, history=history,
                    numerics=asdict(report), max_absolute_error=report.max_absolute_error,
                    actual_sha256=hashlib.sha256(struct.pack('<' + str(len(actual)) + 'f', *actual)).hexdigest(),
                    expected_sha256=self.oracle.last['logits_sha256'],
                    oracle_computed_tokens=self.oracle.last['computed_tokens'])
        self.sut.measure(report, snapshot, history)
        self.assertTrue(report.pass_elementwise, report)
        self.assertTrue(report.top1_stable_within_tolerance, report)
        self.assertEqual(self.sut.history(snapshot), history)

    def hot(self, snapshot, history, *, repeat):
        before = self.backend._refresh_stats()
        start = time.perf_counter_ns()
        with self.sut.materialize(snapshot) as (lease, published):
            actual = self.backend.logits(lease)
            consumed = self.backend.position(lease) + 1
        elapsed = (time.perf_counter_ns() - start) / 1e9
        after = self.backend._refresh_stats()
        self.record(kind='context-hot-activation', length=len(history), repeat=repeat,
                    seconds=elapsed, before=before, after=after, snapshot=snapshot)
        self.assertEqual(after['forward_tokens'], before['forward_tokens'],
                         'Hot activation must not recompute the stored prefix')
        for name in ('allocations', 'cow_pages', 'copied_bytes', 'captures', 'written_file_bytes', 'loaded_file_bytes'):
            self.assertEqual(after[name], before[name], name)
        self.assertEqual(consumed, len(history))
        self.assertEqual(actual, published)
        return actual, elapsed

    def ownership(self, retained):
        pages = {}
        for saved in retained:
            for row in self.pages(saved['audit']).values():
                key = row['allocation_id']
                value = (row['kind'], row['bytes'], row['driver_handle'])
                self.assertEqual(pages.setdefault(key, value), value)
        metrics = self.backend.checkpoint_metrics([saved['checkpoint'] for saved in retained])
        self.assertEqual(metrics['checkpoint_unique_physical_pages'], len(pages))
        self.assertEqual(metrics['checkpoint_unique_physical_bytes'], sum(row[1] for row in pages.values()))
        self.assertEqual(metrics['checkpoint_unique_kv_bytes'],
                         sum(row[1] for row in pages.values() if row[0] != 'indexer_pool'))
        self.assertEqual(metrics['checkpoint_unique_index_bytes'],
                         sum(row[1] for row in pages.values() if row[0] == 'indexer_pool'))
        self.assertEqual(metrics['owned_checkpoint_bytes'], sum(saved['image'].running_bytes for saved in retained))
        direct = self.backend._refresh_stats()
        self.assertEqual(direct['snapshot_live_images'], len(retained))
        self.assertEqual(direct['handle_live_entries'], len(retained))
        cached = self.backend.checkpoint_metrics(self.sut.cache.values())
        self.assertLessEqual(cached['checkpoint_unique_physical_bytes'], metrics['checkpoint_unique_physical_bytes'])
        self.record(kind='context-owned-pages', retained_images=len(retained),
                    retained_checkpoint_metrics=metrics, cache_metrics=cached, direct=direct)
        return metrics

    def probe(self, target):
        fixture = self.profile['fixtures']['prefix']
        prefix = tuple(fixture[i % len(fixture)] for i in range(target - 2))
        self.assertEqual(len(prefix), target - 2)
        before = self.resources('before-prefill-' + str(target))
        start = time.perf_counter_ns()
        root = self.sut.create(prefix)
        prefill_seconds = (time.perf_counter_ns() - start) / 1e9
        after = self.resources('after-prefill-' + str(target))
        self.assertEqual(after['counters']['forward_tokens'] - before['counters']['forward_tokens'], len(prefix))
        activations = []
        for repeat in range(5):
            actual, seconds = self.hot(root, prefix, repeat=repeat)
            activations.append(seconds)
        self.compare(actual, root, prefix, 'stored-prefix')
        retained = [self.retain(root, actual)]
        root_pages = self.pages(retained[0]['audit'])
        prefix_bytes = sum(row['filled_prefix_bytes'] for row in root_pages.values())
        self.assertGreater(prefix_bytes, 0)
        if target == self.profile['execution']['capacity']:
            kv = [row for row in root_pages.values() if row['kind'] in ('k', 'v')]
            self.assertEqual({row['pool'] for row in kv},
                             {row['pool'] for row in kv if row['filled_prefix_bytes'] == row['bytes']},
                             'Longest fixture must retain completed physical K/V pages in every pool')
        initial_metrics = self.ownership(retained)
        self.resources('hot-root-' + str(target))
        tokens = [self.profile['fixtures']['suffix_a'][0], self.profile['fixtures']['suffix_b'][0]]
        self.assertNotEqual(*tokens)
        tips, heads, histories, finals = [], [], [], []
        for branch, token in enumerate(tokens):
            name = 'resource-' + str(target) + '-' + str(branch)
            tip = self.sut.start_generation(root, dict(allowed=[token]), name)
            head = self.sut.store.fork(name, tip)
            history = prefix
            actual, _ = self.hot(tip, history, repeat='before-suffix-' + str(branch))
            self.compare(actual, tip, history, 'before-suffix-' + str(branch))
            parent = retained[0]
            for step in range(2):
                prior_pages = self.pages(parent['audit'])
                expected = self.tail_pages(parent['audit'], len(history))
                before_token = self.backend._refresh_stats()
                start = time.perf_counter_ns()
                generated = self.sut.generate(tip, 1)
                seconds = (time.perf_counter_ns() - start) / 1e9
                after_token = self.backend._refresh_stats()
                self.assertEqual(generated.accepted_tokens, (token,))
                self.assertEqual(generated.boundaries, (generated.snapshot_id,))
                self.assertEqual(after_token['forward_tokens'] - before_token['forward_tokens'], 1)
                self.assertEqual(after_token['cow_pages'] - before_token['cow_pages'], len(expected))
                self.assertEqual(after_token['copied_bytes'] - before_token['copied_bytes'],
                                 sum(prior_pages[key]['bytes'] for key in expected))
                tip = generated.snapshot_id
                history += (token,)
                actual, _ = self.hot(tip, history, repeat='consumed-boundary')
                self.compare(actual, tip, history, 'after-consumed-token')
                self.assertEqual(self.sut.controller(tip)['next_draw'], step + 1)
                child = self.retain(tip, actual)
                child_pages = self.pages(child['audit'])
                self.assertEqual(child_pages.keys(), prior_pages.keys())
                changed = {key for key in prior_pages
                           if prior_pages[key]['allocation_id'] != child_pages[key]['allocation_id']}
                self.assertEqual(changed, expected)
                for key in prior_pages.keys() - expected:
                    self.assertEqual(prior_pages[key]['sha256'], child_pages[key]['sha256'])
                self.unchanged(retained)
                retained.append(child)
                parent = child
                self.ownership(retained)
                self.record(kind='context-consumed-token', target=target, branch=branch, step=step,
                            seconds=seconds, before=before_token, after=after_token,
                            expected_private_pages=sorted(expected), snapshot=tip)
                self.resources('after-token-' + str(target) + '-' + str(branch) + '-' + str(step))
            self.assertEqual(len(history), target)
            head = self.sut.store.move(name, head.revision, tip, name + '-publish')
            tips.append(tip)
            heads.append(head)
            histories.append(history)
            finals.append(actual)

        if target == self.profile['execution']['capacity']:
            objects = {path.name: path.read_bytes() for path in self.sut.store.objects.iterdir()}
            for branch, tip in enumerate(tips):
                decoder = self.sut.controller(tip)
                before_refusal = self.backend._refresh_stats()
                with self.assertRaisesRegex(InvalidOperation, 'context exceeded'):
                    self.sut.feed(tip, (tokens[branch],))
                streamed = []
                with self.assertRaisesRegex(InvalidOperation, 'context exceeded'):
                    for item in self.sut.stream(tip, 1):
                        streamed.append(item)
                self.assertEqual(streamed, [], 'No sampled-but-unconsumed output may escape')
                with self.sut.materialize(tip) as (lease, published):
                    with self.assertRaisesRegex(IntegrityError, 'context capacity'):
                        self.backend._request('feed ' + str(tokens[branch]))
                    self.assertEqual(self.backend.position(lease), target - 1)
                    self.assertEqual(self.backend.logits(lease), finals[branch])
                    self.assertEqual(published, finals[branch])
                after_refusal = self.backend._refresh_stats()
                for name in ('forward_tokens', 'allocations', 'cow_pages', 'copied_bytes', 'captures'):
                    self.assertEqual(after_refusal[name], before_refusal[name], name)
                self.assertEqual(self.sut.controller(tip), decoder)
                self.assertEqual(self.sut.store.head(heads[branch].name), heads[branch])
                self.assertEqual(objects, {path.name: path.read_bytes() for path in self.sut.store.objects.iterdir()})
                self.record(kind='capacity-overflow-refused', capacity=target, branch=branch,
                            api_feed=True, stream=True, direct_native_feed=True,
                            before=before_refusal, after=after_refusal, streamed_tokens=0)
            self.unchanged(retained)
        self.ownership(retained)
        final_resources = self.resources('two-futures-' + str(target))
        self.record(kind='context-length-profile', target=target, prefix_length=len(prefix),
                    prefill_seconds=prefill_seconds, hot_activation_seconds=activations,
                    prefix_filled_bytes=prefix_bytes, root_ownership=initial_metrics,
                    final_resources=final_resources)
        # Return no native-image owners. A cold replay below must rebuild this
        # exact history and restore its external controller without resampling.
        return dict(target=target, prefix_filled_bytes=prefix_bytes, root_ownership=initial_metrics,
                    cold_snapshot=tips[-1], cold_history=histories[-1], cold_head=heads[-1],
                    cold_controller=self.sut.controller(tips[-1]))

    def empty(self, baseline):
        self.sut.evict_all()
        gc.collect()
        self.backend.release_checkpoints()
        self.backend._request('reset')
        after = self.backend._refresh_stats()
        self.balanced(baseline, after)
        self.assertEqual(after['consumed'], 0)
        self.record(kind='resource-workspace-baseline', before=baseline, after=after)

    def test_g3_016_long_context_resource_profile(self):
        self.assertEqual(self.profile['calibration_status'], 'FROZEN')
        baseline_bytes = Path(self.profile['calibration_path']).read_bytes()
        numerical = json.loads(baseline_bytes)
        self.assertEqual(numerical['status'], 'FROZEN')
        self.assertEqual((self.profile['atol'], self.profile['rtol']), (numerical['atol'], numerical['rtol']))
        capacity = self.profile['execution']['capacity']
        lengths = [9, capacity // 2, capacity]
        self.assertEqual(lengths, sorted(set(lengths)))
        self.assertGreater(lengths[1], lengths[0])
        self.record(kind='context-profile-scope', capacity=capacity, tested_final_lengths=lengths,
                    baseline_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
                    scope='Full pinned execution capacity; no claim beyond this context limit')
        baseline = self.backend._refresh_stats()
        self.require_instrumentation(baseline)
        profiles = []
        for length in lengths:
            self.empty(baseline)
            row = self.probe(length)
            self.empty(baseline)
            before = self.backend._refresh_stats()
            start = time.perf_counter_ns()
            actual = self.sut.next_logits(row['cold_snapshot'])
            seconds = (time.perf_counter_ns() - start) / 1e9
            after = self.backend._refresh_stats()
            self.assertEqual(after['forward_tokens'] - before['forward_tokens'], length)
            self.assertEqual(after['loaded_file_bytes'], before['loaded_file_bytes'])
            self.assertEqual(self.sut.controller(row['cold_snapshot']), row['cold_controller'])
            self.assertEqual(self.sut.store.head(row['cold_head'].name), row['cold_head'])
            self.compare(actual, row['cold_snapshot'], row['cold_history'], 'full-cold-replay')
            self.record(kind='context-cold-replay', length=length, seconds=seconds, before=before, after=after)
            self.resources('after-cold-replay-' + str(length))
            self.empty(baseline)
            profiles.append({k: row[k] for k in ('target', 'prefix_filled_bytes', 'root_ownership')})
        self.assertEqual([row['target'] for row in profiles], lengths)
        self.assertTrue(all(a['prefix_filled_bytes'] < b['prefix_filled_bytes']
                            for a, b in zip(profiles, profiles[1:])))
        self.record(kind='context-scaling-complete', profiles=profiles,
                    note='Physical reservation may stay constant for a fixed-capacity arena; filled-prefix bytes and completed work are recorded separately')


if __name__ == '__main__':
    unittest.main()
