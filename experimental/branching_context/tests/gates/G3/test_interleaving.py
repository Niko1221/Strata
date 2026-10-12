"""T33: seeded concurrent branch commands and real numerical-cache eviction."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import threading
import time
import unittest

import test_immutability as isolation
from context_time_travel.numerics import compare_logits
from context_time_travel.store import Conflict, Store

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'native'))
from test_cache_interleaving import evict_during_transition


class InterleavingTests(unittest.TestCase):
    setUpClass = classmethod(isolation.SharedPageTests.setUpClass.__func__)
    setUp = isolation.SharedPageTests.setUp
    record = isolation.SharedPageTests.record
    retain = isolation.SharedPageTests.retain
    unchanged = isolation.SharedPageTests.unchanged

    def capture(self, snapshot, history, phase):
        with self.sut.materialize(snapshot) as (lease, published):
            actual = self.backend.logits(lease)
            consumed = self.backend.position(lease) + 1
        self.assertEqual(consumed, len(history))
        self.assertEqual(actual, published)
        self.assertEqual(self.sut.history(snapshot), history)
        return dict(snapshot=snapshot, history=history, actual=actual,
                    phase=phase, consumed=consumed,
                    canonical=self.sut.snapshot_bytes(snapshot))

    def compare(self, observed):
        expected = self.oracle.forward(observed['history'])
        actual = observed['actual']
        self.assertEqual(len(actual), self.backend.vocab_size)
        self.assertEqual(self.oracle.last['computed_tokens'], len(observed['history']))
        self.assertEqual(self.oracle.last['position_max'], len(observed['history']) - 1)
        report = compare_logits(actual, expected, atol=self.profile['atol'], rtol=self.profile['rtol'])
        self.record(kind='interleaving-numerics', phase=observed['phase'],
                    snapshot=observed['snapshot'], history=observed['history'],
                    consumed=observed['consumed'], numerics=asdict(report),
                    max_absolute_error=report.max_absolute_error,
                    actual_sha256=hashlib.sha256(struct.pack('<' + str(len(actual)) + 'f', *actual)).hexdigest(),
                    expected_sha256=self.oracle.last['logits_sha256'],
                    oracle_computed_tokens=self.oracle.last['computed_tokens'])
        self.sut.measure(report, observed['snapshot'], observed['history'])
        self.assertTrue(report.pass_elementwise, report)
        self.assertTrue(report.top1_stable_within_tolerance, report)
        self.assertEqual(self.sut.snapshot_bytes(observed['snapshot']), observed['canonical'])

    def check_controller(self, state, prefix_length):
        decoder = self.sut.controller(state['snapshot'])
        accepted = len(state['history']) - prefix_length
        self.assertEqual(decoder['next_draw'], state['draw'])
        self.assertEqual(decoder['accepted_in_scope'], accepted)
        self.assertEqual(decoder['grammar_state'], 'odd' if accepted % 2 else 'even')
        self.assertEqual(decoder['stream_key'], hashlib.sha256(state['stream'].encode()).hexdigest())
        self.assertEqual(decoder['rng_algorithm'], 'hmac-sha256-u53-v1')
        self.assertEqual(decoder['status'], 'active')
        counts = {}
        for token in state['history']:
            counts[str(token)] = counts.get(str(token), 0) + 1
        self.assertEqual(decoder['seen'], counts)

    def test_g3_013_stress_interleaving_and_eviction(self):
        self.assertEqual(self.profile['calibration_status'], 'FROZEN')
        baseline_bytes = Path(self.profile['calibration_path']).read_bytes()
        baseline = json.loads(baseline_bytes)
        self.assertEqual(baseline['status'], 'FROZEN')
        self.assertEqual((self.profile['atol'], self.profile['rtol']),
                         (baseline['atol'], baseline['rtol']))
        self.record(kind='stress-frozen-baseline', sha256=hashlib.sha256(baseline_bytes).hexdigest(),
                    atol=baseline['atol'], rtol=baseline['rtol'])
        self.assertTrue(callable(getattr(self.backend, 'audit', None)))
        prefix = tuple(self.profile['fixtures']['prefix'][:9])
        tokens = [self.profile['fixtures']['suffix_a'][0], self.profile['fixtures']['suffix_b'][0]]
        self.assertNotEqual(*tokens)
        root = self.sut.create(prefix)
        initial = self.capture(root, prefix, 'initial-prefix')
        self.compare(initial)
        anchor = self.retain(root, initial['actual'])
        audit = anchor['audit']
        self.assertNotEqual(len(prefix) % audit['qsa_page_cells'], 0)
        self.assertTrue(any(0 < r['filled_prefix_bytes'] < r['bytes']
                            for r in audit['regions'] if r['kind'] in ('k', 'v')))

        # A deterministic real-image counterexample precedes randomized work.
        start = self.sut.start_generation(root, dict(allowed=[tokens[0]]), 'interleaving-race')
        result = evict_during_transition(self.sut, start, self.sut.controller(start))
        self.record(kind='controller-cache-race', result=result, real_threads=2)
        self.assertEqual(result[0], 'ok', 'Concurrent cache eviction broke controller publication: ' + repr(result))
        self.compare(self.capture(result[1], prefix, 'controller-after-concurrent-eviction'))
        self.unchanged([anchor])

        original = {root: self.sut.snapshot_bytes(root)}
        seeds = (0xC013, 1701, 65537, 900001)
        rounds = 16
        coverage = set()
        worker_ids = set()
        observed_count = 2
        consumed_children = 0
        overall_before = self.backend._refresh_stats()
        for seed in seeds:
            rng = random.Random(seed)
            states, stacks, heads = [], [], []
            for branch, token in enumerate(tokens):
                stream = 'stress-' + str(seed) + '-' + str(branch)
                grammar = dict(start='even', transitions={
                    'even': {str(token): 'odd'}, 'odd': {str(token): 'even'}})
                tip = self.sut.start_generation(root, dict(allowed=[token], grammar=grammar), stream)
                state = dict(snapshot=tip, history=prefix, draw=0, stream=stream)
                states.append(state)
                stacks.append([state])
                heads.append(self.sut.store.fork(stream, tip))
                original[tip] = self.sut.snapshot_bytes(tip)
                self.compare(self.capture(tip, prefix, 'before-any-suffix'))
                observed_count += 1

            # Native leases serialize the one model workspace. Requests, branch
            # CAS publications and optional-cache changes really run concurrently.
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix='g3-interleaving') as pool:
                for step in range(rounds):
                    barrier = threading.Barrier(3)
                    actions = [(['generate', 'feed', 'rewind', 'reseed', 'read', 'generate'][step]
                                if step < 6 else rng.choice(['generate', 'generate', 'feed', 'rewind', 'reseed', 'read']))
                               for _ in range(2)]
                    rewinds = [rng.randrange(len(stack)) for stack in stacks]
                    eviction = 'all' if step % 3 == 0 else 'one'
                    target = states[rng.randrange(2)]['snapshot']
                    delays = [rng.randrange(4) / 1000 for _ in range(3)]
                    before = self.backend._refresh_stats()

                    def advance(branch):
                        entered = time.monotonic_ns()
                        identity = threading.get_ident()
                        barrier.wait(30)
                        time.sleep(delays[branch])
                        action = actions[branch]
                        state = dict(states[branch])
                        observed = [self.capture(state['snapshot'], state['history'], 'before-' + action)]
                        old_head = heads[branch]
                        if action in ('generate', 'feed'):
                            if action == 'generate':
                                generated = self.sut.generate(state['snapshot'], 1)
                                self.assertEqual(generated.accepted_tokens, (tokens[branch],))
                                self.assertEqual(generated.boundaries, (generated.snapshot_id,))
                                tip = generated.snapshot_id
                                state['draw'] += 1
                            else:
                                tip = self.sut.feed(state['snapshot'], (tokens[branch],))
                            state.update(snapshot=tip, history=state['history'] + (tokens[branch],))
                        elif action == 'rewind':
                            state = dict(stacks[branch][rewinds[branch]])
                        elif action == 'reseed':
                            stream = 'stress-reseed-' + str(seed) + '-' + str(step) + '-' + str(branch)
                            state.update(snapshot=self.sut.set_random_stream(state['snapshot'], stream),
                                         draw=0, stream=stream)
                        self.check_controller(state, len(prefix))
                        new_head = old_head
                        if action != 'read':
                            operation = 'stress-move-' + str(seed) + '-' + str(step) + '-' + str(branch)
                            new_head = self.sut.store.move(old_head.name, old_head.revision, state['snapshot'], operation)
                            self.assertEqual(new_head.revision, old_head.revision + 1)
                            self.assertEqual(self.sut.store.move(old_head.name, old_head.revision,
                                                                 state['snapshot'], operation), new_head)
                            self.assertEqual(new_head.tip, state['snapshot'])
                        observed.append(self.capture(state['snapshot'], state['history'], 'after-' + action))
                        return dict(state=state, head=new_head, observed=observed,
                                    worker=identity, started_ns=entered, finished_ns=time.monotonic_ns(), action=action)

                    def evict():
                        entered = time.monotonic_ns()
                        identity = threading.get_ident()
                        barrier.wait(30)
                        time.sleep(delays[2])
                        if eviction == 'all':
                            self.sut.evict_all()
                        else:
                            try:
                                del self.sut.cache[target]
                            except KeyError:
                                pass  # Already absent is a valid optional-cache state.
                            self.backend.release_checkpoints()
                        return dict(worker=identity, started_ns=entered, finished_ns=time.monotonic_ns(),
                                    action='evict-' + eviction, target=target)

                    futures = [pool.submit(advance, branch) for branch in range(2)]
                    eviction_future = pool.submit(evict)
                    outcomes = [future.result(timeout=900) for future in futures]
                    evicted = eviction_future.result(timeout=900)
                    after = self.backend._refresh_stats()
                    self.assertEqual(len({row['worker'] for row in outcomes + [evicted]}), 3)
                    self.assertEqual(after['lease_acquires'], after['lease_releases'])
                    self.assertGreaterEqual(after['forward_tokens'] - before['forward_tokens'],
                                            sum(action in ('generate', 'feed') for action in actions))
                    if step == 0:
                        self.assertGreater(after['cow_pages'], before['cow_pages'], 'Actual partial-page COW is required')
                        self.assertGreater(after['copied_bytes'], before['copied_bytes'])
                    self.record(kind='concurrent-round', seed=seed, step=step,
                                commands=[{k: v for k, v in row.items() if k not in ('observed', 'head')}
                                          for row in outcomes] + [evicted], before=before, after=after,
                                delays=delays, rewinds=rewinds)
                    for branch, outcome in enumerate(outcomes):
                        states[branch], heads[branch] = outcome['state'], outcome['head']
                        if outcome['action'] != 'read':
                            stacks[branch].append(dict(outcome['state']))
                        coverage.add(outcome['action'])
                        consumed_children += outcome['action'] in ('generate', 'feed')
                        for observation in outcome['observed']:
                            self.compare(observation)
                            observed_count += 1
                            original.setdefault(observation['snapshot'], observation['canonical'])
                        self.assertEqual(Store(self.tmp.name).head(heads[branch].name), heads[branch])
                    coverage.add(evicted['action'])
                    worker_ids.update(row['worker'] for row in outcomes + [evicted])
                    self.unchanged([anchor])
                    for snapshot, raw in original.items():
                        self.assertEqual(self.sut.snapshot_bytes(snapshot), raw)

                # Two real writers compete at one revision; both immutable
                # candidate futures have already been compared with fresh runs.
                shared = self.sut.store.fork('stress-race-' + str(seed), root)
                race = threading.Barrier(2)

                def publish(branch):
                    race.wait(30)
                    op = 'stress-cas-' + str(seed) + '-' + str(branch)
                    try:
                        head = self.sut.store.move(shared.name, 0, states[branch]['snapshot'], op)
                        return ('won', branch, head, op)
                    except Conflict:
                        return ('conflict', branch)

                pending = [pool.submit(publish, branch) for branch in range(2)]
                results = [future.result(timeout=30) for future in pending]
                self.assertEqual(sorted(row[0] for row in results), ['conflict', 'won'])
                winner = next(row for row in results if row[0] == 'won')
                self.assertEqual(self.sut.store.head(shared.name), winner[2])
                self.assertEqual(self.sut.store.move(shared.name, 0, states[winner[1]]['snapshot'], winner[3]), winner[2])
                self.record(kind='concurrent-CAS', seed=seed, outcomes=[list(r[:2]) for r in results],
                            winner=asdict(winner[2]))

            self.sut.evict_all()
            self.assertEqual(len(self.sut.cache), 0)
            self.assertEqual(list(self.sut.checkpoint_directory.iterdir()), [])
            for state, head in zip(states, heads):
                decoder = self.sut.controller(state['snapshot'])
                self.assertNotIn(state['snapshot'], self.sut.cache)
                before = self.backend._refresh_stats()
                observed = self.capture(state['snapshot'], state['history'], 'all-cache-evicted-cold-replay')
                after = self.backend._refresh_stats()
                self.assertEqual(after['forward_tokens'] - before['forward_tokens'], len(state['history']))
                self.assertEqual(after['loaded_file_bytes'], before['loaded_file_bytes'])
                self.assertEqual(self.sut.controller(state['snapshot']), decoder)
                self.assertEqual(self.sut.store.head(head.name), head)
                self.compare(observed)
                observed_count += 1
                self.record(kind='stress-cold-replay', seed=seed, before=before, after=after,
                            snapshot=state['snapshot'], history=state['history'])
            self.unchanged([anchor])

        self.assertEqual(coverage, {'generate', 'feed', 'rewind', 'reseed', 'read', 'evict-one', 'evict-all'})
        self.assertGreaterEqual(len(worker_ids), 3)
        for snapshot, raw in original.items():
            self.assertEqual(self.sut.snapshot_bytes(snapshot), raw)
        self.record(kind='stress-summary', seeds=seeds, rounds_per_seed=rounds,
                    concurrent_rounds=len(seeds) * rounds, numerical_comparisons=observed_count,
                    consumed_children=consumed_children, coverage=sorted(coverage),
                    before=overall_before, after=self.backend._refresh_stats(),
                    distinct_worker_threads=len(worker_ids), canonical_snapshots_checked=len(original))
        del anchor
        gc.collect()


if __name__ == '__main__':
    unittest.main()
