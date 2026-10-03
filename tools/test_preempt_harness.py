"""CPU-only guards against false passes and malformed parity-test configuration."""
import unittest
import queue
import json
import os
import sys
import tempfile
from pathlib import Path
from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import Mock
from tools.prefill_preempt_test import engine_args, state_differences, HASH_RE, parse_state_match, output_differences, Engine, run_scenario


class HarnessChecks(unittest.TestCase):
    def test_valueless_preempt_flag_preserves_following_model_option(self):
        args = engine_args({'args': ['--prefill-preempt', '--pack', '/model', '--spec', '4']},
                           prefill=2048, preempt=False)
        self.assertEqual(args[:4], ['--pack', '/model', '--spec', '4'])
        self.assertNotIn('--prefill-preempt', args)

    def test_controlled_options_are_not_duplicated(self):
        args = engine_args({'args': ['--adapt-every', '4', '--suffix-draft', '1', '--spec-min-p', '.5']},
                           prefill=2048, preempt=True)
        for flag, value in [('--adapt-every', '100000'), ('--suffix-draft', '0'), ('--spec-min-p', '0')]:
            self.assertEqual(args.count(flag), 1)
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn('--prefill-preempt', args)

    def test_mtp_and_target_mismatch_fail(self):
        self.assertEqual(state_differences({'mtp': 'a', 'gdn': 'a'}, {'mtp': 'b', 'gdn': 'b'}),
                         ['mtp', 'gdn'])

    def test_uncommitted_stale_cells_are_not_semantic_state(self):
        self.assertEqual(state_differences({'kv': 'a', 'stale': 'a'}, {'kv': 'a', 'stale': 'b'}), [])

    def test_legacy_hash_keeps_ple_previous_tokens(self):
        line = "STATE_HASH L=8 gdn=a ple=b tail=c pooled=d kv=e mtp=f stale=0 ple_prev=123,456"
        state = parse_state_match(HASH_RE.search(line))
        self.assertEqual(state["ple_prev0"], "123")
        self.assertEqual(state["ple_prev1"], "456")
        self.assertNotIn("dead", state)

    def test_extended_hash_keeps_all_components(self):
        line = ("STATE_HASH L=8 gdn=a ple=b tail=c pooled=d kv=e mtp=f stale=0 "
                "dead=11 pooled_full=22 block=33 ple_prev=-1,456")
        state = parse_state_match(HASH_RE.search(line))
        self.assertEqual([state[k] for k in ("dead", "pooled_full", "block", "ple_prev0")],
                         ["11", "22", "33", "-1"])

    def test_missing_state_cannot_pass(self):
        self.assertEqual(state_differences({'gdn': 'a'}, {}), ['gdn'])

    def test_matching_early_eos_is_not_a_failure(self):
        eos = {'tokens': [1, 2, 3, 4, 5], 'finish': 'stop'}
        self.assertEqual(output_differences(eos, eos, 'B'), [])

    def test_early_eos_against_full_reference_still_fails(self):
        ref = {'tokens': list(range(128)), 'finish': 'length'}
        got = {'tokens': list(range(5)), 'finish': 'stop'}
        diff = output_differences(ref, got, 'B')
        self.assertEqual(len(diff), 1)
        self.assertIn('length 128 vs 5', diff[0])
        self.assertIn('[5]', diff[0])

    def test_finish_reason_is_gated_even_when_tokens_match(self):
        ref = {'tokens': [1], 'finish': 'length'}
        got = {'tokens': [1], 'finish': 'stop'}
        self.assertTrue(output_differences(ref, got, 'B'))

    def engine_stub(self):
        e = Engine.__new__(Engine)
        e.lines = queue.Queue()
        e.pending = defaultdict(deque)
        e.eof_seen = False
        e._record = Mock()
        return e

    def test_cancel_collection_does_not_discard_interim_output(self):
        e = self.engine_stub()
        for line in ['T 123 id=2', 'DONE 1 8 0 0 stop id=2', 'DONE 0 8 0 0 cancel id=1']:
            e.lines.put(line)
        self.assertEqual(e.collect(1)['finish'], 'cancel')
        self.assertEqual(e.collect(2)['tokens'], [123])

    def test_malformed_generated_count_fails(self):
        e = self.engine_stub()
        e.lines.put('DONE 3 8 0 0 length id=1')
        with self.assertRaisesRegex(RuntimeError, 'received 0'):
            e.collect(1)

    def test_exit_signal_is_reported_without_stderr(self):
        e = self.engine_stub()
        e.proc = Mock()
        e.proc.poll.return_value = -9
        e.log = SimpleNamespace(name='stderr.log')
        e.protocol = SimpleNamespace(name='protocol.jsonl')
        self.assertIn('SIGKILL', str(e._ended()))

    def test_spontaneous_clean_exit_cannot_pass_cleanup(self):
        e = self.engine_stub()
        e.proc = Mock()
        e.proc.poll.return_value = 0
        e.log = Mock(name='stderr.log')
        e.protocol = Mock(name='protocol.jsonl')
        e.pump = Mock()
        with self.assertRaisesRegex(RuntimeError, 'exit=0'):
            e.close()

    def repeated_scenario(self, positions):
        e = self.engine_stub()
        e.gen, e.send, e.state_hash = Mock(), Mock(), Mock(return_value={})
        e.lines.put('PP 2048 8683 1 1')
        for i, pos in enumerate(positions):
            e.lines.put(f'SUSPENDED {pos} 8683 id=1')
            rid = i + 2
            e.lines.put(f'T 7 id={rid}')
            e.lines.put(f'DONE 1 8 0 0 length id={rid}')
            e.lines.put(f'RESUME {pos} id=1')
        e.lines.put('T 7 id=1')
        e.lines.put('DONE 1 8683 0 0 length id=1')
        ref = {'tokens': [7], 'finish': 'length'}
        return run_scenario(e, 'repeat-test', {'trigger': 2048, 'preempts': [2, 3, 4], 'cancel': False},
                            [1] * 8683, [2] * 8, ref, ref, {}, 1, 2048)

    def test_all_repeated_park_boundaries_are_checked(self):
        self.assertEqual(self.repeated_scenario([2048, 4096, 6144]), [])
        failures = self.repeated_scenario([2048, 4097, 6144])
        self.assertTrue(any('non-chunk' in f for f in failures))

    def test_repeated_parks_must_advance_a_chunk(self):
        failures = self.repeated_scenario([2048, 2048, 4096])
        self.assertTrue(any('advance a full chunk' in f for f in failures))

    def test_snapshot_budget_from_config_does_not_disable_boundary_tests(self):
        cfg = {'args': ['--prefill-preempt-snapshot-mib', '1']}
        args = engine_args(cfg, prefill=2048, preempt=True)
        self.assertEqual(args.count('--prefill-preempt-snapshot-mib'), 1)
        self.assertEqual(args[args.index('--prefill-preempt-snapshot-mib') + 1], '4096')

    @unittest.skipIf(os.name == 'nt', 'executable Python fixture uses a Unix shebang')
    def test_real_pipe_records_pump_times_and_clean_quit(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / 'fake-engine'
            exe.write_text('#!' + sys.executable + '\n' + '''import sys
print('INFO expert_slots=100', flush=True)
print('READY 10000', flush=True)
for line in sys.stdin:
    if line.strip() == 'QUIT':
        break
    if line.startswith('GEN '):
        print('RESUME 0 id=2', flush=True)
        print('T 123 id=2', flush=True)
        print('DONE 1 8 1 1 stop id=2', flush=True)
''', encoding='utf-8')
            exe.chmod(0o700)
            e = Engine(str(exe), [], Path(tmp) / 'engine.log')
            try:
                e.gen(2, [1] * 8, 32)
                result = e.collect(2)
                self.assertEqual(result['tokens'], [123])
                observed = result['observed']
                self.assertLessEqual(observed['started'], observed['first_token'])
                self.assertLessEqual(observed['first_token'], observed['done'])
                self.assertEqual(e.info['expert_slots'], '100')
            finally: e.close()
            events = [json.loads(line) for line in (Path(tmp) / 'engine.protocol.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['event'], 'quit_exit')
            self.assertEqual(events[-1]['returncode'], 0)


if __name__ == '__main__':
    unittest.main()
