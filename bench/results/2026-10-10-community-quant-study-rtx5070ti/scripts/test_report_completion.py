"""Portable regression checks for the recovered coding evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent

class CodingCompletionTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / 'data/ALL-METRICS.json').read_text())

    def test_every_local_arm_has_complete_scoring(self):
        count = 0
        for key, model in self.data['models'].items():
            for effort, summary in model['heplus'].items():
                raw = json.loads(gzip.decompress((ROOT / f'data/heplus/{key}-{effort}.json.gz').read_bytes()))
                tasks = raw['tasks']
                self.assertEqual({t['task_id'] for t in tasks}, {f'HumanEval/{i}' for i in range(164)})
                self.assertEqual(len(tasks), 164)
                self.assertEqual(sum(not t.get('harness_error') for t in tasks), 164)
                self.assertEqual(raw['scored_total'], summary['scored_total'])
                self.assertTrue(summary['audit']['complete_scoring'])
                self.assertEqual(raw['passed'], sum(t['passed'] for t in tasks))
                self.assertEqual(summary['score'], round(raw['passed'] / 164 * 100, 1))
                count += 1
        self.assertEqual(count, 20)
        self.assertEqual(self.data['inventory']['partial_he_arms'], [])

    def test_recovered_rows_keep_matching_accounting(self):
        for key, score, first, wall, tokens, coverage in [
            ('base-iq3_xxs', 94.5, 92.1, 865.8, 82843, 99.2),
            ('swift-iq3_s', 95.7, 92.7, 831.7, 61395, 94.4),
        ]:
            summary = self.data['models'][key]['heplus']['low']
            raw = json.loads(gzip.decompress((ROOT / f'data/heplus/{key}-low.json.gz').read_bytes()))
            for field in ['score', 'first_attempt_score', 'total_time', 'total_output_tokens', 'tok_s', 'sum_generation_time', 'throughput_coverage', 'throughput_wall_time', 'throughput_output_tokens']:
                self.assertEqual(summary[field], raw[field])
            self.assertEqual((summary['score'], summary['first_attempt_score'], summary['total_time'], summary['total_output_tokens'], summary['throughput_coverage']), (score, first, wall, tokens, coverage))
            provenance = summary['source_provenance']
            path = ROOT / provenance['adjudication_artifact']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), provenance['packaged_adjudication_sha256'])
            self.assertEqual(sum(t['output_tokens'] for t in raw['tasks']), tokens)

    def test_terminal_model_failures_not_erased(self):
        base = json.loads(gzip.decompress((ROOT / 'data/heplus/base-iq3_xxs-low.json.gz').read_bytes()))
        task = next(t for t in base['tasks'] if t['task_id'] == 'HumanEval/120')
        self.assertFalse(task['passed'])
        self.assertFalse(task['harness_error'])
        self.assertEqual(len(task['attempts']), 2)
        self.assertIn('SyntaxError', task['attempts'][-1]['error'])
        swift = json.loads(gzip.decompress((ROOT / 'data/heplus/swift-iq3_s-low.json.gz').read_bytes()))
        task = next(t for t in swift['tasks'] if t['task_id'] == 'HumanEval/132')
        self.assertFalse(task['passed'])
        self.assertFalse(task['harness_error'])
        self.assertTrue(task['active_cycle'])
        self.assertEqual(task['loop_evidence'], 'exact_cycle')
        self.assertEqual(task['outcome'], 'loop_killed')

    def test_full_matrix_preserves_quality_wall_without_master_score(self):
        board = json.loads((ROOT / 'data/FINAL-SCOREBOARD.json').read_text())
        self.assertEqual(len(board),20)
        for key, model in self.data['models'].items():
            for effort, summary in model['heplus'].items():
                matches = [r for r in board if r['key'] == key and r['arm'] == effort]
                self.assertEqual(len(matches),1)
                self.assertEqual(matches[0]['coding_final'],summary['score'])
                self.assertEqual(matches[0]['coding_wall_s'],summary['total_time'])
                for field in ['rank','overall','composite','decode']:
                    self.assertNotIn(field,matches[0])
        for key in ['hosted_api_reference', 'hosted_api_reference_sol']:
            self.assertIn(key, self.data)

if __name__ == '__main__':
    unittest.main()
