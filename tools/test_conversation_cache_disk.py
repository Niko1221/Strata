import subprocess
import sys
import unittest
from pathlib import Path

from tools.conversation_cache_disk import STATE_KEYS, evidence, verify


def fixture(path='batched'):
    mode = 0 if path == 'batched' else 2
    info = dict(context=4096, kv='int8', kv_resident=0 if mode == 0 else 1024, expert_slots=1000,
                spec=2, mtp_max=1, lookup=0, cvec=0, pcie_frac=.55, conversation_cache_mib=8192,
                conversation_cache_slots=1, conversation_disk_mib=16384, conversation_disk_slots=8, conversation_cache_min_free_mib=2560)
    state = {k: '1234' for k in STATE_KEYS}
    def record(name, reused=0, initial=False):
        return dict(name=name, ids=[1] if initial else [31, 41], text='AZURE-314159', finish='stop', reused=reused, state=dict(state))
    def proof(disk=False):
        return dict(verified=[dict(draft='0123456789abcdef', cells=100, mode=mode, source='disk', resident=64 if mode == 2 else 4096)] if disk else [],
                    draft_prefill=[dict(path='batched' if mode == 0 else 'token', mode=mode, cells=99)],
                    spills=1 if disk else 0, promotion_skips=0, integrity_failures=0,
                    parks=[dict(parked=1, bytes=1234)] if disk else [])
    phases = {
        'baseline': dict(info={**info, 'conversation_cache_mib': 0, 'conversation_disk_mib': 0},
                         records=[record('A', initial=True), record('warm', 100), record('B', initial=True), record('cold')], evidence=proof()),
        'producer': dict(info=info.copy(), records=[record('A', initial=True), record('B', initial=True), record('C', initial=True), record('return', 100)],
                         evidence=proof(True), disk_counts=[0, 0, 1, 2], files_after_close=2),
    }
    for name in ('restart', 'admission', 'foreign', 'corrupt'):
        phases[name] = dict(info=info.copy(), records=[record('return', 100 if name == 'restart' else 0)], evidence=proof(name == 'restart'))
    phases['admission']['info']['conversation_cache_min_free_mib'] = 999999
    phases['admission']['evidence']['promotion_skips'] = 1
    phases['corrupt']['evidence']['integrity_failures'] = 1
    for phase in phases.values():
        phase['disk_bytes'] = [2048] * len(phase['records'])
        phase.setdefault('disk_counts', [0] * len(phase['records']))
    return dict(phases=phases, expected='AZURE-314159', prompt_tokens=100, cache_mib=8192, disk_mib=16384,
                draft_path=path, denial_floor_mib=999999, foreign_asset_before='aaa', foreign_asset_after='bbb', corrupted_files=2)


class DiskGate(unittest.TestCase):
    def test_complete_batched_and_ring_evidence(self):
        verify(fixture())
        verify(fixture('ring'))

    def test_missing_or_wrong_evidence_is_rejected(self):
        mutations = {
            'missing phase': lambda d: d['phases'].pop('restart'),
            'wrong answer': lambda d: d['phases']['restart']['records'][0].update(text='BRONZE'),
            'no disk reuse': lambda d: d['phases']['restart']['records'][0].update(reused=0),
            'changed output': lambda d: d['phases']['restart']['records'][0].update(ids=[99]),
            'changed state': lambda d: d['phases']['restart']['records'][0]['state'].update(kv='bad'),
            'missing state': lambda d: d['phases']['restart']['records'][0]['state'].pop('ple'),
            'incomplete request': lambda d: d['phases']['restart']['records'][0].update(finish='error'),
            'fake disk hit': lambda d: d['phases']['restart']['evidence'].update(verified=[]),
            'RAM instead of disk': lambda d: d['phases']['restart']['evidence']['verified'][0].update(source='ram'),
            'wrong draft payload': lambda d: d['phases']['restart']['evidence']['verified'][0].update(draft='f' * 16),
            'wrong resident mode': lambda d: d['phases']['restart']['evidence']['verified'][0].update(mode=2),
            'no batched prefill': lambda d: d['phases']['producer']['evidence']['draft_prefill'][0].update(path='token'),
            'per-turn write': lambda d: d['phases']['producer'].update(disk_counts=[1, 1, 2, 2]),
            'no eviction write': lambda d: d['phases']['producer'].update(disk_counts=[0, 0, 0, 2]),
            'shutdown write': lambda d: d['phases']['producer'].update(files_after_close=3),
            'wrong engine residency': lambda d: d['phases']['restart']['info'].update(expert_slots=999),
            'wrong cache quota': lambda d: d['phases']['restart']['info'].update(conversation_disk_mib=0),
            'RAM budget exceeded': lambda d: d['phases']['producer']['evidence']['parks'][0].update(bytes=8192 * 1024 * 1024 + 1),
            'disk budget exceeded': lambda d: d['phases']['restart'].update(disk_bytes=[16384 * 1024 * 1024 + 1]),
            'disk entry limit exceeded': lambda d: d['phases']['restart'].update(disk_counts=[9]),
            'missing disk occupancy': lambda d: d['phases']['restart'].update(disk_bytes=[]),
            'foreign reused': lambda d: d['phases']['foreign']['records'][0].update(reused=100),
            'no identity change': lambda d: d.update(foreign_asset_after=d['foreign_asset_before']),
            'no corrupted file': lambda d: d.update(corrupted_files=0),
            'no integrity rejection': lambda d: d['phases']['corrupt']['evidence'].update(integrity_failures=0),
            'no admission rejection': lambda d: d['phases']['admission']['evidence'].update(promotion_skips=0),
            'wrong admission floor': lambda d: d['phases']['admission']['info'].update(conversation_cache_min_free_mib=2560),
            'warm cold-control': lambda d: d['phases']['baseline']['records'][3].update(reused=100),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                data = fixture()
                mutate(data)
                with self.assertRaises(AssertionError):
                    verify(data)

    def test_log_evidence(self):
        parsed = evidence('''strata serve: DRAFT_PREFILL path=batched mode=0 cells=1024
strata serve: SNAPSHOT_VERIFY draft=0123456789abcdef cells=2048 mode=0 source=disk resident=4096
strata serve: conversation cache: parked 2048 tokens in 1.0 ms; parked=1 bytes=8192 evictions=1
strata serve: disk cache: spilled 2048 tokens
strata serve: disk cache: promotion skipped (snapshot integrity check failed)
''')
        self.assertEqual(parsed['verified'], [dict(draft='0123456789abcdef', cells=2048, mode=0, source='disk', resident=4096)])
        self.assertEqual(parsed['parks'], [dict(parked=1, bytes=8192)])
        self.assertEqual(parsed['spills'], 1)
        self.assertEqual(parsed['integrity_failures'], 1)
        self.assertEqual(parsed['promotion_skips'], 1)

    def test_ring_must_wrap_before_restore(self):
        for phase in ('producer', 'restart'):
            for resident in (0, 100, 1024):
                with self.subTest(phase=phase, resident=resident):
                    data = fixture('ring')
                    data['phases'][phase]['evidence']['verified'][0]['resident'] = resident
                    with self.assertRaisesRegex(AssertionError, 'actual draft ring'):
                        verify(data)

    def test_old_log_is_not_residency_proof(self):
        self.assertEqual(evidence('SNAPSHOT_VERIFY draft=0123456789abcdef cells=2048 mode=2 source=disk')['verified'], [])

    def test_dry_run_does_not_open_config_or_engine(self):
        script = Path(__file__).with_name('conversation_cache_disk.py')
        result = subprocess.run([sys.executable, str(script), '--config', '/does-not-exist', '--engine', '/also-missing',
                                 '--output', '/not-created', '--draft-path', 'batched'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('No model loaded', result.stdout)

    def test_verifier_survives_optimized_python(self):
        command = 'from tools.test_conversation_cache_disk import fixture; from tools.conversation_cache_disk import verify; d=fixture(); d["phases"]["restart"]["records"][0]["text"]="wrong"; verify(d)'
        result = subprocess.run([sys.executable, '-O', '-c', command], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('known answer is wrong', result.stderr)


if __name__ == '__main__':
    unittest.main()
