"""Behavioral tests for fresh-request DFlash2 calibration; no models/GPU needed."""
import io
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate as C
import calibrate_dflash2 as D

BASE = ['--dflash', 'draft.gguf', '--prefill', '256', '--spec', '8',
        '--spec-min-p', '0.5', '--max-context', '8192', '--expert-cache', 'auto']


class Process:
    def __init__(self):
        self.stdin = io.StringIO()
        self.closed = False
    def wait(self, timeout=None):
        self.closed = True
        return 0
    def kill(self):
        self.closed = True


class Engine:
    def __init__(self, args, mode='normal', maximum=7):
        self.args, self.mode = list(args), mode
        self.prefill = int(C.arg_value(args, '--prefill')) if C.arg_value(args, '--prefill') not in (None, 'auto') else 256
        reserve = int(C.arg_value(args, '--vram-reserve-mib') or 600)
        cap = C.arg_value(args, '--expert-cache')
        slots = int(cap) if cap not in (None, 'auto') else 4000 - reserve
        free = reserve - 400 + (3000 - slots) * 2 if cap not in (None, 'auto') else reserve - 400
        self.info = dict(prefill_cap=self.prefill, dflash_block=min(int(C.arg_value(args, '--dflash-block') or maximum), maximum), dflash_max=maximum,
                         spec_min_p=float(C.arg_value(args, '--spec-min-p') or .5), vram_free_mib=free, expert_slots=slots,
                         expert_cache_mib=slots * 2)
        self.last, self.proc, self.seen = {}, Process(), []
    def generate(self, ids, max_new, sampling, cancel):
        tune = sampling['strata_tune']
        k, f = tune.get('dflash_block', self.info['dflash_block']), tune.get('spec_min_p', self.info['spec_min_p'])
        assert 1 <= k <= self.info['dflash_max']
        assert sampling['strata_checkpoint'] is False
        assert C.arg_value(self.args, '--prompt-cache') == '0'
        self.seen.append((len(ids), k, f))
        long = len(ids) > 100
        prompt = ({256: 6000, 512: 3200, 1024: 3000}[self.prefill] if long else 30)
        rate = 80 if self.prefill == 512 and k == 6 and f == .85 else 60
        if self.mode == 'bad_hardware' and C.arg_value(self.args, '--pcie-frac') == '0.55':
            rate *= .4
        if self.prefill == 1024:
            rate = 25  # lower residency makes this profile lose despite faster prefill
        if self.mode in ('flat', 'tiny', 'early'):
            prompt, rate = (5000 if long else 30), 60
        if self.mode == 'tiny' and self.prefill == 512:
            prompt, rate = prompt / 1.02, rate * 1.02
        if self.mode == 'heldout' and self.prefill == 512 and len(ids) > 3000:
            prompt = 100000  # coarse winner loses on the extra confirmation document
        n = 16 if self.mode == 'early' and self.prefill == 1024 else max_new
        if self.mode == 'early' and self.prefill == 1024:
            rate = 30
        self.last = dict(generated=n, prompt_ms=prompt, decode_ms=n / rate * 1000,
                         reused=1 if self.mode == 'cached' else 0)
        if self.mode == 'cancel':
            cancel.set()
        yield from [1] * n


class Profiles(unittest.TestCase):
    def run_profile(self, mode='normal', base=BASE, maximum=7, start=None, progress=None, event=None):
        engines = []
        def launch(args):
            e = Engine(args, mode, maximum)
            engines.append(e)
            return e
        result = D.measure_profiles(base, {}, [[1] * 20, [1] * 30, [1] * 40],
                                    lambda i, n: [1] * n, start or launch, say=lambda *_: None,
                                    progress=progress, cancel_event=event)
        self.assertTrue(all(e.proc.closed for e in engines))
        return result, engines

    def test_joint_winner_and_actual_budget_are_persisted(self):
        (settings, report), engines = self.run_profile()
        self.assertEqual(settings['--prefill'], '512')
        self.assertEqual(settings['--dflash-block'], '6')
        self.assertEqual(settings['--spec-min-p'], '0.85')
        self.assertGreater(int(settings['--vram-reserve-mib']), 600)
        self.assertTrue(all(g > C.MIN_GAIN for g in report['paired_gains']))
        self.assertEqual(len(report['confirmation']['candidate']), C.SWEEP_ROUNDS)
        self.assertTrue(all(p['info']['vram_free_mib'] >= D.MIN_FREE_MIB for p in report['profiles'].values()))
        self.assertTrue(any(n > 3000 for e in engines for n, _, _ in e.seen))
        applied = C.apply(BASE, settings)
        self.assertEqual(C.arg_value(applied, '--prefill'), '512')

    def test_expert_cap_and_user_reserve_can_only_become_safer(self):
        base = C.with_arg(BASE, '--expert-cache', '3000')
        base = C.with_arg(base, '--vram-reserve-mib', '700')
        (settings, report), _ = self.run_profile(base=base)
        self.assertLess(int(settings['--expert-cache']), 3000)
        self.assertGreaterEqual(int(settings.get('--vram-reserve-mib', '700')), 700)
        self.assertTrue(all(int(p['info']['expert_slots']) <= 3000 for p in report['profiles'].values()))

    def test_flat_or_subthreshold_gain_keeps_settings(self):
        for mode in ('flat', 'tiny'):
            with self.subTest(mode=mode):
                (settings, _), _ = self.run_profile(mode)
                self.assertEqual(settings, {})

    def test_heldout_confirmation_rejects_coarse_winner(self):
        (settings, report), _ = self.run_profile('heldout')
        self.assertEqual(settings, {})
        self.assertEqual(report['selected'], 'baseline')
        self.assertEqual(len(report['confirmation']['baseline']), C.SWEEP_ROUNDS)

    def test_early_eos_does_not_win_by_output_length(self):
        (settings, _), _ = self.run_profile('early')
        self.assertEqual(settings, {})

    def test_cached_reads_cannot_enter_fresh_profile(self):
        (settings, report), _ = self.run_profile('cached')
        self.assertEqual(settings, {})
        self.assertIn('no complete safe baseline', report['skipped'])

    def test_failed_larger_candidate_does_not_erase_other_results(self):
        engines = []
        def start(args):
            if C.arg_value(args, '--prefill') == '1024':
                raise RuntimeError('out of VRAM')
            e = Engine(args)
            engines.append(e)
            return e
        (settings, report), _ = self.run_profile(start=start)
        self.assertEqual(settings['--prefill'], '512')
        self.assertIn('out of VRAM', report['failed']['prefill-1024'])
        self.assertTrue(all(e.proc.closed for e in engines))

    def test_residency_drift_invalidates_confirmation(self):
        calls = 0
        def start(args):
            nonlocal calls
            calls += 1
            e = Engine(args)
            if calls > 12:
                e.info['expert_slots'] -= calls
            return e
        (settings, report), _ = self.run_profile(start=start)
        self.assertEqual(settings, {})
        self.assertTrue(any('residency changed' in x for x in report['failed'].values()))

    def test_context_budget_bounds_long_documents(self):
        base = C.with_arg(BASE, '--max-context', '2048')
        (_, report), engines = self.run_profile(base=base)
        self.assertTrue(all(n + C.MAX_NEW <= 2048 for e in engines for n, _, _ in e.seen))
        self.assertIn('profiles', report)

    def test_auto_prefill_does_not_hide_the_explicit_chunk_candidate(self):
        base = C.with_arg(BASE, '--prefill', 'auto')
        (_, report), _ = self.run_profile(base=base)
        self.assertIn('prefill-256', report['profiles'])

    def test_loaded_artifact_limits_forward_grid(self):
        (_, _), engines = self.run_profile(maximum=3)
        self.assertTrue(all(k <= 3 for e in engines for _, k, _ in e.seen))

    def test_baseline_winner_does_not_inherit_rejected_hardware(self):
        original = C.with_arg(BASE, '--prefill', '512') + ['--pcie-frac', '0', '--pool-workers', '11', '--adapt-swaps', '0']
        proposal = {'--pcie-frac': '0.55', '--pool-workers': '2'}
        engines = []
        def start(args):
            e = Engine(args, 'bad_hardware')
            engines.append(e)
            return e
        result, report = D.measure_profiles(original, proposal, [[1] * 20],
                          lambda i, n: [1] * n, start, baseline_args=original, say=lambda *_: None)
        self.assertEqual(report['selected'], 'baseline')
        self.assertEqual(result['--pcie-frac'], '0')
        self.assertEqual(result['--pool-workers'], '11')
        self.assertEqual(result['--adapt-swaps'], '0')
        self.assertTrue(all(e.proc.closed for e in engines))

    def test_failed_profile_confirmation_restores_original_calibration(self):
        from tools.test_calibrate import FakeEngine
        original = BASE + ['--dflash-block', '6', '--pcie-frac', '0', '--adapt-swaps', '0']
        original = C.with_arg(original, '--spec-min-p', '0.85')
        class Draft(FakeEngine):
            def __init__(self, args):
                super().__init__(args, lambda *_: 50, info_workers=0)
                self.info.update(dflash_block=7, dflash_max=7)
        with mock.patch.object(D, 'measure_profiles', return_value=({}, {'selected': 'baseline'})) as phase:
            result = C.measure(original, [[1] * 20], Draft, say=lambda *_: None,
                               long_prompt=lambda i, n: [1] * n, dflash2=True)
        self.assertEqual(phase.call_args.kwargs['baseline_args'], original)
        self.assertEqual(result['settings']['--dflash-block'], '6')
        self.assertEqual(result['settings']['--spec-min-p'], '0.85')
        self.assertEqual(result['settings']['--adapt-swaps'], '0')
        self.assertIsNone(result['report']['tok_s'])

    def test_packaged_import_keeps_the_cancellation_exception_identity(self):
        from tools import calibrate as packaged
        event = threading.Event()
        event.set()
        with self.assertRaises(packaged.CalibrationCancelled):
            D.measure_profiles(BASE, {}, [[1]], lambda i, n: [1] * n,
                               lambda a: self.fail('cancelled operation started an engine'),
                               cancel_event=event, calibration=packaged)

    def test_cancellation_closes_the_active_engine(self):
        engines, stopped = [], threading.Event()
        def start(args):
            e = Engine(args, 'cancel')
            engines.append(e)
            return e
        with self.assertRaises(C.CalibrationCancelled):
            self.run_profile(start=start, event=stopped)
        self.assertTrue(engines)
        self.assertTrue(all(e.proc.closed for e in engines))


if __name__ == '__main__':
    unittest.main()
