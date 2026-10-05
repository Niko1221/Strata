"""Contract for the consolidated ./run.sh launcher (run2.sh/run3.sh removed).
Run: python -m unittest discover -s docker -p 'test_*.py'.

The no-argument default is Swift 1.5 IQ3_XXS (plans/run-default-swift-15-iq3xxs-2026-10.md):
the docker line is pinned against bench/results/2026-10-05-run-default-swift/
default-swift-explicit.txt, which is byte-identical to what the live gates ran.  Qwen lines are
pinned too and must stay byte-for-byte what they were before the release axis: --release qwen
--model IQ3_S equals bench/results/2026-10-04-iq3s-tuning/post-tune-iq3s.txt and
--release qwen --model IQ3_XXS equals the launcher-consolidation pin.  Context stays 131072 and
the VRAM budget <= 10240 MiB in every arm.  Stubbed docker/python (see _stub_runtime.py) keep
this GPU- and download-free, like test_runtime_contract.py.
"""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import unittest

from _stub_runtime import stub_python

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'bench' / 'results' / '2026-10-04-launcher-consolidation'
TUNED = ROOT / 'bench' / 'results' / '2026-10-04-iq3s-tuning'


def pinned_env_pairs(fixture, fixtures=FIXTURES):
    """The -e KEY=VAL pairs of the docker run line captured in a fixture."""
    text = (fixtures / fixture).read_text()
    line = next(l for l in text.splitlines() if 'docker run' in l)
    toks = line.split()
    return sorted(toks[i + 1] for i, t in enumerate(toks) if t == '-e' and i + 1 < len(toks))


def launch(*args, env_extra=None):
    """Run ./run.sh --dry-run with docker/python (hipinfo/hfmodel) stubbed; returns (rc, out, err)."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        docker = root / 'docker'
        docker.write_text('#!/bin/sh\nexit 0\n')
        docker.chmod(0o755)
        python = root / 'python'
        stub_python(python)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith('STRATA_') and k != 'HSA_OVERRIDE_GFX_VERSION'}
        env.update(PATH=str(root) + os.pathsep + env['PATH'], PYTHON=str(python), **(env_extra or {}))
        result = subprocess.run(
            ['bash', str(ROOT / 'run.sh'), '--dry-run', '--image', 'test:guard',
             '--hf-cache', directory, '--work', directory, *args],
            env=env, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr


def actual_env_pairs(stdout):
    line = next((l for l in stdout.splitlines() if 'docker run' in l), '')
    toks = line.split()
    return sorted(toks[i + 1] for i, t in enumerate(toks) if t == '-e' and i + 1 < len(toks))


class LauncherContract(unittest.TestCase):
    def test_only_run_sh_launcher(self):
        self.assertEqual(sorted(p.name for p in ROOT.glob('run*.sh')), ['run.sh'])

    def test_launcher_syntax(self):
        subprocess.run(['bash', '-n', str(ROOT / 'run.sh')], check=True)

    def test_swift_default_is_the_gated_line(self):
        rc, out, err = launch()
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out),
                         pinned_env_pairs('default-swift-explicit.txt', SWIFT_DIR))
        self.assertIn('STRATA_MODEL=IQ3_XXS', out)
        self.assertIn('STRATA_MODEL_NAME=swift-1.5-iq3_xxs', out)
        self.assertIn('STRATA_PACK_DIR=/work/packs/swift-iq3_xxs', out)
        self.assertIn('STRATA_EXPERT_CACHE=auto', out)
        self.assertIn('STRATA_VRAM_LATER_MIB=700', out)
        self.assertIn('STRATA_PREFILL_RING=48', out)
        self.assertIn('STRATA_MAX_CONTEXT=131072', out)

    def test_qwen_iq3_s_pin_survives_under_release_qwen(self):
        rc, out, err = launch('--release', 'qwen', '--model', 'IQ3_S')
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out), pinned_env_pairs('post-tune-iq3s.txt', TUNED))

    def test_qwen_iq3xxs_pin_survives_under_release_qwen(self):
        rc, out, err = launch('--release', 'qwen', '--model', 'IQ3_XXS')
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out), pinned_env_pairs('pre-run-iq3xxs.txt'))

    def test_expert_cache_numeric_overrides_auto_with_warning(self):
        rc, out, err = launch('--release', 'qwen', '--model', 'IQ3_S', '--expert-cache', '900')
        self.assertEqual(rc, 0, err)
        self.assertIn('STRATA_EXPERT_CACHE=900', out)
        self.assertIn('overrides the tuned auto sizing', re.sub(r'\x1b\[[0-9;]*m', '', err))

    def test_expert_cache_above_iq3xxs_pin_still_warns(self):
        rc, out, err = launch('--release', 'qwen', '--model', 'IQ3_XXS', '--expert-cache', '900')
        self.assertEqual(rc, 0, err)
        self.assertIn('STRATA_EXPERT_CACHE=900', out)
        self.assertIn('tuned 800', re.sub(r'\x1b\[[0-9;]*m', '', err))

    def test_ornith_refused_before_docker(self):
        rc, out, err = launch('--model', 'ornith')
        self.assertNotEqual(rc, 0)
        self.assertNotIn('docker run', out)
        self.assertIn('IQ3_XXS', re.sub(r'\x1b\[[0-9;]*m', '', err))


SWIFT_DIR = ROOT / 'bench' / 'results' / '2026-10-05-run-default-swift'


class ReleaseAxis(unittest.TestCase):
    """plans/run-default-swift-15-iq3xxs-2026-10.md: the --release axis.  The Swift line is
    pinned byte-for-byte against the captured fixture; Qwen lines must not gain a single -e
    (their historical byte identity is pinned in LauncherContract above)."""

    def test_release_swift_line_matches_the_gated_launch(self):
        rc, out, err = launch('--release', 'swift', '--model', 'IQ3_XXS')
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out),
                         pinned_env_pairs('default-swift-explicit.txt', SWIFT_DIR))

    def test_release_flag_line_equals_the_manual_env_line(self):
        """What users typed yesterday (explicit -e) and what --release does today must give the
        same EFFECTIVE docker environment - docker takes the last -e of a key, so compare the
        last-wins dicts, duplicates included on the manual side."""
        rc1, old, err1 = launch('--model', 'IQ3_XXS',
                               '-e', 'STRATA_HF_REPO=ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF',
                               '-e', 'STRATA_PACK_DIR=/work/packs/swift-iq3_xxs',
                               '-e', 'STRATA_MODEL_NAME=swift-1.5-iq3_xxs',
                               '-e', 'STRATA_EXPERT_CACHE=auto',
                               '-e', 'STRATA_VRAM_LATER_MIB=700',
                               '-e', 'STRATA_PREFILL_RING=48')
        rc2, new, err2 = launch('--release', 'swift', '--model', 'IQ3_XXS')
        self.assertEqual((rc1, rc2), (0, 0), err1 + err2)

        def effective(stdout):
            line = next((l for l in stdout.splitlines() if 'docker run' in l), '')
            toks = line.split()
            env = {}
            for i, t in enumerate(toks):
                if t == '-e' and i + 1 < len(toks) and '=' in toks[i + 1]:
                    k, _, v = toks[i + 1].partition('=')
                    env[k] = v
            return env
        self.assertEqual(effective(old), effective(new))
        self.assertEqual(sorted(f'{k}={v}' for k, v in effective(new).items()),
                         pinned_env_pairs('default-swift-explicit.txt', SWIFT_DIR))

    def test_release_qwen_line_keeps_the_qwen_pin(self):
        rc, qwen, _ = launch('--release', 'qwen', '--model', 'IQ3_S')
        rc2, default, _ = launch()
        self.assertEqual((rc, rc2), (0, 0))
        self.assertNotEqual(actual_env_pairs(qwen), actual_env_pairs(default))
        self.assertEqual(actual_env_pairs(qwen), pinned_env_pairs('post-tune-iq3s.txt', TUNED))

    def test_coder_quant_still_resolves_coder(self):
        rc, out, err = launch('--model', 'IQ1_M')      # no release named: the quant names coder
        self.assertEqual(rc, 0, err)
        self.assertIn('STRATA_MODEL_NAME=qwen3.8-flash-next-coder-iq1_m', out)
        self.assertIn('STRATA_PACK_DIR=/work/packs/coder-iq1_m', out)

    def test_coder_quant_under_qwen_is_refused(self):
        rc, out, err = launch('--release', 'qwen', '--model', 'IQ1_M')
        self.assertNotEqual(rc, 0)
        self.assertNotIn('docker run', out)
        self.assertIn('coder', re.sub(r'\x1b\[[0-9;]*m', '', err))

    def test_unmeasured_combination_warns_and_uses_conservative_pins(self):
        rc, out, err = launch('--release', 'swift', '--model', 'IQ2_XS')
        self.assertEqual(rc, 0, err)
        self.assertIn('not measured on this card', re.sub(r'\x1b\[[0-9;]*m', '', err))
        self.assertIn('STRATA_EXPERT_CACHE=680', out)
        self.assertNotIn('STRATA_PREFILL_RING', out)

    def test_repo_env_still_routes_the_release(self):
        rc, out, err = launch('--model', 'IQ3_XXS', env_extra={
            'STRATA_HF_REPO': 'ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF'})
        self.assertEqual(rc, 0, err)
        self.assertIn('STRATA_MODEL_NAME=swift-1.5-iq3_xxs', out)   # host env alone routes swift
        self.assertIn('STRATA_HF_REPO=ukisai/', out)                 # and is forwarded (it was not
                                                                    # before this change)


if __name__ == '__main__':
    unittest.main()
