"""Contract for the consolidated ./run.sh launcher (run2.sh/run3.sh removed).
Run: python -m unittest discover -s docker -p 'test_*.py'.

Pins the pre-change behavior measured on the gfx1101 host in
bench/results/2026-10-04-launcher-consolidation/ (see its README): the old run.sh
(IQ3_XXS, expert cache 800) and the old run2.sh (IQ3_S, expert cache 680) are both
reproduced by run.sh, and the docker -e environment matches the captured fixtures.
Stubbed docker/python keep this GPU- and download-free, like test_runtime_contract.py.
"""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'bench' / 'results' / '2026-10-04-launcher-consolidation'


def pinned_env_pairs(fixture):
    """The -e KEY=VAL pairs of the docker run line captured in a pre-change fixture."""
    text = (FIXTURES / fixture).read_text()
    line = next(l for l in text.splitlines() if 'docker run' in l)
    toks = line.split()
    return sorted(toks[i + 1] for i, t in enumerate(toks) if t == '-e' and i + 1 < len(toks))


def launch(*args):
    """Run ./run.sh --dry-run with docker/python (hipinfo/hfmodel) stubbed; returns (rc, out, err)."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        docker = root / 'docker'
        docker.write_text('#!/bin/sh\nexit 0\n')
        docker.chmod(0o755)
        python = root / 'python'
        python.write_text(f"#!{sys.executable}\n" + """import os, sys
if sys.argv[1].endswith('hipinfo.py'):
    if '--arch' in sys.argv: print('gfx1101')
    elif '--render-node' in sys.argv: print('/dev/dri/renderD128')
    elif '--vram' in sys.argv: print('12272 1500 10772')
    elif '--reserve-mib' in sys.argv: print('768')
elif sys.argv[1].endswith('hfmodel.py'):
    print('STRATA_CACHED=1; STRATA_ARENA_GB=43')
else:
    os.execv(sys.executable, [sys.executable] + sys.argv[1:])
""")
        python.chmod(0o755)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith('STRATA_') and k != 'HSA_OVERRIDE_GFX_VERSION'}
        env.update(PATH=str(root) + os.pathsep + env['PATH'], PYTHON=str(python))
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

    def test_iq3xxs_default_reproduces_pre_change_run_sh(self):
        rc, out, err = launch()
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out), pinned_env_pairs('pre-run-iq3xxs.txt'))
        self.assertIn('STRATA_MODEL=IQ3_XXS', out)
        self.assertIn('STRATA_EXPERT_CACHE=800', out)

    def test_iq3_s_reproduces_pre_change_run2_sh(self):
        rc, out, err = launch('--model', 'IQ3_S')
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out), pinned_env_pairs('pre-run2-iq3s.txt'))
        self.assertIn('STRATA_MODEL=IQ3_S', out)
        self.assertIn('STRATA_EXPERT_CACHE=680', out)

    def test_iq3xxs_via_model_flag_keeps_800(self):
        rc, out, err = launch('--model', 'IQ3_XXS')
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out), pinned_env_pairs('pre-run-iq3xxs.txt'))

    def test_expert_cache_above_tuned_warns_but_passes_through(self):
        rc, out, err = launch('--model', 'IQ3_S', '--expert-cache', '900')
        self.assertEqual(rc, 0, err)
        self.assertIn('STRATA_EXPERT_CACHE=900', out)
        self.assertIn('tuned 680', re.sub(r'\x1b\[[0-9;]*m', '', err))

    def test_ornith_refused_before_docker(self):
        rc, out, err = launch('--model', 'ornith')
        self.assertNotEqual(rc, 0)
        self.assertNotIn('docker run', out)
        self.assertIn('IQ3_XXS', re.sub(r'\x1b\[[0-9;]*m', '', err))


if __name__ == '__main__':
    unittest.main()
