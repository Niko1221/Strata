"""Contract for the consolidated ./run.sh launcher (run2.sh/run3.sh removed).
Run: python -m unittest discover -s docker -p 'test_*.py'.

IQ3_XXS still reproduces the launcher-consolidation pin exactly (fixtures in
bench/results/2026-10-04-launcher-consolidation/).  IQ3_S - the shipped default since
11a6026 - carries the tuning measured on the gfx1101 host
(bench/results/2026-10-04-iq3s-tuning/, post-tune fixtures): expert cache `auto`,
STRATA_VRAM_LATER_MIB=700, STRATA_PREFILL_RING=48.  Context stays 131072 and the VRAM
budget stays <= 10240 MiB in every arm.  Stubbed docker/python keep this GPU- and
download-free, like test_runtime_contract.py.
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
TUNED = ROOT / 'bench' / 'results' / '2026-10-04-iq3s-tuning'


def pinned_env_pairs(fixture, fixtures=FIXTURES):
    """The -e KEY=VAL pairs of the docker run line captured in a fixture."""
    text = (fixtures / fixture).read_text()
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

    def test_iq3_s_default_uses_the_measured_tuning(self):
        rc, out, err = launch()
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out), pinned_env_pairs('post-tune-iq3s.txt', TUNED))
        self.assertIn('STRATA_MODEL=IQ3_S', out)
        self.assertIn('STRATA_EXPERT_CACHE=auto', out)
        self.assertIn('STRATA_VRAM_LATER_MIB=700', out)
        self.assertIn('STRATA_PREFILL_RING=48', out)
        self.assertIn('STRATA_MAX_CONTEXT=131072', out)

    def test_iq3xxs_via_model_flag_keeps_800(self):
        rc, out, err = launch('--model', 'IQ3_XXS')
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out), pinned_env_pairs('pre-run-iq3xxs.txt'))

    def test_expert_cache_numeric_overrides_auto_with_warning(self):
        rc, out, err = launch('--model', 'IQ3_S', '--expert-cache', '900')
        self.assertEqual(rc, 0, err)
        self.assertIn('STRATA_EXPERT_CACHE=900', out)
        self.assertIn('overrides the tuned auto sizing', re.sub(r'\x1b\[[0-9;]*m', '', err))

    def test_expert_cache_above_iq3xxs_pin_still_warns(self):
        rc, out, err = launch('--model', 'IQ3_XXS', '--expert-cache', '900')
        self.assertEqual(rc, 0, err)
        self.assertIn('STRATA_EXPERT_CACHE=900', out)
        self.assertIn('tuned 800', re.sub(r'\x1b\[[0-9;]*m', '', err))

    def test_ornith_refused_before_docker(self):
        rc, out, err = launch('--model', 'ornith')
        self.assertNotEqual(rc, 0)
        self.assertNotIn('docker run', out)
        self.assertIn('IQ3_XXS', re.sub(r'\x1b\[[0-9;]*m', '', err))


if __name__ == '__main__':
    unittest.main()
