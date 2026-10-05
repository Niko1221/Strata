"""Run with python -m unittest discover -s docker -p 'test_*.py'."""
from pathlib import Path
import subprocess
import os
import sys
import tempfile
import unittest

from _stub_runtime import stub_python

ROOT = Path(__file__).resolve().parents[1]


class RuntimeContract(unittest.TestCase):
    def reject(self, *args):
        result = subprocess.run(['bash', str(ROOT / 'run.sh'), '--dry-run', *args],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        return result.stderr

    def test_context_is_fixed(self):
        self.assertIn('131072', self.reject('--max-context', '65536'))

    def test_budget_cannot_be_raised(self):
        self.assertIn('hard ceiling', self.reject('--budget', '10241'))

    def test_budget_must_leave_reserves(self):
        self.assertIn('hard ceiling', self.reject('--budget', '1280'))

    def test_launcher_syntax(self):
        subprocess.run(['bash', '-n', str(ROOT / 'run.sh'), str(ROOT / 'docker/entrypoint-hip.sh')], check=True)

    def test_tuned_defaults_and_cli_overrides(self):
        # Stub discovery only; exercise the actual launcher argument construction.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docker = root / 'docker'
            docker.write_text('#!/bin/sh\nexit 0\n')
            docker.chmod(0o755)
            python = root / 'python'
            stub_python(python)
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith('STRATA_') and k != 'HSA_OVERRIDE_GFX_VERSION'}
            env.update(PATH=str(root) + os.pathsep + env['PATH'], PYTHON=str(python))
            args = ['bash', str(ROOT / 'run.sh'), '--dry-run', '--image', 'test:guard',
                    '--hf-cache', directory, '--work', directory]
            default = subprocess.run(args, env=env, capture_output=True, text=True, check=True).stdout
            for setting in ('STRATA_MODEL=IQ3_XXS', 'STRATA_MODEL_NAME=swift-1.5-iq3_xxs',
                            'STRATA_PACK_DIR=/work/packs/swift-iq3_xxs',
                            'STRATA_HF_REPO=ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF',
                            'STRATA_PREFILL=2048', 'STRATA_EXPERT_CACHE=auto',
                            'STRATA_POOL_WORKERS=0', 'STRATA_MAX_CONTEXT=131072',
                            'STRATA_VRAM_LATER_MIB=700', 'STRATA_VRAM_RUNTIME_RESERVE_MIB=1024',
                            'STRATA_PREFILL_RING=48'):
                self.assertIn(setting, default)
            qwen = subprocess.run(args + ['--release', 'qwen', '--model', 'IQ3_XXS'], env=env,
                                  capture_output=True, text=True, check=True).stdout
            self.assertIn('STRATA_EXPERT_CACHE=800', qwen)
            self.assertIn('STRATA_VRAM_LATER_MIB=768', qwen)
            self.assertNotIn('STRATA_PREFILL_RING', qwen)  # pins belong to their release+quant
            self.assertNotIn('STRATA_MODEL_NAME', qwen)    # the qwen line gains no -e at all
            override = subprocess.run(args + ['--prefill', '1024', '--expert-cache', '512',
                                             '--pool-workers', '23'], env=env,
                                      capture_output=True, text=True, check=True).stdout
            for key, initial, explicit in [('PREFILL', '2048', '1024'),
                                           ('EXPERT_CACHE', 'auto', '512'),
                                           ('POOL_WORKERS', '0', '23')]:
                self.assertGreater(override.index(f'STRATA_{key}={explicit}'),
                                   override.index(f'STRATA_{key}={initial}'))


class ProcessAccounting(unittest.TestCase):
    def test_clients_are_deduplicated_and_device_filtered(self):
        import importlib.util
        import sys
        import tempfile
        sys.path.insert(0, str(ROOT / 'docker'))
        spec = importlib.util.spec_from_file_location('vram_guard', ROOT / 'docker/vram-guard.py')
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d) / '123' / 'fdinfo'
            directory.mkdir(parents=True)
            fields = 'drm-driver: amdgpu\ndrm-client-id: 7\ndrm-pdev: 0000:03:00.0\ndrm-total-vram: 4096 KiB\n'
            (directory / '1').write_text(fields)
            (directory / '2').write_text(fields)
            (directory / '3').write_text(fields.replace('client-id: 7', 'client-id: 8').replace('4096', '1024'))
            (directory / '4').write_text(fields.replace('03:00', '09:00'))
            self.assertEqual(guard.process_vram(123, '0000:03:00.0', Path(d)), 5120 * 1024)
            self.assertIsNone(guard.process_vram(124, '0000:03:00.0', Path(d)))
            self.assertIsNone(guard.process_vram(123, '0000:05:00.0', Path(d)))


if __name__ == '__main__':
    unittest.main()
