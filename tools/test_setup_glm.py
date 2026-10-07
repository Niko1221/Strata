"""GLM installer checks without a GPU, downloads or a model."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import setup
from tools import glm_install as I, glm_setup as G

class GlmSetup(unittest.TestCase):
    def test_model_menu_cli_dispatches_before_gpu_checks(self):
        for flags in (['--family','glm'],['--model','GLM-5.3']):
            with mock.patch.object(sys,'argv',['setup.py',*flags,'--yes','--no-start']), mock.patch.object(I,'install',return_value=0) as install, mock.patch.object(setup,'gpus',side_effect=AssertionError('Qwen GPU check')):
                self.assertEqual(setup.main(),0)
                self.assertTrue(install.called)

    def test_resource_check_and_resumed_space(self):
        I.check_resources(63.2,421_000_000_000,419_282_314_240)
        with self.assertRaisesRegex(ValueError,'64 GB'): I.check_resources(32,10**12,0)
        with self.assertRaisesRegex(ValueError,'space'): I.check_resources(64,100,420_000_000_000)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'a.safetensors').write_bytes(b'a'*8); (p/'b.safetensors.part').write_bytes(b'b'*3)
            self.assertEqual(I.bytes_needed(p,[('a.safetensors',8),('b.safetensors',10)]),7)

    def test_manifest_is_pinned_and_paths_are_confined(self):
        meta={'sha':'a'*40,'siblings':[{'rfilename':n,'size':10} for n in ('config.json','tokenizer.json','out.safetensors','../evil.safetensors','folder/no.safetensors')]}
        rev,files=I.model_files(meta)
        self.assertEqual(rev,'a'*40); self.assertEqual(len(files),3)
        with self.assertRaises(ValueError): I.model_files({**meta,'sha':'main'})

    def test_config_and_launchers_and_preferences(self):
        for win in (False,True):
            with self.subTest(windows=win),tempfile.TemporaryDirectory() as d:
                root=Path(d); model=root/'model'; model.mkdir(); exe=root/'strata-glm.exe'; exe.touch()
                cfg=root/'strata-glm53.json'; cfg.write_text(json.dumps({'sampling':{'temperature':0.2},'api_key':'saved'}))
                with mock.patch.object(G,'WIN',win),mock.patch.object(G,'check_model'),mock.patch('strata_tokenizer.extract_hf'):
                    path,script=G.prepare(model,exe,root=root,port=8091,kv='bf16',gpu=0,open_browser=False)
                got=json.loads(path.read_text())
                self.assertEqual(got['family'],'glm'); self.assertEqual(got['sampling'],{'temperature':0.2})
                self.assertEqual(got['api_key'],'saved'); self.assertIn('--gpu',got['args']); self.assertIn('bf16',got['args'])
                self.assertNotIn('--open',script.read_text()); self.assertNotIn(b'\r\r\n',script.read_bytes())
                self.assertEqual(got['port'],8091)
                # the server reads the port from --port only (default 8095): the launcher must pass the config's
                self.assertIn('8091',script.read_text())

    def test_unsafe_names_and_remote_without_key_are_rejected(self):
        for kw in ({'name':'../escape'},{'port':0},{'host':'0.0.0.0'}):
            with self.assertRaises(ValueError): G.prepare('missing','missing',**kw)

    def test_glm_config_is_not_upgraded_with_qwen_options(self):
        cfg={'family':'glm','args':['--prefill','layer'],'exe':'glm'}
        with mock.patch.object(setup,'engine_version',side_effect=AssertionError('Qwen version probe')):
            self.assertIs(setup.upgrade_config(Path('missing'),cfg),cfg)

    def test_start_uses_glm_path(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'strata-glm53.json'; p.write_text(json.dumps({'family':'glm','args':[],'exe':'glm'}))
            with mock.patch.object(G,'start_config',return_value=0) as start, mock.patch.object(setup,'gpus',side_effect=AssertionError('GPU probe')):
                self.assertEqual(setup.start(p,8091),0); self.assertEqual(start.call_args.kwargs['port'],8091)

    def test_start_config_passes_the_configs_port(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'strata-glm53.json'; p.write_text(json.dumps({'family':'glm','args':[],'exe':'glm','port':8080,'open_browser':False}))
            with mock.patch('subprocess.call',return_value=0) as call:
                G.start_config(p)
            cmd=call.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--port')+1],'8080')

if __name__=='__main__': unittest.main()
