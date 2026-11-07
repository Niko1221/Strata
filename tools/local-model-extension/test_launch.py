import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import launch


class LauncherTests(unittest.TestCase):
    def fixture(self):
        folder=tempfile.TemporaryDirectory(prefix='strata-extension launcher ')
        self.addCleanup(folder.cleanup)
        root=Path(folder.name)
        app=root/'app.exe';app.write_bytes(b'fixture')
        config=root/'extension.json';config.write_text(json.dumps({'app_executable':str(app)}))
        return root,app,config

    def test_missing_preload_runs_plain_app_and_preserves_existing_environment(self):
        root,app,config=self.fixture()
        with patch.object(launch,'__file__',str(root/'launch.py')),patch.dict(os.environ,{'NODE_OPTIONS':'--trace-warnings'}):
            actual,env,available=launch.plan(config)
        self.assertEqual(actual,app)
        self.assertFalse(available)
        self.assertEqual(env['NODE_OPTIONS'],'--trace-warnings')

    def test_external_preload_uses_a_file_uri_for_spaces_and_never_edits_the_app(self):
        root,app,config=self.fixture()
        for name in ['host-preload.mjs','extension_bridge.mjs','owner_api.mjs']:(root/name).write_text('// fixture')
        original=app.read_bytes()
        with patch.object(launch,'__file__',str(root/'launch.py')),patch.dict(os.environ,{'NODE_OPTIONS':'--trace-warnings'}):
            actual,env,available=launch.plan(config)
        self.assertTrue(available)
        self.assertIn('--trace-warnings --import=file:',env['NODE_OPTIONS'])
        self.assertIn('%20',env['NODE_OPTIONS'])
        self.assertEqual(env['STRATA_LOCAL_MODELS_CONFIG'],str(config.resolve()))
        self.assertEqual(app.read_bytes(),original)

    def test_nonexistent_or_non_app_targets_are_rejected(self):
        root,app,config=self.fixture()
        for target in [str(root/'missing.exe'),str(root),str(root/'app.txt')]:
            config.write_text(json.dumps({'app_executable':target}))
            with self.assertRaises(ValueError):launch.plan(config)
