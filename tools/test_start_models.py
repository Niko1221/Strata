"""Desktop launcher checks: no GPU, downloads, real configs or process mutations."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import start_models
import personal_control


class DesktopLaunchArguments(unittest.TestCase):
    def test_default_uses_installed_config_and_native_only(self):
        args = start_models.parse_args(['--no-browser'])
        self.assertIsNone(args.model)
        self.assertTrue(args.no_browser)
        self.assertFalse(args.start_ollama)

    def test_explicit_installed_tags_and_names_are_accepted(self):
        for model in ('coder-iq1_m', 'iq2_xs', 'custom-native-tag', 'An Installed Native Model'):
            self.assertEqual(start_models.parse_args(['--model', model, '--no-browser']).model, model)

    def test_optional_backend_flag_and_missing_model_value(self):
        self.assertTrue(start_models.parse_args(['--start-ollama']).start_ollama)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            start_models.parse_args(['--model'])


class DesktopLaunchBehavior(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.manager = mock.Mock()
        self.manager.state_dir = self.root / '.strata-mcp'
        self.manager.api_key_for.return_value = None
        self.manager.tracked_server.return_value = None
        self.manager.find_config.return_value = self.root / 'strata-custom-native.json'
        self.manager.configs.return_value = [self.manager.find_config.return_value]
        self.manager.run_python.return_value = 'synthetic-python'
        self.patch = mock.patch.multiple(start_models, ROOT=self.root,
            Strata=mock.Mock(return_value=self.manager), spawn_detached=mock.Mock(return_value=mock.Mock(pid=4242)),
            proc_identity=mock.Mock(return_value='synthetic-process'))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        environment = mock.patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_native_only_reuses_server_without_starting_ollama(self):
        with mock.patch.object(start_models, 'reachable', return_value={'service': 'strata'}) as reachable:
            self.assertEqual(start_models.main(['--no-browser']), 0)
        reachable.assert_called_once_with('http://127.0.0.1:8080/health', None)
        start_models.spawn_detached.assert_not_called()

    def test_cold_launch_uses_latest_config_and_forces_loopback(self):
        with mock.patch.object(start_models, 'reachable', side_effect=[None, {'service': 'strata'}]):
            self.assertEqual(start_models.main(['--no-browser']), 0)
        self.manager.find_config.assert_not_called()
        self.manager.configs.assert_called_once_with()
        command = start_models.spawn_detached.call_args.args[0]
        self.assertEqual(command[command.index('--host') + 1], '127.0.0.1')
        self.assertIn('--lazy', command)
        self.assertEqual(self.manager.save_state.call_args.args[1]['model'], 'custom-native')

    def test_environment_key_is_used_without_command_line_exposure(self):
        with mock.patch.dict(os.environ, {'STRATA_API_KEY': 'synthetic-test-key'}), \
             mock.patch.object(start_models, 'reachable', side_effect=[None, {'service': 'strata'}]) as reachable:
            self.assertEqual(start_models.main(['--no-browser', '--model', 'custom-native']), 0)
        self.assertTrue(all(call.args[1] == 'synthetic-test-key' for call in reachable.call_args_list))
        self.assertNotIn('synthetic-test-key', start_models.spawn_detached.call_args.args[0])
        self.manager.find_config.assert_called_once_with('custom-native')

    def test_newer_shared_settings_are_not_launched_as_models(self):
        config = self.root / 'strata-custom-native.json'
        settings = self.root / 'strata-custom-native.shared-settings.json'
        config.write_text('{}', encoding='utf-8')
        settings.write_text('{"temperature":0.5}', encoding='utf-8')
        os.utime(config, (1, 1))
        os.utime(settings, (2, 2))
        self.manager.configs.side_effect = lambda: sorted(self.root.glob('strata-*.json'),
            key=lambda path: path.stat().st_mtime, reverse=True)
        with mock.patch.object(start_models, 'reachable', side_effect=[None, {'service': 'strata'}]):
            self.assertEqual(start_models.main(['--no-browser']), 0)
        command = start_models.spawn_detached.call_args.args[0]
        self.assertEqual(command[command.index('--config') + 1], str(config))

    def test_convenience_helper_passes_filtered_config_tag_to_manager(self):
        self.manager.configs.return_value = [self.root / 'strata-custom-native.shared-settings.json',
                                            self.root / 'strata-custom-native.json']
        self.manager.describe_config.return_value = {'ready': True, 'model': 'custom-native', 'model_name': 'synthetic-native'}
        self.manager.probe.return_value = {'loaded': True, 'model': 'synthetic-native'}
        tools = mock.Mock(s=self.manager)
        tools.call.return_value = {}
        self.assertEqual(personal_control.start_model(tools, None)['model'], 'synthetic-native')
        self.manager.describe_config.assert_called_once_with(self.root / 'strata-custom-native.json')
        self.assertIn(mock.call('strata_start', {'model': 'custom-native', 'wait_seconds': 0}), tools.call.call_args_list)

    def test_explicit_shared_settings_are_rejected(self):
        self.manager.find_config.return_value = self.root / 'strata-custom-native.shared-settings.json'
        with self.assertRaisesRegex(RuntimeError, 'shared sampling settings'):
            start_models.model_config(self.manager, 'custom-native.shared-settings')

    def test_stop_uses_active_config_key_and_preserves_other_ports(self):
        config = self.root / 'strata-active.json'
        config.write_text('{}', encoding='utf-8')
        self.manager.state.return_value = {'config': config.name}
        self.manager.read_config.return_value = {'api_key': 'synthetic-active-key'}
        original = self.manager.api_key_for
        original.return_value = 'synthetic-other-key'
        def stop():
            self.assertEqual(self.manager.api_key_for(8080), 'synthetic-active-key')
            self.assertEqual(self.manager.api_key_for(9090), 'synthetic-other-key')
            return {'summary': 'Synthetic backend stopped'}
        tools = mock.Mock()
        tools.strata_stop.side_effect = stop
        response = mock.MagicMock()
        response.__enter__.return_value = io.StringIO('{}')
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(start_models, 'reachable', return_value={'service': 'strata'}), \
             mock.patch.object(start_models, 'Tools', return_value=tools), \
             mock.patch.object(start_models.urllib.request, 'build_opener', return_value=opener):
            self.assertEqual(start_models.main(['--stop']), 0)
        self.assertIs(self.manager.api_key_for, original)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header('Authorization'), 'Bearer synthetic-active-key')

    def test_managed_config_key_wins_over_another_models_key(self):
        config = self.root / 'strata-active.json'
        config.write_text('{}', encoding='utf-8')
        self.manager.state.return_value = {'config': config.name}
        self.manager.read_config.return_value = {'api_key': 'synthetic-active-key'}
        self.manager.api_key_for.return_value = 'synthetic-other-model-key'
        with mock.patch.object(start_models, 'reachable', return_value={'service': 'strata'}) as reachable:
            self.assertEqual(start_models.main(['--no-browser']), 0)
        reachable.assert_called_once_with('http://127.0.0.1:8080/health', 'synthetic-active-key')
        self.manager.api_key_for.assert_not_called()

    def test_missing_optional_ollama_is_reported_only_when_requested(self):
        with mock.patch.object(start_models, 'reachable', return_value=None), \
             mock.patch.object(start_models.shutil, 'which', return_value=None), \
             mock.patch.object(Path, 'is_file', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Ollama is missing'):
                start_models.main(['--no-browser', '--start-ollama'])
        start_models.spawn_detached.assert_not_called()

    def test_explicit_ollama_start_uses_installed_binary(self):
        with mock.patch.object(start_models, 'reachable', side_effect=[None, {'service': 'strata'}]), \
             mock.patch.object(start_models.shutil, 'which', return_value='synthetic-ollama'):
            self.assertEqual(start_models.main(['--no-browser', '--start-ollama']), 0)
        command, _, _, environment = start_models.spawn_detached.call_args.args
        self.assertEqual(command, ['synthetic-ollama', 'serve'])
        self.assertEqual(environment['OLLAMA_HOST'], '127.0.0.1:11434')
        self.assertNotIn('OLLAMA_CONTEXT_LENGTH', environment)


if __name__ == '__main__':
    unittest.main()
