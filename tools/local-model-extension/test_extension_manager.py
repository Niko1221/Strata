import json
from pathlib import Path
import tempfile
import unittest

import extension_manager as manager


class ExtensionManagerTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.source = self.folder/'source'
        (self.source/'ui').mkdir(parents=True)
        self.root = self.folder/'extensions'/'strata-local-models'
        self.manifest = {'schema': 1, 'id': 'strata-local-models', 'version': '1.0.0',
                         'bridgeApi': 1, 'app': {'major': 0, 'minor': 2},
                         'entry': 'ui/entry.mjs', 'panel': 'ui/index.html'}
        self.write_manifest()
        (self.source/'ui/entry.mjs').write_text('export const entry=true;')
        (self.source/'ui/index.html').write_text('<html>Current Model</html>')
        self.carrier = self.folder/'requested-context.json'
        self.carrier.write_text('{"tokens":131072}')

    def write_manifest(self):
        (self.source/'manifest.json').write_text(json.dumps(self.manifest))

    def test_install_and_activate_only_own_package_and_preferences(self):
        record = manager.install(self.source, self.root, activate=True)
        value = manager.state(self.root)
        self.assertEqual(value['active'], record)
        self.assertTrue(value['enabled'])
        self.assertIsNone(value['previous'])
        self.assertEqual(manager.read_json(self.root/'preferences.json')['uiCopyStyle'], 'title_case')
        self.assertEqual(self.carrier.read_text(), '{"tokens":131072}')
        manager.verify_release(self.root, record)

    def test_upgrade_and_rollback_preserve_saved_preferences(self):
        first = manager.install(self.source, self.root, activate=True)
        preferences = '{"schema":1,"uiCopyStyle":"title_case","retained":"user"}'
        (self.root/'preferences.json').write_text(preferences)
        self.manifest['version'] = '1.0.1'
        self.write_manifest()
        second = manager.install(self.source, self.root, activate=True)
        self.assertEqual(manager.state(self.root)['previous'], first)
        rolled = manager.rollback(self.root)
        self.assertEqual(rolled['active'], first)
        self.assertEqual(rolled['previous'], second)
        self.assertEqual((self.root/'preferences.json').read_text(), preferences)
        self.assertEqual(self.carrier.read_text(), '{"tokens":131072}')

    def test_reinstall_is_idempotent_and_does_not_lose_previous(self):
        record = manager.install(self.source, self.root, activate=True)
        again = manager.install(self.source, self.root, activate=True)
        self.assertEqual(record, again)
        self.assertIsNone(manager.state(self.root)['previous'])

    def test_unknown_or_non_boolean_features_cannot_replace_a_valid_activation(self):
        first = manager.install(self.source, self.root, activate=True)
        for features in ({'chatEffort': True}, {'chatEffort': 'true'}, {'unknown': True}, [], {}):
            self.manifest['features'] = features
            self.write_manifest()
            with self.assertRaises(ValueError):
                manager.install(self.source, self.root, activate=True)
            self.assertEqual(manager.state(self.root)['active'], first)

    def test_disable_keeps_package_and_can_be_enabled_again(self):
        record = manager.install(self.source, self.root, activate=True)
        value = manager.set_enabled(self.root, False)
        self.assertFalse(value['enabled'])
        self.assertEqual(value['active'], record)
        self.assertTrue(manager.set_enabled(self.root, True)['enabled'])

    def test_corrupt_asset_cannot_be_enabled_but_can_be_disabled(self):
        record = manager.install(self.source, self.root, activate=True)
        (self.root/'releases'/record['release']/'ui/entry.mjs').write_text('changed')
        self.assertFalse(manager.set_enabled(self.root, False)['enabled'])
        with self.assertRaises(ValueError):
            manager.set_enabled(self.root, True)

    def test_failed_update_does_not_change_active_release(self):
        manager.install(self.source, self.root, activate=True)
        previous = (self.root/'active.json').read_bytes()
        self.manifest['bridgeApi'] = 2
        self.write_manifest()
        with self.assertRaises(ValueError):
            manager.install(self.source, self.root, activate=True)
        self.assertEqual((self.root/'active.json').read_bytes(), previous)

    def test_missing_entry_cannot_be_activated(self):
        self.manifest['entry'] = 'ui/missing.mjs'
        self.write_manifest()
        with self.assertRaises(ValueError):
            manager.install(self.source, self.root, activate=True)
        self.assertFalse(manager.state(self.root)['enabled'])

    def test_rollback_without_a_previous_release_does_not_change_activation(self):
        manager.install(self.source, self.root, activate=True)
        previous = (self.root/'active.json').read_bytes()
        with self.assertRaises(ValueError):
            manager.rollback(self.root)
        self.assertEqual((self.root/'active.json').read_bytes(), previous)

    def test_relative_root_traversal_and_other_directories_are_rejected(self):
        for path in ('strata-local-models', self.folder/'unrelated'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                manager.install(self.source, path)
        for name in ('../entry.mjs', 'ui/../entry.mjs', '/ui/entry.mjs', 'ui//entry.mjs', 'ui/private.exe'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                manager.asset_name(name)


if __name__ == '__main__':
    unittest.main()
