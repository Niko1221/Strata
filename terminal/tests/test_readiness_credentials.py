"""Readiness needs management authority without expanding memory-principal access."""
import json
import os
import unittest
from unittest import mock

from strata_control import readiness


class ReadinessCredentials(unittest.TestCase):
    def config(self, keys, enabled=True):
        return json.dumps({"api_key": keys, "hakua_harness": {
            "enabled": enabled, "users": [
                {"key_env": "STRATA_READINESS_MEMORY_KEY"}]}}).encode()

    def test_operator_is_selected_regardless_of_key_order(self):
        with mock.patch.dict(os.environ, {"STRATA_API_KEY": "",
                                          "STRATA_READINESS_MEMORY_KEY": "principal"}):
            for keys in (["principal", "operator"], ["operator", "principal"]):
                with self.subTest(keys=keys):
                    self.assertEqual(readiness._credential(self.config(keys)), "operator")

    def test_memory_only_credentials_cannot_attest_a_managed_server(self):
        with mock.patch.dict(os.environ, {"STRATA_API_KEY": "",
                                          "STRATA_READINESS_MEMORY_KEY": "principal"}):
            with self.assertRaisesRegex(readiness.ReadinessError, "^probe_management_credential_unavailable$"):
                readiness._credential(self.config(["principal"]))

    def test_environment_override_does_not_fall_back_to_an_unaccepted_key(self):
        with mock.patch.dict(os.environ, {"STRATA_API_KEY": "principal",
                                          "STRATA_READINESS_MEMORY_KEY": "principal"}):
            with self.assertRaisesRegex(readiness.ReadinessError, "^probe_management_credential_unavailable$"):
                readiness._credential(self.config(["principal", "operator"]))

    def test_disabled_harness_keeps_the_existing_first_key_policy(self):
        with mock.patch.dict(os.environ, {"STRATA_API_KEY": "",
                                          "STRATA_READINESS_MEMORY_KEY": "principal"}):
            self.assertEqual(readiness._credential(self.config(["principal", "operator"], False)), "principal")


if __name__ == "__main__":
    unittest.main()
