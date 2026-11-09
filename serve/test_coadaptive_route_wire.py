"""Exercise service routing through the actual native sampling serializer."""
import copy
import threading
import unittest
from unittest import mock

from serve import server
from serve.coadaptive import CoAdaptive


class CoAdaptiveRouteWire(unittest.TestCase):
    def run_request(self, sampling, *, enabled=True, mode="live"):
        tok = server.ByteTokenizer()
        engine = server.MockEngine(tok, "ok")
        svc = server.Service(engine, tok, None)
        svc.coadaptive = CoAdaptive({"enabled": enabled, "mode": mode}, pcie_frac=.37)
        generate = engine.generate
        wire, effective = [], []

        def capture(ids, max_new, values, cancel, **kwargs):
            effective.append(copy.deepcopy(values))
            wire.append(server.StrataEngine.sampling_keys(values))
            yield from generate(ids, max_new, values, cancel, **kwargs)

        with mock.patch.object(svc.coadaptive, "choose_route", return_value=.65) as choose, \
                mock.patch.object(engine, "generate", side_effect=capture):
            list(svc.run([1], False, None, 8, sampling, threading.Event()))
        self.assertFalse(svc.fifo.locked())
        return wire, effective, choose

    def test_selected_route_reaches_native_wire_without_changing_other_tuning(self):
        sampling = {"temperature": .7, "strata_tune": {"spec_min_p": .8, "aux_cpus": True},
                    "experimental_speed_projection": False, "reasoning_budget_tokens": 1024}
        before = copy.deepcopy(sampling)
        wire, effective, choose = self.run_request(sampling)
        self.assertIn(" pcie_frac=0.65", wire[0])
        self.assertIn(" spec_min_p=0.8", wire[0])
        self.assertIn(" aux_cpus=1", wire[0])
        self.assertEqual(sampling, before)
        self.assertFalse(effective[0]["experimental_speed_projection"])
        self.assertEqual(effective[0]["reasoning_budget_tokens"], 1024)
        choose.assert_called_once()

    def test_explicit_nested_route_including_zero_wins(self):
        for fraction in (0, .2, 1):
            with self.subTest(fraction=fraction):
                wire, effective, choose = self.run_request({"strata_tune": {"pcie_frac": fraction}})
                self.assertIn(" pcie_frac=" + repr(float(fraction)), wire[0])
                self.assertEqual(effective[0]["strata_tune"]["pcie_frac"], fraction)
                choose.assert_not_called()

    def test_disabled_and_shadow_policy_do_not_change_native_route(self):
        for enabled, mode in ((False, "live"), (True, "shadow")):
            with self.subTest(enabled=enabled, mode=mode):
                wire, effective, _ = self.run_request({"strata_tune": {"spec_min_p": .8}},
                                                      enabled=enabled, mode=mode)
                self.assertNotIn("pcie_frac=", wire[0])
                self.assertEqual(effective[0]["strata_tune"], {"spec_min_p": .8})


if __name__ == "__main__":
    unittest.main()
