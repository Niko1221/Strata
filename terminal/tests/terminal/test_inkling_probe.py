"""The measured drafting choice (strata_control/inkling_probe.py, 0221-DSPARK): the decision rule, what a probe is valid for, the status sentence,
the probe driver on a fake engine, and how the run config's "auto" uses it."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import inkling_kind  # noqa: E402
from strata_control import inkling_family, inkling_probe as P  # noqa: E402


class FakeEngine:
    """An engine on the line protocol: READY, then a DONE line per GEN whose decode time follows a speed table."""

    def __init__(self, args, speeds, fail=False):
        self.args = list(args)
        self.fail = fail
        draft = "off" if "--draft" in args and args[args.index("--draft") + 1] == "off" else "on"
        self.speeds = speeds[draft]
        self.lines = [] if fail else ["INFO x", "READY 32768 stop hold holdb"]
        self.sent = []

    def send(self, line):
        self.sent.append(line)
        if line.startswith("GEN"):
            n = int(line.split()[1])
            ids = line.split()[3].split(",")
            kind = "code" if len(ids) == len(P.CODE_IDS) else "chat"
            tps = self.speeds[kind]
            self.lines.append("DONE %d %d 10.0 %.1f length 0 0 0" % (n, len(ids), n / tps * 1000.0))

    def readline(self, timeout):
        return self.lines.pop(0) if self.lines else None

    def close(self):
        pass


class Decision(unittest.TestCase):
    def test_the_rule_is_the_geometric_mean_against_the_margin(self):
        self.assertEqual(P.MARGIN, 1.10)
        self.assertEqual(P.decide({"code": 1.36, "chat": 1.04}), "on")          # A6000, SQ-20 (measured)
        self.assertEqual(P.decide({"code": 1.59, "chat": 1.06}), "on")          # RTX 5090
        self.assertEqual(P.decide({"code": 2.75, "chat": 1.59}), "on")          # 3060, 5 to 7 tok/s plain
        self.assertEqual(P.decide({"code": 2.8, "chat": 2.6}), "on")            # 4060, SQ-10
        self.assertEqual(P.decide({"code": 0.86, "chat": 0.75}), "off")         # 4060, UD-Q4_K_XL
        self.assertEqual(P.decide({"code": 0.90, "chat": 0.89}), "off")         # RX 7900 XT
        self.assertEqual(P.decide({"code": 1.10, "chat": 1.10}), "on")          # exactly the margin counts
        self.assertEqual(P.decide({"code": 1.0, "chat": 1.0}), "off")
        # one good prompt cannot carry a bad one: 2.0 and 0.6 give 1.095
        self.assertEqual(P.decide({"code": 2.0, "chat": 0.6}), "off")
        self.assertEqual(P.decide({}), "off")
        self.assertEqual(P.decide({"code": 1.2, "chat": 1.0}, margin=1.05), "on")
        self.assertAlmostEqual(P.mixed_gain({"code": 4.0, "chat": 1.0}), 2.0)


class Placement(unittest.TestCase):
    BASE = {"args": ["--native", "/m/Inkling-2-Mini-SQ-10.strata", "--max-context", "32768", "--vram-experts-gb", "2.99", "--ram-experts-gb", "5.61"],
            "parallel": 1, "backend": "cuda", "gpu": 0}

    def test_the_probe_is_valid_for_one_placement(self):
        key = P.placement_key(self.BASE)
        self.assertEqual(key, P.placement_key(dict(self.BASE)))
        # the folder of the model file and unrelated flags do not matter
        moved = dict(self.BASE, args=["--native", "/other/Inkling-2-Mini-SQ-10.strata", "--max-context", "32768", "--vram-experts-gb", "2.99",
                                      "--ram-experts-gb", "5.61", "--host", "127.0.0.1"])
        self.assertEqual(key, P.placement_key(moved))
        def changed(**kw):
            return P.placement_key(dict(self.BASE, **kw))
        self.assertNotEqual(key, changed(parallel=4))
        self.assertNotEqual(key, changed(gpu=1))
        self.assertNotEqual(key, changed(backend="hip"))
        self.assertNotEqual(key, changed(args=["--native", "/m/Inkling-2-Mini-SQ-15.strata", "--max-context", "32768", "--vram-experts-gb", "2.99",
                                                "--ram-experts-gb", "5.61"]))
        self.assertNotEqual(key, changed(args=["--native", "/m/Inkling-2-Mini-SQ-10.strata", "--max-context", "65536", "--vram-experts-gb", "2.99",
                                                "--ram-experts-gb", "5.61"]))
        self.assertNotEqual(key, changed(args=["--native", "/m/Inkling-2-Mini-SQ-10.strata", "--max-context", "32768", "--vram-experts-gb", "7.5",
                                                "--ram-experts-gb", "0.0"]))

    def test_stale_and_decision_of(self):
        cfg = dict(self.BASE)
        self.assertTrue(P.is_stale(cfg))
        self.assertIsNone(P.decision_of(cfg))
        P.record(cfg, {"decision": "on", "key": P.placement_key(cfg), "gains": {"code": 1.4, "chat": 1.1}})
        self.assertFalse(P.is_stale(cfg))
        self.assertEqual(P.decision_of(cfg), "on")
        cfg["parallel"] = 4                                  # the placement changed: measure again
        self.assertTrue(P.is_stale(cfg))
        self.assertIsNone(P.decision_of(cfg))
        self.assertIn("measured for another placement", P.note(cfg))

    def test_the_status_sentence_gives_the_numbers(self):
        cfg = dict(self.BASE)
        self.assertEqual(P.note(cfg), "")
        P.record(cfg, {"decision": "off", "key": P.placement_key(cfg), "gains": {"code": 0.86, "chat": 0.75}, "plain_tps": {"code": 11.2, "chat": 10.8},
                       "margin": 1.10, "tokens": 128})
        text = P.note(cfg)
        self.assertTrue(text.startswith("Drafting off:"), text)
        for part in ("128-token probe", "0.80x", "code 0.86x", "chat 0.75x", "plain decoding 11 tok/s", "needs 1.10x"):
            self.assertIn(part, text)


class Driver(unittest.TestCase):
    SPEEDS = {"off": {"code": 40.0, "chat": 40.0}, "on": {"code": 80.0, "chat": 44.0}}

    def run_probe(self, speeds=None, fail_arm=None, base=None):
        started = []

        def open_engine(args):
            started.append(list(args))
            return FakeEngine(args, speeds or self.SPEEDS, fail=(fail_arm is not None and len(started) == fail_arm))
        res = P.probe(open_engine, base or ["--native", "m.strata", "--max-context", "32768"], key="k")
        return res, started

    def test_plain_first_then_chain_drafting_and_the_numbers_are_recorded(self):
        res, started = self.run_probe()
        self.assertEqual(len(started), 2)
        self.assertEqual(started[0][-2:], ["--draft", "off"])
        self.assertEqual(started[1][-5:], ["--draft", "dspark", "--draft-beam", "1", "--draft-adapt"])
        self.assertEqual(res["decision"], "on")
        self.assertAlmostEqual(res["gains"]["code"], 2.0, places=2)
        self.assertAlmostEqual(res["gains"]["chat"], 1.1, places=2)
        self.assertAlmostEqual(res["plain_tps"]["code"], 40.0, places=1)
        self.assertAlmostEqual(res["mixed_gain"], 1.483, places=2)
        self.assertEqual((res["margin"], res["tokens"], res["key"]), (1.10, 128, "k"))

    def test_a_card_where_drafting_loses_comes_out_off(self):
        res, _ = self.run_probe({"off": {"code": 11.0, "chat": 11.0}, "on": {"code": 9.5, "chat": 6.3}})   # the 4060 with UD-Q4_K_XL
        self.assertEqual(res["decision"], "off")
        self.assertLess(res["mixed_gain"], 1.0)

    def test_drafting_flags_in_the_base_arguments_are_replaced(self):
        _, started = self.run_probe(base=["--native", "m.strata", "--draft", "dspark", "--draft-beam", "4", "--draft-adapt", "--dspark", "d.gguf"])
        for args in started:
            self.assertEqual(args.count("--draft"), 1)
            if "--draft-beam" in args:
                self.assertEqual(args[args.index("--draft-beam") + 1], "1")
            self.assertIn("--dspark", args)                  # the drafter file stays

    def test_an_engine_that_fails_leaves_drafting_off_with_the_reason(self):
        res, _ = self.run_probe(fail_arm=1)
        self.assertEqual(res["decision"], "off")
        self.assertIn("plain engine start", res["error"])
        res, _ = self.run_probe(fail_arm=2)
        self.assertEqual(res["decision"], "off")
        self.assertIn("drafting engine start", res["error"])
        self.assertIn("plain_tps", res)


class RunConfig(unittest.TestCase):
    CFG = {"arch": "inkling2_loop", "backend": "cuda", "args": ["--native", "m.strata", "--max-context", "32768"], "parallel": 1}

    def setUp(self):
        self._saved = inkling_family.EXACT_DRAFTING_BACKENDS

    def tearDown(self):
        inkling_family.EXACT_DRAFTING_BACKENDS = self._saved

    def test_auto_follows_the_probe_once_the_backend_may_draft(self):
        inkling_family.EXACT_DRAFTING_BACKENDS = ("cuda",)
        self.assertEqual(inkling_family.default_drafting("cuda"), "auto")
        cfg = dict(self.CFG)                                 # nothing chosen = auto
        self.assertFalse(inkling_kind.wants_drafter(cfg))    # no probe yet: off, and the launcher is told to probe
        self.assertTrue(inkling_kind.needs_drafting_probe(cfg))
        P.record(cfg, {"decision": "on", "key": P.placement_key(cfg), "gains": {"code": 1.5, "chat": 1.2}})
        self.assertTrue(inkling_kind.wants_drafter(cfg))
        self.assertFalse(inkling_kind.needs_drafting_probe(cfg))
        P.record(cfg, {"decision": "off", "key": P.placement_key(cfg), "gains": {"code": 0.8, "chat": 0.7}})
        self.assertFalse(inkling_kind.wants_drafter(cfg))
        cfg["parallel"] = 4                                  # another placement: measure again
        self.assertTrue(inkling_kind.needs_drafting_probe(cfg))

    def test_nothing_is_probed_or_drafted_while_the_backend_is_not_proven(self):
        self.assertEqual(inkling_family.EXACT_DRAFTING_BACKENDS, ())
        cfg = dict(self.CFG, drafting="auto")
        P.record(cfg, {"decision": "on", "key": P.placement_key(cfg), "gains": {"code": 3.0, "chat": 3.0}})
        self.assertFalse(inkling_kind.wants_drafter(cfg))
        self.assertFalse(inkling_kind.needs_drafting_probe(cfg))
        self.assertFalse(inkling_kind.needs_drafting_probe(dict(self.CFG)))
        # an explicit choice is never overruled, and an explicit off never probes
        self.assertTrue(inkling_kind.wants_drafter(dict(self.CFG, drafting="dspark")))
        self.assertFalse(inkling_kind.needs_drafting_probe(dict(self.CFG, drafting="dspark")))
        self.assertFalse(inkling_kind.needs_drafting_probe(dict(self.CFG, drafting="off")))

    def test_the_status_note_reaches_the_engine_metrics_even_while_drafting_stays_off(self):
        cfg = dict(self.CFG, drafting="off")
        P.record(cfg, {"decision": "on", "key": P.placement_key(cfg), "gains": {"code": 1.5, "chat": 1.2}, "plain_tps": {"code": 20, "chat": 20}})
        inkling_kind.engine_args(cfg, announce=True)
        info = inkling_kind.drafting_info(["--draft", "off"])
        self.assertIn("Drafting on:", info["drafting_note"])
        self.assertIn("stays off until its answers are proven identical", info["drafting_note"])
        self.assertEqual(inkling_kind.drafting_info(["--draft", "off"])["drafting_note"], "")   # said once


if __name__ == "__main__":
    unittest.main()
