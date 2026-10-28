"""cvec=2 (auto) reaches the engine: projection_key passes a bool as cvec=0|1 (as before), the integer 2 and the
string "auto" as cvec=2, and nothing else - an absent or unusable value sends no key, so the engine's on-by-default
holds.  cvec_threshold (a finite number) adds the request's own gate, cvec_thresh=F; the shared-defaults and
run-config validators take the same domain.

    python -m unittest serve.test_projection_key -v
"""
import math
import unittest

from serve.server import (StrataEngine, clean_shared_defaults, sampling_defaults_from_config,
                          history_projection, parse_cvec_gate, asks_cvec_auto)


class ProjectionKey(unittest.TestCase):
    def pk(self, sampling):
        return StrataEngine.projection_key(sampling)

    def test_bool_unchanged(self):
        self.assertEqual(self.pk({"experimental_speed_projection": True}), " cvec=1")
        self.assertEqual(self.pk({"experimental_speed_projection": False}), " cvec=0")

    def test_int_two_is_auto(self):
        self.assertEqual(self.pk({"experimental_speed_projection": 2}), " cvec=2")
        self.assertEqual(self.pk({"experimental_speed_projection": 1}), " cvec=1")
        self.assertEqual(self.pk({"experimental_speed_projection": 0}), " cvec=0")

    def test_auto_string_is_auto(self):
        self.assertEqual(self.pk({"experimental_speed_projection": "auto"}), " cvec=2")
        self.assertEqual(self.pk({"experimental_speed_projection": "Auto"}), " cvec=2")

    def test_absent_or_unusable_sends_nothing(self):
        self.assertEqual(self.pk({}), "")
        self.assertEqual(self.pk({"experimental_speed_projection": None}), "")
        self.assertEqual(self.pk({"experimental_speed_projection": 3}), "")
        self.assertEqual(self.pk({"experimental_speed_projection": -1}), "")
        self.assertEqual(self.pk({"experimental_speed_projection": "2"}), "")   # only "auto" is a string key
        self.assertEqual(self.pk({"experimental_speed_projection": 2.0}), "")   # nor a float

    def test_threshold_rides_any_mode(self):
        self.assertEqual(self.pk({"cvec_threshold": -0.9}), " cvec_thresh=-0.9")
        self.assertEqual(self.pk({"experimental_speed_projection": "auto", "cvec_threshold": -0.9}),
                         " cvec=2 cvec_thresh=-0.9")
        self.assertEqual(self.pk({"experimental_speed_projection": False, "cvec_threshold": -1}),
                         " cvec=0 cvec_thresh=-1")   # the engine ignores it unless auto; the key is still honest

    def test_threshold_unusable_sends_nothing(self):
        self.assertEqual(self.pk({"cvec_threshold": None}), "")
        self.assertEqual(self.pk({"cvec_threshold": True}), "")      # a bool is not a gate
        self.assertEqual(self.pk({"cvec_threshold": "-0.9"}), "")    # a string is not a number
        self.assertEqual(self.pk({"cvec_threshold": math.nan}), "")
        self.assertEqual(self.pk({"cvec_threshold": math.inf}), "")
        self.assertEqual(self.pk({"cvec_threshold": object()}), "")


class SharedDefaults(unittest.TestCase):
    def test_esp_domain(self):
        self.assertEqual(clean_shared_defaults({"experimental_speed_projection": True}),
                         {"experimental_speed_projection": True})
        self.assertEqual(clean_shared_defaults({"experimental_speed_projection": "auto"}),
                         {"experimental_speed_projection": "auto"})
        for bad in (2, "Auto ", "on", 0):
            with self.assertRaises(ValueError):
                clean_shared_defaults({"experimental_speed_projection": bad})

    def test_threshold(self):
        self.assertEqual(clean_shared_defaults({"cvec_threshold": -0.9}), {"cvec_threshold": -0.9})
        self.assertEqual(clean_shared_defaults({"cvec_threshold": -1}), {"cvec_threshold": -1.0})   # floats
        for bad in ("-0.9", True, math.nan, math.inf, object()):
            with self.assertRaises(ValueError):
                clean_shared_defaults({"cvec_threshold": bad})


class RunConfig(unittest.TestCase):
    def test_esp_and_threshold_accepted(self):
        out = sampling_defaults_from_config({"sampling": {"experimental_speed_projection": "auto",
                                                          "cvec_threshold": -0.9}})
        self.assertEqual(out, {"experimental_speed_projection": "auto", "cvec_threshold": -0.9})
        self.assertEqual(sampling_defaults_from_config({"sampling": {"experimental_speed_projection": True}}),
                         {"experimental_speed_projection": True})

    def test_bad_values_refuse_start(self):
        for bad in (2, "on"):
            with self.assertRaises(SystemExit):
                sampling_defaults_from_config({"sampling": {"experimental_speed_projection": bad}})
        for bad in ("-0.9", True, math.nan):
            with self.assertRaises(SystemExit):
                sampling_defaults_from_config({"sampling": {"cvec_threshold": bad}})


class HistoryProjection(unittest.TestCase):
    """/history tells the truth about auto: the gate decides in the engine, so serve keeps "auto" instead
    of reading it truthy as on (the web badge renders it as a third, neutral state)."""

    def test_bool_unchanged(self):
        self.assertTrue(history_projection({"experimental_speed_projection": True}, True))
        self.assertFalse(history_projection({"experimental_speed_projection": False}, True))
        self.assertEqual(history_projection({}, True), True)            # absent = on (the documented default)

    def test_auto_stays_auto(self):
        self.assertEqual(history_projection({"experimental_speed_projection": "auto"}, True), "auto")
        self.assertEqual(history_projection({"experimental_speed_projection": "Auto"}, True), "auto")
        self.assertEqual(history_projection({"experimental_speed_projection": 2}, True), "auto")
        self.assertIs(history_projection({"experimental_speed_projection": 1}, True), True)
        self.assertIs(history_projection({"experimental_speed_projection": 0}, True), False)

    def test_no_vector_is_none(self):
        self.assertIsNone(history_projection({"experimental_speed_projection": "auto"}, False))
        self.assertIsNone(history_projection({"experimental_speed_projection": True}, False))


class CvecGateLine(unittest.TestCase):
    """_pump captures the engine's CVECAUTO verdict line off the control stdout (generate.cpp:15945) for the
    request history; the human line in the log and the batched refusal (stderr-only) are not its domain."""

    def test_measure_line(self):
        g = parse_cvec_gate("CVECAUTO mean=-0.9622 max=-0.0489 thresh=-0.9000 -> steer")
        self.assertEqual(g, {"verdict": "steer", "mean": -0.9622, "max": -0.0489, "thresh": -0.9})
        g = parse_cvec_gate("CVECAUTO mean=0.2833 max=2.3939 thresh=-0.9000 -> stock")
        self.assertEqual(g["verdict"], "stock")

    def test_carriage_return(self):
        g = parse_cvec_gate("CVECAUTO mean=-0.9622 max=-0.0489 thresh=-0.9000 -> steer\r")
        self.assertEqual(g["verdict"], "steer")

    def test_log_only_sentinel_thresh(self):
        # no threshold given: the engine logs lowest() (-FLT_MAX) and the gate stays stock (log-only mode)
        g = parse_cvec_gate("CVECAUTO mean=-0.3053 max=-0.0553 thresh=-340282346638528859811704183484516925440.0000"
                            " -> stock")
        self.assertEqual(g["verdict"], "stock")
        self.assertLess(g["thresh"], -3e38)

    def test_explicit_inf_thresh(self):
        # the P2 harness's log-only mode passes --cvec-threshold -inf and the engine prints it via %.4f as
        # "-inf"; it must parse (mean/max are finite dots and stay strict, a malformed word does not pass)
        g = parse_cvec_gate("CVECAUTO mean=-0.0717 max=3.4941 thresh=-inf -> stock")
        self.assertEqual(g["verdict"], "stock")
        self.assertEqual(g["thresh"], float("-inf"))
        self.assertIsNone(parse_cvec_gate("CVECAUTO mean=-0.9 max=0.0 thresh=-infinite -> stock"))
        self.assertIsNone(parse_cvec_gate("CVECAUTO mean=-0.9 max=0.0 thresh=nan -> stock"))
        self.assertIsNone(parse_cvec_gate("CVECAUTO mean=-0.9 max=0.0 thresh=infi -> stock"))

    def test_anything_else_is_none(self):
        self.assertIsNone(parse_cvec_gate("strata serve: prompt 64 tokens = 0 reused + 64 read"))
        self.assertIsNone(parse_cvec_gate("strata serve: cvec auto: mean -0.9622 max -0.0489 thresh -0.9000 -> steer"))
        self.assertIsNone(parse_cvec_gate("strata serve: cvec auto: batched -> stock"))
        self.assertIsNone(parse_cvec_gate("CVECAUTO mean=1.2.3 max=-0.05 thresh=-0.9000 -> stock"))
        self.assertIsNone(parse_cvec_gate("CVECAUTO mean=-0.9 max=0.0 thresh=-0.9 -> maybe"))
        self.assertIsNone(parse_cvec_gate("CVECBAUTO mean=-0.9 max=0.0 thresh=-0.9 -> steer"))


class AsksCvecAuto(unittest.TestCase):
    """The gate-line scan runs only for requests that asked for auto (the engine prints verdict lines only
    under cvec_auto): same domain history_projection keeps as "auto" - the string, case-insensitive, or the
    integer 2; bools and other ints are not auto."""

    def test_auto_shapes(self):
        self.assertTrue(asks_cvec_auto({"experimental_speed_projection": "auto"}))
        self.assertTrue(asks_cvec_auto({"experimental_speed_projection": "AUTO"}))
        self.assertTrue(asks_cvec_auto({"experimental_speed_projection": 2}))

    def test_not_auto(self):
        for v in (True, False, 0, 1, "true", "on", None):
            self.assertFalse(asks_cvec_auto({"experimental_speed_projection": v}), repr(v))
        self.assertFalse(asks_cvec_auto({}))
        self.assertFalse(asks_cvec_auto(None))


if __name__ == "__main__":
    unittest.main()
