"""Inkling-2-Mini's picture reader follows the Qwen rule (owner 2026-11-26): the run config's `--vision-reader gpu|cpu` (and the INFO line the engine prints:
`vision_reader=gpu|cpu|off`) decide the picture size cap and the silence allowance, and the flag is passed to an engine that knows it (`--info` says
vision_reader=1) and only to one.

    python -m pytest serve/test_inkling_vision_reader.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import inkling_kind as K  # noqa: E402

BASE = ["--native", "m.strata", "--max-context", "32768", "--vision", "v.strata"]
NEW = {"arch": "inkling2_loop", "images": "1", "drafter": "none", "vision_reader": "1"}
OLD = {"arch": "inkling2_loop", "images": "1", "drafter": "none"}


def cfg(backend="cuda", extra=(), **kw):
    return {"arch": "inkling2_loop", "backend": backend, "drafting": "off", "args": BASE + list(extra), **kw}


def run_args(c, info, exe="strata-inkling", **kw):
    with mock.patch.object(K, "probe", return_value=info):
        return K.engine_args(c, exe, announce=False, **kw)


class Reader(unittest.TestCase):
    def test_the_engine_info_line_wins(self):
        self.assertEqual(K.picture_reader(cfg(), {"vision_reader": "gpu"}), "gpu")
        self.assertEqual(K.picture_reader(cfg(extra=["--vision-reader", "gpu"]), {"vision_reader": "cpu"}), "cpu")
        self.assertIsNone(K.picture_reader(cfg(), {"vision_reader": "off"}))

    def test_the_config_flag_is_next(self):
        self.assertEqual(K.picture_reader(cfg(extra=["--vision-reader", "cpu"])), "cpu")
        self.assertEqual(K.picture_reader(cfg(extra=["--vision-reader", "gpu"])), "gpu")
        self.assertIsNone(K.picture_reader(cfg(extra=["--vision-reader", "auto"])))
        self.assertIsNone(K.picture_reader(cfg()))

    def test_the_cap_follows_the_reader(self):
        # an NVIDIA card with the GPU reader: no cap and no allowance
        self.assertFalse(K.cpu_picture_reader(cfg("cuda", ["--vision-reader", "gpu"])))
        self.assertEqual(K.image_cap_pixels(cfg("cuda", ["--vision-reader", "gpu"]), {}), 0)
        # a small AMD or Intel card that reads on the CPU: capped
        for backend in ("hip", "sycl"):
            c = cfg(backend, ["--vision-reader", "cpu"])
            self.assertTrue(K.cpu_picture_reader(c), backend)
            self.assertEqual(K.image_cap_pixels(c, {}), K.DEFAULT_MAX_PIXELS, backend)
        # the engine's own word counts when the config says nothing (auto)
        self.assertEqual(K.image_cap_pixels(cfg("hip"), {}, {"vision_reader": "cpu"}), K.DEFAULT_MAX_PIXELS)
        self.assertEqual(K.image_cap_pixels(cfg("cuda"), {}, {"vision_reader": "gpu"}), 0)
        # the person's setting still wins, on either reader
        self.assertEqual(K.image_cap_pixels(cfg("cuda", ["--vision-reader", "gpu"], inkling_image_max_mp=2), {}), 2_000_000)
        self.assertEqual(K.image_cap_pixels(cfg("hip", ["--vision-reader", "cpu"], inkling_image_max_mp=0), {}), 0)

    def test_nothing_said_keeps_the_old_backend_rule(self):
        self.assertTrue(K.cpu_picture_reader(cfg("cuda")))
        self.assertTrue(K.cpu_picture_reader(cfg("cpu")))
        for gpu in ("hip", "sycl", "metal"):
            self.assertFalse(K.cpu_picture_reader(cfg(gpu)), gpu)
        self.assertEqual(K.image_cap_pixels(cfg("cuda"), {}), K.DEFAULT_MAX_PIXELS)
        self.assertEqual(K.image_cap_pixels(cfg("metal"), {}), 0)


class EngineArgs(unittest.TestCase):
    def test_the_flag_goes_only_to_an_engine_that_knows_it(self):
        got = run_args(cfg(extra=["--vision-reader", "cpu"]), NEW)
        self.assertEqual(got[got.index("--vision-reader") + 1], "cpu")
        got = run_args(cfg(extra=["--vision-reader", "cpu"]), OLD)               # an older engine refuses unknown options (exit 2)
        self.assertNotIn("--vision-reader", got)
        self.assertIn("--vision", got)
        got = run_args(cfg(extra=["--vision-reader", "gpu"]), None)
        self.assertNotIn("--vision-reader", got)

    def test_the_flag_goes_with_the_vision_file_or_not_at_all(self):
        got = run_args(cfg(extra=["--vision-reader", "cpu"]), {**NEW, "images": "0"})        # an engine that reads no images: no --vision, no reader flag
        self.assertNotIn("--vision", got)
        self.assertNotIn("--vision-reader", got)

    def test_a_config_without_the_flag_gets_the_machines_rule_when_the_card_can_be_read(self):
        for gib, want in ((12, "cpu"), (16, "cpu"), (24, "gpu"), (32, "gpu")):
            with mock.patch.object(K, "detected_reader", return_value=(want, float(gib))):
                got = run_args(cfg("cuda"), NEW)
            self.assertEqual(got[got.index("--vision-reader") + 1], want, gib)
        with mock.patch.object(K, "detected_reader", return_value=(None, 0.0)):         # the card cannot be read: the engine decides
            self.assertNotIn("--vision-reader", run_args(cfg("cuda"), NEW))
        with mock.patch.object(K, "detected_reader", return_value=("cpu", 12.0)):         # an older engine: nothing is added
            self.assertNotIn("--vision-reader", run_args(cfg("cuda"), OLD))

    def test_the_persons_flag_is_not_replaced(self):
        with mock.patch.object(K, "detected_reader", return_value=("gpu", 32.0)):
            got = run_args(cfg("cuda", ["--vision-reader", "cpu"]), NEW)
        self.assertEqual(got.count("--vision-reader"), 1)
        self.assertEqual(got[got.index("--vision-reader") + 1], "cpu")
        with mock.patch.object(K, "detected_reader", return_value=("gpu", 32.0)):
            got = run_args(cfg("cuda", ["--vision-reader", "auto"]), NEW)
        self.assertEqual(got[got.index("--vision-reader") + 1], "auto")                  # auto asks the engine

    def test_detected_reader_by_backend(self):
        self.assertEqual(K.detected_reader({"backend": "metal"}), ("gpu", 0.0))
        self.assertEqual(K.detected_reader({"backend": "sycl"}), ("cpu", 0.0))
        self.assertEqual(K.detected_reader({"backend": "cpu"}), (None, 0.0))
        fake = mock.Mock()
        fake.ok.return_value = True
        for gib, want in ((8, "cpu"), (12, "cpu"), (16, "cpu"), (24, "gpu"), (32, "gpu")):
            fake.read.return_value = {"mem_total": int(gib * 2**30)}
            with mock.patch("serve.telemetry.gpu_reader", return_value=fake):
                self.assertEqual(K.detected_reader({"backend": "cuda", "gpu": 0}), (want, float(gib)), gib)
                self.assertEqual(K.detected_reader({"backend": "hip", "gpu": 0}), (want, float(gib)), gib)
        with mock.patch("serve.telemetry.gpu_reader", return_value=fake):
            self.assertEqual(K.detected_reader({"backend": "cuda", "gpu": [0, 1]}), (None, 0.0))     # several cards: the engine decides
        fake.ok.return_value = False
        with mock.patch("serve.telemetry.gpu_reader", return_value=fake):
            self.assertEqual(K.detected_reader({"backend": "cuda"}), (None, 0.0))


if __name__ == "__main__":
    unittest.main()
