"""#7043: pictures for Inkling's CPU picture reader are scaled down before the engine sees them, and only there.

strata-inkling reads a picture with its GPU encoder where it has one (HIP, SYCL, Metal); CUDA has none, so a 4535 x 3004 photo took minutes
on the CPU.  The engine has no option for the size, so serve scales the picture it writes for the engine: at most 0.8 MP (inkling_kind.DEFAULT_MAX_PIXELS: about a minute of the measured CPU reader) on
the CPU backends (a normal screenshot is not touched, byte for byte), the config's `inkling_image_max_mp` (megapixels, 0 = off) otherwise.

    python -m pytest serve/test_inkling_image_cap.py
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import inkling_kind as K  # noqa: E402

try:
    from PIL import Image
except ImportError:                                       # pragma: no cover
    Image = None


def picture(w, h, fmt="PNG", mode="RGB"):
    im = Image.new(mode, (w, h), (200, 30, 30) if mode == "RGB" else 128)
    for x in range(0, w, 7):                              # some structure, so it is not one flat colour
        im.putpixel((x, x % h), (0, 0, 255) if mode == "RGB" else 255)
    out = io.BytesIO()
    im.save(out, format=fmt)
    return out.getvalue()


class Setting(unittest.TestCase):
    def test_the_default_is_on_for_the_cpu_readers_only(self):
        self.assertEqual(K.image_cap_pixels({"backend": "cuda"}, {}), K.DEFAULT_MAX_PIXELS)
        self.assertEqual(K.image_cap_pixels({"backend": "cpu"}, {}), K.DEFAULT_MAX_PIXELS)
        self.assertEqual(K.image_cap_pixels({}, {}), K.DEFAULT_MAX_PIXELS)                       # no backend named: the CUDA default
        for gpu in ("hip", "sycl", "metal"):
            self.assertEqual(K.image_cap_pixels({"backend": gpu}, {}), 0, gpu)

    def test_the_setting_turns_it_off_or_changes_it_on_every_backend(self):
        self.assertEqual(K.image_cap_pixels({"backend": "cuda", "inkling_image_max_mp": 0}, {}), 0)
        self.assertEqual(K.image_cap_pixels({"backend": "cuda", "inkling_image_max_mp": 4}, {}), 4_000_000)
        self.assertEqual(K.image_cap_pixels({"backend": "hip", "inkling_image_max_mp": 2.5}, {}), 2_500_000)
        self.assertEqual(K.image_cap_pixels({"backend": "cuda"}, {"STRATA_INKLING_IMAGE_MAX_MP": "0"}), 0)
        self.assertEqual(K.image_cap_pixels({"backend": "cuda", "inkling_image_max_mp": 6}, {"STRATA_INKLING_IMAGE_MAX_MP": "0"}), 6_000_000)   # the config first

    def test_a_bad_value_is_named(self):
        for bad in (-1, "big", True, float("nan")):
            with self.assertRaises(ValueError, msg=repr(bad)) as cm:
                K.image_cap_pixels({"inkling_image_max_mp": bad}, {})
            self.assertIn("inkling_image_max_mp", str(cm.exception))


@unittest.skipIf(Image is None, "Pillow is not installed")
class Scaling(unittest.TestCase):
    CAP = K.DEFAULT_MAX_PIXELS

    def test_a_picture_at_or_under_the_cap_is_passed_on_byte_for_byte(self):
        for fmt in ("PNG", "JPEG"):
            data = picture(1024, 768, fmt)
            out, note = K.cap_picture(data, self.CAP)
            self.assertIs(out, data, fmt)
            self.assertIsNone(note)

    def test_a_big_photo_is_scaled_to_the_cap_with_its_proportions(self):
        data = picture(4535, 3004, "JPEG")
        out, note = K.cap_picture(data, self.CAP)
        self.assertEqual(out[:3], b"\xff\xd8\xff")                              # a JPEG stays a JPEG
        with Image.open(io.BytesIO(out)) as im:
            w, h = im.size
        self.assertLessEqual(w * h, self.CAP)
        self.assertGreater(w * h, self.CAP * 0.97)                               # close to the cap, not far under it
        self.assertAlmostEqual(w / h, 4535 / 3004, delta=0.01)
        self.assertIn("4535x3004", note)
        self.assertIn(f"{w}x{h}", note)
        self.assertLess(len(out), len(data))

    def test_a_png_stays_a_png(self):
        out, note = K.cap_picture(picture(3000, 2000, "PNG"), self.CAP)
        self.assertEqual(out[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIsNotNone(note)

    def test_off_and_unreadable(self):
        data = picture(4000, 3000, "PNG")
        self.assertEqual(K.cap_picture(data, 0), (data, None))
        junk = b"\x89PNG\r\n\x1a\n" + b"not really"
        self.assertEqual(K.cap_picture(junk, self.CAP), (junk, None))           # the engine says what it thinks of it

    def test_a_transparent_png_and_a_palette_picture(self):
        for mode in ("RGBA", "P", "L"):
            im = Image.new(mode, (2800, 1600))
            out = io.BytesIO()
            im.save(out, format="PNG")
            small, note = K.cap_picture(out.getvalue(), self.CAP)
            self.assertIsNotNone(note, mode)
            with Image.open(io.BytesIO(small)) as r:
                self.assertLessEqual(r.size[0] * r.size[1], self.CAP, mode)

    def test_a_very_wide_picture_keeps_at_least_one_pixel_a_side(self):
        out, note = K.cap_picture(picture(20000, 4, "PNG"), 1000)
        with Image.open(io.BytesIO(out)) as im:
            self.assertGreaterEqual(min(im.size), 1)
            self.assertLessEqual(im.size[0] * im.size[1], 1000)


@unittest.skipIf(Image is None, "Pillow is not installed")
class WhatTheEngineIsGiven(unittest.TestCase):
    def test_write_images_through_the_servers_normalizer(self):
        import tempfile
        from serve.server import Vision
        said = []
        with tempfile.TemporaryDirectory() as d:
            def norm(data, cap=K.DEFAULT_MAX_PIXELS):
                data = Vision.normalize(data)
                data, note = K.cap_picture(data, cap)
                if note:
                    said.append(note)
                return data
            big, screen = picture(4535, 3004, "JPEG"), picture(1024, 768, "PNG")
            paths = K.write_images([big, screen], lambda s: s, norm, folder=d)
            with Image.open(paths[0]) as a, Image.open(paths[1]) as b:
                self.assertLessEqual(a.size[0] * a.size[1], K.DEFAULT_MAX_PIXELS)
                self.assertEqual(b.size, (1024, 768))
            self.assertEqual(Path(paths[1]).read_bytes(), screen)                # the screenshot reached the engine unchanged
        self.assertEqual(len(said), 1)


class Allowance(unittest.TestCase):
    """A request with pictures on the CPU reader's path may stay silent for as long as the pictures take (measured curve, a margin), so it is slow and answered."""

    def test_the_reader_is_the_cpu_except_on_hip_sycl_metal(self):
        for backend in ("cuda", "cpu", None):
            self.assertTrue(K.cpu_picture_reader({"backend": backend}), backend)
        for backend in ("hip", "sycl", "metal"):
            self.assertFalse(K.cpu_picture_reader({"backend": backend}), backend)

    def test_the_allowance_follows_the_pixels(self):
        small, mid, big = (K.picture_allowance_s([px]) for px in (250_000, 1_000_000, 4_000_000))
        self.assertGreater(small, K.ALLOW_BASE_S)
        self.assertGreater(mid, small)
        self.assertGreater(big, mid)
        self.assertGreater(K.picture_allowance_s([1_000_000, 1_000_000]), mid)          # two pictures take longer than one
        self.assertEqual(K.picture_allowance_s([]), 0.0)                                 # no picture: nothing changes
        self.assertLessEqual(K.picture_allowance_s([200_000_000]), K.ALLOW_MAX_S)

    def test_the_allowance_covers_the_measured_time_with_a_margin(self):
        for px in (250_000, 500_000, 1_000_000, 2_000_000):
            self.assertGreaterEqual(K.picture_allowance_s([px]), K.ALLOW_MARGIN * K.picture_read_seconds(px))
        self.assertGreaterEqual(K.ALLOW_MARGIN, 1.5)

    def test_a_picture_that_cannot_be_measured_counts_as_the_cap(self):
        self.assertEqual(K.picture_allowance_s([0], cap=1_000_000), K.picture_allowance_s([1_000_000]))
        self.assertGreater(K.picture_allowance_s([0]), K.picture_allowance_s([1_000_000]))      # (no cap: the engine's own 4096 x 4096)


# an engine that says nothing for 2.5 s after the picture request (the CPU reader), then answers
SLOW = ("import sys, time\n"
        "print('READY 4096 stop', flush=True)\n"
        "for line in sys.stdin:\n"
        "    if line.startswith('QUIT'):\n"
        "        break\n"
        "    if line.startswith(('GEN', 'BGEN')):\n"
        "        time.sleep(2.5)\n"
        "        print('T 90\\nDONE 1 1 0 1 stop', flush=True)\n")


class SilenceWatchdog(unittest.TestCase):
    def run_request(self, picture_allow, silence=1.0):
        import tempfile
        import threading
        from unittest import mock
        import serve.server as server
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "slow_engine.py"
            script.write_text(SLOW, encoding="utf-8")
            real = server.subprocess.Popen

            def popen(cmd, **kw):
                return real([sys.executable, str(script), *cmd[2:]], **kw)
            with mock.patch.object(server.subprocess, "Popen", popen), mock.patch.object(server, "narrate_start", lambda *a, **k: None), \
                    mock.patch.object(server, "end_leftover_engines", lambda *a, **k: None):
                eng = server.StrataEngine("strata", [])
                self.addCleanup(eng.close)
                eng.silence_s = silence
                eng.picture_allow_s = picture_allow
                return list(eng.generate([1, 2, 3], 4, {}, threading.Event(), embeddings="x.bin")), eng

    def test_without_an_allowance_the_silence_ends_the_request(self):
        from serve.server_parts.engines import EngineSilent
        with self.assertRaises(EngineSilent):
            self.run_request(0.0)

    def test_with_the_allowance_the_request_is_answered(self):
        toks, eng = self.run_request(8.0)
        self.assertEqual([t for t in toks if t is not None], [90])
        self.assertEqual(eng.picture_allow_s, 0.0)                                       # used once: the next request starts from engine_silence_s again

    def test_a_larger_silence_of_the_persons_stays(self):
        toks, _ = self.run_request(0.5, silence=8.0)                                     # the allowance is smaller than engine_silence_s: that one counts
        self.assertEqual([t for t in toks if t is not None], [90])


if __name__ == "__main__":
    unittest.main()
