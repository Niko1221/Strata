"""strata-metal against the engine line protocol, on a real model (macOS).

    STRATA_METAL_EXE=build-metal/metal/strata-metal STRATA_METAL_GGUF=<a .gguf> python -m unittest metal.test_strata_metal

Any GGUF with <|im_start|> works (LiquidAI/LFM2-350M-GGUF Q8_0 is a 360 MB hybrid model whose recurrent layers need
the checkpoints, as Qwen3.8-Flash-Next's do).  Skipped without the two variables.  Greedy throughout, so the answers
of two engines given the same ids MUST be the same tokens.
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import unittest

EXE, GGUF = os.environ.get("STRATA_METAL_EXE"), os.environ.get("STRATA_METAL_GGUF")
PAD = 5000                       # the image cells' token id here (an ordinary id of any test model's vocabulary)


def embeddings(path, images, n_embd):
    """strata-vision's file: per image int32 {'SVE1', n, nx, ny, n_embd} then n x n_embd float32 (a fill value)."""
    with open(path, "wb") as f:
        for nx, ny, fill in images:
            f.write(struct.pack("<5i", 0x31455653, nx * ny, nx, ny, n_embd))
            f.write(struct.pack(f"<{nx * ny * n_embd}f", *([fill] * (nx * ny * n_embd))))


class Engine:
    def __init__(self, *extra):
        self.p = subprocess.Popen([EXE, "--serve", "--gguf", GGUF, "--max-context", "4096", "--image-pad-id", str(PAD), *extra],
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.info = {}
        for line in self.p.stdout:
            if line.startswith("INFO "):
                self.info.update(kv.split("=", 1) for kv in line.split()[1:])
            if line.startswith("READY"):
                self.ready = line.split()
                break
        self.im_start = int(self.info["im_start"])
        self.later = []                  # BT / BDONE lines that came while an admission was read
        self.n_embd = int(self.info["n_embd"])

    def send(self, line: str):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def line(self) -> str:
        return self.p.stdout.readline().strip()

    def gen(self, ids, max_new=16, stop_after=None, image=None):
        """-> (tokens, resume, DONE fields); `image`: an embeddings file (GENI)"""
        head = f"GENI {max_new} {image}" if image else f"GEN {max_new}"
        self.send(head + " " + ",".join(map(str, ids)))
        toks, resume = [], None
        while True:
            line = self.line()
            if line.startswith("RESUME "):
                resume = int(line.split()[1])
            elif line.startswith("T "):
                toks.append(int(line[2:]))
                if stop_after is not None and len(toks) == stop_after:
                    self.send("STOP")
            elif line.startswith("DONE "):
                return toks, resume, line.split()
            elif line.startswith("ERR"):
                raise AssertionError(line)

    def admit(self, slot, ids, max_new):
        """BGEN -> (first token, resume, continues)"""
        self.send(f"BGEN {slot} {max_new} " + ",".join(map(str, ids)))
        toks, resume = [], None
        while True:
            line = self.line()
            if line.startswith("RESUME "):
                resume = int(line.split()[1])
            elif line.startswith("T "):
                toks.append(int(line[2:]))
            elif line.startswith(("BT ", "BDONE ")):              # other slots decode between the commands
                self.later.append(line)
            elif line.startswith("BADM "):
                f = line.split()
                return toks, resume, f[1] == str(slot) and f[2] == "1"
            elif line.startswith("ERR"):
                raise AssertionError(line)

    def slots(self, stop_after=None):
        """BT / BDONE until every admitted slot is done -> {slot: (tokens, finish)}"""
        got, done = {}, {}
        while True:
            f = (self.later.pop(0) if self.later else self.line()).split()
            if f[0] == "BT":
                got.setdefault(int(f[1]), []).append(int(f[2]))
                if stop_after and len(got[int(f[1])]) == stop_after[1] and int(f[1]) == stop_after[0]:
                    self.send(f"BSTOP {stop_after[0]}")
            elif f[0] == "BDONE":
                done[int(f[1])] = (got.get(int(f[1]), []), f[3])
                if len(done) == self.admitted:
                    return done

    def close(self):
        self.send("QUIT")
        self.p.wait(timeout=30)


def message(im_start, body):
    return [im_start] + body


@unittest.skipUnless(EXE and GGUF, "set STRATA_METAL_EXE and STRATA_METAL_GGUF")
class TestStrataMetal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = Engine()
        s = cls.e.im_start
        cls.m1 = message(s, list(range(300, 340)))
        cls.m2 = message(s, list(range(500, 560)))
        cls.m3 = message(s, list(range(700, 720)))
        cls.mx = message(s, list(range(900, 930)))

    @classmethod
    def tearDownClass(cls):
        cls.e.close()

    def test_ready(self):
        self.assertEqual(self.e.ready, ["READY", "4096", "stop"])
        self.assertEqual(self.e.info["backend"], "metal")

    def test_divergence_goes_back_to_the_boundary_and_matches_a_fresh_engine(self):
        a = self.m1 + self.m2 + self.m3
        b = self.m1 + self.m2 + self.mx
        self.e.gen(a)
        out_b, resume, _ = self.e.gen(b)
        self.assertEqual(resume, len(self.m1) + len(self.m2))     # the checkpoint before m3's <|im_start|>
        fresh = Engine()
        try:
            out_fresh, resume0, _ = fresh.gen(b)
        finally:
            fresh.close()
        self.assertEqual(resume0, 0)
        self.assertEqual(out_b, out_fresh)

    def test_an_extended_prompt_reads_only_the_new_part(self):
        a = self.m1 + self.m3
        out, _, done = self.e.gen(a)
        nxt = a + out + self.mx
        _, resume, done2 = self.e.gen(nxt)
        self.assertEqual(resume, len(a) + len(out) - 1)            # all held: the prompt and every token but the last
        self.assertEqual(int(done2[14]), len(self.mx) + 1)         # prompt tokens read

    def test_stop_cancels(self):
        toks, _, done = self.e.gen(self.m2 + self.mx, max_new=2000, stop_after=2)
        self.assertEqual(done[5], "cancel")
        self.assertLess(len(toks), 50)

    def test_save_restore(self):
        a = self.m1 + self.m2
        self.e.gen(a, max_new=4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.bin")
            self.e.send(f"SAVE {path}")
            saved = self.e.line().split()
            self.assertEqual(saved[0], "SAVED")
            held = int(saved[1])
            other = Engine()
            try:
                other.send(f"RESTORE {path}")
                restored = other.line().split()
                self.assertEqual(restored[:2], ["RESTORED", str(held)])
                other.send(f"RESTORE {path}.missing")
                self.assertTrue(other.line().startswith("SERR invalid 0 "))
            finally:
                other.close()

    def test_images_are_compared_by_their_content(self):
        with tempfile.TemporaryDirectory() as d:
            one, two = os.path.join(d, "one.bin"), os.path.join(d, "two.bin")
            embeddings(one, [(4, 2, 0.01)], self.e.n_embd)
            embeddings(two, [(4, 2, -0.02)], self.e.n_embd)
            ids = self.m1 + [PAD] * 8 + self.m3
            out1, _, _ = self.e.gen(ids, image=one)
            _, resume, _ = self.e.gen(ids, image=one)                   # the same picture: read again only the end
            self.assertGreaterEqual(resume, len(self.m1) + 8)
            _, resume, _ = self.e.gen(ids, image=two)                   # same ids, another picture: not past m1
            self.assertLessEqual(resume, len(self.m1))
            fresh = Engine()
            try:
                self.assertEqual(fresh.gen(ids, image=one)[0], out1)      # what a fresh engine says for picture one
            finally:
                fresh.close()

    def test_bad_image_requests_are_refused_in_step(self):
        with tempfile.TemporaryDirectory() as d:
            one = os.path.join(d, "one.bin")
            embeddings(one, [(4, 2, 0.01)], self.e.n_embd)
            for ids, image in ((self.m1 + [PAD] * 7 + self.m3, one),        # fewer pads than rows
                               (self.m1 + [PAD] * 8, one),                   # ends in the image
                               (self.m1 + self.m3, one),                     # an image the prompt does not have
                               (self.m1 + [PAD] * 8 + self.m3, os.path.join(d, "missing.bin"))):
                with self.assertRaises(AssertionError):
                    self.e.gen(ids, image=image)
            bad = os.path.join(d, "bad.bin")
            embeddings(bad, [(4, 2, 0.01)], self.e.n_embd + 1)            # the wrong width
            with self.assertRaises(AssertionError):
                self.e.gen(self.m1 + [PAD] * 8 + self.m3, image=bad)
        self.assertEqual(len(self.e.gen(self.m1, max_new=2)[0]), 2)       # still in step

    def test_batch_slots(self):
        b = Engine("--batch", "2")
        try:
            self.assertEqual(b.info["batch_slots"], "2")
            a = self.m1 + self.m2
            solo, _, _ = b.gen(a, max_new=12)                              # the solo path's answer (seq 0)
            first, resume, cont = b.admit(0, a + [self.m3[1]], 12)         # a different prompt in slot 0
            self.assertTrue(cont)
            b.admitted = 1
            got = b.slots()
            self.assertEqual(len(got[0][0]) + 1, 12)                       # the admission's token + 11 windows
            self.assertEqual(got[0][1], "length")
            part, _, _ = b.gen(a, max_new=4)                               # a solo request the server then promotes:
            self.assertEqual(part, solo[:4])
            first, resume, cont = b.admit(1, a + part, 8)                  # BGEN with the prompt + its tokens so far
            self.assertEqual(resume, len(a) + 3)                           # seq 0 copied over: nothing read again
            b.admitted = 1
            got = b.slots()
            self.assertEqual(solo[:4] + first + got[1][0], solo)          # and it goes on as the solo path would have
            b.admit(0, a, 400)
            with self.assertRaises(AssertionError):                       # a busy slot is refused
                b.admit(0, a, 4)
            b.send("BSTOP 0")
            b.admitted = 1
            self.assertEqual(b.slots()[0][1], "cancel")
            # two slots at once: each says what it says alone (a slot sampled another slot's row once - a C++
            # overload took the row index for a draft token - and only this comparison sees that)
            x, y = self.m1 + self.m3, self.m2 + self.mx
            alone_x, _, _ = b.gen(x, max_new=10)
            alone_y, _, _ = b.gen(y, max_new=10)
            fx, _, _ = b.admit(0, x, 10)
            fy, _, _ = b.admit(1, y, 10)
            b.admitted = 2
            got = b.slots()
            self.assertEqual(fx + got[0][0], alone_x)
            self.assertEqual(fy + got[1][0], alone_y)
            b.admit(0, self.m1 + self.m2 + self.m3, 400)                   # two slots at once, one stopped
            b.admit(1, self.m3 + self.m1, 400)
            b.admitted = 2
            got = b.slots(stop_after=(1, 3))
            self.assertEqual(got[1][1], "cancel")
            self.assertLess(len(got[1][0]), 40)
            self.assertIn(got[0][1], ("length", "stop"))
            self.assertEqual(len(b.gen(self.m1, max_new=2)[0]), 2)         # the solo path still in step
        finally:
            b.close()

    def test_refusals_stay_in_step(self):
        for cmd in ("GEN x", "GENI 4 1,2", "VRAM 100", "HELLO"):
            self.e.send(cmd)
            self.assertTrue(self.e.line().startswith("ERR"), cmd)
        self.assertEqual(len(self.e.gen(self.m1, max_new=2)[0]), 2)


if __name__ == "__main__":
    unittest.main()
