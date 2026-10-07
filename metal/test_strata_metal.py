"""strata-metal against the engine line protocol, on a real model (macOS).

    STRATA_METAL_EXE=build-metal/metal/strata-metal STRATA_METAL_GGUF=<a .gguf> python -m unittest metal.test_strata_metal

Any GGUF with <|im_start|> works (LiquidAI/LFM2-350M-GGUF Q8_0 is a 360 MB hybrid model whose recurrent layers need
the checkpoints, as Qwen3.8-Flash-Next's do).  Skipped without the two variables.  Greedy throughout, so the answers
of two engines given the same ids MUST be the same tokens.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest

EXE, GGUF = os.environ.get("STRATA_METAL_EXE"), os.environ.get("STRATA_METAL_GGUF")


class Engine:
    def __init__(self):
        self.p = subprocess.Popen([EXE, "--serve", "--gguf", GGUF, "--max-context", "4096"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.info = {}
        for line in self.p.stdout:
            if line.startswith("INFO "):
                self.info.update(kv.split("=", 1) for kv in line.split()[1:])
            if line.startswith("READY"):
                self.ready = line.split()
                break
        self.im_start = int(self.info["im_start"])

    def send(self, line: str):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def line(self) -> str:
        return self.p.stdout.readline().strip()

    def gen(self, ids, max_new=16, stop_after=None):
        """-> (tokens, resume, DONE fields)"""
        self.send(f"GEN {max_new} " + ",".join(map(str, ids)))
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
        self.assertEqual(resume, len(a) + len(out))
        self.assertEqual(int(done2[14]), len(self.mx))             # prompt tokens read

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

    def test_refusals_stay_in_step(self):
        for cmd in ("GEN x", "GENI 4 /tmp/x 1,2", "VRAM 100", "HELLO"):
            self.e.send(cmd)
            self.assertTrue(self.e.line().startswith("ERR"), cmd)
        self.assertEqual(len(self.e.gen(self.m1, max_new=2)[0]), 2)


if __name__ == "__main__":
    unittest.main()
