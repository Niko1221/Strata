"""#6608 / #6720: Apple's GPU compiler for the M1 / M2 families (Apple7, Apple8: G13 / G14) aborts ("cannot select: s64 = ...") on a device
pointer indexed by `<constant> + (<variable> >> <shift>)`, for example `pl[64 + (j >> 3)]`, and the engine dies with
"mtl: pipeline ink_moe_gu_situ: Compilation failed due to an interrupted connection" before it serves anything.  The same bytes read
through a pointer formed first (`hp = pl + 64; hp[j >> 3]`) compile.  This test fails when a Metal source of strata-inkling indexes a pointer the
old way again.  No GPU, no Mac.

    python -m unittest tools.test_metal_g13_index_pattern
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

METAL = Path(__file__).resolve().parents[2] / "engine" / "src" / "inkling2" / "metal"

# `name[ 64 + (j >> 3) ]`, `name[128u + (j >> 2)]`, `name[ 64 + j >> 3 ]` : a literal added to a shifted variable inside one subscript
PATTERN = re.compile(r"\b[A-Za-z_]\w*\s*\[\s*\d+u?\s*\+\s*\(?\s*[A-Za-z_]\w*\s*>>\s*\d+\s*\)?\s*\]")


def code_lines(path: Path):
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = line.split("//", 1)[0]
        if code.strip():
            yield no, code


class MetalIndexPattern(unittest.TestCase):
    def test_no_literal_plus_shift_subscript(self):
        files = sorted(METAL.glob("*.metal"))
        self.assertTrue(files, "no .metal files found under " + str(METAL))
        bad = []
        for f in files:
            for no, code in code_lines(f):
                if PATTERN.search(code):
                    bad.append(f"{f.name}:{no}: {code.strip()}")
        self.assertEqual(bad, [], "pointer subscripts the Apple7/Apple8 GPU compiler cannot select (form the pointer first, then index):\n" + "\n".join(bad))

    def test_pattern_matches_the_known_bad_lines(self):
        # keeps the regex honest: the lines of 0.2.20 that crashed the compiler
        for line in ("const uint h = uint(pl[64 + (j >> 3)]);", "const uint h = uint(pl[128 + (j >> 3)]);",
                     "const uchar hb = pl[128 + (j >> 2)];", "x = p[ 64u + (k>>3) ];"):
            self.assertTrue(PATTERN.search(line), line)
        for line in ("const uint h = uint(hp[j >> 3]);", "float a = blk[2 + g];", "y = pl[j >> 1];"):
            self.assertFalse(PATTERN.search(line), line)


if __name__ == "__main__":
    unittest.main()
