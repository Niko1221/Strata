"""The checked-in IQ grid tables must stay byte-identical to ggml-common.h."""
from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path

VULKAN = Path(__file__).resolve().parents[3] / "engine" / "vulkan"
GENERATOR = VULKAN / "tools" / "gen_iq_tables.py"
HPP = VULKAN / "include" / "strata" / "vulkan" / "iq_tables.hpp"


def load_generator():
    spec = importlib.util.spec_from_file_location("gen_iq_tables", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IQTables(unittest.TestCase):
    def setUp(self):
        self.tool = load_generator()

    def test_tables_match_the_vendored_ggml_header(self):
        source = self.tool.HEADER.read_text(encoding="utf-8")
        expected = self.tool.read_tables(source)
        text = HPP.read_text(encoding="utf-8")
        for name, (ctype, count, width) in self.tool.TABLES.items():
            match = re.search(rf"static const uint8_t {name.upper()}\[(\d+)\] = \{{(.*?)\}};",
                              text, re.DOTALL)
            self.assertIsNotNone(match, f"missing table {name}")
            self.assertEqual(int(match.group(1)), len(expected[name]))
            got = bytes(int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", match.group(2)))
            self.assertEqual(got, expected[name], f"{name} drifted from ggml-common.h")

    def test_shader_sources_and_spirv_are_present(self):
        for name in ("expert_iq3_xxs_gemv", "expert_iq2_xs_gemv", "expert_iq1_m_gemv"):
            src = VULKAN / "shaders" / f"{name}.comp"
            spv = VULKAN / "shaders" / f"{name}.comp.spv"
            self.assertTrue(src.is_file(), f"missing {src}")
            self.assertIn('#include "iq_tables.glsl"', src.read_text())
            self.assertTrue(spv.is_file() and spv.stat().st_size > 1000, f"missing SPIR-V for {name}")

    def test_generator_rejects_a_wrong_table_count(self):
        # A truncated table must fail loudly rather than emit a short array.
        good = ("GGML_TABLE_BEGIN(uint8_t, kmask_iq2xs, 8)\n"
                "1, 2, 3\nGGML_TABLE_END()\n")
        with self.assertRaises(SystemExit):
            self.tool.read_tables(good)


if __name__ == "__main__":
    unittest.main()
