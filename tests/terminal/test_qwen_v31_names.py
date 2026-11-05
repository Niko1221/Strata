"""Round 0214: Unsloth's v3.1 Qwen3.8-Flash-Next sets in the Terminal's model names.  UD-IQ1_M is also DeepSeek's set: the bare name stays DeepSeek's (as before),
the Qwen set is --family unsloth."""
from __future__ import annotations

import unittest
from unittest import mock

from strata_cli.launch.catalog import Catalog
from tests.terminal.launch_support import HW_12, Home, catalog

PC = {**HW_12, "ram_gb": 64.0, "disk_free_gb": 900.0, "engine_vendor": "nvidia"}


class Names(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(PC))
        p.start()
        self.addCleanup(p.stop)

    def test_the_new_names_resolve_to_the_unsloth_family(self):
        cat = catalog(PC)
        for m in ("UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS-v31"):
            self.assertEqual(cat.resolve(m.lower()), ("unsloth", m))
        self.assertEqual(cat.resolve("ud-iq4_xs"), ("unsloth", "UD-IQ4_XS"))

    def test_ud_iq1_m_bare_is_still_deepseeks_and_unsloth_by_family(self):
        cat = catalog(PC)
        self.assertEqual(cat.resolve("ud-iq1_m"), ("deepseek41", "UD-IQ1_M"))
        self.assertEqual(cat.resolve("ud-iq1_m", "unsloth"), ("unsloth", "UD-IQ1_M"))


if __name__ == "__main__":
    unittest.main()
