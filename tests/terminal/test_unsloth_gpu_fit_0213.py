"""0.2.13 (owner, 2026-11-08): one fit label for every family.  The model's weight files WITHOUT the n-gram / PLE lookup table (the "Needs (RAM + VRAM)" column)
against the graphics memory plus the RAM: "fits", else "may run slower".  Nothing else is said on those rows (no "SSD N GB", no "too big", no RAM floor).

The screenshot PC: a Tesla V100-PCIE-32GB with 31 GB of RAM (63 GB together): every set up to 63 GB of weights said "fits"; the Unsloth sets said "too big" while
IQ3_S, with the same 50 GB of experts, said "fits, a bit slower".
"""
from __future__ import annotations

import unittest

from strata_cli.launch.catalog import FITS, NO, SLOWER, TIGHT, verdict
from strata_control import fit_label
from tests.terminal.launch_support import HW_12, Home, catalog

V100 = {"index": 0, "name": "Tesla V100-PCIE-32GB", "vram_gb": 32.0, "usable": True, "vendor": "nvidia", "arch": 70}
HW_V100_31 = {**HW_12, "ram_gb": 31.0, "disk_free_gb": 5000.0, "gpus": [V100]}


class TheRule(unittest.TestCase):
    def test_the_label_is_the_weights_against_ram_plus_card(self):
        self.assertEqual(fit_label.label(63.0, 31.0, 32.0), "fits")                 # exactly as much as there is
        self.assertEqual(fit_label.label(63.1, 31.0, 32.0), "may run slower")
        self.assertEqual(fit_label.label(35.0, 31.0, 0.0), "may run slower")        # no card: the RAM alone

    def test_the_card_that_counts(self):
        self.assertEqual(fit_label.card_room_gb([{"vram_gb": 12.0}, {"vram_gb": 24.0}]), 24.0)         # the largest card
        self.assertEqual(fit_label.card_room_gb([{"vram_gb": 96.0, "uma": True, "dedicated_gb": 8.0}]), 8.0)   # a chip that shares the RAM: its carve-out only
        self.assertEqual(fit_label.card_room_gb([]), 0.0)


class TheTerminalMenu(unittest.TestCase):
    def rows(self, hw):
        with Home():
            return {(r.family, r.model): r for r in catalog(hw).rows()}

    def test_the_v100_screenshot(self):
        rows = self.rows(HW_V100_31)
        seen = set()
        for (family, model), r in rows.items():
            if family == "deepseek41":
                continue
            seen.add(r.verdict)
            want = FITS if r.memory_gb <= 63.0 else SLOWER
            self.assertEqual((r.verdict, r.words), (want, "fits" if want == FITS else "may run slower"), (family, model, r.memory_gb))
        self.assertEqual(seen, {FITS, SLOWER})
        for m in ("UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS", "UD-IQ4_XS-v31"):         # the Unsloth sets that were "too big"
            self.assertEqual(rows[("unsloth", m)].words, "fits", m)
        for m in ("UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            self.assertEqual(rows[("unsloth", m)].words, "may run slower", m)
        self.assertEqual(rows[("qwen", "IQ3_S")].words, rows[("unsloth", "UD-IQ3_XXS")].words)                # same size, same word
        self.assertEqual(rows[("nvfp4", "NVFP4")].words, "may run slower")                                     # 69 GB
        for key, r in rows.items():
            if key[0] == "deepseek41":                                    # 72 to 306 GB of weights, the Engram table (111 to 203 GB) not counted
                self.assertEqual(r.words, "may run slower", key)

    def test_no_row_uses_the_old_words(self):
        for hw in (HW_V100_31, HW_12, {**HW_12, "ram_gb": 16.0, "gpus": []}):
            for key, r in self.rows(hw).items():
                if r.verdict == NO:                                           # DeepSeek without a card: not about memory
                    self.assertEqual(key[0], "deepseek41")
                    continue
                for old in ("too big", "fits, a bit slower", "fits, part read from SSD", "much read from SSD", "does not fit", "slow here"):
                    self.assertNotIn(old, r.words, (key, r.words))
                self.assertIn(r.words, ("fits", "may run slower", "tight: needs more disk space"), key)

    def test_a_64_gb_pc_with_a_12_gb_card(self):
        rows = self.rows({**HW_12, "disk_free_gb": 5000.0})                # 76 GB together
        self.assertEqual(rows[("unsloth", "UD-IQ4_XS-v31")].words, "fits")
        self.assertEqual(rows[("unsloth", "UD-Q4_K_XL")].words, "may run slower")           # 78 GB
        self.assertEqual(rows[("nvfp4", "NVFP4")].words, "fits")                              # 69 GB

    def test_deepseek_keeps_the_two_messages_that_are_not_about_memory(self):
        rows = self.rows({**HW_V100_31, "disk_free_gb": 100.0})
        self.assertEqual((rows[("deepseek41", "IST-DAS-2.00bpw")].verdict, rows[("deepseek41", "IST-DAS-2.00bpw")].words), (TIGHT, "tight: needs more disk space"))
        no_card = self.rows({**HW_V100_31, "gpus": []})
        self.assertEqual(no_card[("deepseek41", "IST-DAS-2.00bpw")].verdict, NO)             # it cannot run without a card: said as before

    def test_the_catalog_words(self):
        with Home():
            row = next(x for x in catalog(HW_V100_31).strata.model_table(HW_V100_31) if x["family"] == "unsloth")
        by = {x["model"]: x["on_this_pc"] for x in row["sizes"]}
        self.assertEqual({v for k, v in by.items()}, {"fits", "may run slower"})
        self.assertEqual(by["UD-IQ3_XXS"], "fits")
        self.assertEqual(by["UD-Q4_K_XL"], "may run slower")

    def test_the_table_words_in_the_terminal(self):
        self.assertEqual(verdict("fits"), (FITS, "fits"))
        self.assertEqual(verdict("may run slower"), (SLOWER, "may run slower"))
        self.assertEqual(verdict("does not fit: it needs one or more graphics cards")[0], NO)


if __name__ == "__main__":
    unittest.main()
