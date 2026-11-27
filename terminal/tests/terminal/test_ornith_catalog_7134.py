"""#7134: Ornith-1.5-35B-A3B in the shared catalog (setup.OWN_PATH_FAMILIES["ornith"], the Qwen3.6 engine path).  test_ornith_types.py
(eSPiYa, PR #7140) pins the table, the importer and the fit figures; these tests pin the places that read the table: the model list the
app and `strata launch` show (its own group, its own figures), the run config name setup writes and reads back, the picture step, and
the model ids a client may send."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import setup
from strata_control import own_path_fit, pictures
from tests.terminal.launch_support import HW_12, Home, catalog

SIZES = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]


class ModelList(unittest.TestCase):
    def table(self):
        with Home():
            cat = catalog(HW_12)
            return cat, cat.strata.model_table(cat.hardware())

    def test_ornith_is_a_group_of_its_own_after_qwen36(self):
        cat, table = self.table()
        order = [f["family"] for f in table]
        self.assertIn("ornith", order)
        self.assertLess(order.index("qwen36"), order.index("ornith"))
        ornith = next(f for f in table if f["family"] == "ornith")
        self.assertEqual(ornith["title"], "Ornith-1.5-35B-A3B")
        self.assertEqual([s["model"] for s in ornith["sizes"]], SIZES)

    def test_qwen36_keeps_its_own_sizes(self):
        _cat, table = self.table()
        q36 = next(f for f in table if f["family"] == "qwen36")
        self.assertEqual([s["model"] for s in q36["sizes"]], list(setup.OWN_PATH_FAMILIES["qwen36"]["models"]))
        self.assertFalse(set(SIZES) & {s["model"] for s in q36["sizes"]} - {"Q8_0"})

    def test_every_size_carries_its_download_and_the_context_figures(self):
        _cat, table = self.table()
        ornith = next(f for f in table if f["family"] == "ornith")
        for s in ornith["sizes"]:
            size, _sha = setup.ORNITH_FILES[setup.OWN_PATH_FAMILIES["ornith"]["file"].format(q=s["model"])]
            self.assertAlmostEqual(s["download_gb"], size / 1e9, delta=0.1, msg=s["model"])
            fit = s.get("context_fit")
            self.assertEqual(fit, own_path_fit.inputs("ornith-1.5-35b-a3b", s["model"]), s["model"])
            self.assertEqual(fit["bytes_per_token"], own_path_fit.BYTES_PER_TOKEN)

    def test_the_rows_of_strata_launch_list_ornith_under_its_own_title(self):
        with Home():
            cat = catalog(HW_12)
            rows = [r for r in cat.rows(cat.computer().raw) if r.family == "ornith"]
        self.assertEqual([r.model for r in rows], SIZES)
        self.assertEqual({r.family_title for r in rows}, {"Ornith-1.5-35B-A3B"})

    def test_setup_accepts_the_family_and_its_sizes_as_a_pair(self):
        # `--model Q4_K_M` alone is a size of --family ornith, not of any Flash-Next table
        self.assertIn("ornith", setup.PREVIEW_FAMILIES)
        self.assertTrue(set(SIZES) <= set(setup.OWN_PATH_FAMILIES["ornith"]["models"]))
        self.assertEqual(setup.OWN_PATH_FAMILIES["ornith"]["tag"], "ornith-1.5-35b-a3b-")


class RunConfig(unittest.TestCase):
    def test_a_run_config_named_for_an_ornith_size_is_read_back_as_that_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            for q in SIZES:
                cfg = Path(tmp) / f"strata-ornith-1.5-35b-a3b-{q.lower()}.json"
                cfg.write_text(json.dumps({"arch": "qwen35moe", "args": ["--max-context", "65536", "--mtp", "2"],
                                           "model_name": f"ornith-1.5-35b-a3b-{q.lower()}"}), encoding="utf-8")
                got = setup.choices_from_config(cfg)
                self.assertEqual((got["family"], got["model"], got["context"]), ("ornith", q, 65536), q)

    def test_a_qwen36_config_is_still_qwen36(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "strata-qwen36-ud-q4_k_xl.json"
            cfg.write_text(json.dumps({"arch": "qwen35moe", "args": ["--max-context", "32768"]}), encoding="utf-8")
            self.assertEqual(setup.choices_from_config(cfg)["family"], "qwen36")

    def test_the_base_figure_is_per_size_from_the_dense_weights_the_engine_printed(self):
        # measured on an RTX 3090 with engine 0.1.69, `strata launch --family ornith`, 32K ("N GiB on the GPU" of the start): the dense
        # part grows with the size, so a Q8_0 start holds about 2.5 GiB more than the Q5_K_M start the base figure was first taken from
        measured = {"Q4_K_M": 1.65, "Q5_K_M": 1.85, "Q6_K": 2.06, "Q8_0": 4.40}
        self.assertEqual(own_path_fit.DENSE_GPU_GIB_ORNITH, measured)
        state = own_path_fit.STATE_GIB
        for q, dense in measured.items():
            want = own_path_fit.BASE_GIB - 4.40 + dense + own_path_fit.HYBRID_MIN_ORNITH[q] + state
            self.assertAlmostEqual(own_path_fit.inputs("ornith-1.5-35b-a3b", q)["fixed_gb"], want, places=3, msg=q)
        fixed = {q: own_path_fit.inputs("ornith-1.5-35b-a3b", q)["fixed_gb"] for q in measured}
        self.assertEqual(sorted(fixed, key=fixed.get), ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"])
        self.assertGreater(fixed["Q8_0"] - fixed["Q5_K_M"], 2.5)
        # Qwen3.6 keeps its one base figure
        self.assertAlmostEqual(own_path_fit.inputs("qwen3.6-35b-a3b", "Q8_0")["fixed_gb"], own_path_fit.BASE_GIB + 1760 / 1024 + state, places=3)

    def test_the_fit_figures_follow_the_config_name(self):
        for q in SIZES:
            self.assertEqual(own_path_fit.from_model_name(f"ornith-1.5-35b-a3b-{q.lower()}"),
                             own_path_fit.inputs("ornith-1.5-35b-a3b", q), q)
        self.assertIsNone(own_path_fit.from_model_name("ornith-1.5-35b-a3b-ud-q4_k_xl"))
        self.assertIsNotNone(own_path_fit.from_model_name("qwen3.6-35b-a3b-ud-q4_k_xl"))


class PicturesAndNames(unittest.TestCase):
    def test_the_picture_step_leaves_ornith_to_its_installer_like_qwen36(self):
        cfg = {"arch": "qwen35moe", "model_name": "ornith-1.5-35b-a3b-q4_k_m"}
        self.assertIsNone(pictures.family_of(cfg, None))
        self.assertIsNone(pictures.family_of({"model_name": "qwen3.6-35b-a3b-ud-q4_k_xl"}, None))

    def test_the_server_knows_ornith_ids_as_strata_ids(self):
        from serve.server import Service
        for name in ("ornith-1.5-35b-a3b-q4_k_m", "ornith-1.5-35b-a3b-q8_0"):
            self.assertTrue(Service.is_strata_model_id(name), name)
        self.assertFalse(Service.is_strata_model_id("gpt-4o"))
        self.assertTrue(Service._FAMILY_PREFIX.match("ornith-1.5-35b-a3b-q4_k_m"))
        self.assertFalse(Service._FAMILY_PREFIX.match("ornith-1x5-35b-a3b-q4_k_m"))


if __name__ == "__main__":
    unittest.main()
