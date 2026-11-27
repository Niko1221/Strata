"""Ornith-1.5-35B-A3B (GGUF architecture qwen35moe) on Strata: the same engine path as Qwen3.6-35B-A3B - identical architecture,
tensor layout, tokenizer and MTP block - so the loader's checks (strata_control/q36_types.py) and the KV, MTP and picture figures
are shared, and the family is its own files, names, sizes and RAM figures (strata_control/ornith_types.py).

These tests keep the catalog row equal to ornith_types.py, check that every published set (the real tensor type tables, read from
the GGUF headers at the pinned revision: tests/data/ornith_types/*.types.json) passes the engine loader's checks with no tensor
left unclassified, that "use my own files" reads an Ornith file as family ornith and a Qwen3.6 file as family qwen36, and that the
fit figures the app shows exist for both families."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from strata_control import models, ornith_types, own_path_fit, q36_types
from tests.terminal.test_q36_types import q36_gguf

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = sorted((ROOT / "tests" / "data" / "ornith_types").glob("*.types.json"))
GATE_OF = {"Q8_0": 8, "Q6_K": 14, "Q5_K_M": 13, "Q4_K_M": 12}      # GGML type of the routed experts' gate in each published file


def table(fixture: Path) -> dict[str, int]:
    return {n: v["type"] for n, v in json.loads(fixture.read_text(encoding="utf-8"))["tensors"].items()}


def size_of(fixture: Path) -> str:
    return fixture.name.split(".types")[0][len("Ornith-1.5-35B-"):]


class Catalog(unittest.TestCase):
    def test_the_catalog_sizes_are_setups(self):
        import setup
        fam = setup.OWN_PATH_FAMILIES[ornith_types.FAMILY]
        self.assertEqual(tuple(fam["models"]), ornith_types.SIZES)
        self.assertEqual(fam["file"], ornith_types.FILE)
        self.assertEqual(fam["arch"], "qwen35moe")
        self.assertEqual(fam["name"], own_path_fit.FAMILY_NAME_ORNITH)
        self.assertEqual(fam["tag"], "ornith-1.5-35b-a3b-")

    def test_the_engine_path_is_qwen36s(self):
        import setup
        q = setup.OWN_PATH_FAMILIES[q36_types.FAMILY]
        o = setup.OWN_PATH_FAMILIES[ornith_types.FAMILY]
        for key in ("arch", "kv_bytes_per_token", "mtp", "vision", "state_gb"):
            self.assertEqual(o[key], q[key], key)

    def test_the_files_are_the_pinned_repository(self):
        import setup
        repo = "ornith-ai/Ornith-1.5-35B-A3B-GGUF"
        self.assertIn(repo, setup.HF_REVISIONS)
        self.assertEqual(sorted(setup.ORNITH_FILES),
                         sorted([f.name.split(".types")[0] + ".gguf" for f in FIXTURES] + ["mmproj-Ornith-1.5-35B-BF16.gguf"]))
        for f in FIXTURES:
            doc = json.loads(f.read_text(encoding="utf-8"))
            self.assertEqual((doc["repo"], doc["revision"]), (repo, setup.HF_REVISIONS[repo]), f.name)

    def test_the_ram_figures_are_the_file_sizes(self):
        import setup
        fam = setup.OWN_PATH_FAMILIES[ornith_types.FAMILY]
        for q in ornith_types.SIZES:
            size, sha = setup.ORNITH_FILES[ornith_types.FILE.format(q=q)]
            self.assertAlmostEqual(fam["models"][q]["ram_gb"], size / 1e9, delta=0.1, msg=q)
            self.assertEqual(len(sha), 64, q)


class PublishedSets(unittest.TestCase):
    def test_every_published_set_is_accepted_by_the_engine_loader(self):
        self.assertEqual([f.name.split(".types")[0] for f in FIXTURES],
                         sorted(ornith_types.FILE.format(q=q)[:-5] for q in ornith_types.SIZES))
        for f in FIXTURES:
            types = table(f)
            self.assertEqual(q36_types.problems(types), [], f.name)
            self.assertFalse([n for n in types if q36_types.kind_of(n) == "other"], f.name)

    def test_the_published_sizes_are_the_plain_quantizations(self):
        for f in FIXTURES:
            gate = {t for n, t in table(f).items() if n.endswith("ffn_gate_exps.weight")}
            self.assertEqual(gate, {GATE_OF[size_of(f)]}, f.name)

    def test_a_renamed_file_is_taken_as_the_size_its_experts_are(self):
        base = table(next(f for f in FIXTURES if size_of(f) == "Q4_K_M"))
        experts = [n for n in base if n.endswith("ffn_gate_exps.weight") or n.endswith("ffn_up_exps.weight")]
        for q, t in GATE_OF.items():
            self.assertEqual(ornith_types.size_for("my-ornith.gguf", {**base, **{n: t for n in experts}}), (q, False), q)

    def test_a_type_the_engine_cannot_run_is_refused(self):
        types = table(next(f for f in FIXTURES if size_of(f) == "Q4_K_M"))
        down = next(n for n in types if n.endswith("ffn_down_exps.weight"))
        self.assertTrue(q36_types.problems({**types, down: 1}))              # F16 is no down type of the expert kernels
        norm = next(n for n in types if n.endswith("attn_norm.weight"))
        self.assertTrue(q36_types.problems({**types, norm: 2}))             # a norm the f32 loader cannot widen (Q4_0)
        gate = next(n for n in types if n.endswith("ffn_gate_exps.weight"))
        self.assertTrue(q36_types.problems({**types, gate: 14}))            # gate and up must be one type


class Importer(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="strata-ornith-import-")
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def small_table(self, **changes) -> dict[str, int]:
        """The Q4_K_M set's names and types, a few layers of it (enough for each class), with `changes` applied."""
        full = table(next(f for f in FIXTURES if size_of(f) == "Q4_K_M"))
        keep = {n: t for n, t in full.items() if not n.startswith("blk.") or n.split(".")[1] in ("0", "3")}
        return {**keep, **changes}

    def put(self, name: str, types: dict[str, int]) -> Path:
        (self.dir / name).write_bytes(q36_gguf(types))
        return self.dir

    def test_a_published_ornith_file_gives_a_plan_for_family_ornith(self):
        plan = models.plan_import(str(self.put("Ornith-1.5-35B-Q4_K_M.gguf", self.small_table())))
        self.assertEqual(plan["issues"], [])
        self.assertEqual((plan["candidate_selection"]["family"], plan["candidate_selection"]["model"]), ("ornith", "Q4_K_M"))
        self.assertIn("--gguf-dir", plan["candidate_selection"]["setup_args_hint"])

    def test_a_qwen36_file_is_still_family_qwen36(self):
        plan = models.plan_import(str(self.put("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", self.small_table())))
        self.assertEqual(plan["candidate_selection"]["family"], "qwen36")
        self.assertEqual(plan["candidate_selection"]["model"], "UD-Q4_K_XL")

    def test_a_file_under_the_publishers_name_with_a3b_is_taken_as_its_size_and_said(self):
        # the name the local files are kept under (the -A3B form the model is announced by): not the published file name, so
        # the size comes from the routed experts' gate type and the note says the build was not measured
        q5 = table(next(f for f in FIXTURES if size_of(f) == "Q5_K_M"))
        experts = {n: t for n, t in q5.items() if n.endswith("_exps.weight")}
        plan = models.plan_import(str(self.put("Ornith-1.5-35B-A3B-Q5_K_M.gguf", self.small_table(**experts))))
        self.assertEqual((plan["candidate_selection"]["family"], plan["candidate_selection"]["model"]), ("ornith", "Q5_K_M"))
        self.assertTrue(any("not exactly a build Strata has tested" in m for m in plan["messages"]))

    def test_a_renamed_ornith_file_cannot_be_told_from_qwen36_and_is_said(self):
        # the two families are the same architecture: a file that does not name Ornith is taken as the shared family
        plan = models.plan_import(str(self.put("my-ornith.gguf", self.small_table())))
        self.assertEqual((plan["candidate_selection"]["family"], plan["candidate_selection"]["model"]), ("qwen36", "UD-Q4_K_XL"))
        self.assertTrue(any("not exactly a build Strata has tested" in m for m in plan["messages"]))

    def test_the_cli_the_app_calls_gives_the_same_plan(self):
        from strata_cli.cli import main
        folder = self.put("Ornith-1.5-35B-Q4_K_M.gguf", self.small_table())
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["models", "import", str(folder), "--plan", "--output", "json"])
        answer = json.loads(out.getvalue())
        plan = answer.get("plan", answer)
        self.assertEqual(code, 0, out.getvalue()[:500])
        self.assertEqual(plan["candidate_selection"]["family"], "ornith")

    def test_an_unsupported_quant_is_refused_with_the_existing_words(self):
        types = self.small_table()
        down = next(n for n in types if n.endswith("ffn_down_exps.weight"))
        plan = models.plan_import(str(self.put("Ornith-1.5-35B-Q4_K_M.gguf", {**types, down: 1})))
        self.assertIsNone(plan["candidate_selection"])
        self.assertIn("unsupported_or_missing_architecture", plan["issues"])
        self.assertIn(models.UNSUPPORTED_MODEL_TYPE, plan["messages"])


class FitFigures(unittest.TestCase):
    def test_the_formula_inputs_are_there_for_both_families(self):
        q = own_path_fit.inputs(own_path_fit.FAMILY_NAME, "UD-Q4_K_XL")
        o = own_path_fit.inputs(own_path_fit.FAMILY_NAME_ORNITH, "Q4_K_M")
        self.assertEqual(o["bytes_per_token"], q["bytes_per_token"])     # the same KV cache per token
        self.assertGreater(o["fixed_gb"], 0)

    def test_a_config_model_name_finds_the_size(self):
        for name, size in ((own_path_fit.FAMILY_NAME_ORNITH, "Q5_K_M"), (own_path_fit.FAMILY_NAME, "UD-Q4_K_XL")):
            self.assertEqual(own_path_fit.from_model_name(f"{name}-{size.lower()}"), own_path_fit.inputs(name, size), size)
        self.assertIsNone(own_path_fit.from_model_name("qwen3.8-flash-next-gsq-rco-iq3_s"))

    def test_the_context_advice_stays_at_128k(self):
        self.assertEqual(own_path_fit.DEFAULT_CONTEXT, 131072)


class LaunchDisplayName(unittest.TestCase):
    def test_the_display_name_is_the_family(self):
        from strata_cli.launch.harness import Target, _display_name
        t = Target(url="http://127.0.0.1:8080", key="k", model="ornith-1.5-35b-a3b-q5_k_m")
        self.assertIn("Ornith-1.5-35B-A3B", _display_name(t))
        self.assertIn("Qwen3.6-35B-A3B", _display_name(Target(url=t.url, key=t.key, model="qwen3.6-35b-a3b-ud-q4_k_xl")))


if __name__ == "__main__":
    unittest.main()
