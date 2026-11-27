"""Custom builds in the Terminal (0.2.23): the person's own compatible fine-tunes and quantizations (local .strata or .gguf files) are listed
next to the official builds of the same base model, run under their own run config, and removed without touching the model files.

Covers Qwen3.8-Flash-Next (.gguf), Qwen3.6-35B-A3B (.gguf), HY-4 Flash (.strata) and Inkling-2-Mini (.strata and .gguf). The files here are tiny
synthetic ones; no engine runs and nothing is downloaded.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import cmd, custom, settings as S
from strata_cli.launch.catalog import Catalog, pick_row
from strata_cli.launch.runner import Runner, launch_name_of
from strata_control import custom_models
from tests.terminal.launch_support import HW_12, HW_24, Home, catalog, run_cli
from tests.terminal.test_launch_tui import Base, flat, make_ctx, press

GB = 1_000_000_000

# one entry per family: the base build the custom file is run as, the official run config id, the engine and the kind of file
FAMILIES = (
    dict(key="qwen", title="Qwen3.8-Flash-Next", base="IQ3_S", official="iq3_s", exe="component:strata-engine", arch=None, ext=".gguf", size=40 * GB),
    dict(key="qwen36", title="Qwen3.6-35B-A3B", base="UD-Q4_K_XL", official="qwen36-ud-q4_k_xl", exe="component:strata-engine", arch="qwen35moe", ext=".gguf", size=20 * GB),
    dict(key="hy4-flash", title="Hy4-Flash", base="SQ-64", official="hy4-flash-sq-64", exe="component:strata-hy4", arch="hy4flash", ext=".strata", size=120 * GB),
    dict(key="inkling-2-mini", title="Inkling-2-Mini", base="SQ-20", official="inkling-2-mini-sq-20", exe="component:strata-inkling", arch="inkling2_loop", ext=".strata", size=17 * GB),
    dict(key="inkling-2-mini", title="Inkling-2-Mini", base="UD-Q4_K_XL", official="inkling-2-mini-ud-q4_k_xl", exe="component:strata-inkling", arch="inkling2_loop", ext=".gguf", size=16 * GB),
)
LABEL = "My Merge v2"


def official_config(spec: dict, weights: Path) -> dict:
    cfg = {"exe": spec["exe"], "args": ["--native", str(weights), "--max-context", "32768"], "tokenizer": str(weights.parent / "tok"), "model_name": spec["official"], "port": 8080}
    if spec["arch"]:
        cfg["arch"] = spec["arch"]
    return cfg


def custom_config(spec: dict, weights: Path, label: str = LABEL) -> dict:
    cfg = official_config(spec, weights)
    cfg.update(custom_models.run_fields(label, spec["key"], spec["base"], spec["size"], weights))
    cfg["model_name"] = custom_models.model_alias(spec["key"], label)
    cfg["drafting"] = "auto"
    return cfg


class Layout:
    """An install folder with the official run config of one spec and (optionally) a custom one, and the person's own tiny weights file."""

    def __init__(self, home: Home, spec: dict, both: bool = True):
        self.spec = spec
        self.install = Path(home.env["STRATA_INSTALL_DIR"])
        self.install.mkdir(parents=True, exist_ok=True)
        for old in self.install.glob("strata-*.json"):                                             # one layout at a time (the sub-tests share a home)
            old.unlink()
        self.files = home.base / "own-files"
        self.files.mkdir(parents=True, exist_ok=True)
        self.weights = self.files / f"my-merge{spec['ext']}"
        self.weights.write_bytes(b"synthetic weights, never read as a model")
        self.official_weights = self.files / f"official{spec['ext']}"
        self.official_weights.write_bytes(b"official synthetic weights")
        self.official = self.install / f"strata-{spec['official']}.json"
        self.official.write_text(json.dumps(official_config(spec, self.official_weights)), encoding="utf-8")
        self.tag = custom_models.run_tag(spec["key"], LABEL)
        self.custom = self.install / f"strata-{self.tag}.json"
        if both:
            self.custom.write_text(json.dumps(custom_config(spec, self.weights)), encoding="utf-8")


def spec_id(spec: dict) -> str:
    return f"{spec['key']}-{spec['base']}{spec['ext']}"


class CustomRowsTests(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def test_the_run_config_id_is_its_own_and_is_never_an_official_one(self):
        for spec in FAMILIES:
            with self.subTest(spec_id(spec)):
                lay = Layout(self.home, spec, both=False)
                self.assertTrue(lay.tag.startswith(custom_models.tag_prefix(spec["key"]) + "custom-"), lay.tag)
                self.assertNotEqual(lay.tag, spec["official"])
                self.assertEqual(custom.run_tag(spec["key"], LABEL), lay.tag)
                # a name that tries to leave the folder or to be an official id is only ever a flat slug
                for evil in ("../../etc/passwd", "C:\\x\\y", "iq3_s", "a/b\\c", "\x1b[31mred"):
                    t = custom.run_tag(spec["key"], evil)
                    self.assertRegex(t, r"^[a-z0-9][a-z0-9.-]*$")
                    self.assertIn("custom-", t)
                    self.assertNotIn("/", t)
                    self.assertNotIn("..", t)

    def test_a_custom_build_is_listed_beside_the_official_one_with_its_name_and_size(self):
        for spec in FAMILIES:
            with self.subTest(spec_id(spec)):
                lay = Layout(self.home, spec)
                cat = catalog(HW_24)
                rows = cat.rows(cat.computer().raw)
                official = next(r for r in rows if r.id == spec["official"])
                mine = next(r for r in rows if r.custom and r.id == lay.tag)
                self.assertFalse(official.custom)
                self.assertEqual((mine.family, mine.model), (official.family, official.model))     # its parent family, run as the base model
                self.assertEqual(mine.display_name, LABEL)
                self.assertIn(LABEL, mine.title)
                self.assertEqual(mine.custom_bytes, spec["size"])
                self.assertAlmostEqual(mine.download_gb, round(spec["size"] / 1e9, 1))
                self.assertEqual(mine.key, (mine.family, mine.model, lay.tag))
                self.assertEqual(official.key, (official.family, official.model))
                ids = [r.id for r in rows]
                self.assertEqual(len(ids), len(set(ids)))                                        # both coexist, nothing replaced
                self.assertGreater(ids.index(lay.tag), ids.index(spec["official"]))              # under the family of its base model
                self.assertFalse(mine.recommended)
                # the list lookups find the right row for each
                self.assertIs(pick_row(rows, S.LaunchSettings(family=spec["key"], model=spec["base"], custom_id=lay.tag)), mine)
                self.assertIs(pick_row(rows, S.LaunchSettings(family=spec["key"], model=spec["base"])), official)
                # an official config is still the official one
                self.assertEqual(cat.tag_of(spec["key"], spec["base"]), spec["official"].lower())
                self.assertEqual(cat.tag_of(spec["key"], spec["base"], lay.tag), lay.tag)


    def test_the_fit_follows_the_size_of_the_files(self):
        """The same base model with smaller and bigger files of the person's own: the row's size is the files' size, and bigger files never need less memory."""
        for spec in FAMILIES:
            with self.subTest(spec_id(spec)):
                got = {}
                for factor in (0.5, 1.5):
                    lay = Layout(self.home, spec)
                    cfg = json.loads(lay.custom.read_text())
                    cfg["custom_bytes"] = int(spec["size"] * factor)
                    lay.custom.write_text(json.dumps(cfg), encoding="utf-8")
                    cat = catalog(HW_24)
                    rows = cat.rows(cat.computer().raw)
                    mine = next(r for r in rows if r.custom and r.id == lay.tag)
                    self.assertAlmostEqual(mine.download_gb, round(spec["size"] * factor / 1e9, 1))
                    got[factor] = mine.fit_gb
                self.assertGreaterEqual(got[1.5], got[0.5])
                if spec["key"] in ("qwen", "qwen36"):                                         # the scaled estimate: strictly more for more bytes
                    self.assertGreater(got[1.5], got[0.5])

    def test_a_config_that_names_a_base_model_that_is_not_listed_is_skipped_not_an_error(self):
        lay = Layout(self.home, FAMILIES[0], both=False)
        bad = custom_config(FAMILIES[0], lay.weights)
        bad["base_model"] = "NOT-A-SIZE"
        (lay.install / "strata-qwen-custom-odd.json").write_text(json.dumps(bad), encoding="utf-8")
        (lay.install / "strata-qwen-custom-broken.json").write_text("{not json", encoding="utf-8")
        cat = catalog(HW_24)
        rows = cat.rows(cat.computer().raw)
        self.assertFalse([r for r in rows if r.custom])

    def test_an_official_config_is_never_taken_for_a_custom_one(self):
        lay = Layout(self.home, FAMILIES[0], both=False)
        self.assertEqual(custom.entries(catalog(HW_24)), [])
        self.assertIsNone(custom.read_custom(lay.official))
        with self.assertRaises(ValueError):
            custom.remove_reference(lay.official)
        self.assertTrue(lay.official.exists())

    def test_the_launch_profile_of_a_custom_build_is_not_the_official_ones(self):
        for spec in FAMILIES:
            with self.subTest(spec_id(spec)):
                a = launch_name_of(S.LaunchSettings(family=spec["key"], model=spec["base"], context=32768))
                b = launch_name_of(S.LaunchSettings(family=spec["key"], model=spec["base"], context=32768, custom_id=custom_models.run_tag(spec["key"], LABEL)))
                self.assertNotEqual(a, b)


class CustomLaunchFlagsTests(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_24, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)

    def plan(self, *args):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", *args])
        self.assertEqual(code, 0, err or out)
        return json.loads(out)["plan"]

    def test_model_takes_the_id_of_a_custom_build_and_runs_it_as_its_own_run_config(self):
        for spec in FAMILIES:
            with self.subTest(spec_id(spec)):
                lay = Layout(self.home, spec)
                plan = self.plan("--model", lay.tag)
                text = json.dumps(plan)
                self.assertIn(lay.tag, text)
                self.assertIn(spec["base"], text)
                # the display name works too, in any letter case
                self.assertIn(lay.tag, json.dumps(self.plan("--model", LABEL.lower())))
                s, _ = cmd.settings_from_args(_parse(["--model", lay.tag]), catalog(HW_24))
                self.assertEqual((s.family, s.model, s.custom_id, s.custom_label), (spec["key"], spec["base"], lay.tag, LABEL))
                self.assertEqual(s.drafting, "auto")                                              # fine-tunes keep the base model's vocabulary
                self.assertEqual(Path(s.model_path), lay.weights)                                 # a second setup run reproduces the same run config
                self.assertIn(f"--model {lay.tag}", S.flags_text(S.LaunchSettings(family=s.family, model=s.model, custom_id=lay.tag)))

    def test_custom_label_needs_the_files_it_names(self):
        code, out, err = run_cli(["launch", "--plan", "--model", "iq3_s", "--custom-label", "x"])
        self.assertNotEqual(code, 0)
        self.assertIn("--model-path", out + err)

    def test_the_label_is_cleaned_never_a_path(self):
        self.assertEqual(custom.clean_label("  a\x1b[0m\n b   "), "a [0m b")
        self.assertLessEqual(len(custom.clean_label("x" * 500)), 80)
        self.assertEqual(custom.slug("../../Windows/System32"), "windows-system32")


def _parse(argv):
    import argparse
    from strata_cli.launch import cmd
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="command")
    cmd.add_parsers(sp)
    return parser.parse_args(["launch", *argv])


class CustomImportFlowTests(unittest.TestCase):
    """--model-path FILE (the person's own file): the plan says it is no tested build, the build gets the name the person gave (or the file's own
    name), a run config of its own and Drafting Auto; setup is told --custom-label and the one file."""

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def plan_for(self, spec: dict, weights: Path) -> dict:
        return {"source": str(weights), "model_dir": str(weights.parent), "scope": "local_gguf_candidate", "files": [], "file_count": 1,
                "total_bytes": spec["size"], "issues": [], "plain_issues": [], "messages": [], "missing_shards": [], "folder_holds": [],
                "file_kind": "strata" if spec["ext"] == ".strata" else "gguf", "file_set_status": "plausible",
                "candidate_selection": {"family": spec["key"], "model": spec["base"], "basis": "same_layout_other_build", "custom": True,
                                        "suggested_label": weights.stem, "family_title": spec["title"], "base_family": spec["key"], "base_model": spec["base"]}}

    def test_an_import_of_a_custom_file_is_named_and_gets_its_own_run_config(self):
        for spec in FAMILIES:
            with self.subTest(spec_id(spec)):
                lay = Layout(self.home, spec, both=False)
                with mock.patch("strata_control.models.plan_import", return_value=self.plan_for(spec, lay.weights)):
                    s = S.LaunchSettings(model_path=str(lay.weights))
                    r = Runner(s, catalog=catalog(HW_24), emit=lambda e: None)
                    r.check_own_files()
                self.assertEqual((s.family, s.model), (spec["key"], spec["base"]))
                self.assertEqual(s.custom_label, lay.weights.stem)                                # the suggested label: the file's own name
                self.assertEqual(s.custom_id, custom_models.run_tag(spec["key"], lay.weights.stem))
                self.assertNotEqual(s.custom_id, spec["official"])
                self.assertEqual(s.drafting, "auto")
                args = r.setup_args()
                self.assertEqual(args[args.index("--custom-label") + 1], lay.weights.stem)
                self.assertEqual(args[args.index("--gguf-dir") + 1], str(lay.weights))           # the one file, so a folder that also holds the official build is no problem
                self.assertEqual(args[args.index("--family") + 1], spec["key"])
                self.assertEqual(args[args.index("--model") + 1], spec["base"])
                # the settings line says the same in flags
                line = S.flags_text(s)
                self.assertIn("--custom-label", line)
                self.assertIn("--model-path", line)

    def test_the_name_the_person_typed_wins_over_the_suggestion(self):
        spec = FAMILIES[3]
        lay = Layout(self.home, spec, both=False)
        with mock.patch("strata_control.models.plan_import", return_value=self.plan_for(spec, lay.weights)):
            s = S.LaunchSettings(model_path=str(lay.weights), custom_label=custom.clean_label(LABEL))
            r = Runner(s, catalog=catalog(HW_24), emit=lambda e: None)
            r.check_own_files()
        self.assertEqual(s.custom_label, LABEL)
        self.assertEqual(s.custom_id, lay.tag)

    def test_a_published_build_is_not_made_a_custom_one(self):
        spec = FAMILIES[0]
        lay = Layout(self.home, spec, both=False)
        plan = self.plan_for(spec, lay.weights)
        plan["candidate_selection"] = {"family": spec["key"], "model": spec["base"], "basis": "known_shard_names_only", "custom": False}
        with mock.patch("strata_control.models.plan_import", return_value=plan):
            s = S.LaunchSettings(model_path=str(lay.weights))
            Runner(s, catalog=catalog(HW_24), emit=lambda e: None).check_own_files()
        self.assertIsNone(s.custom_id)
        self.assertIsNone(s.custom_label)
        self.assertNotIn("--custom-label", Runner(s, catalog=catalog(HW_24)).setup_args())

    def test_starting_an_installed_custom_build_again_keeps_its_run_config(self):
        for spec in FAMILIES:
            with self.subTest(spec_id(spec)):
                lay = Layout(self.home, spec)
                s = S.LaunchSettings()
                custom.use_installed(s, custom.find(catalog(HW_24), lay.tag))
                with mock.patch("strata_control.models.plan_import", return_value=self.plan_for(spec, lay.weights)):
                    r = Runner(s, catalog=catalog(HW_24), emit=lambda e: None)
                    r.check_own_files()
                self.assertEqual((s.custom_id, s.custom_label, s.family, s.model), (lay.tag, LABEL, spec["key"], spec["base"]))
                args = r.setup_args()
                self.assertEqual(args[args.index("--custom-label") + 1], LABEL)                    # a second setup run writes the same run config
                self.assertEqual(r.installed_state()[0], lay.custom)                              # and the launch reads the custom run config, not the official one

    def test_two_files_with_the_same_name_do_not_share_a_run_config(self):
        spec = FAMILIES[3]
        lay = Layout(self.home, spec)                                                               # one custom build called LABEL already exists
        other = lay.files / "other" / f"my-merge{spec['ext']}"
        other.parent.mkdir()
        other.write_bytes(b"another file")
        s = S.LaunchSettings(family=spec["key"], model=spec["base"], model_path=str(other))
        custom.use_import(s, catalog(HW_24), LABEL)
        self.assertNotEqual(s.custom_id, lay.tag)
        self.assertTrue(s.custom_id.startswith(lay.tag))
        s2 = S.LaunchSettings(family=spec["key"], model=spec["base"], model_path=str(lay.weights))
        custom.use_import(s2, catalog(HW_24), LABEL)
        self.assertEqual(s2.custom_id, lay.tag)                                                     # the same files again: the same run config

    def test_the_run_config_keys_a_card_without_the_installer_writes_match_controls(self):
        """Intel Arc: adopt.py has no setup.py behind it, so strata_cli writes the same custom keys itself (size of every shard, alias, display name)."""
        spec = FAMILIES[0]
        lay = Layout(self.home, spec, both=False)
        for i in (1, 2, 3):
            (lay.files / f"split-model-0000{i}-of-00003.gguf").write_bytes(b"x" * (1000 * i))
        s = S.LaunchSettings(family=spec["key"], model=spec["base"], model_path=str(lay.files / "split-model-00001-of-00003.gguf"))
        custom.use_import(s, catalog(HW_24), "Split Merge")
        keys = custom.run_config_keys(s)
        self.assertEqual(keys, custom_models.run_fields("Split Merge", spec["key"], spec["base"], 6000, s.model_path))
        self.assertEqual(custom.custom_alias(s), custom_models.model_alias(spec["key"], "Split Merge"))
        self.assertTrue(keys["custom"])

    def test_the_unsupported_words_name_all_four_families_and_keep_the_custom_api_guidance(self):
        for name in ("Qwen3.8-Flash-Next", "Qwen3.6-35B-A3B", "HY-4 Flash", "Inkling-2-Mini", "Custom model API", "llama-server"):
            self.assertIn(name, custom.UNSUPPORTED_WORDS)

    def test_the_cli_says_what_a_custom_plan_is(self):
        from strata_cli import cli
        plan = self.plan_for(FAMILIES[2], Path("/x/my-merge.strata"))
        plan["adoption_status"] = "requires_checks"
        text = cli._import_words(plan)
        self.assertIn(".strata file(s)", text)
        self.assertIn("your own build of Hy4-Flash", text)
        self.assertIn("never copied", text)


class CustomModelScreenTests(Base):
    def test_the_model_screen_lists_a_custom_build_with_its_name_and_a_custom_marker_beside_the_official_one(self):
        for spec in FAMILIES:
            with self.subTest(spec_id(spec)):
                lay = Layout(self.home, spec)

                async def case(app, pilot, ctx, spec=spec):
                    await press(pilot, "enter")
                    t = flat(app)
                    self.assertIn("Which model?", t)
                    self.assertIn(LABEL, t)
                    self.assertIn("[Custom]", t)
                    self.assertIn(f"{spec['base']}", t)                                              # the official row is still there
                    self.assertNotIn(str(lay.weights), t)
                self.run_app(case, ctx=make_ctx(HW_24, S.LaunchSettings(family=spec["key"], model=spec["base"], custom_id=lay.tag, model_path=str(lay.weights))))

    def test_choosing_the_custom_row_selects_its_run_config_and_choosing_the_official_row_goes_back(self):
        spec = FAMILIES[3]
        lay = Layout(self.home, spec)

        async def choose(app, pilot, ctx, key):
            await press(pilot, "enter")                                                              # the Model screen
            app.screen.menu().select(key)
            await pilot.pause(0.1)
            await press(pilot, "enter")                                                              # choose the row under the cursor
            return ctx.settings

        async def custom_row(app, pilot, ctx):
            s = await choose(app, pilot, ctx, (spec["key"], spec["base"], lay.tag))
            self.assertEqual((s.family, s.model, s.custom_id, s.custom_label), (spec["key"], spec["base"], lay.tag, LABEL))
            self.assertEqual(s.drafting, "auto")
            self.assertEqual(Path(s.model_path), lay.weights)
        self.run_app(custom_row, ctx=make_ctx(HW_24))

        async def official_row(app, pilot, ctx):
            s = await choose(app, pilot, ctx, (spec["key"], spec["base"]))
            self.assertEqual((s.family, s.model, s.custom_id, s.custom_label), (spec["key"], spec["base"], None, None))
            self.assertIsNone(s.model_path)
        self.run_app(official_row, ctx=make_ctx(HW_24))

    def test_the_own_files_screen_asks_for_a_name_prefilled_with_the_suggestion(self):
        spec = FAMILIES[2]
        lay = Layout(self.home, spec, both=False)
        plan = CustomImportFlowTests.plan_for(None, spec, lay.weights)

        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            screen = app.screen
            with mock.patch("strata_control.models.plan_import", return_value=plan), mock.patch("strata_control.models.own_path_problem", return_value=None):
                screen.check_own(str(lay.weights))
                await pilot.pause(0.2)
            from textual.widgets import Input
            box = screen.query_one("#label", Input)
            self.assertFalse(box.disabled)
            self.assertEqual(box.value, lay.weights.stem)                                            # the suggestion, editable
            self.assertIn("your own build", flat(app))
            screen.finish_custom("My HY4 build")
            s = ctx.settings
            self.assertEqual((s.custom_label, s.family, s.model), ("My HY4 build", spec["key"], spec["base"]))
            self.assertEqual(s.custom_id, custom_models.run_tag(spec["key"], "My HY4 build"))
            self.assertEqual(s.model_path, str(lay.weights))
        self.run_app(case, ctx=make_ctx(HW_24))


class RemoveKeepsTheWeightsTests(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def test_remove_deletes_only_the_run_config_and_never_the_model_files(self):
        for spec in FAMILIES:
            with self.subTest(spec_id(spec)):
                lay = Layout(self.home, spec)
                log = lay.install / f"strata-{lay.tag}.log"
                log.write_text("log", encoding="utf-8")
                before = {p: p.read_bytes() for p in lay.files.iterdir() if p.is_file()}
                code, out, err = run_cli(["models", "remove", lay.tag, "--install-dir", str(lay.install)])           # no --yes: only says what it would do
                self.assertEqual(code, 0, err)
                self.assertTrue(lay.custom.exists())
                self.assertIn("--yes", out)
                code, out, err = run_cli(["models", "remove", lay.tag, "--install-dir", str(lay.install), "--yes", "--output", "json"])
                self.assertEqual(code, 0, err or out)
                self.assertFalse(lay.custom.exists())
                self.assertFalse(log.exists())
                self.assertTrue(json.loads(out)["kept_files"])
                self.assertTrue(lay.official.exists())                                                          # the official run config is untouched
                after = {p: p.read_bytes() for p in lay.files.iterdir() if p.is_file()}
                self.assertEqual(before, after)                                                                 # every model file is still there, byte for byte
                self.assertTrue(lay.weights.exists())

    def test_remove_refuses_an_official_install(self):
        lay = Layout(self.home, FAMILIES[0])
        code, out, err = run_cli(["models", "remove", FAMILIES[0]["official"], "--install-dir", str(lay.install), "--yes"])
        self.assertNotEqual(code, 0)
        self.assertTrue(lay.official.exists())
        self.assertTrue(lay.official_weights.exists())

    def test_remove_reference_only_takes_files_beside_the_run_config(self):
        lay = Layout(self.home, FAMILIES[3])
        victim = lay.files / "strata-keepme.log"                     # a file with a run config's look elsewhere is never taken
        victim.write_text("x", encoding="utf-8")
        removed = custom.remove_reference(lay.custom)
        self.assertEqual(removed, [lay.custom.name])
        self.assertTrue(victim.exists())
        self.assertTrue(lay.weights.exists())

    def test_models_list_shows_the_custom_marker_the_name_and_the_size(self):
        lay = Layout(self.home, FAMILIES[3])
        code, out, err = run_cli(["models", "list", "--install-dir", str(lay.install)])
        self.assertEqual(code, 0, err)
        self.assertIn(lay.tag, out)
        self.assertIn("[Custom]", out)
        self.assertIn("17 GB", out)
        self.assertIn(LABEL, out)


if __name__ == "__main__":
    unittest.main()
