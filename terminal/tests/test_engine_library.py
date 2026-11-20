"""Settings > Engine and `strata components engines | use | remove`: the engines installed on this computer, listed, switched and deleted.

The same stub packages the updater tests use (real schema-2 packages, a real test start). Proves: an update keeps every older engine; the list shows
size, the engine in use and the newest; a switch to an older engine and back to the newest holds across a fresh read (a restart); the newest is
selected without a pin; the engine in use and an engine a program uses cannot be deleted; a running model stops a switch with one plain answer;
a package this computer cannot run is shown with its reason and cannot be switched to; the DeepSeek family is listed and switched with its own pin
and the two families never touch each other's state."""
import json
import os
import shutil
import stat
import unittest
from pathlib import Path
from unittest import mock

from strata_control import dsv41_engine, dsv41_paths, engine_component as ec, engine_library as lib, preset_marks, updater
from tests.terminal import update_fixture as fx
from tests.terminal.test_updater import PLATFORM, WINDOWS, UpdaterCase


def folder_of(version):
    return f"{version}-{PLATFORM}-cuda-cuda13"


class Library(UpdaterCase):
    def setUp(self):
        super().setUp()
        self.install_current("0.1.45")
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        updater.run_update(self.ctx(), control=False)           # 0.1.45 stays installed, 0.1.46 is in use
        patch = mock.patch.object(lib, "_process_uses", return_value=False)
        patch.start()
        self.addCleanup(patch.stop)

    def listing(self, **over):
        return lib.list_engines(self.ctx(**over))

    def qwen(self, **over):
        return {e["version"]: e for e in lib.rows_of(self.listing(**over), lib.QWEN)}

    def test_the_list_shows_every_installed_engine_with_size_and_the_one_in_use(self):
        rows = self.qwen()
        self.assertEqual(sorted(rows), ["0.1.45", "0.1.46"])
        self.assertTrue(rows["0.1.46"]["active"] and rows["0.1.46"]["newest"])
        self.assertFalse(rows["0.1.45"]["active"])
        self.assertFalse(rows["0.1.45"]["newest"])
        for row in rows.values():
            self.assertGreater(row["bytes"], 0)
            self.assertTrue(row["fits"] and row["why_not"] is None)
        listing = self.listing()
        self.assertEqual(listing["total_bytes"], sum(r["bytes"] for r in rows.values()))
        self.assertEqual([f["family"] for f in listing["families"]], ["qwen"], "no DeepSeek engine installed: no DeepSeek group")
        self.assertEqual(rows["0.1.46"]["delete_blocked"], "This is the engine in use. Switch to another one first.")
        self.assertTrue(rows["0.1.45"]["can_delete"] and rows["0.1.45"]["can_use"])

    def test_switch_to_an_older_engine_and_back_to_the_newest_survives_a_restart(self):
        out = lib.use_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        self.assertEqual((out["status"], out["from"], out["to"], out["automatic"]), ("switched", "0.1.46", "0.1.45", False))
        self.assertEqual(self.resolved(), "0.1.45")
        self.assertEqual(ec.pinned_folder(self.components), folder_of("0.1.45"))
        # a fresh read of the folder (what a restart sees)
        self.assertEqual(self.qwen()["0.1.45"]["active"], True)
        self.assertEqual(self.qwen()["0.1.46"]["active"], False)
        self.assertEqual(ec.status(self.components, backend="cuda")["engine"]["version"], "0.1.45")
        # both are still installed: nothing was removed by a switch
        self.assertEqual(self.engine_folders(), [folder_of("0.1.45"), folder_of("0.1.46")])
        back = lib.use_engine(self.ctx(), "qwen", folder_of("0.1.46"))
        self.assertEqual((back["status"], back["to"], back["automatic"]), ("switched", "0.1.46", True))
        self.assertIsNone(ec.pinned_folder(self.components), "the newest follows updates again: no pin")
        self.assertEqual(self.resolved(), "0.1.46")
        again = lib.use_engine(self.ctx(), "qwen", folder_of("0.1.46"))
        self.assertEqual((again["status"], again["changed"]), ("already", False))

    def test_a_switch_marks_tuning_presets_for_the_short_recheck_and_deletes_nothing_of_the_person(self):
        self.assertEqual(len(preset_marks.pending(self.state_base)), 1, "the update to 0.1.46 left the presets owing a re-check")
        lib.use_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        self.assertEqual(preset_marks.pending(self.state_base), [], "back on the engine the presets were measured on: nothing owed")
        lib.use_engine(self.ctx(), "qwen", folder_of("0.1.46"))
        self.assertEqual(len(preset_marks.pending(self.state_base)), 1)
        self.assert_user_data_untouched()

    def test_an_update_after_a_switch_makes_the_new_engine_the_one_in_use_and_clears_the_pin(self):
        lib.use_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        self.publish(("0.1.47", fx.GOOD_ENGINE))
        out = updater.run_update(self.ctx(), control=False)
        self.assertEqual(out["results"][0]["status"], "updated")
        self.assertEqual(self.resolved(), "0.1.47")
        self.assertIsNone(ec.pinned_folder(self.components))
        self.assertEqual(self.engine_folders(), [folder_of("0.1.45"), folder_of("0.1.46"), folder_of("0.1.47")])

    def test_the_engine_in_use_cannot_be_deleted_and_an_older_one_can(self):
        with self.assertRaisesRegex(updater.UpdateError, "engine in use"):
            lib.delete_engine(self.ctx(), "qwen", folder_of("0.1.46"))
        self.assertEqual(self.engine_folders(), [folder_of("0.1.45"), folder_of("0.1.46")])
        out = lib.delete_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        self.assertEqual((out["status"], out["version"]), ("deleted", "0.1.45"))
        self.assertGreater(out["freed_bytes"], 0)
        self.assertEqual(self.engine_folders(), [folder_of("0.1.46")])
        self.assertIsNone(updater.State(self.components).section("engine").get("previous"), "the deleted engine is not the one to roll back to")
        with self.assertRaisesRegex(updater.UpdateError, "not installed"):
            lib.delete_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        self.assertEqual(self.resolved(), "0.1.46")
        self.assert_user_data_untouched()

    def test_an_engine_a_running_program_uses_cannot_be_deleted(self):
        with mock.patch.object(lib, "_process_uses", side_effect=lambda folder: folder.name == folder_of("0.1.45")):
            row = self.qwen()["0.1.45"]
            self.assertFalse(row["can_delete"])
            self.assertIn("Stop the model first", row["delete_blocked"])
            with self.assertRaisesRegex(updater.UpdateError, "Stop the model first"):
                lib.delete_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        self.assertTrue((self.components / "engine" / folder_of("0.1.45")).is_dir())

    def test_deleting_a_pinned_older_version_after_switching_away_clears_the_old_marks(self):
        lib.use_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        lib.use_engine(self.ctx(), "qwen", folder_of("0.1.46"))
        lib.delete_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        state = updater.State(self.components).section("engine")
        self.assertNotIn(folder_of("0.1.45"), (state.get("previous"), state.get("pinned")))

    def test_a_running_model_stops_a_switch_with_one_plain_answer_and_changes_nothing(self):
        busy = lambda: [{"profile": "app-1"}]                        # noqa: E731
        with self.assertRaises(updater.Busy) as caught:
            lib.use_engine(self.ctx(serving=busy), "qwen", folder_of("0.1.45"))
        self.assertEqual(caught.exception.exit_code, 7)
        self.assertEqual(self.resolved(), "0.1.46")
        self.assertIsNone(ec.pinned_folder(self.components))
        # deleting an engine nothing uses is not a change to the running one
        out = lib.delete_engine(self.ctx(serving=busy), "qwen", folder_of("0.1.45"))
        self.assertEqual(out["status"], "deleted")

    def test_a_package_this_computer_cannot_run_is_listed_with_its_reason_and_cannot_be_switched_to(self):
        other = self.components / "engine" / folder_of("0.1.99")
        shutil.copytree(self.components / "engine" / folder_of("0.1.46"), other)
        manifest = json.loads((other / ec.MANIFEST).read_text())
        manifest.update(version="0.1.99", min_driver={"nvidia": "999.0"})
        (other / ec.MANIFEST).write_text(json.dumps(manifest))
        row = self.qwen()["0.1.99"]
        self.assertFalse(row["fits"] or row["can_use"])
        self.assertIn("driver 999.0", row["why_not"])
        self.assertFalse(row["newest"], "the newest is the newest one that can run here")
        self.assertTrue(self.qwen()["0.1.46"]["newest"])
        with self.assertRaises(updater.UpdateError):
            lib.use_engine(self.ctx(), "qwen", folder_of("0.1.99"))
        self.assertEqual(self.resolved(), "0.1.46")
        self.assertTrue(row["can_delete"], "an engine that cannot run here can still be deleted to free the space")

    def test_a_damaged_package_is_listed_and_can_be_deleted(self):
        broken = self.components / "engine" / "0.1.50-broken"
        broken.mkdir()
        (broken / "junk.bin").write_bytes(b"x" * 5000)
        rows = self.qwen()
        row = rows["0.1.50"]
        self.assertFalse(row["fits"])
        self.assertEqual(row["bytes"], 5000)
        lib.delete_engine(self.ctx(), "qwen", "0.1.50-broken")
        self.assertFalse(broken.exists())

    def test_a_version_or_newest_names_an_engine_for_the_terminal(self):
        listing = self.listing()
        self.assertEqual(lib.find_entry(listing, "0.1.45")["folder"], folder_of("0.1.45"))
        self.assertEqual(lib.find_entry(listing, "newest")["folder"], folder_of("0.1.46"))
        with self.assertRaisesRegex(updater.UpdateError, "no installed engine matches"):
            lib.find_entry(listing, "0.9.9")

    def test_a_path_is_not_a_folder_name(self):
        for bad in ("../x", "a/b", ".rejected", ""):
            with self.assertRaises(updater.UpdateError):
                lib.delete_engine(self.ctx(), "qwen", bad)
            with self.assertRaises(updater.UpdateError):
                lib.use_engine(self.ctx(), "qwen", bad)

    def test_rollback_and_use_share_one_truth(self):
        updater.rollback_engine(self.ctx())
        self.assertTrue(self.qwen()["0.1.45"]["active"])
        lib.use_engine(self.ctx(), "qwen", folder_of("0.1.46"))
        out = updater.rollback_engine(self.ctx())
        self.assertEqual((out["status"], out["to"]), ("rolled-back", "0.1.45"), "use remembered where it came from")


class Cli(UpdaterCase):
    def setUp(self):
        super().setUp()
        self.install_current("0.1.45")
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        updater.run_update(self.ctx(), control=False)
        for target, name, value in ((lib, "_process_uses", False), (updater, "serving_now", [])):
            patch = mock.patch.object(target, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(updater, "detect_hardware", return_value=dict(self.hardware))
        patch.start()
        self.addCleanup(patch.stop)

    def cli(self, *args):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        from strata_cli.cli import main
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    def common(self):
        return ["--components-dir", str(self.components)]

    def test_engines_use_remove_from_the_terminal(self):
        code, text, _ = self.cli("components", "engines", *self.common())
        self.assertEqual(code, 0, text)
        self.assertIn("0.1.46", text)
        self.assertIn("in use", text)
        code, out, _ = self.cli("components", "engines", "--output", "json", *self.common())
        self.assertEqual(json.loads(out)["families"][0]["engines"][0]["version"], "0.1.46")
        code, text, err = self.cli("components", "use", "0.1.45", *self.common())
        self.assertEqual(code, 0, text + err)
        self.assertIn("Engine 0.1.45 is now in use (was 0.1.46)", text)
        self.assertEqual(self.resolved(), "0.1.45")
        code, text, err = self.cli("components", "remove", "0.1.45", *self.common())
        self.assertEqual(code, 2, "the engine in use is not deleted")
        self.assertIn("engine in use", err)
        code, text, err = self.cli("components", "use", "newest", *self.common())
        self.assertEqual(code, 0, text + err)
        self.assertIn("follows updates again", text)
        code, text, err = self.cli("components", "remove", "0.1.45", *self.common())
        self.assertEqual(code, 0, text + err)
        self.assertIn("Engine 0.1.45 deleted", text)
        self.assertEqual(self.engine_folders(), [folder_of("0.1.46")])


@unittest.skipIf(WINDOWS, "the DeepSeek stub is a shell script")
class DeepSeek(UpdaterCase):
    """Two DeepSeek packages beside the Qwen ones. The family has its own folder (engine-dsv41/) and its own choice (.chosen)."""

    def setUp(self):
        super().setUp()
        self.install_current("0.1.45")
        self.dest = self.components.parent / dsv41_paths.DIR
        for version in ("0.1.55.1", "0.1.56"):
            folder = self.dest / f"{version}-{PLATFORM}-cuda-cuda13"
            folder.mkdir(parents=True)
            exe = folder / dsv41_paths.exe_name()
            exe.write_text(f"#!/bin/sh\necho 'strata-dsv41 {version}'\n")
            exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        self.config = self.root / "ds-run.json"
        self.config.write_text(json.dumps({"exe": str(self.dest / f"0.1.56-{PLATFORM}-cuda-cuda13" / dsv41_paths.exe_name()), "args": [], "lib_dirs": []}))
        for target, name, value in ((dsv41_paths, "saved_configs", [self.config]), (lib, "_process_uses", False)):
            patch = mock.patch.object(target, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

    def exe_in_config(self):
        return Path(json.loads(self.config.read_text())["exe"]).parent.name.split("-")[0]

    def deepseek(self):
        return {e["version"]: e for e in lib.rows_of(lib.list_engines(self.ctx()), lib.DEEPSEEK)}

    def test_both_families_are_listed_each_with_its_engine_in_use(self):
        listing = lib.list_engines(self.ctx())
        self.assertEqual([f["family"] for f in listing["families"]], ["qwen", "deepseek"])
        rows = self.deepseek()
        self.assertEqual(sorted(rows), ["0.1.55.1", "0.1.56"])
        self.assertTrue(rows["0.1.56"]["active"] and rows["0.1.56"]["newest"] and not rows["0.1.55.1"]["active"])
        self.assertEqual(rows["0.1.56"]["delete_blocked"], "This is the engine in use. Switch to another one first.")
        self.assertTrue(lib.rows_of(listing, "qwen")[0]["active"], "the Qwen engine in use is its own")

    def qwen_section(self):
        return json.loads(json.dumps(updater.State(self.components).section("engine")))

    def test_switching_deepseek_does_not_touch_the_qwen_engine_and_the_other_way_round(self):
        qwen_before = self.qwen_section()
        out = lib.use_engine(self.ctx(), "deepseek", f"0.1.55.1-{PLATFORM}-cuda-cuda13", str(self.dest))
        self.assertEqual((out["status"], out["from"], out["to"], out["profiles_repointed"]), ("switched", "0.1.56", "0.1.55.1", 1))
        self.assertEqual(self.exe_in_config(), "0.1.55.1")
        self.assertEqual((self.dest / dsv41_paths.PIN).read_text().strip(), f"0.1.55.1-{PLATFORM}-cuda-cuda13")
        self.assertEqual(dsv41_engine.find(self.dest).parent.name.split("-")[0], "0.1.55.1", "a launch from the Terminal follows the choice")
        self.assertEqual(self.resolved(), "0.1.45", "the Qwen engine is as it was")
        self.assertEqual(self.qwen_section(), qwen_before, "no Qwen state was written")
        # Qwen: switch and delete leave the DeepSeek folder and choice alone
        before = sorted(p.name for p in self.dest.iterdir())
        self.install_current("0.1.44")
        lib.use_engine(self.ctx(), "qwen", folder_of("0.1.44"))
        lib.delete_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        self.assertEqual(sorted(p.name for p in self.dest.iterdir()), before)
        self.assertEqual(self.exe_in_config(), "0.1.55.1")
        self.assertEqual((self.dest / dsv41_paths.PIN).read_text().strip(), f"0.1.55.1-{PLATFORM}-cuda-cuda13")

    def test_back_to_the_newest_drops_the_choice(self):
        lib.use_engine(self.ctx(), "deepseek", f"0.1.55.1-{PLATFORM}-cuda-cuda13", str(self.dest))
        out = lib.use_engine(self.ctx(), "deepseek", f"0.1.56-{PLATFORM}-cuda-cuda13", str(self.dest))
        self.assertTrue(out["automatic"])
        self.assertFalse((self.dest / dsv41_paths.PIN).exists())
        self.assertEqual(self.exe_in_config(), "0.1.56")

    def test_a_launch_upgrade_does_not_replace_a_chosen_package(self):
        lib.use_engine(self.ctx(), "deepseek", f"0.1.55.1-{PLATFORM}-cuda-cuda13", str(self.dest))
        with mock.patch("strata_control.updater.open_feed", side_effect=AssertionError("must not look at the feed")):
            self.assertIsNone(dsv41_engine.upgrade(self.dest, "http://127.0.0.1:1/none"))

    def test_deepseek_delete_blocks_the_one_in_use_and_frees_an_older_one(self):
        with self.assertRaisesRegex(updater.UpdateError, "engine in use"):
            lib.delete_engine(self.ctx(), "deepseek", f"0.1.56-{PLATFORM}-cuda-cuda13", str(self.dest))
        out = lib.delete_engine(self.ctx(), "deepseek", f"0.1.55.1-{PLATFORM}-cuda-cuda13", str(self.dest))
        self.assertEqual(out["status"], "deleted")
        self.assertEqual(sorted(self.deepseek()), ["0.1.56"])

    def test_a_stale_choice_is_ignored(self):
        (self.dest / dsv41_paths.PIN).write_text("0.0.1-gone\n")
        self.assertEqual(dsv41_engine.find(self.dest).parent.name.split("-")[0], "0.1.56")
        self.assertTrue(self.deepseek()["0.1.56"]["active"])

    def test_a_running_model_stops_a_deepseek_switch(self):
        with self.assertRaises(updater.Busy):
            lib.use_engine(self.ctx(serving=lambda: [{"profile": "app-1"}]), "deepseek", f"0.1.55.1-{PLATFORM}-cuda-cuda13", str(self.dest))
        self.assertEqual(self.exe_in_config(), "0.1.56")


if __name__ == "__main__":
    unittest.main()
