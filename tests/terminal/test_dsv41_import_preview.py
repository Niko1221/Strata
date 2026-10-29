"""The app's "I already have the model" preview recognises an official DeepSeek-V4.1-Flash set (IST-DAS 1.75 / 2.00 / 2.25 bpw), under the
official file names or the first release's, from the pinned names and sizes. Qwen previews are unchanged (tests/terminal/test_models*.py)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import dsv41_sets, models

T = dsv41_sets.tables()


def files_of(model, legacy=False, engram="official", drop=(), wrong=()):
    entry = T["sets"][model]
    pinned = dict(entry["legacy"] if legacy else entry["shards"])
    out = {n: (s + 1 if n in wrong else s) for n, s in pinned.items() if n not in drop}
    if engram:
        name = next(n for n in T["engram"] if (n.startswith("engram") == (engram == "legacy")))
        out[name] = T["engram"][name]
    return out


class DeepSeekPreview(unittest.TestCase):
    def preview(self, listing, picked=None):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            target = folder
            if picked:
                (folder / picked).write_bytes(b"GGUF")
                target = folder / picked
            with mock.patch.object(models, "_dsv41_files", return_value=listing):
                return models.plan_import(str(target)), folder

    def test_the_tables_come_from_setup_and_cover_the_three_official_sizes(self):
        self.assertEqual(sorted(T["sets"]), ["IST-DAS-1.75bpw", "IST-DAS-2.00bpw", "IST-DAS-2.25bpw"])
        self.assertEqual(list(T["engram"]), ["DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"])

    def test_each_official_set_is_recognised_with_its_size_name(self):
        for model in T["sets"]:
            for legacy in (False, True):
                if legacy and not T["sets"][model]["legacy"]:
                    continue                                    # 1.75 has no first-release names
                plan, folder = self.preview(files_of(model, legacy=legacy, engram="legacy" if legacy else "official"))
                self.assertEqual(plan["file_set_status"], "plausible", (model, legacy, plan["issues"]))
                self.assertEqual((plan["candidate_selection"]["family"], plan["candidate_selection"]["model"]), ("deepseek41", model))
                self.assertEqual(plan["file_count"], 6)                 # the set's five files and the Engram, the sixth shard
                self.assertEqual(plan["candidate_selection"]["setup_args_hint"][-2:], ["--gguf-dir", str(folder)])
                self.assertEqual(plan["issues"], [])
                self.assertTrue(plan["messages"])
                self.assertEqual(bool(plan["already_set_up"] is None or "config" in plan["already_set_up"]), True)

    def test_without_the_engram_it_is_still_the_set_and_says_what_setup_fetches(self):
        plan, _ = self.preview(files_of("IST-DAS-2.00bpw", engram=None))
        self.assertEqual(plan["file_set_status"], "plausible")
        self.assertEqual(plan["file_count"], 5)
        self.assertTrue(any("n-gram table" in m and "downloads it" in m for m in plan["messages"]))

    def test_a_missing_or_short_shard_names_the_file(self):
        gone = sorted(T["sets"]["IST-DAS-2.00bpw"]["shards"])[2]
        plan, _ = self.preview(files_of("IST-DAS-2.00bpw", drop=(gone,)))
        self.assertEqual(plan["file_set_status"], "invalid_or_incomplete")
        self.assertIn("incomplete_or_invalid_shard_set", plan["issues"])
        self.assertEqual(plan["missing_shards"], [gone])
        self.assertIsNone(plan["candidate_selection"])
        short = sorted(T["sets"]["IST-DAS-2.00bpw"]["shards"])[1]
        plan, _ = self.preview(files_of("IST-DAS-2.00bpw", wrong=(short,)))
        self.assertEqual(plan["missing_shards"], [short])

    def test_two_sets_in_one_folder_ask_for_a_file_and_a_picked_file_decides(self):
        listing = {**files_of("IST-DAS-2.00bpw"), **files_of("IST-DAS-2.25bpw")}
        plan, _ = self.preview(listing)
        self.assertIn("multiple_model_groups", plan["issues"])
        self.assertEqual(sorted(h["model"] for h in plan["folder_holds"]), ["IST-DAS-2.00bpw", "IST-DAS-2.25bpw"])
        self.assertIsNone(plan["candidate_selection"])
        one = sorted(T["sets"]["IST-DAS-2.25bpw"]["shards"])[0]
        plan, _ = self.preview(listing, picked=one)
        self.assertEqual(plan["candidate_selection"]["model"], "IST-DAS-2.25bpw")
        self.assertEqual(plan["file_set_status"], "plausible")

    def test_one_shard_picked_is_the_whole_set_not_an_unsupported_model(self):
        first = sorted(T["sets"]["IST-DAS-2.00bpw"]["shards"])[0]
        plan, _ = self.preview(files_of("IST-DAS-2.00bpw"), picked=first)
        self.assertEqual(plan["candidate_selection"]["model"], "IST-DAS-2.00bpw")
        self.assertNotIn("unsupported_or_missing_architecture", plan["issues"])
        self.assertFalse(any("not supported" in m for m in plan["messages"]))

    def test_other_files_are_not_claimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(models, "_dsv41_files", return_value={"Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf": 5, "x.gguf": 3}):
                self.assertIsNone(models._dsv41_plan(Path(tmp), Path(tmp).stat()))


if __name__ == "__main__":
    unittest.main()
