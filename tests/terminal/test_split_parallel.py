"""The all-cards split engine and the parallel setting: no collision, and the number means the split's lanes.

The split's run config (strata_control/split_engine.split_config) always carries `--batch N`.  The launcher used to write a launch-profile override
`parallel=<card fit>` beside it even when nobody asked, and `strata serve --profile` (strata_cli/cli.py _effective) refuses a parallel override next to
--batch: every 2-card NVIDIA start with the split on Auto stopped with "parallel override conflicts with --batch/--slots in run config args".
Now: the launcher writes no parallel override when the run config holds the slots; `--parallel N` is the split's lanes; the app's Concurrent requests
(a profile override on the saved run config) is accepted for a split config and sets --batch in the engine's arguments.  Nothing is started.
"""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from strata_cli import cli
from strata_cli.launch import settings as S
from strata_cli.launch.runner import Runner
from strata_control import profiles, split_engine as SE
from tests.terminal.launch_support import HW_12, Home, catalog

A6000 = {"name": "NVIDIA RTX A6000", "vram_gb": 48.0, "usable": True, "vendor": "nvidia"}
HW_2XA6000 = {**HW_12, "ram_gb": 125.0, "disk_free_gb": 900.0, "engine_vendor": "nvidia", "gpus": [{**A6000, "index": i} for i in range(2)]}
R9700 = {"name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "usable": True, "vendor": "amd", "arch": "gfx1201"}
HW_4XR9700 = {**HW_12, "ram_gb": 251.0, "disk_free_gb": 900.0, "engine_vendor": "amd", "gpus": [{**R9700, "index": i} for i in range(4)]}

SPLIT_ARGS = ["--native", "/m/x-00001.gguf", "--mtp", "/m/mtp/rt", "--draft-vocab", "/e/draft_vocab_en.bin", "--max-context", "131072",
              "--spec", "6", "--lookup", "15", "--conversation-cache-mib", "4096", "--batch", "6"]
LAYER_ARGS = ["--pack", "/m/packs/x", "--native", "/m/x-00001.gguf", "--spec", "4", "--mtp", "/m/mtp/rt"]


def split_cfg(lanes: int = 6, **more) -> dict:
    args = list(SPLIT_ARGS)
    args[args.index("--batch") + 1] = str(lanes)
    return {"exe": "/e/strata-split-2-sm120", "args": args, "tokenizer": "/m/tokenizer", "model_name": "m", "port": 8080,
            "split_engine_note": "Using the all-cards split engine on 2 cards.", "drafting": "mtp", **more}


def layer_cfg(**more) -> dict:
    return {"exe": "/e/strata", "args": list(LAYER_ARGS), "tokenizer": "/m/tokenizer", "model_name": "m", "port": 8080, **more}


class Base(unittest.TestCase):
    hw = HW_2XA6000

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def profile_for(self, cfg: dict, *, parallel=None, hw=None, four_card=None) -> dict:
        """What Runner.ensure_profile saves as the launch profile's overrides for this written run config."""
        cfgfile = Path(self.home.base) / "cfg.json"
        cfgfile.write_text(json.dumps(cfg), encoding="utf-8")
        s = S.LaunchSettings(family="qwen", model="UD-Q4_K_XL", context=131072, tune="skip", parallel=parallel)
        r = Runner(s, catalog=catalog(hw or self.hw))
        if four_card is not None:
            r.four_card_parallel = four_card
        r.ensure_profile(cfgfile)
        prof, _rev = profiles.load(r.profile)
        return dict(prof.get("overrides") or {})


class TerminalPath(Base):
    def test_the_split_run_config_gets_no_parallel_override_by_default(self):
        """The 2-card Auto start: before, the card-fit default (4) was written beside the split's --batch and the server refused to start."""
        ov = self.profile_for(split_cfg())
        self.assertNotIn("parallel", ov)
        values, _o = cli._effective(split_cfg(), {"overrides": ov}, None)      # the check that refused the start
        self.assertEqual(values["parallel"], None)

    def test_a_layer_split_run_config_keeps_the_fitted_default(self):
        ov = self.profile_for(layer_cfg())
        self.assertEqual(ov.get("parallel"), 4, "48 GB card: the fitted default is unchanged on the layer split")
        cli._effective(layer_cfg(), {"overrides": ov}, None)

    def test_parallel_n_on_the_split_is_not_a_profile_override_either(self):
        ov = self.profile_for(split_cfg(lanes=3), parallel=3)
        self.assertNotIn("parallel", ov)
        cli._effective(split_cfg(lanes=3), {"overrides": ov}, None)

    def test_the_amd_four_card_default_does_not_collide_with_the_split(self):
        """AMD gfx1201 x4 starts the split by itself too; amd_four_card.apply put parallel=8 in the run config and the launcher mirrored it into the profile."""
        cfg = split_cfg(lanes=8, parallel=8, exe="/e/strata-split-4-amd")
        ov = self.profile_for(cfg, hw=HW_4XR9700, four_card=8)
        self.assertNotIn("parallel", ov)
        values, origins = cli._effective(cfg, {"overrides": ov}, None)
        self.assertEqual((values["parallel"], origins["parallel"]), (8, "run_config"))

    def test_the_conflict_check_still_stops_a_plain_batch_config(self):
        """Only the split's run config is exempt: another config with its own --batch and a parallel override is still refused."""
        cfg = layer_cfg(args=LAYER_ARGS + ["--batch", "6"])
        with self.assertRaises(profiles.ProfileError):
            cli._effective(cfg, {"overrides": {"parallel": 4}}, None)


class LanesFromTheFlag(Base):
    def run_apply(self, parallel, use_split):
        """Runner.apply_split_engine with the rule stubbed: what the rule sees in the config, and what is left of it afterwards."""
        seen = {}

        def fake(cfg, **_kw):
            seen["parallel"] = cfg.get("parallel")
            if use_split:
                new = split_cfg(lanes=int(cfg.get("parallel") or 6), parallel=cfg.get("parallel"))
                cfg.clear()
                cfg.update(new)
            return SE.Decision(use_split, ["Using the all-cards split engine on 2 cards."] if use_split else ["no"], 2, quiet=not use_split)

        s = S.LaunchSettings(family="qwen", model="UD-Q4_K_XL", context=131072, tune="skip", parallel=parallel)
        r = Runner(s, catalog=catalog(self.hw))
        cfg = layer_cfg()
        with mock.patch.object(SE, "apply_to_config", fake), contextlib.redirect_stdout(io.StringIO()):
            r.apply_split_engine(cfg)
        return seen["parallel"], cfg

    def test_parallel_n_reaches_the_rule_and_becomes_the_lane_count(self):
        seen, cfg = self.run_apply(3, True)
        self.assertEqual(seen, 3)
        self.assertEqual(cfg["args"][cfg["args"].index("--batch") + 1], "3")

    def test_without_the_flag_the_rule_sees_nothing_and_the_default_lanes_stay(self):
        seen, cfg = self.run_apply(None, True)
        self.assertIsNone(seen)
        self.assertEqual(cfg["args"][cfg["args"].index("--batch") + 1], "6")

    def test_a_refusal_leaves_the_layer_split_config_without_the_extra_key(self):
        seen, cfg = self.run_apply(3, False)
        self.assertEqual(seen, 3)
        self.assertNotIn("parallel", cfg)
        self.assertEqual(cfg["args"], LAYER_ARGS)


class AppPath(Base):
    """The app's Settings 'Concurrent requests = N' is a profile override on the saved run config (strata_control/profiles); `strata serve --profile` reads it."""

    def engine_argv(self, cfg: dict, overrides: dict) -> list[str]:
        from serve.server import engine_args
        values, _o = cli._effective(cfg, {"overrides": overrides}, None)
        eff = {**cfg, "args": list(cfg["args"])}
        if values["parallel"] is not None:
            eff["parallel"] = values["parallel"]
        with contextlib.redirect_stdout(io.StringIO()):
            return engine_args(eff)

    def test_concurrent_requests_on_a_split_config_sets_its_lanes(self):
        argv = self.engine_argv(split_cfg(lanes=6), {"host": "127.0.0.1", "port": 8080, "parallel": 3})
        self.assertEqual(argv.count("--batch"), 1)
        self.assertEqual(argv[argv.index("--batch") + 1], "3")

    def test_no_override_keeps_the_lanes_the_launcher_wrote(self):
        argv = self.engine_argv(split_cfg(lanes=5), {"host": "127.0.0.1", "port": 8080})
        self.assertEqual(argv[argv.index("--batch") + 1], "5")

    def test_the_layer_split_still_takes_its_batch_from_parallel(self):
        argv = self.engine_argv(layer_cfg(), {"host": "127.0.0.1", "port": 8080, "parallel": 3})
        self.assertEqual(argv[argv.index("--batch") + 1], "3")


if __name__ == "__main__":
    unittest.main()
