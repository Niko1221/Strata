"""The gate of CLEAN-1A: in every machine class x model set, the app's plain Load and the app's Models-screen Load start the engine as `strata launch` does,
apart from the allow-list (tests/snapshot/allowlist.py).  Reads the golden files, so it needs no Node and no machine; test_config_snapshot.py keeps the golden files true."""

from __future__ import annotations

import json
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.snapshot import allowlist  # noqa: E402
from tests.snapshot.generate import GOLDEN  # noqa: E402


def flat(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(flat(v, f"{prefix}/{k}"))
        return out
    return {prefix: value}


def private_cache_args(args):
    argv = list(args)
    for flag, value in allowlist.HAKUA_PRIVATE_CACHE.items():
        if argv.count(flag) != 1:
            raise ValueError("Missing/duplicate fork cache policy flag: " + flag)
        at = argv.index(flag)
        if argv[at + 1:at + 2] != [value]:
            raise ValueError("Changed fork cache policy: " + flag)
        del argv[at:at + 2]
    return argv


def comparable(rec: dict, allowed_cfg: set[str]) -> dict:
    argv = list(rec["engine"]["argv"])
    principal = rec["config_file"].get("hakua_harness")
    if principal is not None:
        expected = {"enabled": True, "users": [{"principal": "local", "key_env": "STRATA_HAKUA_LOCAL_KEY",
                    "root": "<ROOT>/host-data/hakua-memory/local"}]}
        if principal != expected:
            raise ValueError("Unexpected fork principal policy")
        argv = private_cache_args(argv)
    cfg = {k: v for k, v in rec["config_file"].items() if k not in allowed_cfg}
    if principal is not None:
        cfg['args'] = private_cache_args(cfg['args'])
    env = {k: v for k, v in rec["engine"]["env_set"].items() if k not in allowed_cfg and k not in allowlist.ENV_BY_DESIGN}
    return {"config": flat(cfg), "argv": argv, "env": env, "fetches": rec.get("fetches")}


class EntryPointsAgree(unittest.TestCase):
    maxDiff = None

    def cells(self):
        for path in sorted(GOLDEN.glob("*/*.json")):
            yield path.parent.name + "/" + path.stem, json.loads(path.read_text(encoding="utf-8"))["entries"]

    def check(self, other: str, allowed: set[str]):
        seen = 0
        for cell, entries in self.cells():
            with self.subTest(cell=cell, entry=other):
                want, got = comparable(entries["launch"], allowed), comparable(entries[other], allowed)
                self.assertEqual(want["argv"], got["argv"], f"{cell}: engine arguments")
                self.assertEqual(want["env"], got["env"], f"{cell}: engine environment")
                self.assertEqual(want["config"], got["config"], f"{cell}: run config")
                self.assertEqual(want["fetches"], got["fetches"], f"{cell}: the drafter downloads asked for")
            seen += 1
        self.assertGreaterEqual(seen, 32)

    def test_the_plain_load_starts_the_engine_as_launch_does(self):
        self.check("app_plain", set(allowlist.BY_DESIGN))

    def test_the_models_screen_load_differs_only_by_the_choice_it_saves(self):
        self.check("app_models", set(allowlist.BY_DESIGN) | set(allowlist.MODELS_SCREEN))

    def test_no_app_entry_writes_request_slots(self):
        for cell, entries in self.cells():
            for entry in ("app_plain", "app_models"):
                self.assertNotIn("parallel", entries[entry]["config_file"], f"{cell}/{entry}")
                self.assertNotIn("--batch-mtp", entries[entry]["engine"]["argv"], f"{cell}/{entry}")

    def test_app_architecture_and_principal_contract_are_explicit(self):
        for cell, entries in self.cells():
            expected = "deepseek41" if cell.endswith("/deepseek") else "qwen4exp"
            for entry in ("app_plain", "app_models"):
                rec = entries[entry]
                self.assertEqual(rec["config_file"].get("arch"), expected, cell)
                self.assertEqual("hakua_harness" in rec["config_file"], expected == "qwen4exp", cell)
                comparable(rec, set(allowlist.BY_DESIGN))

    def test_private_cache_changes_are_rejected_instead_of_ignored(self):
        rec = next(entries["app_plain"] for cell, entries in self.cells() if cell.endswith("/iq3_s"))
        changed = copy.deepcopy(rec)
        at = changed["engine"]["argv"].index("--prompt-cache")
        changed["engine"]["argv"][at + 1] = "1"
        with self.assertRaisesRegex(ValueError, "fork cache policy"):
            comparable(changed, set(allowlist.BY_DESIGN))


if __name__ == "__main__":
    unittest.main()
