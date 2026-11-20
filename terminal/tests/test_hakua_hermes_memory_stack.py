"""Focused tests for Hakua↔Hermes memory stack (no GPU).

Covers persona sync, Ebbinghaus remember/recall, graph node, Obsidian export,
sleep consolidation, and harness persona_sha256 snippet generation.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from hakua_hermes_memory_stack import (  # noqa: E402
    HakuaHermesMemoryStack,
    StackPaths,
    forgetting_retention_probe,
)


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class TestHakuaHermesMemoryStack(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hakua-hermes-"))
        self.hermes = self.tmp / "hermes"
        self.hakua = self.tmp / "hakua"
        self.vault = self.tmp / "vault"
        self.hermes.mkdir()
        (self.hermes / "memories").mkdir()
        self.hakua.mkdir()
        self.vault.mkdir()
        self.soul_hash = _write(self.hermes / "SOUL.md", "# SOUL\nYou are Hermes Agent under Hakua.\n")
        self.mem_hash = _write(
            self.hermes / "memories" / "MEMORY.md",
            "# Memory\n\n§ [core] 2026-10-29: Prefer Japanese status\nKeep replies concise.\n",
        )
        self.env_patch = {
            "HERMES_HOME": str(self.hermes),
            "HAKUA_MEMORY_HOME": str(self.hakua),
            "HAKUA_HERMES_PRINCIPAL": "hermes",
            "HAKUA_MEMORY_SOT": "hakua",
            "OBSIDIAN_VAULT_PATH": str(self.vault),
        }
        self._old = {k: os.environ.get(k) for k in self.env_patch}
        os.environ.update(self.env_patch)
        self.stack: HakuaHermesMemoryStack | None = None

    def tearDown(self) -> None:
        if self.stack is not None:
            self.stack.close()
            self.stack = None
        for key, value in self._old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_01_forgetting_curve_math(self) -> None:
        r0 = forgetting_retention_probe(0.0, 3.0)
        r3 = forgetting_retention_probe(3.0, 3.0)
        r30 = forgetting_retention_probe(30.0, 3.0)
        self.assertEqual(r0, 1.0)
        self.assertAlmostEqual(r3, 0.367879, places=4)
        self.assertLess(r30, r3)

    def test_02_sync_persona_to_hakua_profile(self) -> None:
        self.stack = HakuaHermesMemoryStack()
        result = self.stack.sync_persona(direction="to_hakua")
        self.assertEqual(result["direction"], "to_hakua")
        self.assertTrue((self.hakua / "profiles" / "hermes" / "SOUL.md").is_file())
        self.assertTrue((self.hakua / "profiles" / "hermes" / "MEMORY.md").is_file())
        approvals = result["persona_sha256"]
        self.assertEqual(approvals["SOUL.md"], self.soul_hash)
        self.assertEqual(approvals["MEMORY.md"], self.mem_hash)
        soul = self.hakua / "profiles" / "hermes" / "SOUL.md"
        self.assertFalse(soul.is_symlink())
        self.assertEqual(soul.stat().st_nlink, 1)

    def test_03_remember_recall_graph_and_sleep(self) -> None:
        self.stack = HakuaHermesMemoryStack()
        self.stack.sync_persona(direction="to_hakua")
        saved = self.stack.remember(
            "User prefers Japanese なんj status updates for Hakua memory work.",
            importance=0.9,
            tags=["user-preference", "communication"],
        )
        self.assertTrue(saved.get("saved"))
        self.assertIn("ebbinghaus", saved)
        self.assertIn("graph", saved)

        recalled = self.stack.recall("Japanese status", top_k=5)
        self.assertGreaterEqual(recalled["count"], 1)
        blob = json.dumps(recalled, ensure_ascii=False)
        self.assertIn("Japanese", blob)

        sleep = self.stack.sleep(prune=True, reindex=True, sync_obsidian=True)
        self.assertEqual(sleep["mode"], "hakua_sleep")
        self.assertIn("ebbinghaus_sleep", sleep)
        self.assertIn("obsidian", sleep)
        self.assertTrue((self.vault / "Hakua" / "SOUL.md").is_file())
        self.assertTrue((self.vault / "Hakua" / "MEMORY.md").is_file())

        snippet = self.stack.harness_user_snippet(key_env="HAKUA_HERMES_KEY")
        self.assertEqual(snippet["principal"], "hermes")
        self.assertEqual(snippet["key_env"], "HAKUA_HERMES_KEY")
        self.assertIn("persona_sha256", snippet)
        self.assertRegex(snippet["persona_sha256"]["SOUL.md"], r"^[0-9a-f]{64}$")

    def test_04_stack_paths_resolve_from_env(self) -> None:
        paths = StackPaths.resolve()
        self.assertEqual(paths.hermes_home, self.hermes.resolve())
        self.assertEqual(paths.hakua_home, self.hakua.resolve())
        self.assertEqual(paths.principal, "hermes")
        self.assertEqual(paths.obsidian_vault, self.vault.resolve())


if __name__ == "__main__":
    unittest.main()
