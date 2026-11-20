"""Unit and integration test for Hermes Agent Strata plugin & provider profile.

Verifies:
1. ProviderProfile registration (strata profile).
2. hermes-strata tool registration and functionality (status, memory_sync, inspect_gguf, infer).
3. Hakua SOUL.md and MEMORY.md parsing and context formatting.
4. GGUF header inspection logic.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

# Setup MILSPEC logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_hermes_strata")

HERMES_REPO = Path(os.environ.get('STRATA_TEST_HERMES_REPO', '.')).resolve()
if os.environ.get('STRATA_TEST_HERMES_REPO') and str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))


def _load_plugin_module(plugin_dir: Path, module_name: str = "hermes_strata"):
    init_file = plugin_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_file,
        submodule_search_locations=[str(plugin_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {init_file}")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = module_name
    mod.__path__ = [str(plugin_dir)]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


class MockContext:
    """Mock Hermes plugin registration context."""

    def __init__(self) -> None:
        self.tools: dict[str, dict] = {}

    def register_tool(self, name: str, toolset: str, schema: dict, handler: Any, check_fn: Any, description: str) -> None:
        self.tools[name] = {
            "name": name,
            "toolset": toolset,
            "schema": schema,
            "handler": handler,
            "check_fn": check_fn,
            "description": description,
        }


@unittest.skipUnless(os.environ.get('STRATA_TEST_HERMES_REPO'), 'external Hermes plugin checkout required: set STRATA_TEST_HERMES_REPO')
class TestHermesStrataPlugin(unittest.TestCase):
    """Test suite for Strata Hermes plugin."""

    @classmethod
    def setUpClass(cls) -> None:
        plugin_path = HERMES_REPO / "plugins" / "hermes-strata"
        cls.plugin_mod = _load_plugin_module(plugin_path, "hermes_strata")

    def test_01_provider_profile(self) -> None:
        """Test Strata model provider profile."""
        try:
            from providers import get_provider_profile
            strata_p = get_provider_profile("strata")
            self.assertIsNotNone(strata_p)
            self.assertEqual(strata_p.name, "strata")
            self.assertEqual(strata_p.base_url, "http://127.0.0.1:8080/v1")
            extra, top = strata_p.build_api_kwargs_extras(model="deepseek-v4.1-flash")
            self.assertEqual(extra.get("thinking"), {"type": "enabled"})
            logger.info("Test 01 (Provider profile): PASSED")
        except Exception as exc:
            logger.error("Test 01 failed: %s", exc)
            raise

    def test_02_tool_registration(self) -> None:
        """Test hermes-strata tool registration."""
        try:
            ctx = MockContext()
            self.plugin_mod.register(ctx)

            self.assertIn("strata_status", ctx.tools)
            self.assertIn("strata_infer", ctx.tools)
            self.assertIn("strata_memory_sync", ctx.tools)
            self.assertIn("strata_inspect_gguf", ctx.tools)
            logger.info("Test 02 (Tool registration): PASSED (registered 4 tools)")
        except Exception as exc:
            logger.error("Test 02 failed: %s", exc)
            raise

    def test_03_memory_sync_and_soul(self) -> None:
        """Test Hakua SOUL.md reading and MEMORY.md stats & context formatting."""
        core = sys.modules["hermes_strata.core"]

        soul_path = core.resolve_soul_path()
        logger.info("Resolved SOUL.md path: %s", soul_path)
        self.assertIsNotNone(soul_path, "SOUL.md should exist")

        directives = core.read_soul_directives()
        self.assertGreater(len(directives), 50, "SOUL.md directives should have content")
        self.assertIn("Hermes Agent", directives)

        memories = core.read_memory_entries(max_entries=5)
        logger.info("Found %d recent memory entries", len(memories))

        context = core.build_system_context(include_soul=True, include_memory=True, max_memories=3)
        self.assertIn("CORE IDENTITY & DIRECTIVES (SOUL.md)", context)
        logger.info("Test 03 (Memory sync & SOUL): PASSED")

    def test_04_gguf_inspection(self) -> None:
        """Test arbitrary GGUF header parser."""
        core = sys.modules["hermes_strata.core"]

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tf:
            # GGUF header: magic="GGUF" (4s), version=3 (I), tensor_count=128 (Q), metadata_kv_count=42 (Q)
            header = struct.pack("<4sIQQ", b"GGUF", 3, 128, 42)
            tf.write(header)
            # pad with 1MB dummy data
            tf.write(b"\x00" * (1024 * 1024))
            temp_path = tf.name

        try:
            res = core.inspect_gguf_file(temp_path)
            self.assertTrue(res.get("valid"))
            self.assertEqual(res.get("gguf_version"), 3)
            self.assertEqual(res.get("tensor_count"), 128)
            self.assertEqual(res.get("metadata_kv_count"), 42)
            self.assertIn("compatibility", res)
            logger.info("Test 04 (GGUF inspection): PASSED")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_05_status_handler(self) -> None:
        """Test status tool handler output formatting."""
        res_raw = self.plugin_mod._status_handler()
        res = json.loads(res_raw)
        self.assertIn("alive", res)
        self.assertIn("hardware", res)
        self.assertEqual(res["hardware"]["gpu"], "NVIDIA GeForce RTX 5060 Ti 16GB (sm_120 Blackwell)")
        self.assertEqual(res["hardware"]["system_ram"], "32GB (Zero-VRAM CPU Drafter pool)")
        logger.info("Test 05 (Status probe): PASSED (alive=%s)", res.get("alive"))


if __name__ == "__main__":
    unittest.main()
