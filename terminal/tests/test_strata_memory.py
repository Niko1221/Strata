"""tests/test_strata_memory.py - Test suite for Strata Unified Long-Term Memory.

Tests:
1. Status retrieval of SOUL.md, MEMORY.md, Hermes SQLite, and hakua CompositeMemory.
2. Reading SOUL.md and MEMORY.md.
3. Adding and recalling memory items across subsystems.
4. CLI execution.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from strata_memory_mcp import StrataUnifiedMemory
import strata_memory_mcp as memory_module


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Exercise the actual file/SQLite fallback without reading or writing a personal profile."""
    hermes = tmp_path / 'hermes'
    (hermes / 'memories').mkdir(parents=True)
    (hermes / 'SOUL.md').write_text('# Fixture persona\nSynthetic test content.\n', encoding='utf-8')
    (hermes / 'memories/MEMORY.md').write_text('# Fixture memory\nOnly synthetic records.\n', encoding='utf-8')
    with sqlite3.connect(hermes / 'ebbinghaus_memory.db') as database:
        database.execute('''CREATE TABLE memories (
            memory_id INTEGER PRIMARY KEY, content TEXT, encoded TEXT, cues TEXT, tags TEXT,
            salience REAL, valence REAL, strength REAL, rehearsal_count INTEGER,
            retrieval_count INTEGER, source TEXT, session_id TEXT, created_at REAL,
            updated_at REAL, state TEXT)''')
    database.close()
    hakua = tmp_path / 'hakua'
    monkeypatch.setattr(memory_module, 'HERMES_DIR', hermes)
    monkeypatch.setattr(memory_module, 'HAKUA_MEM_DIR', hakua)
    monkeypatch.setattr(memory_module, 'HakuaHermesMemoryStack', None)
    # This suite tests the documented optional-dependency fallback. The stack
    # integration has separate consent/profile tests and must not be auto-loaded.
    monkeypatch.setitem(sys.modules, 'hakua_memory.composite', None)
    for key, value in {'HOME': tmp_path, 'USERPROFILE': tmp_path, 'HERMES_HOME': hermes, 'HAKUA_MEMORY_HOME': hakua}.items():
        monkeypatch.setenv(key, str(value))


def test_strata_memory_status() -> None:
    mem = StrataUnifiedMemory()
    status = mem.status()
    assert status["status"] == "healthy"
    assert "soul" in status
    assert "memory_md" in status
    assert "hermes_ebbinghaus" in status
    assert "hakua_composite" in status
    assert status["soul"]["exists"] is True


def test_strata_memory_read_soul() -> None:
    mem = StrataUnifiedMemory()
    soul_data = mem.read_soul()
    assert soul_data["exists"] is True
    assert "content" in soul_data
    assert len(soul_data["content"]) > 0
    assert "sha256" in soul_data


def test_strata_memory_read_memory_md() -> None:
    mem = StrataUnifiedMemory()
    mem_data = mem.read_memory_md(limit=10)
    assert mem_data["exists"] is True
    assert isinstance(mem_data["sections"], list)
    if mem_data["sections"]:
        first = mem_data["sections"][0]
        assert "header" in first
        assert "body" in first


def test_strata_memory_remember_and_recall() -> None:
    mem = StrataUnifiedMemory()
    test_tag = "pytest-test-marker"
    test_content = "Antigravity and Strata pair-programming verification checkpoint."

    save_res = mem.remember(content=test_content, importance=0.85, tags=[test_tag])
    assert save_res["saved"] is True
    assert isinstance(save_res['hermes_memory_id'], int)

    recall_res = mem.recall(query=test_tag, top_k=5)
    assert recall_res["count"] >= 1
    found = any(test_tag in str(r.get("metadata", {})) or test_tag in r.get("content", "") for r in recall_res["results"])
    assert found is True


def test_remember_reports_failure_when_no_storage_backend_is_available() -> None:
    mem = StrataUnifiedMemory()
    mem.get_hermes_db_path().unlink()
    result = mem.remember(content='Synthetic unsaved record', tags=['fixture'])
    assert result['saved'] is False
    assert result['hermes_memory_id'] is None
    assert mem.recall('Synthetic unsaved record')['count'] == 0


def test_strata_memory_cli_status() -> None:
    # Match the in-process missing-optional-dependencies fixture while running
    # the real __main__, parser and JSON CLI in a separate interpreter.
    bootstrap = "import runpy,sys; sys.modules['hakua_hermes_memory_stack']=None; sys.modules['hakua_memory.composite']=None; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    cmd = [sys.executable, '-c', bootstrap, str(REPO_ROOT / "tools" / "strata_memory_mcp.py"), "--cli", "status"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    out_json = json.loads(proc.stdout)
    assert out_json["status"] == "healthy"
    assert out_json["soul"]["exists"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
