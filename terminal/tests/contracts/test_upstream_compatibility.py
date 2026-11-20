"""tests/contracts/test_upstream_compatibility.py - Verification of T01 upstream compatibility & fork contract preservation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_INTAKE = REPO_ROOT / "_docs" / "upstream-intake"

def test_fork_feature_inventory() -> None:
    contracts_file = DOCS_INTAKE / "fork-contracts.json"
    assert contracts_file.exists(), "fork-contracts.json must exist"
    data = json.loads(contracts_file.read_text(encoding="utf-8"))
    contracts = data.get("contracts", [])
    expected_ids = {f"F{i:02d}" for i in range(1, 12)}
    found_ids = {c["id"] for c in contracts}
    assert expected_ids == found_ids, f"Missing fork contracts: {expected_ids - found_ids}"
    for c in contracts:
        assert c["invariants"], f"Contract {c['id']} must define invariants"
        assert c["verification"], f"Contract {c['id']} must define verification criteria"

def test_changes_csv_classification() -> None:
    csv_file = DOCS_INTAKE / "changes.csv"
    assert csv_file.exists(), "changes.csv must exist"
    with open(csv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) >= 400, "Should classify all upstream changes from U0 to U1"
    valid_classes = {"APPLIED_UPSTREAM", "ADAPTED_EQUIVALENT", "ALREADY_EQUIVALENT", "NOT_APPLICABLE", "BLOCKED"}
    for r in rows:
        assert r["classification"] in valid_classes, f"Invalid classification in row: {r}"

def test_source_lock_milestones() -> None:
    lock_file = DOCS_INTAKE / "source-lock.json"
    assert lock_file.exists(), "source-lock.json must exist"
    data = json.loads(lock_file.read_text(encoding="utf-8"))
    m = data["milestones"]
    assert m["U0_common_ancestor"] == "011c9e600a93bbc9fd451c563501b5831c32bbca"
    assert m["F0_fork_base"] == "ddc8f9e19cafb981ed91caaab0992ca814e6eca9"
    assert m["U1_upstream_target"] == "7dd5cf18e68c639a8773e096c5c79f971ea2a9f5"
    assert m["C1_upstream_fix"] == "c487c875d05075113965bc61d822b27179f70639"

def test_upstream_off_contract() -> None:
    """Verifies that default config and runtime schema do not inject invalid upstream flags."""
    # When all fork features (spin8, turboquant) are off/default, base arguments remain clean
    from strata_control.user_args import apply
    cfg = {"args": ["--context", "32k", "--port", "18501"]}
    applied = apply(cfg)
    assert applied == []
    assert cfg["args"] == ["--context", "32k", "--port", "18501"]
