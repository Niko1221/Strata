"""Control-only packaging releases must not relabel an unchanged native engine."""
from pathlib import Path
import sys

import pytest
import strata_cli

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
import package_terminal


def source(tmp_path, monkeypatch):
    monkeypatch.setattr(package_terminal, "ROOT", tmp_path)
    monkeypatch.setattr(strata_cli, "__file__", str(tmp_path / "strata_cli" / "__init__.py"))
    (tmp_path / "CMakeLists.txt").write_text("project(strata VERSION 0.4.1 LANGUAGES C CXX)\n")
    return tmp_path


def test_control_version_can_advance_without_native_or_product_version(tmp_path, monkeypatch):
    root = source(tmp_path, monkeypatch)
    (root / "VERSION").write_text("0.4.1\n")
    (root / "CONTROL_VERSION").write_text("0.4.2\n")
    assert package_terminal._source_version() == "0.4.2"
    assert strata_cli.version() == "0.4.2"


def test_archive_version_and_legacy_source_still_work(tmp_path, monkeypatch):
    root = source(tmp_path, monkeypatch)
    assert package_terminal._source_version() == "0.4.1"
    assert strata_cli.version() == "0.4.1"
    (root / "VERSION").write_text("0.4.2\n")
    assert strata_cli.version() == "0.4.2"


@pytest.mark.parametrize("invalid", ["0.4.2-beta.1", "0.4", "0.04.2", "0.4.2\n0.4.3", ""])
def test_invalid_control_identity_cannot_fall_back_to_engine(tmp_path, monkeypatch, invalid):
    root = source(tmp_path, monkeypatch)
    (root / "CONTROL_VERSION").write_text(invalid)
    with pytest.raises(ValueError, match="CONTROL_VERSION"):
        package_terminal._source_version()
    with pytest.raises(ValueError, match="CONTROL_VERSION"):
        strata_cli.version()
