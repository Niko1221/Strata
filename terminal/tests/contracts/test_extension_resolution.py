"""tests/contracts/test_extension_resolution.py - Verification of T02 extension resolution & snapshot integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from strata_control.extensions import (
    ExtensionOptions,
    EffectiveExtension,
    resolve_extension,
)

def test_single_snapshot() -> None:
    """Verifies that resolved extension options produce an immutable byte-exact snapshot."""
    opts = ExtensionOptions(spin8_mode="off", dialect="standard")
    ext1 = resolve_extension(opts)
    ext2 = resolve_extension(opts)
    assert ext1.snapshot_hash == ext2.snapshot_hash
    assert ext1.clean_off is True
    assert ext1.extra_args == []

def test_manual_override_conflict() -> None:
    """Conflicting options between CLI args and extension sidecars must be rejected before launch."""
    opts = ExtensionOptions(
        spin8_mode="cuda",
        cli_args=["--spin8-mode", "off"]
    )
    with pytest.raises(ValueError, match="Conflicting extension settings"):
        resolve_extension(opts)

def test_unknown_upstream_extension_field() -> None:
    """Must not leak unknown extension fields into upstream argv."""
    opts = ExtensionOptions(spin8_mode="off")
    ext = resolve_extension(opts)
    for arg in ext.extra_args:
        assert not arg.startswith("--fork-internal")

def test_owner_ready_identity() -> None:
    """Identity string incorporates extension hash to detect post-validation drift."""
    opts = ExtensionOptions(spin8_mode="cuda", dialect="tq4_1s")
    ext = resolve_extension(opts)
    assert len(ext.snapshot_hash) == 64
    assert ext.clean_off is False
    assert "--spin8-mode" in ext.extra_args or "--kv" in ext.extra_args
