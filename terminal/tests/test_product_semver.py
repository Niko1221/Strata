"""Release numbering must be valid SemVer and must not go backwards."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "product_version", Path(__file__).resolve().parents[2] / "tools/bump_hakua_product_version.py"
)
versioning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(versioning)


@pytest.mark.parametrize("value", [
    "0.4.1-beta.1", "1.0.0", "2.3.4-rc.12+build.001", "1.0.0-0.3.7",
    "0.4.0-preview.20261103.7",
])
def test_valid_semver(value):
    assert versioning.PRODUCT_RE.fullmatch(value)


@pytest.mark.parametrize("value", [
    "v0.4.1", "01.2.3", "1.02.3", "1.2.03", "1.2", "1.2.3-beta.01",
    "1.2.3-", "1.2.3+", "1.2.3-beta..1", "1.2.3-beta_1", "1.2.3-beta.\u0661",
])
def test_invalid_semver(value):
    assert versioning.PRODUCT_RE.fullmatch(value) is None


@pytest.mark.parametrize("current,channel,tags,expected", [
    ("0.4.0-preview.20261103.7", "beta", [], "0.4.1-beta.1"),
    ("0.4.1-beta.1", "beta", ["v0.4.1-beta.9"], "0.4.1-beta.10"),
    ("0.4.1-beta.1+test", "beta", [], "0.4.1-beta.2"),
    ("0.4.1-beta.9", "rc", [], "0.4.1-rc.1"),
    ("0.4.1-rc.1", "beta", [], "0.4.2-beta.1"),
    ("1.0.0", "beta", [], "1.0.1-beta.1"),
    ("1.0.0-beta.extra", "beta", [], "1.0.1-beta.1"),
])
def test_next_prerelease(current, channel, tags, expected):
    assert versioning.next_prerelease(channel, current, tags) == expected
