"""Terminal clients for the shared Strata server."""

import re
from pathlib import Path

SCHEMA_VERSION = 1

#: The version people are told: the Strata app's own (apps/desktop/src-tauri/Cargo.toml `version`; tests/terminal/test_tuiqa_version.py keeps the
#: two equal, except for a Terminal-only release such as 0.2.3.1 or 0.2.4, whose app stays 0.2.3). On the Hakua fork this matches
#: repository-root ``VERSION`` (product SemVer). `version()` below is the Control build number, which the engine-compatibility
#: checks compare, so it stays numeric. CONTROL_VERSION owns it in a source checkout; archives carry it in VERSION.
PRODUCT_VERSION = "0.6.1"

_BUILD_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")


def version() -> str:
    """Control build: source CONTROL_VERSION, archive VERSION, or the legacy CMake version."""
    root = Path(__file__).resolve().parents[1]
    control = root / "CONTROL_VERSION"
    if control.is_file():
        text = control.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", text):
            raise ValueError("CONTROL_VERSION must be a three-part release version")
        return text
    try:
        text = (root / "VERSION").read_text(encoding="utf-8").strip()
        # Hakua product previews live in VERSION for packaging; build number must stay dotted-numeric.
        if text and _BUILD_VERSION.fullmatch(text):
            return text
    except OSError:
        pass
    try:
        m = re.search(r"^\s*project\([^)]*?VERSION\s+([0-9]+(?:\.[0-9]+)+)",
                      (root / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
        if m:
            return m.group(1)
    except OSError:
        pass
    return "unknown"
