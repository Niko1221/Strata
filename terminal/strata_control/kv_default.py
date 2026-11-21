"""Whether a run config names `--kv` itself, or leaves the format to the engine.

This tree's engine is the Strata release it was changed from plus the suffix of what it adds: **0.1.63-q4u32**.
The suffix is the marker: an upstream engine has no suffix and no q4u32 default, so a config for one names the format
it always named and that start stays byte-identical to the release.  A config for a q4u32 build on a **gfx12** card
leaves `--kv` out above 8K and lets the engine choose (q4u32 there, where its prompt path was measured); up to 8K
the release named fp16 and so does this rule.

The flavour is read from the engine itself: the suffix of its own version line (setup.py's binary_engine_flavor), or
the manifest's "flavor" field for a packaged engine.  An engine whose flavour cannot be told counts as one without it.
"""

from __future__ import annotations

#: the variant this tree's engine carries (CMakeLists.txt STRATA_ENGINE_FLAVOR): the build that picks the KV format
#: itself, q4u32 on gfx12 (the measured one, A.23)
KV_ENGINE_FLAVOR = "q4u32"

#: the cards whose q4u32 prompt path was measured (A.17-A.23, Windows gfx1201)
GFX12_ARCHS = ("gfx1200", "gfx1201")


def version_tuple(version) -> tuple:
    """("0.1.59.1", (0, 1, 59, 1), None) -> a tuple of numbers; anything unreadable is ().

    A part that is not a number ends the version: "0.1.59.1-dev" is (0, 1, 59, 1), the numbers up to the word.
    """
    if isinstance(version, tuple):
        return version
    out = []
    for part in str(version or "").split("."):
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def flavor_of(version: str) -> str:
    """The variant suffix of a version string ("0.1.63-q4u32" -> "q4u32"), or "" when it has none."""
    text = str(version or "")
    if "-" not in text:
        return ""
    return text.rsplit("-", 1)[1].strip()


def is_gfx12(arch) -> bool:
    """True for a gfx12 card ("gfx1201", "gfx1201:sramecc-:xnack-")."""
    text = str(arch or "").lower()
    return any(text == a or text.startswith(a + ":") for a in GFX12_ARCHS)


def engine_picks_kv(flavor: str, *, hip: bool = False, arch: str = "") -> bool:
    """True when the engine's own default is the measured one: a q4u32 build on a gfx12 card."""
    if str(flavor or "") != KV_ENGINE_FLAVOR:
        return False
    return bool(hip) and is_gfx12(arch)


def kv_flag(flavor: str, context: int, named: str | None, *, hip: bool = False, arch: str = "") -> str | None:
    """The value `--kv` gets in a run config, or None when the flag is left out.

    `named` is the format the person asked for (None: they did not).  `context` is the token count the config writes.
    This is setup's own rule, unchanged: a format the person named is kept, and without a name the config names what
    the release named - 8-bit above 8K, 16-bit up to it (setup.py's own `kv = "fp16" if ctx <= 8192 else (a.kv or
    "int8")`).  The one change: above 8K a q4u32 build on a gfx12 card names nothing at all - it picks the measured
    format itself.  Up to 8K it keeps naming fp16, so those starts stay as they were.

    `q4u32w` is setup's name for q4u32 plus the exact-cell window: the config names `--kv q4u32`, and setup writes
    the window into the engine env (STRATA_KV_WINDOW=1, S=256, R=64).
    """
    if named == "q4u32w":
        return "q4u32"
    if named:
        return named
    if context > 8192 and engine_picks_kv(flavor, hip=hip, arch=arch):
        return None
    return "int8" if context > 8192 else "fp16"


def kv_value(flavor: str, context: int, named: str | None, *, hip: bool = False, arch: str = "") -> str:
    """The format to name in a config that always names one, and the format the engine will run with.

    Above 8K on a gfx12 card with a q4u32 build that is q4u32 (54.5 % of int8's bytes); everywhere else the format
    the config named, as before.  `q4u32w` (setup's window choice) runs as q4u32 with the window env beside it.
    """
    if named == "q4u32w":
        return "q4u32"
    if named:
        return named
    if context > 8192 and engine_picks_kv(flavor, hip=hip, arch=arch):
        return "q4u32"
    return "int8" if context > 8192 else "fp16"
