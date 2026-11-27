"""Custom builds: a person's own model files (a fine-tune, a merge, another maker's quantization) of a model Strata's engines already run.

The files stay where they are. Strata writes ONE small run config for them, `strata-<family>-custom-<name>.json`, beside the official ones, so an
official install of the same size is never replaced. The run config records `"custom": true`, the name the person chose (`display_name`), the
family and the closest published size it is run as (`base_family`, `base_model`), the files' real size (`custom_bytes`) and where they are
(`custom_source`). Removing a custom build removes that run config (and Strata's own start script and log for it); the model files are never
touched.

This module holds the pure rules (names, the config fields, the fit with the real file size). It does not read model files and does not write.
"""

from __future__ import annotations

import contextlib
import re
import threading
import unicodedata

MAX_LABEL = 80
MAX_SLUG = 40
BADGE = "Custom"
#: setup family id -> the tag setup gives its run configs (strata-<tag><size>.json); the Flash-Next family has none
FAMILY_TAGS = {"qwen": "", "swift": "swift-", "coder": "coder-", "unsloth": "unsloth-", "gyro": "gyro-", "qwen36": "qwen36-",
               "ornith": "ornith-1.5-35b-a3b-", "hy4-flash": "hy4-flash-", "inkling-2-mini": "inkling-2-mini-"}
#: the family a person sees (the app's and the Terminal's display groups)
DISPLAY_FAMILY = {"qwen": "Qwen3.8-Flash-Next", "swift": "Qwen3.8-Flash-Next", "coder": "Qwen3.8-Flash-Next", "unsloth": "Qwen3.8-Flash-Next",
                  "gyro": "Qwen3.8-Flash-Next", "qwen36": "Qwen3.6-35B-A3B", "ornith": "Ornith-1.5-35B-A3B", "hy4-flash": "HY-4 Flash",
                  "inkling-2-mini": "Inkling-2-Mini"}
#: what a custom build may be written for: the model families the native engines run (HY-4 Flash only as .strata: its engine has no GGUF loader)
SUPPORTED_FAMILIES = frozenset(FAMILY_TAGS)
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f  ‪-‮⁦-⁩]")


def badge() -> str:
    return BADGE


def clean_label(text: object, fallback: str = "") -> str:
    """A name for the screens: printable characters only, one line, single spaces, at most MAX_LABEL characters. It is only ever shown (and put in
    the JSON); a path is never made from it (see `slug`)."""
    raw = str(text if text is not None else "")
    raw = _CONTROL.sub(" ", raw)
    raw = " ".join(raw.split())[:MAX_LABEL].strip()
    return raw or fallback


def slug(text: object) -> str:
    """The part of a run config's file name made from the name: lower-case a-z, 0-9 and single dashes, at most MAX_SLUG characters; never empty,
    never a path (no dots, slashes or drive letters survive)."""
    norm = unicodedata.normalize("NFKD", str(text if text is not None else "")).encode("ascii", "ignore").decode("ascii").lower()
    out = re.sub(r"[^a-z0-9]+", "-", norm).strip("-")[:MAX_SLUG].strip("-")
    return out or "build"


def tag_prefix(base_family: str) -> str:
    """The tag setup gives its run configs of this family; "qwen-" for the Flash-Next family, whose own tag is empty (so a custom file is
    strata-qwen-custom-<name>.json and not strata-custom-<name>.json)."""
    if base_family not in FAMILY_TAGS:
        raise ValueError(f"not a model family Strata's engines run: {base_family!r}")
    return FAMILY_TAGS[base_family] or "qwen-"


def run_tag(base_family: str, label: object, same_source: object = None, existing: "dict[str, object] | None" = None) -> str:
    """The tag of a custom run config (the file is strata-<tag>.json): <family tag>custom-<slug of the name>. `existing` maps tags already in use
    to the source path they were made for: the same files again give the same tag (the run config is refreshed), other files with the same name
    get -2, -3, ... so two builds never share one run config and no official tag can be produced."""
    base = f"{tag_prefix(base_family)}custom-{slug(label)}"
    used = existing or {}
    cand, n = base, 1
    while cand in used and not _same_path(used[cand], same_source):
        n += 1
        cand = f"{base}-{n}"
    return cand


def _same_path(a: object, b: object) -> bool:
    if not a or not b:
        return False
    import os
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def model_alias(base_family: str, label: object) -> str:
    """The name the server lists the build under in /v1/models: <family name>-custom-<slug>."""
    name = {"qwen": "qwen3.8-flash-next", "swift": "swift-qwen3.8-flash-next", "coder": "coder-qwen3.8-flash-next", "unsloth": "qwen3.8-flash-next-unsloth",
            "gyro": "qwen3.8-flash-next-gyro", "qwen36": "qwen3.6-35b-a3b", "ornith": "ornith-1.5-35b-a3b", "hy4-flash": "hy4-flash",
            "inkling-2-mini": "inkling-2-mini"}.get(base_family, base_family)
    return f"{name}-custom-{slug(label)}"


def run_fields(label: object, base_family: str, base_model: str, weight_bytes: int, source: object = None) -> dict:
    """The keys a custom run config adds."""
    fields = {"custom": True, "display_name": clean_label(label, "Custom build"), "base_family": str(base_family), "base_model": str(base_model),
              "custom_bytes": int(weight_bytes or 0)}
    if source:
        fields["custom_source"] = str(source)
    return fields


def is_custom_cfg(cfg: object) -> bool:
    return isinstance(cfg, dict) and cfg.get("custom") is True and isinstance(cfg.get("base_family"), str)


def summary_fields(cfg: dict) -> dict:
    """What `strata models list` adds to a custom run config's row (nothing for an official one)."""
    if not is_custom_cfg(cfg):
        return {}
    return {"custom": True, "display_name": clean_label(cfg.get("display_name"), "Custom build"), "base_family": cfg["base_family"],
            "base_model": str(cfg.get("base_model") or "") or None,
            "custom_bytes": int(cfg["custom_bytes"]) if isinstance(cfg.get("custom_bytes"), int) and cfg["custom_bytes"] >= 0 else None,
            "display_family": DISPLAY_FAMILY.get(cfg["base_family"]), "badge": BADGE}


def own_files_to_remove(config_path: "object") -> list:
    """What removing a custom build deletes: its run config and Strata's own start script and log for it (never the model files, never a folder).
    The caller deletes only paths in this list."""
    from pathlib import Path
    p = Path(str(config_path))
    stem = p.stem
    return [p, p.with_name(stem + ".log"), p.with_name(stem + ".json.bak"), p.with_name(stem + ".bat"), p.with_name(stem + ".sh"), p.with_name(stem + ".cmd")]


# ---- the fit, with the file's real size on top of the closest published build ------------------------------------------------------------------------

GB = 1e9
_LOCK = threading.Lock()


def derived_build(fit_module, base_model: str, weight_bytes: int | None) -> dict:
    """The fit module's entry for `base_model`, moved to the real size of the files: the difference to the published file goes to the routed experts
    (a fine-tune keeps the dense part, a heavier or lighter quantization changes the experts). Without a size the published entry is returned as it is."""
    base = dict(fit_module.BUILDS[base_model])
    if not weight_bytes or weight_bytes <= 0:
        return base
    actual = round(weight_bytes / GB, 2)
    delta = actual - float(base["file_gb"])
    base["file_gb"] = actual
    base["footprint_gb"] = round(max(float(base.get("dense_gb", 0.0)) + 0.1, float(base["footprint_gb"]) + delta), 2)
    base["bodies_gb"] = round(max(0.1, float(base["bodies_gb"]) + delta), 2)
    return base


@contextlib.contextmanager
def temporary_build(fit_module, base_model: str, weight_bytes: int | None):
    """Inside the block `fit_module.BUILDS[key]` is the derived build (the fit functions take a build name); it is gone afterwards, so the
    published lists are never changed."""
    key = f"custom:{base_model}:{int(weight_bytes or 0)}"
    with _LOCK:
        fit_module.BUILDS[key] = derived_build(fit_module, base_model, weight_bytes)
        try:
            yield key
        finally:
            fit_module.BUILDS.pop(key, None)


def fit_placement(cfg: dict, vram_gib: float | None, ram_gib: float, ctx: int | None = None, unified: bool = False, vision: bool = False) -> dict | None:
    """Where the experts of a custom build go on this computer ({"vram_gb", "ram_gb", "ssd_gb"}, GB), by the closest published build's rules and
    the real file size. Inkling-2-Mini and HY-4 Flash only; None for a family whose fit does not depend on the file size (Qwen3.6: see `own_path_inputs`)."""
    if not is_custom_cfg(cfg):
        return None
    fam, base, size = cfg["base_family"], str(cfg.get("base_model") or ""), cfg.get("custom_bytes")
    if fam == "inkling-2-mini":
        from strata_control import inkling_fit as module
    elif fam == "hy4-flash":
        from strata_control import hy4_fit as module
    else:
        return None
    if base not in module.BUILDS:
        return None
    with temporary_build(module, base, size) as key:
        return module.place(key, vram_gib, ram_gib, unified=unified, ctx=ctx, vision=vision)


def own_path_inputs(cfg: dict) -> dict | None:
    """Qwen3.6 (and Ornith): the context rule's inputs for the closest published size (own_path_fit.inputs); the files' size does not enter that rule."""
    if not is_custom_cfg(cfg) or cfg["base_family"] not in ("qwen36", "ornith"):
        return None
    from strata_control import own_path_fit
    name = own_path_fit.FAMILY_NAME if cfg["base_family"] == "qwen36" else own_path_fit.FAMILY_NAME_ORNITH
    return own_path_fit.inputs(name, str(cfg.get("base_model") or ""))
