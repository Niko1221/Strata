"""Custom builds in the Terminal: the person's own compatible fine-tunes and quantizations (local .strata or .gguf files).

Control (strata_control/models.py and setup.py) writes one run config per custom build, strata-<family tag>custom-<slug>.json, with
"custom": true, "display_name", "base_family", "base_model", "custom_bytes" and "custom_source" in it. Nothing here reads or changes the
weights: the Terminal lists those run configs next to the official builds of the same base model, runs them, and removes only the run
config when asked. The person's own files are never copied, moved or deleted.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

BADGE = "Custom"
MAX_CFG_BYTES = 1024 * 1024
KEPT_WORDS = "Only the run settings were removed; your original model files were left where they are."
FAMILIES_LINE = "Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, HY-4 Flash and Inkling-2-Mini"
UNSUPPORTED_WORDS = (f"Strata runs compatible fine-tunes and quantizations of {FAMILIES_LINE} (local .strata or .gguf files). "
                     "For any other model type, add it as a Custom model API (llama-server) instead.")


def clean_label(text: object) -> str:
    """The display name a person typed, as one short printable line (never a path, never control characters)."""
    try:
        from strata_control import custom_models
        fn = getattr(custom_models, "clean_label", None)
        if fn is not None:
            return str(fn(text))
    except Exception:                                    # noqa: BLE001 - the local rule below is the same in words
        pass
    line = "".join(c if c.isprintable() and c not in "\x1b\x7f" else " " for c in str(text or ""))
    return re.sub(r"\s+", " ", line.replace("/", " ").replace("\\", " ")).strip()[:80]


def slug(text: object) -> str:
    try:
        from strata_control import custom_models
        fn = getattr(custom_models, "slug", None)
        if fn is not None:
            return str(fn(text))
    except Exception:                                    # noqa: BLE001
        pass
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", clean_label(text).lower())).strip("-")[:40] or "build"


def run_tag(base_family: str, label: object, source: object = None, catalog: Any = None) -> str:
    """The run config's id (the part between strata- and .json) for a custom build of `base_family` (Control's rule, strata_control/custom_models.py:
    the same files again give the same id, other files with the same name get -2, -3)."""
    from strata_control import custom_models
    existing = {e["id"]: e["source"] for e in entries(catalog)} if catalog is not None else None
    return custom_models.run_tag(base_family, label, source, existing)


def size_words(num_bytes: object) -> str:
    try:
        gb = float(num_bytes) / 1e9
    except (TypeError, ValueError):
        return ""
    return f"{gb:.0f} GB" if gb >= 10 else f"{gb:.1f} GB" if gb > 0 else ""


def read_custom(path: Path) -> dict | None:
    """One run config as a custom entry, or None when it is no custom build (or cannot be read: a list never stops for that)."""
    try:
        if path.stat().st_size > MAX_CFG_BYTES:
            return None
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict) or cfg.get("custom") is not True:
        return None
    name = path.name
    tag = name[len("strata-"):-len(".json")] if name.startswith("strata-") and name.endswith(".json") else path.stem
    shown = clean_label(cfg.get("display_name")) or clean_label(cfg.get("model_name")) or tag
    return {"id": tag, "path": str(path), "display_name": shown, "base_family": str(cfg.get("base_family") or ""),
            "base_model": str(cfg.get("base_model") or ""), "bytes": int(cfg.get("custom_bytes") or 0) if str(cfg.get("custom_bytes") or "0").isdigit() else 0,
            "source": str(cfg.get("custom_source") or ""), "cfg": cfg}


def entries(catalog: Any) -> list[dict]:
    """Every custom build installed here, by id."""
    found: dict[str, dict] = {}
    try:
        configs = catalog.installed_configs()
    except Exception:                                    # noqa: BLE001 - a list is never a reason to stop
        return []
    for d in configs:
        path = Path(str(d.get("path") or ""))
        item = read_custom(path) if path.name else None
        if item and item["id"] not in found:
            found[item["id"]] = item
    return sorted(found.values(), key=lambda e: e["id"])


def find(catalog: Any, name: str) -> dict | None:
    """The custom build a person named: its id (strata-<id>.json or just the id), or its display name, in any letter case."""
    want = str(name or "").strip()
    if want.lower().startswith("strata-"):
        want = want[len("strata-"):]
    if want.lower().endswith(".json"):
        want = want[:-5]
    low = want.lower()
    items = entries(catalog)
    return next((e for e in items if e["id"].lower() == low), None) or next((e for e in items if e["display_name"].lower() == low), None)


def use_installed(s: Any, entry: dict) -> None:
    """Point launch settings at an installed custom build: its base family and model (the fit and the engine path), its id, and the files it
    was made from (so a second setup run, for a new context or pictures, writes the same run config)."""
    s.family, s.model = entry["base_family"], entry["base_model"]
    s.custom_id, s.custom_label = entry["id"], entry["display_name"]
    source = entry.get("source") or ""
    s.model_path = source if source and os.path.exists(source) else None
    if s.drafting is None:
        s.drafting = "auto"


def use_import(s: Any, catalog: Any, label: object) -> str:
    """The files in s.model_path become a custom build called `label`; returns its id."""
    s.custom_label = clean_label(label) or "Custom build"
    if s.custom_id and any(e["id"] == s.custom_id and e["display_name"] == s.custom_label for e in entries(catalog)):
        return s.custom_id                                  # an installed build picked again: its run config stays the same one
    s.custom_id = run_tag(str(s.family), s.custom_label, s.model_path, catalog)
    if s.drafting is None:
        s.drafting = "auto"                              # fine-tunes keep the base model's vocabulary: Auto uses the drafter and falls back by itself
    return s.custom_id


def weight_bytes(source: str | os.PathLike) -> int:
    """The size of the person's files (read only): the file, every shard of a split set, or the model files of a folder."""
    p = Path(source)
    try:
        if p.is_file():
            m = re.match(r"(.+)-\d{5}-of-(\d{5})\.gguf\Z", p.name, re.I)
            if m:
                return sum(q.stat().st_size for q in p.parent.glob(m.group(1) + "-*-of-" + m.group(2) + ".gguf") if q.is_file())
            return p.stat().st_size
        if p.is_dir():
            return sum(q.stat().st_size for q in p.iterdir()
                       if q.is_file() and q.suffix.lower() in (".gguf", ".strata") and not q.name.lower().startswith("mmproj")
                       and "vision" not in q.name.lower() and "engram" not in q.name.lower())
    except OSError:
        return 0
    return 0


def run_config_keys(s: Any, model_dir: str | os.PathLike | None = None) -> dict:
    """The keys a custom build's run config carries (the ones Control's setup writes too): what lists, fit and remove read."""
    from strata_control import custom_models
    source = str(s.model_path or model_dir or "")
    return custom_models.run_fields(s.custom_label or s.custom_id, str(s.family), str(s.model), weight_bytes(source) if source else 0, source or None)


def custom_alias(s: Any) -> str:
    """The name the server lists the build under in /v1/models (Control's rule)."""
    from strata_control import custom_models
    return custom_models.model_alias(str(s.family), s.custom_label or s.custom_id)


def clear(s: Any) -> None:
    s.custom_id = s.custom_label = None


def remove_reference(path: str | os.PathLike) -> list[str]:
    """Remove one custom build: its run config (and the backup beside it) only. The weights, the packs and every other file stay.
    Returns the names removed. A run config that is no custom build is refused, so this can never take an official install."""
    p = Path(path)
    item = read_custom(p)
    if item is None:
        raise ValueError(f"{p.name} is not a custom build, so Strata does not remove it from here.")
    from strata_control import custom_models
    removed = []
    for target in custom_models.own_files_to_remove(p):  # the run config, Strata's own start script and log for it: never a model file, never a folder
        if target.parent != p.parent or not target.name.startswith("strata-"):
            continue
        try:
            if target.is_file() and not target.is_symlink():
                target.unlink()
                removed.append(target.name)
        except OSError:
            continue
    return removed
