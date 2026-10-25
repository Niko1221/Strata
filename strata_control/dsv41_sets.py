"""The official DeepSeek-V4.1-Flash GGUF sets (IST-DAS 1.75 / 2.00 / 2.25 bpw), recognised by their pinned file names and sizes.

The Terminal's setup downloads them (setup.py MULTI_FILE_FAMILIES, DSV41_ISTA_FILES) and accepts the files the first release was downloaded as
(DSV41_LEGACY_FILES: model-2.00bpw-0000N-of-00004.gguf, engram-IQ4_NL.gguf). The app's "I already have the model" preview used to know only the
Qwen fingerprints, so a folder with such a set ended in "the folder holds more than one model" (the Engram is a second file) or, for one shard,
"this model type is not supported by the Strata engine". This module is how Control says what a folder of these files is, from the names and
sizes alone: no tensor is read and no 40 GB file is hashed in a preview (setup checks the SHA-256 once, when the set is adopted).

The tables are setup.py's own (read from it, never copied), so a new size added there is recognised here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TABLES: dict | None = None
FAMILY = "deepseek41"


def tables() -> dict | None:
    """{model: {"label", "shards": {name: bytes}, "legacy": {name: bytes}}} for the official sets and {"engram": {name: bytes}}; None when
    setup.py cannot be read (the preview then behaves as before)."""
    global _TABLES
    if _TABLES is not None:
        return _TABLES or None
    try:
        spec = importlib.util.spec_from_file_location("strata_setup_tables", ROOT / "setup.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)            # type: ignore[union-attr]
        family = module.MULTI_FILE_FAMILIES[FAMILY]
        ista, legacy, renamed = module.DSV41_ISTA_FILES, module.DSV41_LEGACY_FILES, module.DSV41_LEGACY_NAMES
        result: dict = {"sets": {}, "engram": {}}
        for model, entry in family["models"].items():
            if entry.get("recipe") != "ista":
                continue                                # Unsloth's 508 GB set is not an "already have it" case of the app
            shards = {name: ista[name][0] for name in entry["shards"]}
            old = {renamed[name]: legacy[renamed[name]][0] for name in entry["shards"] if name in renamed and renamed[name] in legacy}
            result["sets"][model] = {"label": entry.get("label", model), "shards": shards, "legacy": old}
        for name in (module.DSV41_ISTA_ENGRAM, renamed.get(module.DSV41_ISTA_ENGRAM)):
            if name and (name in ista or name in legacy):
                result["engram"][name] = (ista.get(name) or legacy[name])[0]
        _TABLES = result
    except Exception:                                   # noqa: BLE001 - a preview must never fail because of this
        _TABLES = {}
    return _TABLES or None


def match(files: dict[str, int]) -> list[dict]:
    """The official sets the files {name: bytes} of one folder belong to, best first: complete sets before partial ones.
    Each: {"model", "label", "complete", "present": [names], "missing": [names], "wrong_size": [names], "legacy": bool, "engram": name|None}.
    A set counts when at least one of its shards is there; names and sizes are the pinned ones."""
    known = tables()
    if not known:
        return []
    engram = next((name for name in known["engram"] if files.get(name) == known["engram"][name]), None)
    found = []
    for model, entry in known["sets"].items():
        for legacy, pinned in ((False, entry["shards"]), (True, entry["legacy"])):
            if not pinned:
                continue
            present = [n for n in pinned if n in files]
            if not present:
                continue
            wrong = [n for n in present if files[n] != pinned[n]]
            missing = [n for n in pinned if n not in files]
            found.append({"model": model, "label": entry["label"], "legacy": legacy, "present": sorted(present), "missing": missing,
                          "wrong_size": wrong, "complete": not missing and not wrong, "engram": engram,
                          "bytes": sum(pinned[n] for n in present if n not in wrong) + (known["engram"][engram] if engram else 0)})
    found.sort(key=lambda row: (not row["complete"], len(row["missing"]) + len(row["wrong_size"]), row["model"]))
    return found
