"""Strata Dev (docs/DEV_BUILD.md): model overlays from ``dev/models/*.json`` on top of a run configuration (``strata-<model>.json``).

Only with ``STRATA_DEV=1``. Live never reads ``dev/models``, and ``apply`` then returns the configuration unchanged. It is called by
``presets.overlay`` before the saved tuning, so every start path (the app's load, the Terminal, a restart) uses it, and a tuned
preset still goes on top of it. Nothing here writes a run configuration; the overlay lives only in the started engine.

One overlay file::

    {
      "model": "qwen3.8-flash-next-*",          # which run configs: their "model_name" (or the config's file name), a glob
      "label": "Qwen schnell",                   # shown in the app
      "source": {"hf": "org/repo", "file": "x.gguf"} | {"gguf": "C:/models/x.gguf"} | {"server": "http://..."},
      "quant": "Q4_K_M",                         # what the app shows; informational
      "context": 65536,                          # --max-context
      "vision": false,                           # the run config's "vision"
      "args": {"--kv": "q8_0", "--mtp": null},  # engine options: a value sets, null removes, true is a switch
      "env": {"NAME": "value"},                  # environment: null removes
      "thinking": "low",                         # reasoning_effort for requests that name none
      "sampling": {"temperature": 0.6},          # merged into "sampling"
      "chat_template": "templates/x.jinja",      # a template file (relative to this json): a copy of the tokenizer folder uses it
      "profiles": {"schnell": {...}, ...},       # named overlays on top of this one (same keys)
      "profile": "schnell"                       # the default profile; the app's choice (dev-model-profiles.json) wins
    }

Built-in profiles (``PROFILES``: "schnell", "coder", "max-qualitaet") are there for every overlay; a file's own ``profiles`` with the
same name replaces them. Never raises: a broken file is skipped and named in ``info["warnings"]``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from strata_control import tune_knobs

PROFILES: dict[str, dict[str, Any]] = {
    # Fast answers: little thinking, the smaller context the engine starts quicker with.
    "schnell": {"label": "Schnell", "thinking": "low", "context": 32768, "sampling": {"temperature": 0.7}},
    # Code: steady output and thinking on.
    "coder": {"label": "Coder", "thinking": "high", "sampling": {"temperature": 0.2, "top_p": 0.95}},
    # Best answers the model can give here: most thinking, the model card's sampling.
    "max-qualitaet": {"label": "Max Qualität", "thinking": "high", "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}},
}
KEYS = ("context", "vision", "args", "env", "thinking", "sampling", "chat_template")


def enabled(env: dict[str, str] | None = None) -> bool:
    return (env if env is not None else os.environ).get("STRATA_DEV") == "1"


def models_dir(env: dict[str, str] | None = None) -> Path | None:
    """``STRATA_DEV_MODELS``, else ``dev/models`` beside ``STRATA_DEV_CONFIG`` (the host's dev-mode.mjs uses the same rule)."""
    env = env if env is not None else os.environ
    if env.get("STRATA_DEV_MODELS"):
        return Path(env["STRATA_DEV_MODELS"])
    if env.get("STRATA_DEV_CONFIG"):
        return Path(env["STRATA_DEV_CONFIG"]).resolve().parent / "dev" / "models"
    return None


def choices_file(env: dict[str, str] | None = None) -> Path | None:
    """The app's profile choice per overlay (written by the host): ``<STRATA_HOME>/dev-model-profiles.json``."""
    env = env if env is not None else os.environ
    # The host's store.mjs: STRATA_HOME, else ~/.strata (Strata Dev always sets STRATA_HOME, so Live's ~/.strata is not read there).
    return (Path(env["STRATA_HOME"]) if env.get("STRATA_HOME") else Path.home() / ".strata") / "dev-model-profiles.json"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def load_overlays(directory: Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Every ``*.json`` in the folder (``_`` at the start: skipped), as ``{"id", "file", **data}``, and the warnings."""
    out: list[dict[str, Any]] = []
    warnings: list[str] = []
    if directory is None or not directory.is_dir():
        return out, warnings
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        data = _read_json(path, None)
        if not isinstance(data, dict) or not isinstance(data.get("model"), str) or not data["model"].strip():
            warnings.append(f"dev/models/{path.name}: needs a JSON object with \"model\"; it is skipped")
            continue
        out.append({**data, "id": path.stem, "file": str(path)})
    return out, warnings


def matches(overlay: dict[str, Any], cfg: dict[str, Any], config_name: str | None = None) -> bool:
    pattern = overlay["model"].strip().lower()
    names = [str(cfg.get("model_name") or "").lower(), str(config_name or "").lower()]
    return any(n and fnmatch.fnmatchcase(n, pattern) for n in names)


def _merge_layer(cfg: dict[str, Any], layer: dict[str, Any], base_dir: Path, info: dict[str, Any]) -> dict[str, Any]:
    out = cfg
    args = dict(layer.get("args") or {})
    if layer.get("context") is not None:
        args["--max-context"] = str(int(layer["context"]))
    if args:
        out["args"] = tune_knobs.apply_to_argv([str(a) for a in out.get("args") or []], args)
    if isinstance(layer.get("env"), dict):
        out["env"] = tune_knobs.apply_to_env(dict(out.get("env") or {}), layer["env"])
    if isinstance(layer.get("vision"), bool):
        out["vision"] = layer["vision"]
    if isinstance(layer.get("thinking"), str) and layer["thinking"]:
        out["reasoning_effort"] = layer["thinking"]
    if isinstance(layer.get("sampling"), dict):
        out["sampling"] = {**(out.get("sampling") or {}), **layer["sampling"]}
    if isinstance(layer.get("chat_template"), str) and layer["chat_template"]:
        template = Path(layer["chat_template"])
        template = template if template.is_absolute() else base_dir / template
        tokenizer = _with_template(out.get("tokenizer"), template, info)
        if tokenizer:
            out["tokenizer"] = tokenizer
    return out


def _with_template(tokenizer: Any, template: Path, info: dict[str, Any]) -> str | None:
    """A copy of the tokenizer folder with ``chat_template.jinja`` replaced, in the Dev data folder. The original stays as it is."""
    home = os.environ.get("STRATA_HOME")
    if not isinstance(tokenizer, str) or not Path(tokenizer).is_dir() or not template.is_file() or not home:
        info.setdefault("warnings", []).append(f"chat_template {template}: needs the run config's tokenizer folder and the template file; not applied")
        return None
    text = template.read_bytes()
    key = hashlib.sha256(str(Path(tokenizer).resolve()).encode() + b"\0" + text).hexdigest()[:16]
    target = Path(home) / "dev-tokenizers" / key
    if not (target / "chat_template.jinja").is_file():
        shutil.copytree(tokenizer, target, dirs_exist_ok=True)
        (target / "chat_template.jinja").write_bytes(text)
    return str(target)


def apply(cfg: dict[str, Any], *, env: dict[str, str] | None = None, config_name: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The run config with the matching overlays (and the chosen profile) on top, and what was done. Outside Dev: unchanged."""
    info: dict[str, Any] = {"applied": False}
    if not enabled(env):
        return cfg, info
    try:
        overlays, warnings = load_overlays(models_dir(env))
        chosen = _read_json(choices_file(env), {})
        out = json.loads(json.dumps(cfg))
        used = []
        for ov in overlays:
            if not matches(ov, out, config_name):
                continue
            base = Path(ov["file"]).parent
            out = _merge_layer(out, {k: ov[k] for k in KEYS if k in ov}, base, info)
            name = (chosen.get(ov["id"]) if isinstance(chosen, dict) else None) or ov.get("profile")
            profiles = {**PROFILES, **(ov.get("profiles") or {})}
            if name and name in profiles:
                out = _merge_layer(out, profiles[name], base, info)
            elif name:
                warnings.append(f"dev/models/{ov['id']}.json: profile \"{name}\" is not known")
            used.append({"overlay": ov["id"], "profile": name or None})
        if warnings:
            info.setdefault("warnings", []).extend(warnings)
        if used:
            info.update({"applied": True, "overlays": used})
            return out, info
        return cfg, info
    except Exception as e:  # noqa: BLE001 - a start never fails because of a Dev overlay
        info["error"] = f"{type(e).__name__}: {e}"
        return cfg, info
