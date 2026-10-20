"""Bounded, read-only discovery of setup-written local model run configs.

The setup installer writes strata-*.json beside setup.py. A configured model is
not automatically a compatible or loadable model: component presence and native
engine compatibility are reported separately.

Finding an existing install: a terminal archive is a new folder with no run configs in it. The places where Strata
installs keep run configs (the Terminal's own folder, the folders earlier installs remembered in the user's settings file,
and the desktop app's install folder, which the Terminal shares) are searched read-only by strata_control/discovery.py,
the same function the launch screens use; the first place that has any is the main one. Nothing is copied or moved:
the configs name their engine and model files by absolute path, so the old install and its data folder keep
serving the models.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from strata_control import discovery
from strata_control.doctor import inspect as inspect_components
from strata_control.gguf_preflight import inspect_metadata
from strata_control.profiles import ProfileError


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}\Z")
MAX_CONFIG_BYTES = 1024 * 1024
MAX_CONFIGS = 128
MAX_IMPORT_SHARDS = 128
SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf\Z", re.I)
SUPPORTED_SELECTIONS = frozenset({"qwen/Q2_0", "qwen/IQ2_XS", "qwen/IQ3_XXS", "qwen/IQ3_S",
                                   "swift/IQ2_XS", "swift/IQ3_XXS", "coder/IQ1_M",
                                   "unsloth/UD-IQ4_XS", "unsloth/UD-Q4_K_XL", "gyro/Gyro-S", "gyro/Gyro-M"})


def settings_path() -> Path:
    """The user's Strata settings file: the same place setup.py writes it (%APPDATA%/Strata/settings.json, or
    $XDG_CONFIG_HOME/strata/settings.json). The desktop app sets STRATA_SETTINGS_FILE to a file of its own (in its data folder), so it
    never reads an older command-line install's list of Strata folders."""
    return discovery.own_settings_file()


def remembered_installs() -> list[Path]:
    """Earlier Strata folders recorded in the settings file, newest first. Read-only; a missing or unreadable file
    is an empty list."""
    return discovery._installs_in(discovery._read_json(settings_path()))        # noqa: SLF001


def _has_configs(directory: Path) -> bool:
    try:
        return directory.is_dir() and bool(_paths(directory))
    except (ProfileError, OSError):
        return False


def resolve_install(override: str | None = None) -> tuple[Path, str]:
    """(the main install folder, how it was found): explicit (an argument or STRATA_INSTALL_DIR), here (this folder has run
    configs), remembered (an earlier install from the settings file), app (the desktop app's install folder, shared with the
    Terminal) or default (the app's folder, with no model in it yet)."""
    raw = override or os.environ.get("STRATA_INSTALL_DIR")
    if raw is not None:
        path = Path(raw).expanduser()
        if not path.is_absolute() or not path.is_dir():
            raise ProfileError("install directory must be an existing absolute directory", 5)
        return path.resolve(), "explicit"
    return discovery.find_install(here=ROOT)


def all_installs(override: str | None = None) -> list[tuple[Path, str]]:
    """Every folder with run configs, main one first: what `strata models list` shows and what the launch screens read."""
    directory, source = resolve_install(override)
    if source == "explicit":
        return [(directory, source)]
    found = [(p.path, p.source) for p in discovery.places(here=ROOT)]
    return found or [(directory, source)]


def install_dir(override: str | None = None) -> Path:
    return resolve_install(override)[0]


def _model_id(path: Path) -> str | None:
    if not path.name.startswith("strata-") or not path.name.endswith(".json") or \
            path.name.endswith(".shared-settings.json"):
        return None
    value = path.name[len("strata-"):-len(".json")]
    return value if MODEL_ID.fullmatch(value) else None


def _paths(directory: Path) -> list[tuple[str, Path]]:
    try:
        entries = [(name, path) for path in directory.iterdir() if (name := _model_id(path)) is not None]
    except OSError:
        raise ProfileError("cannot read installed model directory", 5) from None
    if len(entries) > MAX_CONFIGS:
        raise ProfileError("more than 128 model configs need inspection", 5)
    return sorted(entries, key=lambda item: item[0])


def _read(path: Path) -> tuple[bytes, dict]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise ProfileError("run config must not be a symlink", 5)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            details = os.fstat(stream.fileno())
            if not stat.S_ISREG(details.st_mode) or details.st_size > MAX_CONFIG_BYTES:
                raise ProfileError("run config must be a regular file under 1 MiB", 5)
            raw = stream.read(MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            raise ProfileError("run config exceeds 1 MiB", 5)
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError("run config is unreadable or invalid JSON", 5) from None
    if not isinstance(value, dict) or not isinstance(value.get("exe"), str) or not value["exe"] or \
            not isinstance(value.get("tokenizer"), str) or not value["tokenizer"] or \
            not isinstance(value.get("args"), list) or \
            not all(isinstance(arg, str) for arg in value["args"]):
        raise ProfileError("run config lacks engine, tokenizer or string arguments", 5)
    return raw, value


def _short(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return "".join(char if char.isprintable() and char not in "\x1b\x7f" else " " for char in value)[:120]


def _arg(args: list[str], flag: str) -> str | None:
    try:
        index = args.index(flag)
        return args[index + 1] if index + 1 < len(args) else None
    except ValueError:
        return None


def _summary(model_id: str, path: Path, raw: bytes, cfg: dict, report: dict) -> dict:
    context = _arg(cfg["args"], "--max-context")
    context_tokens = int(context) if isinstance(context, str) and 1 <= len(context) <= 8 and \
        context.isascii() and context.isdecimal() and 1 <= int(context) <= 16_777_216 else None
    backend = cfg.get("backend")
    return {"id": model_id, "name": _short(cfg.get("model_name")) or model_id,
            "source_config": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "backend": _short(backend) if backend is not None else None,
            "context_tokens": context_tokens,
            "component_status": "present" if report["ok"] else "missing_or_invalid",
            "issue_codes": [item.get("code") for item in report["issues"]][:32],
            "execution_compatibility": "unverified"}


def _inspect_one(model_id: str, path: Path) -> tuple[dict, dict]:
    raw, cfg = _read(path)
    try:
        report = inspect_components(path)
        after, _ = _read(path)
    except (OSError, ValueError):
        raise ProfileError("run config changed or became invalid during inspection", 6) from None
    if after != raw:
        raise ProfileError("run config changed during inspection; retry", 6)
    return _summary(model_id, path, raw, cfg, report), report


def list_installed(override: str | None = None) -> dict:
    directory, source = resolve_install(override)
    places = all_installs(override)
    models = []
    for place, place_source in places:
        try:
            entries = _paths(place) if place.is_dir() else []
        except ProfileError as exc:
            models.append({"id": place.name, "install_dir": str(place), "component_status": "invalid", "error": str(exc),
                           "execution_compatibility": "unverified"})
            continue
        for model_id, path in entries:
            if any(m.get("id") == model_id for m in models):
                continue                                   # the same model in a later place: the main one wins
            try:
                summary, _ = _inspect_one(model_id, path)
            except ProfileError as exc:
                summary = {"id": model_id, "source_config": str(path), "component_status": "invalid",
                           "error": str(exc), "execution_compatibility": "unverified"}
            summary["install_dir"] = str(place)
            summary["install_dir_source"] = place_source
            models.append(summary)
    return {"scope": "local_run_configs", "install_dir": str(directory), "install_dir_source": source,
            "install_dirs": [str(p) for p, _ in places],
            "models": models,
            "model_count": len(models), "discovery_limit": MAX_CONFIGS}


def inspect(model_id: str, override: str | None = None) -> dict:
    if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id):
        raise ProfileError("model ID must be a setup config name without a path")
    directory, path = None, None
    for place, _src in all_installs(override):
        try:
            choices = dict(_paths(place)) if place.is_dir() else {}
        except ProfileError:
            continue
        if model_id in choices:
            directory, path = place, choices[model_id]
            break
    if path is None:
        raise ProfileError(f"model {model_id!r} is not configured in this installation", 5)
    summary, report = _inspect_one(model_id, path)
    return {"scope": "local_run_config", "install_dir": str(directory), "model": summary,
            "components": report}


def _gguf_header(path: Path) -> dict:
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise ProfileError("GGUF shard is a symlink", 5)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            details = os.fstat(stream.fileno())
            if not stat.S_ISREG(details.st_mode) or details.st_size < 8:
                raise ProfileError("GGUF shard is not a nonempty regular file", 5)
            header = stream.read(8)
    except OSError:
        raise ProfileError("GGUF shard is unreadable", 5) from None
    if header[:4] != b"GGUF":
        raise ProfileError("file does not begin with a GGUF header", 5)
    version = int.from_bytes(header[4:8], "little")
    if version not in (2, 3):
        raise ProfileError("GGUF header version is not recognized", 5)
    return {"name": path.name, "path": str(path), "bytes": details.st_size,
            "gguf_version": version}


def _candidate_selection(shards: list[Path]) -> dict | None:
    fingerprint_file = ROOT / "data" / "gguf_fingerprints.json"
    try:
        if fingerprint_file.stat().st_size > 128 * 1024:
            raise ProfileError("model fingerprint catalog exceeds 128 KiB", 5)
        catalog = json.loads(fingerprint_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise ProfileError("model fingerprint catalog is unavailable", 5) from None
    names = {item.name for item in shards}
    for selection in sorted(SUPPORTED_SELECTIONS):
        row = catalog.get(selection) if isinstance(catalog, dict) else None
        expected = row.get("files") if isinstance(row, dict) else None
        if isinstance(expected, list) and expected and names == {Path(name).name for name in expected}:
            family, model = selection.split("/", 1)
            expected_experts = row.get("expert_bytes")
            return {"family": family, "model": model, "basis": "known_shard_names_only",
                    "expected_expert_types": sorted(expected_experts) if isinstance(expected_experts, dict) else [],
                    "expected_tensor_count": row.get("tensors") if type(row.get("tensors")) is int else None,
                    "setup_args_hint": ["--family", family, "--model", model,
                                        "--gguf-dir", str(shards[0].parent)]}
    return None


def _candidate_by_header(shards: list[Path], tensors: int, payload: int, experts: set[str]) -> dict | None:
    """A file set whose NAME setup does not know (renamed, or downloaded elsewhere) but whose header is exactly a known one: the same tensor
    count, the same payload size and the same expert formats as a row of the fingerprint table."""
    try:
        catalog = json.loads((ROOT / "data" / "gguf_fingerprints.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    for selection in sorted(SUPPORTED_SELECTIONS):
        row = catalog.get(selection) if isinstance(catalog, dict) else None
        if not isinstance(row, dict) or row.get("tensors") != tensors or row.get("bytes") != payload:
            continue
        expected = row.get("expert_bytes")
        if not isinstance(expected, dict) or set(expected) != experts:
            continue
        family, model = selection.split("/", 1)
        return {"family": family, "model": model, "basis": "header_size_and_tensors",
                "expected_expert_types": sorted(expected), "expected_tensor_count": tensors,
                "setup_args_hint": ["--family", family, "--model", model, "--gguf-dir", str(shards[0].parent)]}
    return None


def _already_set_up(shards: list[Path], candidate: dict | None) -> dict | None:
    """Is this model already installed (a run config of any known install names the file) and what was prepared beside it (pack, draft
    layer, picture reader)? Read-only; None when the file set is not one Strata knows."""
    if not shards or candidate is None:
        return None
    first = shards[0]
    cfg = discovery.run_config_for(first, discovery.places(here=ROOT))
    beside = discovery.prepared_beside(first)
    if cfg is None and not any(beside.get(k) for k in ("pack", "mtp", "mmproj")):
        return {"config": None, "data_dir": str(beside["data_dir"]) if beside.get("data_dir") else None, "pack": None, "mtp": None, "mmproj": None}
    return {"config": str(cfg) if cfg else None, **{k: (str(v) if v else None) for k, v in beside.items()}}


UNSUPPORTED_MODEL_TYPE = ("This model type isn't supported by the Strata engine. "
                          "Add it as a Custom model API (llama-server) instead.")
# The picture readers Strata ships a pinned copy of (setup downloads and checks one of these next to the model).
# (name, size in bytes) as setup.py pins them: setup would delete a file of that name with another size, so only an exact
# match is called "the official one".
PINNED_PROJECTORS = {"mmproj-qwen3.8-flash-next-bf16.gguf": 907543008, "mmproj-f16.gguf": 899283584}
OUR_PROJECTOR_TYPES = {"qwen3vl_merger"}


def _assess_projector(path: Path) -> dict:
    """What a picture-reader file (mmproj*.gguf) next to a model is, in words; never blocks the import."""
    try:
        size = path.stat().st_size
    except OSError:
        size = -1
    row = {"name": path.name, "status": "unreadable", "pinned": PINNED_PROJECTORS.get(path.name.lower()) == size}
    try:
        info = inspect_metadata(path)
    except ProfileError:
        row["plain"] = f"{path.name} could not be read as a picture reader, so it is not used."
        return row
    row["architecture"], row["projector_type"] = info["architecture"], info.get("projector_type")
    if info["architecture"] != "clip":
        row["status"] = "not_a_projector"
        row["plain"] = f"{path.name} is not a picture reader (mmproj) file, so it is not used."
    elif info.get("projector_type") not in OUR_PROJECTOR_TYPES:
        row["status"] = "unsupported"
        row["plain"] = (f"{path.name} reads pictures for a different kind of model. "
                        "Strata cannot use it; add that model as a Custom model API (llama-server) instead.")
    elif row["pinned"]:
        row["status"] = "ok"
        row["plain"] = f"{path.name} is the picture reader Strata uses for this model family; pictures will work."
    else:
        row["status"] = "other_name"
        row["plain"] = (f"{path.name} is a picture reader of the right kind but not the file Strata checks; "
                        "Strata uses the official one (mmproj-Qwen3.8-Flash-Next-BF16.gguf) and "
                        "downloads it when it is not in the folder.")
    return row


def plan_import(raw_path: str) -> dict:
    """Inspect a local candidate without copying, parsing tensors or adopting it."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ProfileError("model import path must be absolute")
    try:
        details = path.lstat()
    except OSError:
        raise ProfileError("model import path does not exist", 5) from None
    if stat.S_ISLNK(details.st_mode):
        raise ProfileError("model import path must not be a symlink", 5)
    if stat.S_ISDIR(details.st_mode):
        try:
            shards = []
            ignored_auxiliary = []
            for child in path.iterdir():
                if child.name.lower().endswith(".gguf"):
                    if child.name.lower().startswith("mmproj"):
                        ignored_auxiliary.append(child.name)
                        continue
                    shards.append(child)
                    if len(shards) > MAX_IMPORT_SHARDS:
                        raise ProfileError("more than 128 GGUF candidates in the directory", 5)
        except OSError:
            raise ProfileError("cannot inspect GGUF directory", 5) from None
    elif stat.S_ISREG(details.st_mode) and path.name.lower().endswith(".gguf"):
        ignored_auxiliary = []
        try:                       # the picture reader that sits next to a chosen model file
            ignored_auxiliary = [c.name for c in path.parent.iterdir()
                                 if c.name.lower().startswith("mmproj") and c.name.lower().endswith(".gguf")][:8]
        except OSError:
            pass
        if path.name.lower().startswith("mmproj"):
            raise ProfileError("this is a picture reader (mmproj), not a model. Choose the model's own "
                               ".gguf file; Strata finds the reader next to it", 5)
        match = SHARD.fullmatch(path.name)
        if match:
            total = int(match.group(3))
            if not 1 <= total <= MAX_IMPORT_SHARDS:
                raise ProfileError("GGUF shard count must be in 1..128", 5)
            shards = [path.with_name(f"{match.group(1)}-{i:05d}-of-{total:05d}.gguf")
                      for i in range(1, total + 1)]
        else:
            shards = [path]
    else:
        raise ProfileError("model import needs a GGUF file or directory", 5)
    if not shards:
        raise ProfileError("no GGUF files found at the model import path", 5)
    shards.sort(key=lambda item: item.name)
    groups = {}
    for item in shards:
        match = SHARD.fullmatch(item.name)
        if match:
            key = (match.group(1), int(match.group(3)))
            groups.setdefault(key, set()).add(int(match.group(2)))
        else:
            groups.setdefault((item.name, 1), set()).add(1)
    issues = []
    folder_holds = []
    if len(groups) != 1:
        issues.append("multiple_model_groups")
        # a folder with several models: say which ones Strata knows, so the person can point at one of them
        for key in sorted(groups):
            members = [item for item in shards if (SHARD.fullmatch(item.name) and (SHARD.fullmatch(item.name).group(1),
                       int(SHARD.fullmatch(item.name).group(3))) == key) or (not SHARD.fullmatch(item.name) and (item.name, 1) == key)]
            try:
                known = _candidate_selection(members)
            except ProfileError:
                known = None
            folder_holds.append({"file": members[0].name if members else key[0], "files": len(members),
                                 "family": known["family"] if known else None, "model": known["model"] if known else None})
    if len(groups) == 1:
        (_, total), indices = next(iter(groups.items()))
        if total > MAX_IMPORT_SHARDS or indices != set(range(1, total + 1)):
            issues.append("incomplete_or_invalid_shard_set")
    files = []
    for item in shards:
        try:
            files.append(_gguf_header(item))
        except ProfileError as exc:
            issues.append(f"{item.name}: {exc}")
    metadata = None
    observed_experts: set[str] = set()
    observed_tensor_count = 0
    observed_payload = 0
    if files and not issues:
        for item in shards:
            try:
                inspected = inspect_metadata(item)
            except ProfileError as exc:
                issues.append(f"{item.name}: {exc}")
                continue
            if metadata is None:
                metadata = inspected
            observed_experts.update(inspected["expert_tensor_types"])
            observed_tensor_count += inspected["tensor_count_scanned"]
            observed_payload += int(inspected.get("payload_bytes_required") or 0)
            if inspected["architecture"] is not None and inspected["architecture"] != "qwen4exp":
                issues.append(f"{item.name}: unsupported_architecture")
    if metadata is not None:
        if metadata["architecture"] != "qwen4exp":
            issues.append("unsupported_or_missing_architecture")
        if metadata["tokenizer_model"] != "gpt2" or not metadata["tokenizer_tokens_present"] or \
                not metadata["tokenizer_merges_present"]:
            issues.append("unsupported_or_incomplete_tokenizer_metadata")
    projectors = [_assess_projector(path.parent / n if path.is_file() else path / n)
                  for n in sorted(ignored_auxiliary)[:8]]
    messages = []
    if any(i.endswith("unsupported_architecture") or i == "unsupported_or_missing_architecture" for i in issues):
        messages.append(UNSUPPORTED_MODEL_TYPE)
    elif projectors:
        messages.extend(p["plain"] for p in projectors)
    candidate = _candidate_selection(shards)
    if candidate is None and metadata is not None and not issues:
        candidate = _candidate_by_header(shards, observed_tensor_count, observed_payload, observed_experts)
    if candidate is None:
        issues.append("unrecognized_or_unsupported_model_variant")
    elif metadata is not None:
        expected = set(candidate["expected_expert_types"])
        if not observed_experts or observed_experts - expected:
            issues.append("missing_or_unexpected_expert_quantization")
    return {"source": str(path), "scope": "local_gguf_candidate", "files": files,
            "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files),
            "metadata": metadata, "candidate_selection": candidate,
            "observed_expert_types": sorted(observed_experts),
            "observed_tensor_count": observed_tensor_count,
            "ignored_auxiliary_files": sorted(ignored_auxiliary),
            "projectors": projectors, "messages": messages, "already_set_up": _already_set_up(shards, candidate),
            "reads_pictures": any(p["status"] in ("ok", "other_name") for p in projectors) and
            not any(i.endswith("unsupported_architecture") for i in issues),
            "folder_holds": folder_holds,
            "file_set_status": "plausible" if not issues else "invalid_or_incomplete",
            "issues": issues, "adoption_status": "requires_architecture_quantization_tokenizer_and_engine_checks",
            "execution_compatibility": "unverified", "would_write": False,
            "notes": ["A known shard name is only a selection hint; it does not prove provenance or file integrity",
                      "Tensor directory, model-pack, engine version and hardware checks remain before adoption",
                      "No model is adopted or copied by this preview"]}
