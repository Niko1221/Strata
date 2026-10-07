"""A YAML launch file layered over a model's setup-generated JSON config.

No hardware detection, downloads or installation happen here. The generated config
keeps the model paths and hardware choices; the YAML file owns everyday settings.
"""
from __future__ import annotations

import ipaddress
import json
import math
import os
import re
from pathlib import Path

from serve import runconfig

ROOT = Path(__file__).resolve().parents[1]
FLAGS = {"context": "--max-context", "kv": "--kv", "vram_reserve_mib": "--vram-reserve-mib"}
SERVER_KEYS = {key for key in runconfig.SPEC if "." not in key} - {"vram_reserve_mib"}
KEYS = SERVER_KEYS | set(FLAGS) | {"model", "config", "host", "port", "api_key", "api_key_env", "sampling"}


def installed(root: Path = ROOT) -> dict[str, Path]:
    """Only model configs, excluding shared settings and other strata-*.json files."""
    found = {}
    for path in sorted(root.glob("strata-*.json")):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(cfg, dict) and isinstance(cfg.get("exe"), str) and isinstance(cfg.get("args"), list):
                found[path.stem[len("strata-"):]] = path
        except (OSError, ValueError):
            continue
    return found


def read(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise ValueError("YAML needs PyYAML in the setup environment; run ./setup.sh first") from None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError:
        # Parser diagnostics can include the line containing an API key.
        raise ValueError(f"{path.name}: invalid YAML; check indentation and quoting") from None
    if not isinstance(raw, dict) or not all(isinstance(k, str) for k in raw):
        raise ValueError(f"{path.name}: expected a YAML mapping with setting names")
    unknown = raw.keys() - KEYS
    if unknown:
        raise ValueError(f"{path.name}: unknown settings: {', '.join(sorted(unknown))}")
    return raw


def source(raw: dict, path: Path, root: Path = ROOT) -> Path:
    if ("model" in raw) == ("config" in raw):
        raise ValueError("set exactly one of model (an installed name) or config (a setup JSON file)")
    if "config" in raw:
        value = raw["config"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("config must be a path to a setup JSON file")
        base = Path(value).expanduser()
        base = base if base.is_absolute() else path.parent / base
        if base.suffix.lower() != ".json":
            raise ValueError("config must point to a setup JSON file")
        return base.resolve()
    model = raw["model"]
    if not isinstance(model, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", model):
        raise ValueError("model must be an installed name from make models")
    models = installed(root)
    if model.lower() not in models:
        names = ", ".join(models) or "none; run ./setup.sh first"
        raise ValueError(f"model {model!r} is not installed; installed models: {names}")
    return models[model.lower()].resolve()


def set_arg(args: list[str], flag: str, value) -> list[str]:
    """Replace every occurrence, including --flag=value, so no stale value wins."""
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 2
        elif args[i].startswith(flag + "="):
            i += 1
        else:
            out.append(args[i])
            i += 1
    if value is not None:
        out += [flag, str(value)]
    return out


def arg_value(args: list[str], flag: str):
    value = None
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
    return value


def network(host, key):
    if not isinstance(host, str) or not host.strip() or host != host.strip():
        raise ValueError("host must be an address, such as 127.0.0.1 or 0.0.0.0")
    if not isinstance(key, str):
        raise ValueError("api_key must be a string")
    if ":" in host:
        raise ValueError("IPv6 hosts are not supported by the server listener; use an IPv4 address or hostname")
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host == "localhost"
    if not local and not key.strip():
        raise ValueError("LAN access requires an API key: set STRATA_API_KEY or api_key in the launch file")
    if key and not key.strip():
        raise ValueError("api_key must not contain only whitespace")


def load(path: str | Path, root: Path = ROOT, *, validate_network=True) -> dict:
    path = Path(path).resolve()
    raw = read(path)
    base = source(raw, path, root)
    cfg = json.loads(base.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("exe"), str) or \
            not isinstance(cfg.get("args"), list) or not all(isinstance(x, str) for x in cfg["args"]):
        raise ValueError(f"{base.name}: expected a setup config with exe and args")
    # Setup uses absolute paths. Older or hand-written configs may use paths relative to cwd.
    cwd = Path(cfg.get("cwd") or base.parent)
    cfg["cwd"] = str((base.parent / cwd).resolve() if not cwd.is_absolute() else cwd)
    if cfg.get("tokenizer") and not Path(cfg["tokenizer"]).is_absolute():
        cfg["tokenizer"] = str(Path(cfg["cwd"]) / cfg["tokenizer"])

    for key in SERVER_KEYS & raw.keys():
        value = runconfig.check(key, raw[key], cfg)
        if value is None:
            cfg.pop(key, None)
        else:
            cfg[key] = value
    if "sampling" in raw:
        sampling = raw["sampling"]
        if not isinstance(sampling, dict):
            raise ValueError("sampling must be a mapping")
        changes = {f"sampling.{k}": v for k, v in sampling.items()}
        if changes:
            cfg, _ = runconfig.apply(cfg, changes)

    for key, flag in FLAGS.items():
        if key not in raw:
            continue
        value = raw[key]
        if key == "context":
            if type(value) is not int or not 1 <= value < 2**31:
                raise ValueError("context must be a whole number from 1 to 2147483647")
            # Setup owns experimental RoPE choices. Reuse a configured extension only within its range.
            if value > 262144:
                method = arg_value(cfg["args"], "--rope-scaling")
                factor = float(arg_value(cfg["args"], "--rope-scale") or 1)
                if method not in ("linear", "yarn") or not math.isfinite(factor) or not value <= 262144 * factor:
                    raise ValueError("context above 262144 needs a covering RoPE configuration; run setup with "
                                     "--setup --context <tokens> --no-start first")
        elif key == "kv":
            if value not in ("fp16", "int8", "q4_0", "k8v4"):
                raise ValueError("kv must be fp16, int8, q4_0 or k8v4")
        else:
            value = runconfig.check(key, value, cfg)
        cfg["args"] = set_arg(cfg["args"], flag, value)
    # Starting locally is always the default, even if the installed model once served the LAN.
    cfg["host"] = raw.get("host", "127.0.0.1")
    port = raw.get("port", cfg.get("port", 8080))
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be a whole number from 1 to 65535")
    cfg["port"] = port
    key_env = raw.get("api_key_env", "STRATA_API_KEY")
    if not isinstance(key_env, str) or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", key_env):
        raise ValueError("api_key_env must name an environment variable")
    if "api_key_env" in raw and not os.environ.get(key_env, "").strip():
        raise ValueError(f"set the {key_env} environment variable before starting Strata")
    if key_env in os.environ and not os.environ[key_env].strip():
        raise ValueError(f"{key_env} is empty; set a key or unset the variable for a local launch")
    cfg["api_key"] = os.environ.get(key_env) or raw.get("api_key", cfg.get("api_key", ""))
    if validate_network:
        network(cfg["host"], cfg["api_key"])
    return cfg


def settings_text(path: Path, cfg: dict) -> str:
    """Save web Settings to the overlay, keeping model selection and inherited paths intact."""
    import yaml

    raw, before = read(path), load(path, validate_network=False)
    for key in runconfig.SPEC:
        value = runconfig.value_of(cfg, key)
        if value == runconfig.value_of(before, key):
            continue
        if key.startswith("sampling."):
            raw.setdefault("sampling", {})[key.split(".", 1)[1]] = value
        else:
            raw[key] = value
    return yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
