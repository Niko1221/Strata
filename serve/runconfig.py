"""serve/runconfig.py - #564: the web page's Settings view of the run config (strata-<model>.json).

A short list of documented keys can be read and changed from Strata's own page (GET / POST /config).  Everything
else in the file - the keys setup writes, the network and security keys (host, api_key, cors_origins,
trusted_origins, allowed_hosts), the MCP servers and before_load (which run programs), sampling keys not listed
here - is kept as it is: a change touches only the keys it names (as setup run again keeps the user's keys, #629).
The earlier file is kept as strata-<model>.json.bak.  The server reads the config when it starts, so a change is
used from the next start on.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

# (key, kind, help).  kind: "bool", "int>=0", "num>=0", ("enum", values), "names", or ("sampling", check) for a key
# of the "sampling" block, ("arg", flag) for an engine option kept in "args".
EDITABLE = [
    ("sampling.temperature", ("sampling", "num>=0"),
     "Default temperature for requests that send none (0 = greedy, the default without a sampling block)"),
    ("sampling.top_p", ("sampling", "0<x<=1"), "Default top_p for requests that send none"),
    ("sampling.top_k", ("sampling", "1..64"), "Default top_k for requests that send none (1-64)"),
    ("sampling.min_p", ("sampling", "0<=x<=1"), "Default min_p for requests that send none"),
    ("reasoning_budget_tokens", "int>=0", "Cap the thinking of every request at this many tokens (0 or empty: no cap)"),
    ("fit_max_tokens", "bool", "Shorten a max_tokens that does not fit the context instead of answering 400"),
    ("anthropic_thinking", ("enum", ["model", "on_request"]),
     "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)"),
    ("effort_position", ("enum", ["start", "end"]),
     "Where a non-default reasoning effort goes: start (the default) or end (keeps the cache when it changes)"),
    ("aliases", "names", "Other model names the server lists and answers to (comma-separated)"),
    ("idle_unload_s", "num>=0", "Unload the model after this many seconds without requests (0 or empty: never)"),
    ("lazy_load", "bool", "Start without loading the model; the first request loads it (text only)"),
    ("engine_silence_s", "num>=0", "End a request when the engine says nothing for this long (default 300 s, 0 = wait)"),
    ("api_monitor", "bool", "Keep the last 100 requests' prompts and answers in memory for /api-monitor"),
    ("open_browser", "bool", "Open the chat page in the browser when the model is ready"),
    ("vram_reserve_mib", ("arg", "--vram-reserve-mib"),
     "VRAM in MiB the engine leaves free for other programs (engine default 700)"),
    # the on-disk layer (off unless set here): the conversation cache's disk tier and the system prompt cache
    ("conversation_cache_spill_dir", ("arg", "--conversation-cache-spill-dir", "str"),
     "Folder the RAM cache writes the conversations it evicts to (giving it turns the disk tier on)"),
    ("conversation_cache_disk_mib", ("arg", "--conversation-cache-disk-mib"),
     "The spill folder's limit in MiB (engine default 8192; 0 = off)"),
    ("conversation_cache_spill_when_full", ("arg", "--conversation-cache-spill-when-full",
                                            ("enum", ["evict-oldest", "reject"])),
     "When the spill folder is full: evict-oldest (the default) or reject (keep it all, save nothing new)"),
    ("conversation_cache_spill_max_age_days", ("arg", "--conversation-cache-spill-max-age-days"),
     "Delete spilled conversations older than this many days (0 or empty: never)"),
    ("system_prompt_cache", ("flag", "--system-prompt-cache"),
     "Keep the system prompt's prefill checkpoint on disk and reuse it after a restart"),
    ("system_prompt_cache_dir", ("arg", "--system-prompt-cache-dir", "str"),
     "The system prompt cache's own folder (not the spill folder)"),
    ("system_prompt_cache_mib", ("arg", "--system-prompt-cache-mib"),
     "The system prompt cache folder's limit in MiB (engine default 2048)"),
    ("system_prompt_cache_slots", ("arg", "--system-prompt-cache-slots"),
     "How many system prompt variants are kept (engine default 2; 0 = no cap)"),
    ("system_prompt_cache_max_age_days", ("arg", "--system-prompt-cache-max-age-days"),
     "Delete system prompt variants older than this many days (0 or empty: never)"),
    ("system_prompt_cache_key", ("arg", "--system-prompt-cache-key", "str"),
     "Optional identity added to the system prompt's cache key"),
]
SPEC = {k: kind for k, kind, _ in EDITABLE}


def _arg(cfg: dict, flag: str):
    a = cfg.get("args") if isinstance(cfg.get("args"), list) else []
    if flag in a[:-1]:
        v = str(a[a.index(flag) + 1])
        return int(v) if v.isdigit() else v
    return None


def _arg_raw(cfg: dict, flag: str):
    """The value after `flag` as written (a folder, an enum name), or None."""
    a = cfg.get("args") if isinstance(cfg.get("args"), list) else []
    return str(a[a.index(flag) + 1]) if flag in a[:-1] else None


def _has_flag(cfg: dict, flag: str) -> bool:
    """Whether a bare engine flag (no value of its own) is in the config's args."""
    a = cfg.get("args") if isinstance(cfg.get("args"), list) else []
    return flag in a


def _arg_kind(kind: tuple):
    """What an ("arg", flag[, spec]) key carries: "num" (the default), "str", or ("enum", values)."""
    return kind[2] if len(kind) > 2 else "num"


def _frontend_kind(kind):
    """What the page's Settings view needs for a key: its input kind and its choices (if any)."""
    k = kind[0] if isinstance(kind, tuple) else kind
    if k == "flag":
        return "bool", {}
    if k == "arg":
        spec = _arg_kind(kind)
        if isinstance(spec, tuple) and spec[0] == "enum":
            return "enum", {"choices": spec[1]}
        return ("string" if spec == "str" else "number"), {}
    if k == "enum":
        return "enum", {"choices": kind[1]}
    return ("number" if k in ("sampling", "int>=0", "num>=0") else k), {}


def value_of(cfg: dict, key: str):
    kind = SPEC[key]
    if isinstance(kind, tuple) and kind[0] == "sampling":
        s = cfg.get("sampling")
        return s.get(key.split(".", 1)[1]) if isinstance(s, dict) else None
    if isinstance(kind, tuple) and kind[0] == "arg":
        return _arg(cfg, kind[1]) if _arg_kind(kind) == "num" else _arg_raw(cfg, kind[1])
    if isinstance(kind, tuple) and kind[0] == "flag":
        return _has_flag(cfg, kind[1])
    return cfg.get(key)


def view(cfg: dict, path: str | Path) -> dict:
    """GET /config: the editable keys with their values (None: not set, the default applies)."""
    out = []
    for key, kind, help_ in EDITABLE:
        front, extra = _frontend_kind(kind)
        out.append({"key": key, "value": value_of(cfg, key), "help": help_, "kind": front, **extra})
    return {"file": Path(path).name, "keys": out,
            "note": "Saved to the run config; used from the next start of the model."}


def _number(key, v, whole=False, lo=0.0, hi=None, lo_open=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (whole and v != int(v)) or \
            (v <= lo if lo_open else v < lo) or (hi is not None and v > hi):
        rng = f"{'more than' if lo_open else 'at least'} {lo:g}" + (f" and at most {hi:g}" if hi is not None else "")
        raise ValueError(f"{key}: expected a {'whole ' if whole else ''}number, {rng}, not {v!r}")
    return int(v) if whole else v


def check(key: str, v, cfg: dict):
    """The value to store for `key`, or a ValueError naming what is expected.  None removes the key (its default)."""
    if key not in SPEC:
        raise ValueError(f"{key!r} cannot be changed here (only the keys the Settings view lists)")
    if v is None:
        return None
    kind = SPEC[key]
    k = kind[0] if isinstance(kind, tuple) else kind
    if k == "bool":
        if not isinstance(v, bool):
            raise ValueError(f"{key}: expected true or false, not {v!r}")
        if key == "lazy_load" and v and cfg.get("vision"):
            raise ValueError("lazy_load: lazy loading is text-only, and this model reads images (\"vision\")")
        return v
    if k == "flag":
        if not isinstance(v, bool):
            raise ValueError(f"{key}: expected true or false, not {v!r}")
        return v
    if k == "enum":
        if v not in kind[1]:
            raise ValueError(f"{key}: expected one of {', '.join(kind[1])}, not {v!r}")
        return v
    if k == "names":
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() and len(x) <= 128 for x in v):
            raise ValueError(f"{key}: expected a list of model names, not {v!r}")
        return v or None
    if k == "int>=0":
        return _number(key, v, whole=True)
    if k == "num>=0":
        return _number(key, v)
    if k == "arg":
        spec = _arg_kind(kind)
        if spec == "str":
            if not isinstance(v, str) or not v.strip() or any(ord(c) < 32 for c in v):
                raise ValueError(f"{key}: expected a non-empty value, not {v!r}")
            return v.strip()
        if isinstance(spec, tuple) and spec[0] == "enum":
            if v not in spec[1]:
                raise ValueError(f"{key}: expected one of {', '.join(spec[1])}, not {v!r}")
            return v
        return _number(key, v, whole=True)
    rule = kind[1]                                     # a sampling key: the server's own rules
    if rule == "num>=0":
        return float(_number(key, v))
    if rule == "0<x<=1":
        return float(_number(key, v, hi=1, lo_open=True))
    if rule == "0<=x<=1":
        return float(_number(key, v, hi=1))
    return _number(key, v, whole=True, lo=1, hi=64)


def apply(cfg: dict, changes: dict) -> tuple[dict, list[str]]:
    """The config with `changes` ({key: value or None}) made, and the keys that changed.  Every value is checked
    first, so a request with one bad value changes nothing (ValueError)."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError('send {"set": {"<key>": <value or null>, ...}}')
    checked = {k: check(k, v, cfg) for k, v in changes.items()}
    new = json.loads(json.dumps(cfg))                  # a deep copy: the caller's dict is left alone
    changed = []
    for key, v in checked.items():
        if value_of(new, key) == v:
            continue
        kind = SPEC[key]
        if isinstance(kind, tuple) and kind[0] == "sampling":
            s = dict(new.get("sampling") or {}) if isinstance(new.get("sampling"), dict) else {}
            sub = key.split(".", 1)[1]
            if v is None:
                s.pop(sub, None)
            else:
                s[sub] = v
            if s:
                new["sampling"] = s
            else:
                new.pop("sampling", None)
        elif isinstance(kind, tuple) and kind[0] == "arg":
            a = list(new.get("args") or [])
            flag = kind[1]
            if flag in a[:-1]:
                i = a.index(flag)
                if v is None:
                    del a[i:i + 2]
                else:
                    a[i + 1] = str(v)
            elif v is not None:
                a += [flag, str(v)]
            new["args"] = a
        elif isinstance(kind, tuple) and kind[0] == "flag":
            a = list(new.get("args") or [])
            flag = kind[1]
            if flag in a:
                if not v:
                    a.remove(flag)
            elif v:
                a.append(flag)
            new["args"] = a
        elif v is None:
            new.pop(key, None)
        else:
            new[key] = v
        changed.append(key)
    return new, changed


def save(path: str | Path, cfg: dict) -> Path:
    """The config written whole (a temporary file moved over the old one), the earlier one kept as <name>.bak."""
    path = Path(path)
    bak = path.with_name(path.name + ".bak")
    shutil.copyfile(path, bak)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return bak


def load(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{Path(path).name} is not a JSON object")
    return cfg
