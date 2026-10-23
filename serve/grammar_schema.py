"""JSON Schema → GBNF for the grammar path: the vendored llama.cpp converter (the grammar_schema CLI
that ships beside strata.exe) behind a small LRU.  A schema converts once; the GBNF text is hashed
(serve/grammar.py) and registered with the engine once per engine generation."""
import json
import os
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path

from serve import grammar as grammar_api
from serve.structured import prepare_format

CACHE_CAP = 64
_cache: OrderedDict = OrderedDict()
_lock = threading.Lock()


def schema_text(schema: dict) -> str:
    """The cache key: a canonical JSON dump, so equal schemas convert once."""
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


def converter_exe(exe_dir) -> Path:
    name = "grammar_schema.exe" if os.name == "nt" else "grammar_schema"
    return Path(exe_dir) / name


def schema_to_grammar(exe_dir, schema: dict) -> str:
    """Convert with the grammar_schema CLI.  Raises ValueError on any failure (no exe, schema the
    converter refuses); the caller decides the fallback."""
    key = schema_text(schema)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    exe = converter_exe(exe_dir) if exe_dir is not None else None
    if exe is None or not exe.exists():
        raise ValueError(f"the schema converter is not installed next to the engine ({exe})")
    try:
        proc = subprocess.run([str(exe)], input=key, capture_output=True, text=True,
                              encoding="utf-8", timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise ValueError(f"schema converter failed to run: {e}") from e
    if proc.returncode != 0:
        raise ValueError("schema conversion failed: " + ((proc.stderr or "").strip()[:200] or f"exit {proc.returncode}"))
    text = proc.stdout
    with _lock:
        _cache[key] = text
        if len(_cache) > CACHE_CAP:
            _cache.popitem(last=False)
    return text


def grammar_ok(engine) -> bool:
    """The engine said it can constrain decoding (INFO grammar=1, 0.2.3+).  Fail-closed: an engine
    that never answered the question is treated as unable, exactly like a missing logit_bias."""
    return (getattr(engine, "info", None) or {}).get("grammar") == 1


def exe_dir_of(engine):
    spawn = getattr(engine, "spawn", None)
    return Path(spawn[0]).parent if spawn else None


def plan_response_format(engine, response_format, messages, with_tools=False):
    """(grammar text, prompt messages, validator) for a response_format.

    The prompt path (no grammar capability, a turn with tools, text kind, plain-text format, or a
    converter that fails): prepare_format's system directive rides along and the whole answer is
    buffered and validated.  The grammar path: the messages stay untouched (a constrained model needs
    no formatting lecture; the test suite pins that the directive is absent), and the validator is
    kept only as the end-of-stream canary (grammar.grammar_canary).  Tools + a schema stay on the
    prompt path: a tool call is not JSON, and the tool-call grammar (serve/grammar_tools.py) is the
    constraint that fits that turn.
    """
    original = messages
    grammar_text = None
    if response_format is not None and grammar_ok(engine) and not with_tools:
        kind = response_format.get("type") if isinstance(response_format, dict) else None
        if kind == "json_object":
            grammar_text = grammar_api.JSON_OBJECT_GBNF
        elif kind == "json_schema":
            spec = response_format.get("json_schema")
            if isinstance(spec, dict) and isinstance(spec.get("schema"), dict):
                try:
                    grammar_text = schema_to_grammar(exe_dir_of(engine), spec["schema"])
                except ValueError:
                    grammar_text = None       # the converter refused: the prompt path knows what to do
    messages, validator = prepare_format(response_format, messages, with_tools=with_tools)   # shape checks + validator
    if grammar_text is not None:
        messages = original                    # the constraint rides in the sampler, not in the prompt
    return grammar_text, messages, validator
