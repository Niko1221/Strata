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


def schema_to_grammar(exe_dir, schema: dict, *, tool_value=False, forbidden_sequence=None) -> str:
    """Convert a schema. Tool-value mode preserves the template's unquoted root strings and rejects
    unsupported string constraints instead of allowing a prompt-path fallback to weaken them."""
    key = schema_text(schema)
    cache_key = (key, bool(tool_value), forbidden_sequence or "")
    with _lock:
        if cache_key in _cache:
            _cache.move_to_end(cache_key)
            return _cache[cache_key]
    exe = converter_exe(exe_dir) if exe_dir is not None else None
    if exe is None or not exe.exists():
        raise ValueError(f"the schema converter is not installed next to the engine ({exe})")
    args = [str(exe)]
    if tool_value:
        args.append("--tool-value")
    if forbidden_sequence:
        args.extend(("--forbid-sequence", forbidden_sequence))
    try:
        proc = subprocess.run(args, input=key, capture_output=True, text=True,
                              encoding="utf-8", timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise ValueError(f"schema converter failed to run: {e}") from e
    if proc.returncode != 0:
        raise ValueError("schema conversion failed: " + ((proc.stderr or "").strip()[:200] or f"exit {proc.returncode}"))
    text = proc.stdout
    with _lock:
        _cache[cache_key] = text
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


def plan_response_format(engine, response_format, messages, with_tools=False, allow=True):
    """(grammar text, prompt messages, validator) for a response_format.

    The prompt path (no grammar capability, a turn with tools, text kind, plain-text format, or a
    converter that fails): prepare_format's system directive rides along and the whole answer is
    buffered and validated.  The grammar path: the messages stay untouched (a constrained model needs
    no formatting lecture; the test suite pins that the directive is absent), and the validator is
    kept only as the end-of-stream canary (grammar.grammar_canary).  Tools + a schema stay on the
    prompt path: a tool call is not JSON, and the tool-call grammar (serve/grammar_tools.py) is the
    constraint that fits that turn.

    `allow` is the caller's per-request opt-out (`strata_grammar: false`): the prompt path then takes
    the turn even though the engine could constrain it.  It exists so a caller can compare the two
    paths on the same request, and so an operator can turn the constraint off for a prompt the
    converter's grammar does not fit.
    """
    original = messages
    grammar_text = None
    if response_format is not None and allow and grammar_ok(engine) and not with_tools:
        kind = response_format.get("type") if isinstance(response_format, dict) else None
        if kind == "json_object":
            grammar_text = grammar_api.JSON_OBJECT_GBNF
        elif kind == "json_schema":
            spec = response_format.get("json_schema")
            if isinstance(spec, dict) and isinstance(spec.get("schema"), dict):
                grammar_text = schema_to_grammar(exe_dir_of(engine), spec["schema"])
    messages, validator = prepare_format(response_format, messages, with_tools=with_tools)   # shape checks + validator
    if grammar_text is not None:
        messages = original                    # the constraint rides in the sampler, not in the prompt
    return grammar_text, messages, validator
