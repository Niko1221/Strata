"""Constrained tool-name decoding: the engine-side wire format.

A model that writes `<tool_call><function=NAME>` (or `<tool_call><invoke name="NAME">`) can
hallucinate a NAME no declared tool has (`TaskCreate`, `execute_bash`); the parser drops it and
the raw markup leaks into the reply.  The engine can remove the possibility at the sampler: while
the reply is *inside* a `<tool_call>` envelope (and past the `<think>` reasoning block), the
tokens after a name opener are masked so only a declared name can follow.

The engine streams token ids and has no detokenizer, so this module turns the declared names
and the model's structural markers into ids.  The wire format (see
`engine/include/strata/core/tool_name_constraint.hpp`) is:

    v2|<initial>|<think_open>|<think_close>|<call_open>|<call_close>|#<trigger>|<tail>|...[|#<trigger>|<tail>|...]

`<initial>` is `0`/`1` (whether the generation prompt opens `<think>`).  The four markers are the
ids of `<think>`, `</think>`, `<tool_call>` and `</tool_call>` (the reasoning pair may be empty to
disable reasoning tracking).  Then one group per name form, each starting with `#`: `<trigger>` is
the token ids of the shared prefix of every `OPENER + NAME + CLOSER` sequence of that form, computed
as the longest common prefix (with the bare opener too) so a BPE tokenizer that merges the opener's
last character into the name's first token still matches; each `<tail>` is the remainder of one
sequence after that prefix (the name and its closer).  Every record is a comma-separated list of
ids, records joined by `|`.

Two forms are sent: `<function=NAME>` and `<invoke name="NAME">`.  The `<function=` form also
admits the reserved name `function_calls`, the wrapper some models write before an `<invoke>`
(`<tool_call><function=function_calls><invoke name="bash">`); the invoke form then constrains the
real name.  When a BPE trigger stops at `<function` (Qwen merges `=` into the name), the
`<function_calls>` tag shares it, so it is admitted as a tail too.

`build()` returns `None` (and nothing is sent) when a request declares no tools, or the model's
vocabulary lacks the markers.  A request without the key pays nothing in the engine.
"""
from __future__ import annotations

VERSION = "v2"
_TRIGGER = "<function="
_INVOKE = '<invoke name="'
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_CALL_OPEN = "<tool_call>"
_CALL_CLOSE = "</tool_call>"
WRAPPER_NAME = "function_calls"
# (opener, closer) per name form
FORMS = ((_TRIGGER, ">"), (_INVOKE, '">'))


def engine_supports(engine) -> bool:
    """Whether `engine` reads this wire format: it reports `tool_name_constraint=2` at start (the CUDA/HIP
    engine with per-row verify bias), and is not the all-cards split engine (which has no logit bias)."""
    if engine is None or getattr(engine, "split_engine", False):
        return False
    return getattr(engine, "info", {}).get("tool_name_constraint") == int(VERSION[1:])


def _ids(tok, text: str, special: bool) -> list[int]:
    try:
        return [int(x) for x in tok.encode(text, parse_special=special)]
    except Exception:
        return []


def _common_prefix(seqs: list[list[int]]) -> list[int]:
    first = seqs[0]
    n = len(first)
    for s in seqs[1:]:
        j = 0
        bound = min(n, len(s))
        while j < bound and first[j] == s[j]:
            j += 1
        n = j
        if n == 0:
            break
    return first[:n]


def _form(tok, opener: str, closer: str, names: list[str], extra: tuple[str, ...] = ()) -> str | None:
    """One `#<trigger>|<tail>|...` group, or None when the form cannot be split into trigger + names.  `extra` are whole
    tags that stay allowed when they share the trigger (`<function_calls>` starts with `<function` in a BPE vocabulary)."""
    full = [_ids(tok, opener + name + closer, False) for name in names]
    bare = _ids(tok, opener, False)
    if not bare or any(not s for s in full):
        return None
    # The bare opener bounds the trigger: with one name (or names sharing a prefix) the trigger
    # must still stop before the name, or a hallucinated name would never be masked.
    trigger = _common_prefix(full + [bare])
    for tag in extra:                        # a tag the trigger also fires on must stay writable
        ids = _ids(tok, tag, False)
        if len(ids) > len(trigger) and ids[:len(trigger)] == trigger:
            full.append(ids)
    tails = [s[len(trigger):] for s in full]
    if not trigger or any(not t for t in tails):
        return None
    return "|".join(["#" + ",".join(map(str, trigger))] + [",".join(map(str, t)) for t in tails])


def build(tok, declared_names, thinking: bool) -> str | None:
    """The `tool_name_constraint` record for a request, or None to disable the feature.

    `tok` is the tokenizer (`encode(text, parse_special=...)`); `declared_names` the declared
    tools' names (any non-string or empty name is skipped); `thinking` whether the generation
    prompt opens `<think>` (so the engine starts inside the reasoning block).  Returns None when
    there is nothing to constrain or the vocabulary cannot express the markers."""
    names = list(dict.fromkeys(n for n in declared_names if isinstance(n, str) and n))
    if not names:
        return None
    call_open = _ids(tok, _CALL_OPEN, True)
    call_close = _ids(tok, _CALL_CLOSE, True)
    if not call_open or not call_close:
        return None
    think_open = _ids(tok, _THINK_OPEN, True)
    think_close = _ids(tok, _THINK_CLOSE, True)
    if not think_open or not think_close:
        think_open = think_close = []        # a model with no reasoning block: tracking off
    groups = []
    for opener, closer in FORMS:
        form_names = names + [WRAPPER_NAME] if opener == _TRIGGER and WRAPPER_NAME not in names else names
        g = _form(tok, opener, closer, form_names, ("<" + WRAPPER_NAME + ">",) if opener == _TRIGGER else ())
        if g:
            groups.append(g)
    if not groups:
        return None
    parts = [VERSION, "1" if thinking else "0",
             ",".join(map(str, think_open)),
             ",".join(map(str, think_close)),
             ",".join(map(str, call_open)),
             ",".join(map(str, call_close))]
    return "|".join(parts + groups)
