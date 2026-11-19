"""Constrained tool-name decoding: the engine-side wire format.

A model that writes `<tool_call><function=NAME>` can hallucinate a NAME no declared tool has
(`TaskCreate`, `Delete`); the parser drops it and the raw markup leaks into the reply.  The
engine can remove the possibility at the sampler: while the reply is *inside* a `<tool_call>`
envelope (and past the `<think>` reasoning block), the tokens after `<function=` are masked so
only a declared name can follow.

The engine streams token ids and has no detokenizer, so this module turns the declared names
and the model's structural markers into ids.  The wire format (see
`include/strata/core/tool_name_constraint.hpp`) is:

    <initial>|<trigger>|<think_open>|<think_close>|<call_open>|<call_close>|<name+term>|...

`<initial>` is `0`/`1` (whether the generation prompt opens `<think>`).  `<trigger>` is the
token ids of the shared prefix of every `<function=NAME>` sequence, computed as the longest
common prefix so a BPE tokenizer that merges `=` into the name's first token still matches;
each `<name+term>` is the remainder of one `<function=NAME>` sequence after that prefix (the
name and its `>` closer).  The four markers are the ids of `<think>`, `</think>`,
`<tool_call>` and `</tool_call>` (the reasoning pair may be empty to disable reasoning
tracking).  Every record is a comma-separated list of ids, records joined by `|`.

The feature is opt-in and per-request: `build()` returns `None` (and nothing is sent) when a
request declares no tools, or defers tools (the client holds some back, so the declared set is
incomplete), or the model's vocabulary lacks the markers.  A request without the key pays
nothing in the engine.
"""
from __future__ import annotations

_TRIGGER = "<function="
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_CALL_OPEN = "<tool_call>"
_CALL_CLOSE = "</tool_call>"


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


def build(tok, declared_names, thinking: bool) -> str | None:
    """The `tool_name_constraint` record for a request, or None to disable the feature.

    `tok` is the tokenizer (`encode(text, parse_special=...)`); `declared_names` the declared
    tools' names (any non-string or empty name is skipped); `thinking` whether the generation
    prompt opens `<think>` (so the engine starts inside the reasoning block).  Returns None when
    there is nothing to constrain or the vocabulary cannot express the markers."""
    names = [n for n in declared_names if isinstance(n, str) and n]
    if not names:
        return None
    # One full <function=NAME> sequence per name: the exact token ids the model writes.
    full = [_ids(tok, _TRIGGER + name + ">", False) for name in names]
    if any(not s for s in full):
        return None
    if len(full) == 1:
        if len(full[0]) < 2:
            return None                      # the whole tag is one token: nothing to split into trigger + name
        trigger, tails = full[0][:-1], [full[0][-1:]]
    else:
        trigger = _common_prefix(full)
        tails = [s[len(trigger):] for s in full]
    tails = [t for t in tails if t]
    if not trigger or not tails:
        return None
    think_open = _ids(tok, _THINK_OPEN, True)
    think_close = _ids(tok, _THINK_CLOSE, True)
    call_open = _ids(tok, _CALL_OPEN, True)
    call_close = _ids(tok, _CALL_CLOSE, True)
    if not call_open or not call_close:
        return None
    if not think_open or not think_close:
        think_open = think_close = []        # a model with no reasoning block: tracking off
    parts = ["1" if thinking else "0",
             ",".join(map(str, trigger)),
             ",".join(map(str, think_open)),
             ",".join(map(str, think_close)),
             ",".join(map(str, call_open)),
             ",".join(map(str, call_close))]
    parts += [",".join(map(str, t)) for t in tails]
    return "|".join(parts)
