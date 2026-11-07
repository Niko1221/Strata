"""#3862: tool-call grammars. The template envelope and each parameter schema are constrained
by overlap-aware automata, with generated rule names independent of user-provided properties."""
import re

from serve.frontend import parse_tool_call

# the vendored converter's literal escapes (json-schema-to-grammar.cpp GRAMMAR_LITERAL_ESCAPES)
_ESCAPES = {"\r": "\\r", "\n": "\\n", '"': '\\"', "\\": "\\\\"}


def gbnf_escape(text: str) -> str:
    return "".join(_ESCAPES.get(c, c) for c in text)


def _lit(text: str) -> str:
    return f'"{gbnf_escape(text)}"'


def _rule_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]+", "-", text.lower()).strip("-") or "x"


def _rule_heads(rules: list) -> set[str]:
    return {line.partition(" ::= ")[0] for line in rules if " ::= " in line}


def _fresh_rule_name(base: str, rules: list) -> str:
    heads = _rule_heads(rules)
    name, suffix = base, 0
    while any(head == name or head.startswith(name + "-") or name.startswith(head + "-") for head in heads):
        name = f"{base}-generated-{suffix}"
        suffix += 1
    return name


def _check_unique_rules(rules: list) -> None:
    names = [line.partition(" ::= ")[0] for line in rules if " ::= " in line]
    if len(names) != len(set(names)):
        raise ValueError("the generated tool grammar contains duplicate rule names")


def exclude_seq(seq: str, name: str, rules: list, build) -> str:
    """Emit a grammar that refuses any text containing `seq`, including overlapping matches.

    `build(prefix, seqs)` is the exclusion constructor: the Aho-Corasick emitter ported from llama.cpp
    (serve/grammar_schema.exclusion_grammar, via grammar_schema --ac-grammar).  It returns a complete rule
    block whose start rule is `prefix`; nothing here spells an automaton, so the server and the converter
    share one implementation and cannot drift apart."""
    fresh = _fresh_rule_name(name, rules)
    rules.extend(build(fresh, [seq]).splitlines())
    return fresh


def exclude_seqs(seqs: list, name: str, rules: list, build) -> str:
    """Emit a grammar that refuses any of the sequences, preserving overlapping prefixes.

    One automaton holds every sequence at once, so the state count is the SUM of the sequence states, not
    their product: `</think>` + `` ```json `` + `<tool_call>` costs 29 states rather than 9*8*12."""
    seqs = list(dict.fromkeys(s for s in seqs if s))
    if not seqs:
        raise ValueError("at least one excluded sequence is required")
    fresh = _fresh_rule_name(name, rules)
    rules.extend(build(fresh, seqs).splitlines())
    return fresh



# the exact strings the template writes around a call (verbatim, keep them in sync with chat_template.jinja)
CALL_HEAD = "<tool_call>\n<function="            # ... + NAME + ">\n"
PARAM_HEAD = "<parameter="                    # ... + KEY + ">\n"
PARAM_TAIL = "\n</parameter>\n"
PARAM_END = "</parameter>"                    # the closer alone: frontend.param_end ends a value at this string,
                                              # with or without the template's leading newline (see _call_rule)
CALL_TAIL = "</function>\n</tool_call>"
CALL_JOIN = "\n"                              # calls in one turn are joined by a bare newline
PROSE_PREFIX = "\n\n"                         # prose and calls are separated by a blank line
THINK_TAIL = "\n</think>\n\n"                 # reasoning ends, then the template's blank line
THINK_CLOSE = "</think>\n"                    # the closer as a model writes it, + one newline (see with_think_prologue)
THINK_BANNED = ["</think>", "```json"]       # the reason region refuses the closer (it opens the answer) and the model's fenced-JSON answer habit (free-text reasoning that never ends)
CALL_TAG = "<tool_call>"                      # a call's opening tag: refused inside the reasoning (no schema there)


def _strip_comment(line: str) -> str:
    """Drop a GBNF '#' comment: the parser skips from '#' to end of line unless the '#' sits in a
    quoted literal or a character class; the same scan lets a multi-line rule be joined without a
    trailing comment eating the joined tail."""
    out, quoted, cls, esc = [], False, False, False
    for ch in line:
        if ch == "#" and not quoted and not cls:
            break
        out.append(ch)
        if esc:                                   # a backslash escapes the next character
            esc = False
        elif ch == "\\" and (quoted or cls):
            esc = True
        elif ch == '"' and not cls:
            quoted = not quoted
        elif not quoted:
            if ch == "[":
                cls = True
            elif ch == "]":
                cls = False
    return "".join(out)


def _inline(grammar_text: str, ns: str, rules: list) -> str:
    """Splice the converter's full grammar under the namespace `ns` (every rule renamed `ns-<rule>`,
    literals and character classes left alone) and return the entry rule (the converter's root)."""
    parsed = []
    current = None
    def flush():
        if current is None:
            return
        if not current[1]:
            raise ValueError(f"the schema converter produced an empty rule body: {current[0]}")
        parsed.append((current[0], current[1]))
    for line in grammar_text.splitlines():
        line = _strip_comment(line).strip()
        if not line:
            continue
        m = re.match(r"^([a-zA-Z][a-zA-Z0-9-]*)\s*::=\s*(.*)$", line)
        if m:
            flush()
            current = [m.group(1), m.group(2).strip()]
        elif current is not None:                       # the converter's root grammars span lines
            current[1] = (current[1] + " " + line).strip()
        else:
            raise ValueError(f"the schema converter produced a rule that cannot be inlined: {line!r}")
    flush()
    if not any(old == "root" for old, _ in parsed):
        raise ValueError("the schema converter produced no root rule")
    names = {old for old, _ in parsed}
    token = re.compile(r'"(?:\\.|[^"\\])*"|\[(?:\\.|[^\]])*\]|([a-zA-Z][a-zA-Z0-9-]*)')

    def rename(body: str) -> str:
        out, pos = [], 0
        for m in token.finditer(body):
            out.append(body[pos:m.start()])
            out.append(f"{ns}-{m.group(1)}" if m.group(1) in names else m.group(0))
            pos = m.end()
        out.append(body[pos:])
        return "".join(out)

    old_names = [old for old, _ in parsed]
    if len(old_names) != len(set(old_names)):
        raise ValueError("the schema converter produced duplicate rule names")
    for old, body in parsed:
        rules.append(f"{ns}-{old} ::= {rename(body)}")
    return f"{ns}-root"


def _check_framing(name: str, keys: list, schema) -> None:
    """Refuse a tool whose envelope this server could write but not read back.

    The grammar spells the template's markers around the name and each parameter key, and
    serve/frontend.parse_tool_call reads them back by cutting at the first '>'.  A name or key holding '>'
    therefore makes the two disagree: a tool named `ev>il` is emitted as `<function=ev>il>` and read as `ev`,
    so the call the grammar guarantees is delivered as a different tool (measured, likewise `a>b` -> `a` as a
    parameter key).  Rather than blacklist characters, the check runs the exact text the grammar demands
    through the real parser, so any future framing change is covered by construction.
    """
    body = ("<function=" + name + ">\n"
            + "".join(PARAM_HEAD + key + ">\n" + "x" + PARAM_TAIL for key in keys)
            + CALL_TAIL)
    try:
        call = parse_tool_call(body, schema if isinstance(schema, dict) else None)
    except ValueError as e:
        raise ValueError(f"tool {name!r} cannot be spelled in the call format: {e}") from None
    if call.name != name:
        raise ValueError(f"tool name {name!r} would be read back as {call.name!r}; a name cannot contain '>'")
    if list(call.arguments) != list(keys):
        bad = next((k for k, got in zip(keys, call.arguments) if k != got), keys[len(call.arguments):] or keys)
        raise ValueError(f"parameter {bad!r} of tool {name!r} would be read back as {list(call.arguments)!r}; "
                         "a parameter name cannot contain '>'")


def _call_rule(tool: dict, idx: int, rules: list, convert) -> str:
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("a tool has no name to build a grammar for")
    ns = f"call-{_rule_name(name)}-{idx}"
    schema = tool.get("parameters")
    props = schema.get("properties") if isinstance(schema, dict) else None
    required = set(schema.get("required") or []) if isinstance(schema, dict) else set()
    _check_framing(name, list((props or {}).keys()), schema)
    parts = [_lit(CALL_HEAD + name + ">\n")]
    for prop_idx, (key, sub) in enumerate((props or {}).items()):
        pns = f"{ns}-param-{prop_idx}"
        sub = sub if isinstance(sub, dict) else {}
        # The converter applies raw framing only to a string/const/enum root; object, array,
        # and scalar roots retain their ordinary JSON spelling. Passing tool mode for every
        # parameter lets it resolve a root $ref before choosing that representation and fail
        # closed for ambiguous schemas instead of silently switching to prompt validation.
        # The document rides along because a JSON Pointer is rooted at it: `#/properties/x`
        # names a sibling parameter, which this per-parameter call would otherwise not see.
        value = _inline(convert(sub, True, PARAM_END, schema), pns, rules)
        rules.append(f"{pns} ::= {_lit(PARAM_HEAD + key + '>\n')} {value} {_lit(PARAM_TAIL)}")
        parts.append(pns if key in required else f"( {pns} )?")

    parts.append(_lit(CALL_TAIL))
    rules.append(f"{ns} ::= " + " ".join(parts))
    return ns


def build_tool_grammar(tools: list, tool_choice: str, name, thinking: bool, convert, build) -> str:
    """tools: the template tool list ([{name, parameters}]); tool_choice: "auto" | "required" | "named"
    (from frontend.tool_choice_of); name: the function for "named" (must be one of the tools);
    thinking: the reply opens inside a <|im_start|>assistant\n<think> block, so the grammar carries
    the reasoning and the closing </think> too; convert: JSON Schema -> GBNF text (the cached converter);
    build: an exclusion constructor (prefix, sequences) -> GBNF rules, the Aho-Corasick emitter."""
    if not tools:
        raise ValueError("no tools to build a grammar for")
    rules: list = []
    calls = []
    named_rule = None
    for i, t in enumerate(tools):
        r = _call_rule(t, i, rules, convert)
        calls.append(r)
        if t.get("name") == name:
            named_rule = r
    if tool_choice == "named" and named_rule is None:
        raise ValueError(f"tool_choice names {name!r}, which is not one of the tools")
    rules.append("calls-one ::= " + " | ".join(calls))
    rules.append(f'calls-rest ::= {_lit(CALL_JOIN)} calls-one calls-rest | ""')
    rules.append("calls ::= calls-one calls-rest")
    if tool_choice == "named":
        rules.append(f"answer ::= {named_rule}")
    elif tool_choice == "required":
        rules.append("answer ::= calls")
    elif tool_choice == "auto":
        prose = exclude_seq("<tool_call>", "prose", rules, build)
        rules.append(f'answer ::= {prose} | calls | {prose} {_lit(PROSE_PREFIX)} calls')
    else:
        raise ValueError(f"tool_choice {tool_choice!r} needs no grammar")
    if thinking:
        # the reasoning region also refuses the call's own opening tag: a call written up there is free
        # text, so it carries no schema - measured, a rescued one arrived missing a required parameter
        # and a second call rode beside it.  Banning the tag leaves the answer region as the only place
        # a call can appear, which is the region the grammar actually constrains.
        reason = exclude_seqs(THINK_BANNED + [CALL_TAG], "reason", rules, build)
        # a little whitespace after the closer, then the call.  Bounded for the reason
        # with_think_prologue gives: an unbounded run is where a masked model stalls.
        rules.append(f'ws ::= ( " " | {_lit(chr(9))} | {_lit(chr(10))} ){{0,24}}')
        rules.append(f"root ::= {reason} {_lit(THINK_CLOSE)} ws answer")
    else:
        rules.append("root ::= answer")
    _check_unique_rules(rules)
    return "\n".join(rules) + "\n"


def tool_grammar_choice(capable: bool, tools, tool_choice, mcp: bool):
    """Pure routing: -> ("grammar", kind, name) when the sampler should carry the call,
    ("prompt", method, None) when the prompt path takes the turn (method declared in the response for
    MCP, the case an operator asked about), ("none", None, None) when no call is on the table.
    A tool_choice the old code degrades to "auto" (unknown shape, a name outside the list) keeps the
    prompt path: the grammar would be a policy change on a value the server already ignores."""
    from serve.frontend import tool_choice_of
    kind, name = tool_choice_of(tool_choice)
    if not tools or kind == "none":
        return "none", None, None
    if mcp:
        return "prompt", "prompt_and_validate", None
    if not capable:
        return "prompt", None, None
    if kind == "named" and name not in {t.get("name") for t in tools}:
        return "prompt", None, None
    if kind not in ("auto", "required", "named"):
        return "prompt", None, None
    return "grammar", kind, name


def plan_tool_grammar(engine, tools, tool_choice, mcp, thinking, allow=True):
    """One request's tool-call routing: -> (grammar_text, declared_method).

    grammar_text is the call grammar to register (None: the prompt path takes the turn);
    declared_method is what a prompt-path response says it did (MCP: the case an operator
    asked about). A capable engine with a missing or insufficient converter fails closed so
    a strict tool request cannot silently downgrade to prompt-only validation.

    `allow` is the caller's per-request opt-out (`strata_grammar: false`): route to the prompt
    path even though the engine could spell the call.  Same switch as plan_response_format's."""
    from serve import grammar_schema as grammar_conv
    kind, gkind, gname = tool_grammar_choice(grammar_conv.grammar_ok(engine) and allow,
                                             tools, tool_choice, mcp)
    if kind == "none":
        return None, None
    if kind == "prompt":
        return None, ("prompt_and_validate" if mcp and grammar_conv.grammar_ok(engine) else None)
    exe_dir = grammar_conv.exe_dir_of(engine)
    text = build_tool_grammar(
        tools, gkind, gname, thinking,
        lambda sub, raw_string, forbidden, document=None: grammar_conv.schema_to_grammar(
            exe_dir, sub, tool_value=raw_string, forbidden_sequence=forbidden, document=document),
        exclusion_builder(engine))
    return text, None


def exclusion_builder(engine):
    """The exclusion-grammar constructor for one engine: (prefix, sequences) -> GBNF rules.

    Callers that build a grammar without going through plan_tool_grammar (with_think_prologue's reasoning
    region) use this, so every exclusion in the server comes from the one Aho-Corasick emitter."""
    from serve import grammar_schema as grammar_conv
    exe_dir = grammar_conv.exe_dir_of(engine)
    return lambda prefix, seqs: grammar_conv.exclusion_grammar(exe_dir, prefix, seqs)


def with_think_prologue(grammar_text: str, build) -> str:
    """A response_format grammar on a model whose template opens the answer inside a think block:
    the reasoning region rides as free text that never spells the template's closer, then the
    closer, then the constrained answer. Without this the grammar's first token lands inside the
    open think region and the server reads the whole answer as reasoning.

    The closer is '</think>' + ONE newline, and the answer body grows a leading whitespace rule -
    not the template's full '\n
</think>


\n\n'. The model that closes tightly lands on the answer's first token; the model
    that pads the closer with blank lines rides the answer's own leading whitespace. A grammar
    that requires both template newlines strands a padding model: every token it then writes is
    legal only as free text, so the mask never opens the answer (measured on the rig). The serve
    parser splits reasoning at '</think>' alone, so both models agree where the answer begins.

    The leading whitespace is BOUNDED. A model whose answer habit is a fenced block used to have
    the fence as its only way out of that whitespace; without it the model walks the whitespace
    rule instead, and an unbounded rule let it do that for a whole output budget (measured: 76K
    characters of newlines, finish_reason=length). The cap strands it after 24 characters, and
    the answer opens. The fence itself stays banned in the answer: it parses, but it is not the
    JSON the caller asked for, and every client json.loads(content) fails on it.
    """
    rules: list = []
    answer = _inline(grammar_text, "answer", rules)
    reason = exclude_seqs(THINK_BANNED, "reason", rules, build)
    rules.append(f'ws ::= ( " " | {_lit(chr(9))} | {_lit(chr(10))} ){{0,24}}')   # bounded: see the docstring
    rules.append(f"answer ::= ws {answer}")
    rules.append(f"root ::= {reason} {_lit(THINK_CLOSE)} answer")
    return "\n".join(rules) + "\n"
