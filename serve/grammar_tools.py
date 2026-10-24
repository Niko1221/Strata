"""#3862: tool-call grammars. The template serialization (chat_template.jinja, assistant tool_calls) is a
rigid shape - "<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>" -
so a grammar can carry it token by token: a constrained model cannot write a broken call, invent a tool,
or drop a required parameter. The builder emits exactly the template's literals; the routing in server.py
hands it the schema converter (convert=...) and passes the result to the engine under a content hash.

The prose guard (the "auto" alternation) excludes the literal "<tool_call>": prose may contain any text
that never spells it, so a model that starts a call is bound to finish it. The exclusion is a chain of
rules - one state per character of the forbidden sequence - and it is exact for the sequences used here,
which carry no border: no proper suffix of a prefix repeats the sequence's start, so a mismatched
character always restarts the chain at state 0."""
import re

# the vendored converter's literal escapes (json-schema-to-grammar.cpp GRAMMAR_LITERAL_ESCAPES);
# inside a character class the range characters ride escaped too (GRAMMAR_RANGE_LITERAL_ESCAPE_RE)
_ESCAPES = {"\r": "\\r", "\n": "\\n", '"': '\\"', "\\": "\\\\"}
_CLASS_ESCAPES = dict(_ESCAPES, **{"-": "\\-", "]": "\\]"})


def gbnf_escape(text: str) -> str:
    return "".join(_ESCAPES.get(c, c) for c in text)


def _lit(text: str) -> str:
    return f'"{gbnf_escape(text)}"'


def _rule_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]+", "-", text.lower()).strip("-") or "x"


def exclude_seq(seq: str, name: str, rules: list) -> str:
    """Rules for any text that never spells `seq`; the rule named `name` is state 0. One rule per
    character: [^c] restarts the chain, "c" advances it; the last state simply refuses the final
    character, so the sequence can never complete. Returns the entry rule name."""
    for k, ch in enumerate(seq):
        cls = f"[^{_CLASS_ESCAPES.get(ch, ch)}]"
        head = name if k == 0 else f"{name}-{k}"
        if k == len(seq) - 1:
            rules.append(f'{head} ::= {cls} {name} | ""')
        else:
            rules.append(f'{head} ::= {cls} {name} | {_lit(ch)} {name}-{k + 1} | ""')
    return name


def exclude_seqs(seqs: list, name: str, rules: list) -> str:
    """Rules for any text that never spells any of `seqs` - the product of the per-sequence
    restart chains, one rule per reachable state (the entry rule is `name`; the others are
    `name-<i>-<j>...`). A character that advances an axis takes only its tracking branch (so a
    completing character is refused by that state alone); a character that completes a sequence
    is refused by the state before it; every other character - including one that merely restarts
    a partial match - rides the free class and the text flows as it would unconstrained. Returns
    the entry rule name.

    The single-sequence shape is exactly exclude_seq's chain; use this for the think prologue,
    which must refuse both the template's closer and the model's fenced-JSON answer habit
    (measured on the rig: the habit pours out as free reasoning and the answer never starts).
    """
    lens = [len(s) for s in seqs]

    def nxt(v, ch):
        # a mismatch restarts that axis to 0, as the single-sequence chain does; this keeps every
        # transition off its own state, which the grammar parser demands
        return tuple(j + 1 if ch == s[j] else 0 for j, s in zip(v, seqs))

    def label(v):
        return name if not any(v) else name + "-" + "-".join(str(x) for x in v)

    chars = sorted({c for s in seqs for c in s})
    entry = (0,) * len(seqs)
    states, order = {entry}, [entry]
    heads = {}
    for v in order:
        branches, refused = [], set()
        for ch in chars:
            w = nxt(v, ch)
            if any(w[k] == lens[k] for k in range(len(seqs))):
                refused.add(ch)                    # completing a sequence: the state refuses the char
            elif w != entry:
                if w not in states:
                    states.add(w)
                    order.append(w)
                branches.append((ch, w))
                refused.add(ch)                    # advancing: only its tracking branch may take it, so the
                                                   # refusal of a completing char cannot leak through a
                                                   # parallel restart stack
            # else: the char restarts every axis and rides the free class (any text stays legal - the bug
            # that excluded these made a bare "h" or "t" in the reasoning unmatchable and starved the
            # model's thinking down to letter-free loops)
        cls = "".join(_CLASS_ESCAPES.get(c, c) for c in sorted(refused))
        body = [f"[^{cls}] {label(entry)}"] + [f"{_lit(ch)} {label(w)}" for ch, w in branches]
        heads[v] = f'{label(v)} ::= ' + " | ".join(body) + ' | ""'
    for v in order:
        rules.append(heads[v])
    return name


# the exact strings the template writes around a call (verbatim, keep them in sync with chat_template.jinja)
CALL_HEAD = "<tool_call>\n<function="            # ... + NAME + ">\n"
PARAM_HEAD = "<parameter="                    # ... + KEY + ">\n"
PARAM_TAIL = "\n</parameter>\n"
CALL_TAIL = "</function>\n</tool_call>"
CALL_JOIN = "\n"                              # calls in one turn are joined by a bare newline
PROSE_PREFIX = "\n\n"                         # prose and calls are separated by a blank line
THINK_TAIL = "\n</think>\n\n"                 # reasoning ends, then the template's blank line
THINK_CLOSE = "</think>\n"                    # the closer as a model writes it, + one newline (see with_think_prologue)
THINK_BANNED = ["</think>", "```json"]       # the reason region refuses the closer (it opens the answer) and the model's fenced-JSON answer habit (free-text reasoning that never ends)


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

    for old, body in parsed:
        rules.append(f"{ns}-{old} ::= {rename(body)}")
    return f"{ns}-root"


def _call_rule(tool: dict, idx: int, rules: list, convert) -> str:
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("a tool has no name to build a grammar for")
    ns = f"call-{_rule_name(name)}-{idx}"
    schema = tool.get("parameters")
    props = schema.get("properties") if isinstance(schema, dict) else None
    required = set(schema.get("required") or []) if isinstance(schema, dict) else set()
    parts = [_lit(CALL_HEAD + name + ">\n")]
    for key, sub in (props or {}).items():
        pns = f"{ns}-{_rule_name(key)}"
        if isinstance(sub, dict) and sub.get("type") == "string":
            value = exclude_seq("\n</parameter>", f"{pns}-text", rules)
        else:
            value = _inline(convert(sub if isinstance(sub, dict) else {}), pns, rules)
        rules.append(f"{pns} ::= {_lit(PARAM_HEAD + key + '>\n')} {value} {_lit(PARAM_TAIL)}")
        parts.append(pns if key in required else f"( {pns} )?")
    parts.append(_lit(CALL_TAIL))
    rules.append(f"{ns} ::= " + " ".join(parts))
    return ns


def build_tool_grammar(tools: list, tool_choice: str, name, thinking: bool, convert) -> str:
    """tools: the template tool list ([{name, parameters}]); tool_choice: "auto" | "required" | "named"
    (from frontend.tool_choice_of); name: the function for "named" (must be one of the tools);
    thinking: the reply opens inside a <|im_start|>assistant\n<think> block, so the grammar carries
    the reasoning and the closing </think> too; convert: JSON Schema -> GBNF text (the cached converter)."""
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
        prose = exclude_seq("<tool_call>", "prose", rules)
        rules.append(f'answer ::= {prose} | calls | {prose} {_lit(PROSE_PREFIX)} calls')
    else:
        raise ValueError(f"tool_choice {tool_choice!r} needs no grammar")
    if thinking:
        reason = exclude_seqs(THINK_BANNED, "reason", rules)
        # one newline after the closer, then free whitespace before the call (see with_think_prologue)
        rules.append(f'ws ::= ( " " | {_lit(chr(9))} | {_lit(chr(10))} )*')
        rules.append(f"root ::= {reason} {_lit(THINK_CLOSE)} ws answer")
    else:
        rules.append("root ::= answer")
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


def plan_tool_grammar(engine, tools, tool_choice, mcp, thinking):
    """One request's tool-call routing: -> (grammar_text, declared_method).

    grammar_text is the call grammar to register (None: the prompt path takes the turn);
    declared_method is what a prompt-path response says it did (MCP: the case an operator
    asked about). A missing converter is not an error - the prompt path has always served
    tool calls - so it quietly yields the prompt path too."""
    from serve import grammar_schema as grammar_conv
    kind, gkind, gname = tool_grammar_choice(grammar_conv.grammar_ok(engine), tools, tool_choice, mcp)
    if kind == "none":
        return None, None
    if kind == "prompt":
        return None, ("prompt_and_validate" if mcp and grammar_conv.grammar_ok(engine) else None)
    try:
        exe_dir = grammar_conv.exe_dir_of(engine)
        text = build_tool_grammar(
            tools, gkind, gname, thinking,
            lambda sub: grammar_conv.schema_to_grammar(exe_dir, sub))
    except ValueError:
        return None, None           # the prompt path serves the call, as it always did
    return text, None


def with_think_prologue(grammar_text: str) -> str:
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
    """
    rules: list = []
    answer = _inline(grammar_text, "answer", rules)
    reason = exclude_seqs(THINK_BANNED, "reason", rules)
    rules.append(f'ws ::= ( " " | {_lit(chr(9))} | {_lit(chr(10))} )*')
    # a model whose answer habit is a fenced block must be able to write it - banning the fence left
    # it with no legal continuation right after the closer and the stream degenerated (measured);
    # the canary's JSON extractor strips the fence, so the answer it validates is the JSON
    rules.append(f'fenced ::= {_lit("```json")} ws {answer} ws {_lit("```")}')
    rules.append(f"answer ::= ws ( fenced | {answer} )")
    rules.append(f"root ::= {reason} {_lit(THINK_CLOSE)} answer")
    return "\n".join(rules) + "\n"
