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
            rules.append(f'{head} ::= {cls} {name}-0 | ""')
        else:
            rules.append(f'{head} ::= {cls} {name}-0 | {_lit(ch)} {name}-{k + 1} | ""')
    return name


# the exact strings the template writes around a call (verbatim, keep them in sync with chat_template.jinja)
CALL_HEAD = "<tool_call>\n<function="            # ... + NAME + ">\n"
PARAM_HEAD = "<parameter="                    # ... + KEY + ">\n"
PARAM_TAIL = "\n</parameter>\n"
CALL_TAIL = "</function>\n</tool_call>"
CALL_JOIN = "\n"                              # calls in one turn are joined by a bare newline
PROSE_PREFIX = "\n\n"                         # prose and calls are separated by a blank line
THINK_TAIL = "\n</think>\n\n"                 # reasoning ends, then the template's blank line


def _inline(grammar_text: str, ns: str, rules: list) -> str:
    """Splice the converter's full grammar under the namespace `ns` (every rule renamed `ns-<rule>`,
    literals and character classes left alone) and return the entry rule (the converter's root)."""
    parsed = []
    for line in grammar_text.splitlines():
        line = line.strip()
        if not line or line.startswith(";;"):
            continue
        m = re.match(r"^([a-zA-Z][a-zA-Z0-9-]*)\s*::=\s*(.+)$", line)
        if not m:
            raise ValueError(f"the schema converter produced a rule that cannot be inlined: {line!r}")
        parsed.append((m.group(1), m.group(2)))
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
        reason = exclude_seq("</think>", "reason", rules)
        rules.append(f"root ::= {reason} {_lit(THINK_TAIL)} answer")
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
