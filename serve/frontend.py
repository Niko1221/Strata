"""serve/frontend.py - plan v0.3 P8: the request/response layer between API clients and the engine.

Everything here is text: no model, no GPU. The engine consumes token ids and produces text deltas; this module

  * normalizes OpenAI Chat Completions and Anthropic Messages requests into the chat template's message form,
  * renders the model's own Jinja chat template (pack `tokenizer/chat_template.jinja`) exactly as Hugging Face
    does (checked against the pack's `chat_golden.json`, 10 cases incl. thinking and tools),
  * parses the streamed output incrementally into reasoning (`<think>...</think>`), content and tool calls in
    the template's XML form (`<tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call>`),
    never emitting a partial tag to the client.

Design note: plan v0.3 describes a single C++ process. The frontend is Python first because the template is Jinja
and the formats change often; the engine boundary is token ids in, text deltas out, which keeps a later C++ port
(or a pybind wrapper) a local change.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ template
class TemplateRequestError(jinja2.exceptions.TemplateError, ValueError):
    """The template refused the request's messages (e.g. "No user query found in messages."): a ValueError, so the
    client gets a 400 with the template's message instead of a dropped connection (#365)."""


class ChatTemplate:
    """The model's chat template, rendered with the same Jinja settings as transformers' apply_chat_template."""

    def __init__(self, path: str | Path):
        def raise_exception(message):
            raise TemplateRequestError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        self.source = Path(path).read_text(encoding="utf-8")
        self.template = env.from_string(self.source)
        self.caps = self._detect_caps()

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        # Do not use completed empty assistant turns as examples for the next reply. Keep the final message.
        # (#843.  STRATA_KEEP_EMPTY_TURNS=1 renders them as before.)
        messages = [m for i, m in enumerate(messages)
                    if i == len(messages) - 1 or os.environ.get("STRATA_KEEP_EMPTY_TURNS") == "1" or not (isinstance(m, dict) and m.get("role") == "assistant"
                                                      and not _text_of(m.get("content")).strip()
                                                      and not _has_image(m.get("content")) and not m.get("tool_calls"))]
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                    **kwargs)

    def _detect_caps(self) -> dict[str, bool]:
        """llama.cpp's capability names, checked at load time against this template and Strata's tool-call format.
        These are rendering hints, not a guarantee that the model will follow a request."""
        def render(messages, tools=None):
            try:
                return self.render(messages, tools=tools)
            except Exception as exc:                # noqa: BLE001 - a hint only: never stop the server for a probe
                # A template may reject a role or feature (a custom one may fail in any way): the feature is off,
                # discovery still answers for the others, and the start-up goes on.
                LOGGER.debug("chat template capability probe failed: %s", exc, exc_info=True)
                return ""

        user = {"role": "user", "content": "strata_caps_user"}
        tools = [{"name": f"strata_caps_call_{i}", "description": "strata_caps_description",
                  "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}}}
                 for i in range(2)]
        tool_prompt = render([user], tools)

        def calls_supported(count):
            calls = [{"name": f"strata_caps_call_{i}", "arguments": {"arg": f"strata_caps_arg_{i}"}}
                     for i in range(count)]
            messages = [user, {"role": "assistant", "content": "",
                               "tool_calls": [{"function": call} for call in calls]}]
            replies = [f"strata_caps_result_{i}" for i in range(count)]
            messages += [{"role": "tool", "content": reply} for reply in replies] + [user]
            prompt = render(messages, tools)
            # Instructions include example XML and literal tag names, which are not model output. Parse only
            # complete calls to our probe functions, using the same body parser as OutputParser.
            bodies = re.findall(r"<tool_call>\s*(<function=strata_caps_call_\d+>.*?</function>)\s*</tool_call>",
                                prompt, re.S)
            try:
                parsed = [parse_tool_call(body) for body in bodies]
            except ValueError:
                return False
            return [{"name": call.name, "arguments": call.arguments} for call in parsed] == calls \
                and all(reply in prompt for reply in replies)

        history = [user, {"role": "assistant", "content": "strata_caps_answer",
                          "reasoning_content": "strata_caps_reasoning"}, user]
        return {"supports_tools": all(s in tool_prompt for s in
                                      ("strata_caps_call_0", "strata_caps_description", "<tool_call>", "<function=")),
                "supports_tool_calls": calls_supported(1),
                "supports_system_role": "strata_caps_system" in render(
                    [{"role": "system", "content": "strata_caps_system"}, user]),
                "supports_parallel_tool_calls": calls_supported(2),
                "supports_preserve_reasoning": "strata_caps_reasoning" in render(history)}


# ------------------------------------------------------------------------------------------------ requests
# The content-part types the readers take as text (a part with no type that has a "text" field reads as text too)
TEXT_PARTS = ("text", "input_text", None)


def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in
                   TEXT_PARTS)


IMAGE_PARTS = ("image_url", "input_image", "image")

# Thinking levels.  The model's template knows low, medium and xhigh (its default; "high" means xhigh), and
# enable_thinking=false for none.  Clients spell these many ways; everything maps onto those four.
EFFORT = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
          "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh", "maximum": "xhigh"}


# DeepSeek-V4.1's template takes low, medium, high (its default) and max, or a whole number 1..100 (the "Reasoning
# Effort: N" line of its prompt).  use_family("deepseek41") switches the table; the other families keep the one above.
EFFORT_DEEPSEEK = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
                   "low": "low", "medium": "medium", "high": "high", "xhigh": "max", "max": "max", "maximum": "max"}
_EFFORT_QWEN = dict(EFFORT)
_NUMERIC_EFFORT = False


def use_family(family: str | None) -> None:
    """The reasoning-effort words of the model family this server runs (module state, like ENV_TO_USER)."""
    global _NUMERIC_EFFORT
    EFFORT.clear()
    EFFORT.update(EFFORT_DEEPSEEK if family == "deepseek41" else _EFFORT_QWEN)
    _NUMERIC_EFFORT = family == "deepseek41"


def effort_kwargs(value) -> dict:
    """A reasoning effort as given by a client -> the template's kwargs.  Unknown values are a 400, not a crash."""
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    if _NUMERIC_EFFORT and isinstance(value, int) and not isinstance(value, bool):
        if not 1 <= value <= 100:
            raise ValueError(f"unknown reasoning effort {value!r}: use none, low, medium, high, max or 1..100")
        return {"reasoning_effort": value}
    key = str(value).strip().lower()
    if key not in EFFORT:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, medium or high"
                         + (", max or 1..100" if _NUMERIC_EFFORT else ""))
    level = EFFORT[key]
    return {"enable_thinking": False} if level is None else {"reasoning_effort": level}


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget (budget_tokens) -> a level: under 2K low, under 8K medium, else high."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "low" if n < 2048 else "medium" if n < 8192 else EFFORT["high"]}


def _has_image(content) -> bool:
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS for p in content)


def _image_source(part: dict) -> str:
    """An image part's source as one string: a data: URL, an http(s) URL or a local file path.
    OpenAI: {"type": "image_url", "image_url": {"url": ...}} (or "image_url": "..."), Responses-style
    {"type": "input_image", "image_url": ...}; Anthropic: {"type": "image", "source": {"type": "base64",
    "media_type": ..., "data": ...}} or {"source": {"type": "url", "url": ...}}."""
    if part.get("type") == "image":
        src = part.get("source") or {}
        if src.get("type") == "base64":
            return f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
        return src.get("url") or src.get("path") or ""
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return url or ""


def _parts_of(content):
    """Message content for the template: a string when there is no image (unchanged behaviour), otherwise the
    template's list form - text items and image items, in order - whose image items carry their source."""
    if not _has_image(content):
        return _text_of(content)
    items = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in IMAGE_PARTS:
            items.append({"type": "image", "source": _image_source(part)})
        elif part.get("type") in TEXT_PARTS and "text" in part:
            items.append({"type": "text", "text": part.get("text", "")})
    return items


# The blocks a messages request may carry: text, the image parts, thinking, tool_use and tool_result.
# redacted_thinking is allowed only to be dropped without an error: another server's encrypted reasoning, which
# carries nothing this server can read - the same reason responses.py's _reasoning_text gives for another server's
# encrypted_content.
ANTHROPIC_BLOCKS = ("text",) + IMAGE_PARTS + ("thinking", "tool_use", "tool_result", "redacted_thinking")


def _check_parts(content, path: str, kinds):
    """A request's content parts (on chat) or content blocks (on messages), looked at while the request is converted.
    A part whose type this server does not read (a file, audio, a document, Anthropic's tool_reference,
    search_result, server_tool_use / web_search_tool_result blocks, ...) is left out of the prompt exactly as it always
    was, with a debug line: real clients (Claude Code, Codex, opencode, Cline, Continue, Open WebUI) send such parts
    next to text the model can answer from, so refusing them would break requests that work.  A "refusal" part is read
    as its text, as /v1/responses does.  Content that is not a list, and a part that is not an object, passes through
    unchanged."""
    if not isinstance(content, list):
        return content
    out = content
    for j, part in enumerate(content):
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "refusal":
            if out is content:
                out = list(content)
            out[j] = {"type": "text", "text": part.get("refusal") or ""}
        elif kind not in kinds:
            LOGGER.debug("%s[%d].type: content part of type %r is not read by this server; left out of the prompt",
                         path, j, kind)
    return out


def images_of(messages: list[dict]) -> list[str]:
    """The image sources of the rendered conversation, in prompt order (the template renders one
    <|vision_start|><|image_pad|><|vision_end|> per image item, message by message)."""
    return [item["source"] for m in messages if isinstance(m.get("content"), list)
            for item in m["content"] if item.get("type") == "image"]


# #537: a literal <think> / </think> inside a message's text is plain text, not the model's reasoning markers.  The
# tokenizer matches those two strings as their special tokens everywhere (GGUF token type 4, as llama.cpp does), so
# a user quoting "</think>" used to hand the model a real end-of-reasoning token.  Before the template is rendered
# they are swapped for these private-use characters, and the server encodes the spans they mark as ordinary text.
THINK_TAGS = {"<think>": "\U000F0E01", "</think>": "\U000F0E02"}
# #554: the vision markers the same way.  They are special tokens only where the template writes them for an image
# item; the same strings in a message's text (an agent reading chat_template.jinja, a tool result quoting it) became
# the same special ids, so text with <|vision_start|><|image_pad|> before a picture took that picture's embeddings
VISION_TAGS = {"<|vision_start|>": "\U000F0E03", "<|image_pad|>": "\U000F0E04", "<|vision_end|>": "\U000F0E05",
               "<|video_pad|>": "\U000F0E06"}
# #2026: `</tool_call>` the same way.  A user asking for the string verbatim handed the model the real end-of-call token
# (type 4, matched everywhere); it copied that token into its answer and the reply came back empty.  An assistant
# message keeps its own: a client replaying the model's raw call text sends the tokens the model wrote.
# 0.1.44: `<tool_call>`, `<tool_response>` and `</tool_response>` are type 4 too (ids 248058, 248066, 248070), so a quoted
# one handed the model the real call / tool-result marker in the same way (found by 47Hunter47 in #2026).  The template
# writes the tool-result markers around a tool message itself; only the text inside a message is marked.
TOOL_LITERAL_TAGS = {"</tool_call>": "\U000F0E07", "<tool_call>": "\U000F0E08", "<tool_response>": "\U000F0E09",
                     "</tool_response>": "\U000F0E0A"}
LITERAL_TAGS = {**THINK_TAGS, **VISION_TAGS, **TOOL_LITERAL_TAGS}
THINK_MARKS = {v: k for k, v in LITERAL_TAGS.items()}
CONTROL_MARK0 = 0xF0E10      # the control tokens' marks start here, clear of the two sets above


def literal_tags(controls) -> dict[str, str]:
    """LITERAL_TAGS plus a mark for each control token's text (`controls`: the tokenizer's CONTROL literals,
    <|im_start|>, <|im_end|>, <|endoftext|>, ...).  The rendered prompt is encoded with those literals parsed, so one
    written inside a message - a file an agent reads, a pasted chat template - opened or ended a turn there.  Longest
    first, as the tokenizer matches them: a literal inside a longer one is not marked before it."""
    tags = {**LITERAL_TAGS, **{c: chr(CONTROL_MARK0 + k) for k, c in enumerate(c for c in controls
                                                                                if c not in LITERAL_TAGS)}}
    return dict(sorted(tags.items(), key=lambda t: -len(t[0])))


def _mark(text: str, tags: dict[str, str] = LITERAL_TAGS) -> str:
    for tag, mark in tags.items():
        text = text.replace(tag, mark)
    return text


def _mark_deep(v, tags: dict[str, str] = LITERAL_TAGS):
    if isinstance(v, str):
        return _mark(v, tags)
    if isinstance(v, dict):
        return {k: _mark_deep(x, tags) for k, x in v.items()}
    if isinstance(v, list):
        return [_mark_deep(x, tags) for x in v]
    return v


def _has_tag(v, tags: dict[str, str] = LITERAL_TAGS) -> bool:
    if isinstance(v, str):
        return any(tag in v for tag in tags)
    if isinstance(v, dict):
        return any(_has_tag(x, tags) for x in v.values())
    if isinstance(v, list):
        return any(_has_tag(x, tags) for x in v)
    return False


_TOOL_RESULT_OPEN, _TOOL_RESULT_CLOSE = "<tool_response>", "</tool_response>"


def _keep_tool_result_wrap(content, tags: dict[str, str]):
    """A user message whose whole text is <tool_response>...</tool_response> (a client that formats a tool result the way
    the template does for a tool message) keeps that outer pair as the markers - the template tells such a turn from a
    real question by exactly that pair - and only the text inside is marked.  None: not such a message."""
    o, c = _TOOL_RESULT_OPEN, _TOOL_RESULT_CLOSE

    def wrap(text: str, first: bool, last: bool) -> str:
        i = text.index(o) + len(o) if first else 0
        j = text.rindex(c) if last else len(text)
        head = _mark(text[:i - len(o)], tags) + o if first else ""
        tail = c + _mark(text[j + len(c):], tags) if last else ""
        return head + _mark(text[i:j], tags) + tail

    if isinstance(content, str):
        body = content.strip()
        if len(body) >= len(o) + len(c) and body.startswith(o) and body.endswith(c):
            return wrap(content, True, True)
        return None
    if not (isinstance(content, list) and content and all(isinstance(x, dict) and "text" in x for x in content)):
        return None
    texts = [x["text"] for x in content]
    if not all(isinstance(t, str) for t in texts):
        return None
    body = "".join(texts).strip()
    if not (len(body) >= len(o) + len(c) and body.startswith(o) and body.endswith(c)):
        return None
    if not (texts[0].lstrip().startswith(o) and texts[-1].rstrip().endswith(c)):
        return None          # the pair is split across parts: mark it all, as before
    if len(content) == 1:
        return [{**content[0], "text": wrap(texts[0], True, True)}]
    return ([{**content[0], "text": wrap(texts[0], True, False)}] +
            [_mark_deep(x, tags) for x in content[1:-1]] +
            [{**content[-1], "text": wrap(texts[-1], False, True)}])


def mark_think_literals(messages: list[dict], tools: list[dict] | None, tags: dict[str, str] = LITERAL_TAGS):
    """#537: (messages, tools) with every literal <think> / </think> (#554: and vision marker) in their text swapped for
    LITERAL_TAGS' marks (or `tags`': literal_tags() adds the control tokens' texts), and
    whether there was one (None: no change, the same objects back - a prompt without them renders as it always did).
    An assistant message whose content opens with a whole <think>...</think> block (clients that send the reasoning
    inline) keeps that one block as the model's markers, as before."""
    if not _has_tag(messages, tags) and not _has_tag(tools, tags):
        return messages, tools, False
    out = []
    own = {t: v for t, v in tags.items() if t not in TOOL_LITERAL_TAGS}     # what an assistant message keeps as written
    for m in messages:
        m = dict(m)
        content = m.get("content")
        mine = own if m.get("role") == "assistant" else tags
        for k, v in m.items():
            if k != "role":
                m[k] = _mark_deep(v, mine)
        if m.get("role") == "user":
            kept = _keep_tool_result_wrap(content, tags)
            if kept is not None:
                m["content"] = kept
        if m.get("role") == "assistant" and isinstance(content, str) and content.lstrip().startswith("<think>") \
                and "</think>" in content:
            i, j = content.index("<think>") + len("<think>"), content.index("</think>")
            m["content"] = (content[:i] + _mark(content[i:j], own) + "</think>" +
                            _mark(content[j + len("</think>"):], own))
        out.append(m)
    return out, _mark_deep(tools, tags), True


_THINK_MARK_RE = re.compile("|".join(THINK_MARKS))


def unmark_think_literals(prompt: str, tags: dict[str, str] = LITERAL_TAGS) -> tuple[str, list[tuple[int, int]]]:
    """The rendered prompt with THINK_TAGS' marks turned back into the tags' text, and the (start, end) spans of those
    tags in it: the server encodes them as ordinary text (the tokenizer's encode_plain_spans)."""
    marks = THINK_MARKS if tags is LITERAL_TAGS else {v: k for k, v in tags.items()}
    mark_re = _THINK_MARK_RE if tags is LITERAL_TAGS else re.compile("|".join(map(re.escape, marks)))
    out, spans, pos, n = [], [], 0, 0
    for m in mark_re.finditer(prompt):
        out.append(prompt[pos:m.start()])
        n += m.start() - pos
        tag = marks[m.group(0)]
        out.append(tag)
        spans.append((n, n + len(tag)))
        n += len(tag)
        pos = m.end()
    out.append(prompt[pos:])
    return "".join(out), spans


# Config "env_to_user" (opt-in): opencode writes its <env> block (working directory, today's date) near the START of
# the system message, before ~40K characters of instructions and tool definitions that are the same in every session.
# The prefix cache only reuses a prompt's start, so a new folder or a new day read all of it again (~14K tokens, 8-11 s).
# Moved to the front of the first user message, the system turn (and the tools the template writes into it) is the same
# for every session and the disk prefix cache restores it.  The model still sees the same lines, one turn later.
ENV_TO_USER = False
_ENV_BLOCK = re.compile(r"(?:[^\n]*environment you are running in:\n)?<env>\n.*?\n</env>\n?", re.S)


def _env_to_user(messages: list[dict]) -> list[dict]:
    if not ENV_TO_USER or not messages or messages[0].get("role") != "system":
        return messages
    sys_text = messages[0].get("content")
    if not isinstance(sys_text, str):
        return messages
    m = _ENV_BLOCK.search(sys_text)
    if m is None:
        return messages
    for i in range(1, len(messages)):
        if messages[i].get("role") != "user":
            continue
        block = m.group(0).rstrip("\n")
        c = messages[i].get("content")
        if isinstance(c, str):
            nc = block + "\n\n" + c
        elif isinstance(c, list):
            nc = [{"type": "text", "text": block + "\n\n"}] + c
        else:
            return messages
        out = list(messages)
        out[0] = dict(messages[0], content=sys_text[:m.start()] + sys_text[m.end():])
        out[i] = dict(messages[i], content=nc)
        return out
    return messages


def _late_system_to_user(messages: list[dict]) -> list[dict]:
    """The chat template takes a system message only at the start ("System message must be at the beginning").
    Clients also send them mid-conversation - Claude Code's hook context as {"role": "system"} after the first user
    turn, some OpenAI clients a late "developer" message (issue #56) - so those become user messages, in place:
    merging them into the first one would change the prompt's start and cost the conversation cache every turn."""
    return [dict(m, role="user") if m.get("role") == "system" and i > 0 else m for i, m in enumerate(messages)]


def _object_list(value, name: str) -> list[dict]:
    """#460: a request's "messages" (or a message's "tool_calls") as a list of objects.  Some clients send the array
    double-encoded, as a JSON string, which used to be iterated character by character and crashed on m.get: such a
    string is decoded.  Anything that is still not a list of objects is a ValueError, which the server answers with
    a 400 naming the field.  None (or no field) is an empty list, as a missing field always was."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError(f"{name} must be a list of objects (a string was sent that is not JSON)") from None
    if not isinstance(value, list) or not all(isinstance(m, dict) for m in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


_TOOL_ORDERS: "OrderedDict[frozenset, tuple]" = OrderedDict()   # a set of tool names -> the order it first came in
_TOOL_ORDERS_LOCK = threading.Lock()
_TOOL_ORDERS_MAX = 64


def _stable_tool_order(tools: list[dict], name_of) -> list[dict]:
    """The same set of tools comes back in the order it was first seen.  Clients send their tools in the order their
    MCP servers finished registering, which can differ from one request (or one client restart) to the next, and the
    template renders the tool list before the system prompt: one swap moves the shared prefix back thousands of
    tokens and the whole conversation is read again (opencode issue 23571).  A request's first sight of a tool set
    is untouched (so its prompt is byte for byte what the client sent); later requests with the same names in
    another order are put back in that first order.  Duplicate names, or a single tool: left as sent."""
    if len(tools) < 2:
        return tools
    names = [name_of(t) for t in tools]
    if len(set(names)) != len(names):
        return tools
    key = frozenset(names)
    with _TOOL_ORDERS_LOCK:
        order = _TOOL_ORDERS.get(key)
        if order is None:
            _TOOL_ORDERS[key] = tuple(names)
            while len(_TOOL_ORDERS) > _TOOL_ORDERS_MAX:
                _TOOL_ORDERS.popitem(last=False)
            return tools
        _TOOL_ORDERS.move_to_end(key)
    pos = {n: k for k, n in enumerate(order)}
    return sorted(tools, key=lambda t: pos[name_of(t)])


def _tool_list(value, wrapper: str | None) -> list[dict]:
    """#592: a request's "tools" as a list of tool objects, each with a name - in the OpenAI shape
    {"type": "function", "function": {"name": ...}} (`wrapper` "function"; a bare {"name": ...} is still taken), or
    in the Anthropic shape {"name": ...} (`wrapper` None).  A value that is not (a string such as "auto", a list of
    names, an object without a name) is a ValueError - a 400 naming the field - where it used to take the request
    thread down with no reply at all.  No value (or an empty one) is no tools, as always.  A tool's schema -
    "parameters" in the OpenAI shape, "input_schema" in the Anthropic one - is an object when it is there, for the
    same reason: a string or a list passed and raised later instead, on the model's first call of that tool."""
    if not value:
        return []
    shape = ('{"type": "function", "function": {"name": ..., "parameters": {...}}}' if wrapper else
             '{"name": ..., "input_schema": {...}}')
    try:
        tools = _object_list(value, "tools")
    except ValueError:
        raise ValueError(f"tools must be a list of tool objects, each {shape}") from None
    key = "parameters" if wrapper else "input_schema"
    for i, t in enumerate(tools):
        fn = t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ValueError(f"tools[{i}] has no name: each tool must be {shape}")
        schema = fn.get(key)
        if schema is not None and not isinstance(schema, dict):
            # it reached parse_tool_call and the stream parser as the tool's schema, whose .get("properties") raised
            # AttributeError in the request thread - after the 200 and whatever the model had said before the call
            raise ValueError(f'tools[{i}] ({fn["name"]}): "{key}" must be an object (the JSON schema of its '
                             f"parameters), not {type(schema).__name__}")
    return _stable_tool_order(tools, lambda t: (t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t)["name"])


def tool_arguments(raw) -> dict:
    """A call's arguments from a client's history -> the mapping the template renders (both APIs).  A JSON object
    (as text or already parsed) is used as it is; empty is {}; anything else (cut-off or malformed JSON, a list, a
    number) is kept as text under "arguments" instead of refusing the whole conversation (#510)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return {"arguments": raw}
        return value if isinstance(value, dict) else {"arguments": raw}
    return {"arguments": json.dumps(raw, ensure_ascii=False)}


def require_messages(messages: list) -> None:
    """A chat request with no message at all is a 400 (OpenAI: "empty array. Expected an array with minimum length 1"); the
    template would otherwise render just the assistant turn and the model would answer nothing."""
    if not messages:
        raise ValueError("messages must be a non-empty list: send at least one message")


def openai_to_messages(req: dict) -> tuple[list[dict], list[dict] | None, dict]:
    """OpenAI Chat Completions -> (template messages, template tools, template kwargs)."""
    messages = []
    for i, m in enumerate(_object_list(req.get("messages"), "messages")):
        role = m.get("role")
        if role == "developer":
            role = "system"
        # an image is read in a user, tool or assistant message; the system prompt takes text only, as _text_of reads
        images = role in ("user", "tool", "assistant")
        content = _check_parts(m.get("content"), f"messages[{i}].content",
                               TEXT_PARTS + IMAGE_PARTS if images else TEXT_PARTS)
        out = {"role": role, "content": _parts_of(content) if images else _text_of(content)}
        if m.get("reasoning_content"):
            out["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            calls = []
            for c in _object_list(m["tool_calls"], "tool_calls"):
                fn = c.get("function", c)
                if not isinstance(fn, dict):
                    raise ValueError("tool_calls must be a list of objects (each with a \"function\" object)")
                calls.append({"function": {"name": fn.get("name"), "arguments": tool_arguments(fn.get("arguments"))}})
            out["tool_calls"] = calls
        messages.append(out)
    tools = [t.get("function", t) if t.get("type") == "function" else t
             for t in _tool_list(req.get("tools"), "function")] or None
    kwargs = {}
    # OpenAI Chat Completions: "reasoning_effort"; Responses style: "reasoning": {"effort": ...}
    reasoning = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else {}
    given = req.get("reasoning_effort")
    # `false` is a level (thinking off, effort_kwargs): an `or` between the two fields dropped it
    kwargs.update(effort_kwargs(given if given is not None and given != "" else reasoning.get("effort")))
    # the vLLM / llama.cpp convention: {"chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"}}
    for k, v in (req.get("chat_template_kwargs") or {}).items():
        if k == "enable_thinking" and not v:
            kwargs = {"enable_thinking": False}
        elif k == "reasoning_effort" and "enable_thinking" not in kwargs:
            kwargs.update(effort_kwargs(v))
    return _env_to_user(_late_system_to_user(messages)), tools, kwargs


BILLING_HEADER = "x-anthropic-billing-header:"


def pin_billing_stamp(system: str) -> str:
    """Claude Code starts its system prompt with `x-anthropic-billing-header: cc_version=2.1.170.bf4;
    cc_entrypoint=sdk-cli; cch=b145e;`.  cch changes on EVERY request and the version's 4th part on every session, so
    the prompt changed ~22K tokens in (after the tool list) on every turn and the conversation cache could only reuse
    up to its last 16K checkpoint: half of every agent prompt was read again.  Both stamps are pinned to f's, as
    llama.cpp does (ggml-org/llama.cpp#21793); only a header at the very start of the system text is touched, and
    only inside its first 160 characters."""
    if not system.startswith(BILLING_HEADER):
        return system
    s = list(system)
    cch = system.find("cch=", len(BILLING_HEADER))
    if 0 <= cch <= 160:
        v, end = cch + 4, system.find(";", cch + 4)
        if end > v and end - v <= 16:
            s[v:end] = "f" * (end - v)
    cv = system.find("cc_version=")
    if 0 <= cv <= 160:
        v, end = cv + len("cc_version="), system.find(";", cv)
        if v < end and end - v <= 64:
            parts = system[v:end].split(".")
            if len(parts) > 3:
                tail = v + len(".".join(parts[:3])) + 1
                s[tail:end] = "f" * (end - tail)
    return "".join(s)


def anthropic_to_messages(req: dict, think_unasked: bool = True) -> tuple[list[dict], list[dict] | None, dict]:
    """Anthropic Messages -> (template messages, template tools, template kwargs).  `think_unasked`: a request
    without "thinking", an effort or a budget gets the template's default (it thinks), as through 0.1.31; False
    renders it without thinking (#278, the config's "anthropic_thinking": "on_request")."""
    messages = []
    system = req.get("system")
    if system:
        messages.append({"role": "system",
                         "content": pin_billing_stamp(_text_of(_check_parts(system, "system", TEXT_PARTS)))})
    for i, m in enumerate(_object_list(req.get("messages"), "messages")):
        content = _check_parts(m.get("content"), f"messages[{i}].content", ANTHROPIC_BLOCKS)
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        if m.get("role") == "user" and _has_image(content) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            messages.append({"role": "user", "content": _parts_of(content)})
            continue
        text, reasoning, calls, parts = [], [], [], []
        for j, block in enumerate(content or []):
            kind = block.get("type")
            if kind == "text":
                text.append(block.get("text", ""))
                parts.append(block)
            elif kind in IMAGE_PARTS:
                parts.append(block)
            elif kind == "thinking":
                reasoning.append(block.get("thinking", ""))
            elif kind == "tool_use":
                calls.append({"function": {"name": block.get("name"), "arguments": tool_arguments(block.get("input"))}})
            elif kind == "tool_result":
                # A tool's image (Claude Code's Read of a picture) reaches the encoder like a user's, as the OpenAI
                # path's tool messages already do; a text-only result is one string, as before.
                messages.append({"role": "tool", "content": _parts_of(_check_parts(
                    block.get("content"), f"messages[{i}].content[{j}].content", TEXT_PARTS + IMAGE_PARTS))})
        if text or calls or reasoning or parts:
            # an image sent beside tool results stays in this turn instead of being dropped
            out = {"role": m["role"], "content": _parts_of(parts) if _has_image(parts) else "".join(text)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
    tools = [{"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}
             for t in _tool_list(req.get("tools"), None)] or None
    kwargs = {}
    # Anthropic: "thinking": {"type": "disabled"} or {"type": "enabled", "budget_tokens": N};
    # "output_config": {"effort": "low" | "medium" | "high"}
    thinking = req.get("thinking")
    effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
    # 0.1.45: Anthropic's budget_tokens is a real budget, not only a level: the thinking is closed (wrap-up, </think>,
    # the answer) after that many tokens.  A reasoning_budget_tokens of the request's own wins.
    bt = thinking.get("budget_tokens") if isinstance(thinking, dict) and thinking.get("type") != "disabled" else None
    if isinstance(bt, int) and not isinstance(bt, bool) and bt > 0 and req.get("reasoning_budget_tokens") is None:
        req["reasoning_budget_tokens"] = bt
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        kwargs["enable_thinking"] = False
    elif effort:
        kwargs.update(effort_kwargs(effort))
    elif isinstance(thinking, dict) and thinking.get("budget_tokens"):
        kwargs.update(budget_effort(thinking["budget_tokens"]))
    elif thinking is None and not req.get("reasoning_budget_tokens") and not think_unasked:
        # Opt-in (the config's "anthropic_thinking": "on_request"; the default thinks as 0.1.31 did, since a
        # client that never asks would otherwise lose the thinking on every turn).  Anthropic's thinking is
        # opt-in there. Claude Code's helper calls (a session title, a topic check) ask for none
        # and allow a few dozen tokens, which the model otherwise spent thinking and answered with no text at all.
        # A config's reasoning_effort still applies: Service.with_shared sets output_config before this runs.  A
        # request that gives its own reasoning_budget_tokens (#123) asks for thinking, so it thinks as before.
        kwargs["enable_thinking"] = False
    return _env_to_user(_late_system_to_user(messages)), tools, kwargs


# ------------------------------------------------------------------------------------------------ output parser
@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])


@dataclass
class Event:
    kind: str                     # "reasoning" | "content" | "tool_call"
    text: str = ""
    call: ToolCall | None = None


THINK_END = "</think>"
CALL_START = "<tool_call>"
CALL_END = "</tool_call>"


PARAM_END = "</parameter>"
FUNC_START = "<function="
FUNC_END = "</function>"
# #3441: the opener a model writes when it drifts by a token inside the envelope the template documents:
# `<=NAME>` (the word `function` dropped), `<function:NAME>`, `<function =NAME>`, or the bare `<NAME>`
_LOOSE_OPEN = re.compile(r"<\s*(?:function\s*[=:]\s*|=\s*)?([A-Za-z0-9_.\-]+)\s*>")
_LOOSE_PART = re.compile(r"<\s*(?:function[ \t]*[=:]?\s*|=\s*)?[A-Za-z0-9_.\-]*\s*$")


# what ends a value after its `</parameter>`: the call format's own tags, and the Anthropic XML form's (`</invoke>`,
# `<parameter name=`) that Coder-L writes in place of `</function>` (seen live: `limit` = "25\\n</parameter>\\n</invoke>")
FOLLOWERS = (FUNC_END, CALL_END, "<parameter=", "</<", "</invoke>", "<parameter name=")


# the model's empty stub: a lone `<` line closed by a parameter closer, after a value or right after the function opener
# (or a lone `<` line, left open: whatever follows it is the next opener or closer)
# (or a stray further closer, or an opener cut before its `>`: `<parameter=NAME` + line break)
STUB = re.compile(r"(?:</parameter>\s*)+|<(?:\s*</parameter>\s*|[ \t]*\n\s*(?:</parameter>\s*)?)")


def broken_opener(s: str) -> int:
    """Length of the `<parameter=NAME` line of `s` when its `>` is missing (the line ends first), else 0."""
    line = s.split("\n", 1)
    return len(line[0]) if s.startswith("<parameter=") and len(line) > 1 and ">" not in line[0] else 0


def stub_follows(after: str):
    """`after` (no leading whitespace) begins with a stub: True when what ends the value is certain, None when the
    follower of a lone `<` line has not arrived, False when text follows it (a `<` line of the value itself);
    False too when there is no stub."""
    m = STUB.match(after)
    if not m:
        return False
    rest = after[m.end():]
    if m.group().rstrip().endswith(PARAM_END) or rest.startswith(FOLLOWERS):
        return True
    return None if not rest or any(t.startswith(rest) for t in FOLLOWERS) else False


def stub_pending(text: str) -> bool:
    """`text` (no leading whitespace) is a stub still arriving: `<` and whitespace, then a prefix of the closer."""
    return text.startswith("<") and PARAM_END.startswith(text[1:].lstrip()) and not text.startswith("<parameter")


def param_end(text: str, final: bool = False, names=None) -> int:
    """Where a parameter value in `text` ends: the first `</parameter>` followed (after whitespace) by a call-format
    follower - `</function>`, `</tool_call>`, `<parameter=` (any name, with or without its `>`), a damaged `<=NAME>`,
    a garbled `</<`, the Anthropic form's `</invoke>` / `<parameter name=` - or by the end of the text when `final`.  (A bare `<NAME>` or `</NAME>` of a declared parameter not
    set yet is rewritten to `<parameter=NAME>` before this, fix_param_openers.)  Any other tag or plain text after it stays in
    the value: a file that documents the call format (#210) or holds markup is part of the value, even a whole call
    written there: a `<function=` in the value that is not closed yet owns the next closer.
    -1: none yet; -2: a candidate whose follower has not arrived (streaming; `final` accepts it)."""
    at, inner_end = text.find(PARAM_END), -1
    while at >= 0:
        after = text[at + len(PARAM_END):].lstrip()
        if after.startswith(FOLLOWERS) or stub_follows(after):
            follows = True
        elif after.startswith("<="):        # a damaged `<=NAME>` opener, maybe still arriving
            follows = True if ">" in after.split("\n", 1)[0] else (None if "\n" not in after else False)
        elif not after or PARAM_END.startswith(after) or stub_pending(after) or stub_follows(after) is None or any(t.startswith(after) for t in (*FOLLOWERS, "<=")):
            follows = None                  # the follower has not arrived
        elif names and "\n" not in after and ">" not in after and any(
                t.startswith(after) for n in names for t in ("<%s>" % n, "</%s>" % n)):
            follows = None                  # a bare `<NAME>` / `</NAME>` opener still arriving
        else:
            follows = False
        if follows and text[:at].count(FUNC_START) > text[:at].count(FUNC_END):
            inner_end, follows = at, False  # closes the parameter of a call written inside the value
        if follows:
            return at
        if follows is None:
            return at if final else -2
        at = text.find(PARAM_END, at + 1)
    return inner_end if final else -1


def held_from(text: str, names=None) -> int:
    """Where the text a parameter closer still holds begins: the last `</parameter>` whose follower has not arrived
    (param_end answers -2), or a cut-off prefix of one; len(text) when nothing is held.  Nothing from there on is
    value until param_end decides (final=True: at the end of the output)."""
    at = text.rfind(PARAM_END)
    if at >= 0 and param_end(text[at:], names=names) == -2:
        return at
    return len(text) - max((k for k in range(1, len(PARAM_END)) if text.endswith(PARAM_END[:k])), default=0)


def empty_value(rest: str) -> bool:
    """A value that begins with the call's own `</function>`: the parameter is empty, its closer left out."""
    return rest.lstrip().startswith(FUNC_END)


def split_second_call(buf: str, final: bool = False) -> str:
    """recover: a `<tool_call>` at a line start inside a parameter's value, whose call is closed and not followed by
    a later `</parameter>` of the value, is a second call the model began before closing the first: the first gets its
    closers.  (One that a closer follows is part of the value, #210.)  A call whose end has not arrived
    or whose follower has not is left alone until `final`."""
    for m in re.finditer(r"\n[ \t]*" + re.escape(CALL_START) + r"\s*" + re.escape(FUNC_START), buf):
        k = m.start()
        if buf.rfind(PARAM_START, 0, k) < 0 or buf.rfind(PARAM_START, 0, k) < buf.rfind(PARAM_END, 0, k):
            continue
        j = buf.find(CALL_END, k)
        if j < 0 and not final:
            return buf
        tail = buf[j + len(CALL_END):].lstrip() if j >= 0 else ""
        if j >= 0 and PARAM_END in tail:
            continue
        if j >= 0 and not final:
            return buf
        return buf[:k] + "\n" + PARAM_END + "\n" + FUNC_END + "\n" + CALL_END + buf[k:]
    return buf


NOT_PARAMS = ("function", "think", "parameter")


def cut_name(text: str) -> str:
    """A tag's name as written after `=`: it ends at the first whitespace, `<` or `>` (the one cut for function and
    parameter names)."""
    return re.split(r"[\s<>]", text, 1)[0]


def func_name(text: str) -> str:
    """The tool a `<function=` opener names (`text`: what follows `=`); a stray space before the name is skipped."""
    return cut_name(text.lstrip())


def param_name(opener: str, names=None) -> str | None:
    """The parameter a `<parameter=` opener names (`opener`: the text between `=` and `>`), cut by cut_name.  None -
    the parameter is dropped - when that is empty, one of the call's own tags (NOT_PARAMS), or not among `names`
    (when there are any)."""
    name = cut_name(opener)
    if not name or name in NOT_PARAMS or (names and name not in names):
        return None
    return name


def trim_value(value: str, name: str | None, names=None) -> str:
    """A parameter value as written: the newline before the closer is not part of it; nor is a closer `</NAME>` of
    the parameter (or of a declared one) written instead of `</parameter>`.  (The newline after the opener is the
    caller's: the streaming parser has skipped it already.)"""
    if value.endswith("\n"):
        value = value[:-1]
    for n in {name, *(names or ())} - {None}:
        if value.endswith("</%s>" % n):
            value = value[:-len(n) - 3]
            return value[:-1] if value.endswith("\n") else value
    return value


def _names_of(schemas, head: str) -> set:
    """The declared parameter names of the tool a call opener's name (`head`) names; empty when unknown."""
    props = (((schemas or {}).get(head.strip()) or {}).get("parameters") or {}).get("properties") or {}
    return set(props)


def call_end(text: str, schemas: dict | None = None, final: bool = False) -> int:
    """Where a tool call's body ends (#210): the `</tool_call>` after the call's own `</function>`, found by walking
    its parameters with param_end, so a value may contain either tag.  -1: not complete yet.  A body that is not in
    the call format ends at the first `</tool_call>`, as before."""
    s = text.lstrip()
    pos = len(text) - len(s)
    if not s.startswith("<function="):
        return text.find(CALL_END) if not "<function=".startswith(s) else -1
    gt = text.find(">", pos)
    if gt < 0:
        return -1
    names = _names_of(schemas, text[text.index(FUNC_START, pos) + len(FUNC_START):gt])
    pos = gt + 1
    while True:
        rest = text[pos:]
        s = rest.lstrip()
        pos += len(rest) - len(s)
        stub = STUB.match(s) if stub_follows(s) else None
        if stub:
            pos += stub.end()
        elif broken_opener(s):
            pos += broken_opener(s)
        elif s.startswith("<parameter="):
            gt = text.find(">", pos)
            if gt < 0:
                return -1
            end = param_end(text[gt + 1:], final, names)
            if end < 0:
                return -1
            pos = gt + 1 + end + len(PARAM_END)
        elif s.startswith(FUNC_END):
            rest = text[pos + len(FUNC_END):]
            s = rest.lstrip()
            if s.startswith(CALL_END):
                return pos + len(FUNC_END) + len(rest) - len(s)
            return -1 if CALL_END.startswith(s) else text.find(CALL_END, pos)
        elif not s or "<parameter=".startswith(s) or FUNC_END.startswith(s) or stub_pending(s):
            return -1
        else:
            return text.find(CALL_END, pos)


def tool_choice_of(tool_choice) -> tuple[str, str | None]:
    """A client's `tool_choice` in either API's shape -> ("auto" | "none" | "required" | "named", name).  OpenAI:
    "auto" / "none" / "required", {"type": "function", "function": {"name": N}} or the flat {"type": "function",
    "name": N}; Anthropic: {"type": "auto" | "any" | "none"} and {"type": "tool", "name": N}.  A value it does not
    know is ("unknown", None)."""
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        return (tool_choice, None) if tool_choice in ("auto", "none", "required") else ("unknown", None)
    if isinstance(tool_choice, dict):
        kind = tool_choice.get("type")
        if kind in ("auto", "none"):
            return kind, None
        if kind in ("any", "required"):
            return "required", None
        if kind in ("function", "tool"):
            fn = tool_choice.get("function")
            name = (fn.get("name") if isinstance(fn, dict) else None) or tool_choice.get("name")
            if isinstance(name, str) and name:
                return "named", name
    return "unknown", None


def forced_call(tool_choice, tools: list[dict] | None) -> str | None:
    """`tool_choice` -> the text that opens the call the reply must make, or None (the model decides).
    There is no grammar here: the server writes this opening itself, so the model can only go on with a call.
    "required" / Anthropic "any": any of the tools; a named function: that one.  A value it cannot honour (an
    unknown shape, a name that is not one of the tools, "required" with no tools) is logged and acts as "auto",
    not a 400: a client's odd choice must not stop its request.  "none" is handled by the caller (no tools)."""
    kind, name = tool_choice_of(tool_choice)
    names = {t.get("name") for t in tools or [] if isinstance(t, dict)}
    if kind in ("auto", "none"):
        return None
    if kind == "required" and len(names) == 1:     # one tool to call: name it, the model cannot invent another
        kind, name = "named", next(iter(names))
    if kind == "required" and names:
        return CALL_START + "\n<function="
    if kind == "named" and name in names:
        return CALL_START + f"\n<function={name}>\n"
    print(f"[strata] tool_choice {json.dumps(tool_choice)[:200]} is not supported here (or names no tool of the "
          "request): the model decides, as with \"auto\"", flush=True)
    return None


PARAM_START = "<parameter="


def function_end(text: str, schemas: dict | None = None, final: bool = False) -> int:
    """Where a `<function=NAME>...</function>` at the start of `text` ends (just past `</function>`), walking its
    parameters with param_end like call_end does; -1: not complete (or not in the call format)."""
    s = text.lstrip()
    pos = len(text) - len(s)
    if not s.startswith(FUNC_START):
        return -1
    gt = text.find(">", pos)
    if gt < 0:
        return -1
    names = _names_of(schemas, text[text.index(FUNC_START, pos) + len(FUNC_START):gt])
    pos = gt + 1
    while True:
        rest = text[pos:]
        s = rest.lstrip()
        pos += len(rest) - len(s)
        stub = STUB.match(s) if stub_follows(s) else None
        if stub:
            pos += stub.end()
        elif broken_opener(s):
            pos += broken_opener(s)
        elif s.startswith(PARAM_START):
            gt = text.find(">", pos)
            if gt < 0:
                return -1
            end = param_end(text[gt + 1:], final, names)
            if end < 0:
                return -1
            pos = gt + 1 + end + len(PARAM_END)
        elif s.startswith(FUNC_END):
            return pos + len(FUNC_END)
        else:
            return -1


_JSON_FENCE = re.compile(r"```[ \t]*(?:json)?[ \t]*\r?\n\s*", re.I)


def json_tool_call(body: str, schemas: dict) -> ToolCall | None:
    """`{"name": N, "arguments": {...}}` (or "parameters", or the arguments as a JSON string) of a declared tool -> its
    call; anything else -> None.  #2037: the object may sit in a Markdown fence (```json ... ```)."""
    body = body.strip()
    m = _JSON_FENCE.match(body)
    if m:
        body = body[m.end():].rstrip()
        body = body[:-3] if body.endswith("```") else body
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or obj["name"] not in schemas:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return ToolCall(name=obj["name"], arguments=coerce_args(args, schemas[obj["name"]]))


_BOOLS = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}


def coerce_param(value: str, declared) -> object:
    """A parameter value by its declared JSON-schema type, shared by the whole-call and the streaming parser.
    "string" stays raw; boolean also reads True/False/yes/no/1/0 (any case); integer/number read numeric strings;
    array/object and an undeclared type are JSON-decoded.  Whatever does not convert stays the raw string."""
    if declared == "string":
        return value
    text = value.strip()
    if declared == "boolean" and text.lower() in _BOOLS:
        return _BOOLS[text.lower()]
    try:
        v = json.loads(text)
    except ValueError:
        return value
    if declared == "boolean" and not isinstance(v, bool):
        return value
    return v


def coerce_args(args: dict, schema: dict | None) -> dict:
    """A call's arguments by the tool's declared parameter types - the one place every ToolCall passes through (the JSON
    form, the whole-call parser).  A value that is not a string (a JSON call already carrying a number or boolean) is
    left alone; a string is coerced by its declared type (coerce_param); an undeclared parameter is JSON-decoded."""
    props = ((schema or {}).get("parameters") or {}).get("properties") or {}
    out = {}
    for k, v in args.items():
        declared = (props.get(k) or {}).get("type")
        out[k] = coerce_param(v, declared) if isinstance(v, str) else v
    return out


_FUSED = re.compile(r"<function=([^<>\s=]+)\s*(?=<parameter=)")


def fix_fused(buf: str, schemas: dict) -> str:
    """`<function=Read <parameter=file_path>` - the tool's name (func_name) fused with its first parameter's opener,
    or ended by a line break (the model left out the `>` after the name) - is `<function=Read>\n<parameter=file_path>`
    when the name is a declared tool."""
    return _FUSED.sub(lambda m: "<function=%s>\n" % m.group(1) if func_name(m.group(1)) in schemas else m.group(0), buf)


_DAMAGED_PARAM = re.compile(r"(?:</parameter>|<function=[^>\n]*>)\s*<=([^>\s]+)>")


_BARE_PARAM = re.compile(r"</parameter>\s*</?([^<>\s=/]+)>")


def fix_param_openers(buf: str, schemas: dict, start: int = 0) -> str:
    """recover: inside a call, `<=NAME>` after the function opener or a `</parameter>` is `<parameter=NAME>` when NAME
    is a declared parameter of that call's tool (seen from Coder-L); so is a bare `<NAME>` or `</NAME>` after a `</parameter>` when
    NAME is declared and not set yet in the call.  Only text from `start` on is rewritten."""
    out, last = [], 0
    ms = [(m.start(1) - 2, m) for m in _DAMAGED_PARAM.finditer(buf)]
    ms += [(m.start(1) - (2 if buf[m.start(1) - 1] == "/" else 1), m) for m in _BARE_PARAM.finditer(buf)]
    for cut, m in sorted(ms, key=lambda t: t[0]):
        if cut < start or cut < last:
            continue
        f = buf.rfind(FUNC_START, 0, m.start(1))
        name = func_name(buf[f + len(FUNC_START):]) if f >= 0 else ""
        props = ((schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
        bare = buf[cut] == "<" and buf[cut:cut + 2] != "<="
        if m.group(1) in props and not (bare and PARAM_START + m.group(1) + ">" in buf[f:cut]):
            out += [buf[last:cut], PARAM_START + m.group(1) + ">"]
            last = m.end()
    return buf[:0].join(out) + buf[last:] if out else buf


def parse_tool_call(body: str, schema: dict | None = None) -> ToolCall:
    """`<function=NAME>\\n<parameter=P>\\nVALUE\\n</parameter>...</function>` -> ToolCall. Values are JSON-decoded
    when the tool's schema says the parameter is not a string (or, without a schema, when they parse as JSON
    objects/arrays/numbers/booleans)."""
    body = body.strip()
    if not body.startswith("<function=") or ">" not in body:
        raise ValueError("malformed tool call: " + body[:80])
    name = func_name(body[len("<function="):])      # `<function= capture>`: the model's stray space
    rest = body[body.index(">") + 1:]
    props = ((schema or {}).get("parameters") or {}).get("properties") or {}
    args = {}
    while "<parameter=" in rest:
        rest = rest[rest.index("<parameter=") + len("<parameter="):]
        gt = rest.find(">")
        if gt < 0:      # cut inside the parameter name: drop the opener, keep the complete parameters
            break
        if "\n" in rest[:gt]:      # an opener cut before its `>`: the line is dropped
            continue
        pname = param_name(rest[:gt], props)
        rest = rest[gt + 1:]
        if empty_value(rest):                   # the call's closer where the value should be: empty, no closer
            value, rest = "", rest[rest.index(FUNC_END):]
        else:
            end = param_end(rest, final=True, names=set(props))
            value = rest[:end] if end >= 0 else rest
            rest = rest[end + len(PARAM_END):] if end >= 0 else ""
        if pname:
            args[pname] = trim_value(value.removeprefix("\n"), pname, props)
    return ToolCall(name=name, arguments=coerce_args(args, schema))


RCALL_MAX = 32768        # the most of a `<tool_call>` inside the reasoning that is held while waiting for its end (#804)


class OutputParser:
    """Incremental parser of the model's text. Feed deltas; get events. A tag split across deltas is held back
    until it is complete, so clients never see `<tool_` or `</thi`."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False,
                 recover: bool = False, declared_only: bool = False):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.lead = False
        self.schemas = {t.get("name"): t for t in tools or []}
        # #804/#3075: in the answer too, a call counts only for a tool the request declared.  With tools declared a
        # call for another name stays text (the Responses spelling `namespace__tool` of a `namespace.tool` counts as
        # that tool); `declared_only` extends the rule to a request that declared none (Chat Completions, Anthropic:
        # the client has no tool to run).  The Responses route leaves it off: a Codex compaction request keeps its
        # conversation's tools in the prompt only, and its calls come back as before.
        self.declared_only = declared_only
        # PR #3556: the native namespace__member spelling of a declared namespace.member tool is accepted without advertising a
        # duplicate tool.  Only an unambiguous alias counts (two namespaces can end in the same member), and an exactly declared
        # name always wins over an alias.
        aliases = {}
        for name in self.schemas:
            if isinstance(name, str) and "." in name:
                namespace, member = name.rsplit(".", 1)
                aliases.setdefault(f"{namespace}__{member}", set()).add(name)
        self.aliases = {alias: next(iter(names)) for alias, names in aliases.items()
                        if len(names) == 1 and alias not in self.schemas}
        self.unread = 0              # #3441: `<tool_call>` blocks that opened like a call and were returned as text
        self.unread_names: list[str] = []   # the names (first few) of those that named a tool the request did not declare
        # stream_tools: a tool call is also reported while it is being written - "tool_start" (its name and id) as
        # soon as the name is known, then "tool_args" pieces of its JSON arguments (string parameters character by
        # character; other types whole, once complete) - before the final "tool_call".  Without it, a client sees
        # nothing until the call is complete, which for a large file write can be many minutes.
        self.stream_tools = stream_tools
        # #804/#1058: calls found inside the reasoning wait here as [raw text, ToolCall | None] until the turn shows
        # they were acts: only whitespace (or more calls) after them, then the end of the turn or `</think>`.  The
        # reasoning text before them is tracked (code fence, inline code, current line) to tell an act from a quote.
        self.pending: list[list] = []
        self.rescued = 0             # calls delivered from the reasoning
        self.refused = 0             # declared calls kept as reasoning text (quoted, or the turn was cut)
        self.fence, self.line, self.ticks = "", "", 0
        # recover (opt-in, the config's "tool_call_recovery"): a declared tool's call written next to the template's
        # form is the call - `<parameter=NAME>` or `<=NAME>` as the opener, JSON inside <tool_call>, a bare <function=NAME> at the
        # start of a line outside code, or a second call in the same <tool_call> (which otherwise merges into one).
        self.recover = recover
        self.alt_end = ""            # the closer `</NAME>` of a call opened as `<NAME>`
        self.bare = False            # the call being read opened with <function= alone (no <tool_call> around it)
        self.batch = False           # the call being finished is followed by another in the same wrapper
        self._reset_scan()

    @property
    def clean(self) -> bool:
        """Nothing is held back: a clean point to stop the thinking at (a parser that rewrites the model's text, as
        serve/dsml.py's does, also counts what it holds)."""
        return not self.buf

    def _ok_at(self, p: int) -> bool:
        """self.buf[p] would open a call: at the start of a line, outside a code fence and inline code."""
        snap = (self.fence, self.line, self.ticks)
        self._track(self.buf[:p])
        ok = self._opener_ok()
        self.fence, self.line, self.ticks = snap
        return ok

    def _callable(self, name: str) -> bool:
        """#3075: may a call for `name` be a call?  A tool the request declared (or its Responses spelling); with
        none declared, only where `declared_only` is off."""
        if not self.schemas:
            return not self.declared_only
        return name in self.schemas or name in self.aliases

    def _loose_open(self, after: str):
        """#3441: `after` (text after a `<tool_call>`, whitespace dropped) opens a call in a drifted form that names a
        declared tool -> (name, length of the opener); "wait" while it may still become one; None otherwise."""
        m = _LOOSE_OPEN.match(after)
        if m:
            name = m.group(1)
            return (name, m.end()) if self._callable(name) else None
        if ">" in after or "\n" in after or not after.startswith("<"):
            return None
        if _LOOSE_PART.match(after):
            bare = after[1:].strip().lstrip("=").strip()
            if after[1:].lstrip().startswith(("=", "function")):
                return "wait"
            if any(k.startswith(bare) for k in list(self.schemas) + list(self.aliases) if isinstance(k, str)):
                return "wait"
        return None

    def _follower(self, after: str) -> str:
        """recover: what a `<tool_call>` followed (after whitespace) by `after` opens: "call" (`<function=`), "param"
        (a declared tool's name written as `<parameter=NAME>`), "json" (`{`), "wait" (not decided yet) or "text"."""
        if after.startswith(FUNC_START):
            return "call"
        if after.startswith(PARAM_START):
            gt = after.find(">")
            if gt < 0:
                return "wait" if "\n" not in after else "text"
            return "param" if after[len(PARAM_START):gt] in self.schemas else "text"
        if after.startswith("<="):
            # `<=NAME>`: the template's `<function=NAME>` with its "function" token lost (seen from Coder-L)
            gt = after.find(">")
            if gt < 0:
                return "wait" if "\n" not in after else "text"
            return "eq" if after[2:gt] in self.schemas else "text"
        if after.startswith("<") and after[1:2].isalpha():
            # `<NAME>`: the bare tool name as the tag (seen from Coder-L), one line, a declared tool only
            gt = after.find(">")
            if gt < 0:
                return "wait" if "\n" not in after else "text"
            return "name" if after[1:gt] in self.schemas else "text"
        if after.startswith("{"):
            return "json" if self.schemas else "text"
        if after.startswith("```") or (after and "```".startswith(after)):
            # #2037: the JSON call inside a Markdown fence (```json ... ```) in the explicit wrapper
            m = _JSON_FENCE.match(after)
            if m:
                rest = after[m.end():]
                if rest.startswith("{"):
                    return "json" if self.schemas else "text"
                return "wait" if not rest else "text"
            return "wait" if "\n" not in after else "text"
        if not after or FUNC_START.startswith(after) or PARAM_START.startswith(after) or after == "<=":
            return "wait"
        return "text"

    def _bare_opener(self) -> tuple[int, bool]:
        """recover: the first `<function=` in self.buf that opens a call without the wrapper (start of a line, outside
        code, a declared tool) -> (position, decided); (-1, False) for none.  A name still arriving is undecided."""
        p = self.buf.find(FUNC_START)
        while p >= 0:
            if self._ok_at(p):
                gt = self.buf.find(">", p)
                raw_name = self.buf[p + len(FUNC_START):gt if gt >= 0 else len(self.buf)]
                name = raw_name.strip()
                name = self.aliases.get(name, name)
                if gt < 0 and re.fullmatch(r"\s*[^<>\s]*\s*", raw_name):
                    return p, False
                if gt >= 0 and name in self.schemas:
                    return p, True
            p = self.buf.find(FUNC_START, p + 1)
        return -1, False

    @property
    def line(self) -> str:
        """Materialize the current line only for a decision, newline, or state snapshot."""
        if self._line_blocks is not None:
            self._line_blocks.extend(self._line_parts)
            self._line_parts = ["".join(self._line_blocks)]
            self._line_blocks = None
        elif len(self._line_parts) > 1:
            self._line_parts = ["".join(self._line_parts)]
        return self._line_parts[0] if self._line_parts else ""

    @line.setter
    def line(self, text: str):
        self._line_parts = [text] if text else []
        self._line_blocks = None

    def _track(self, text: str) -> str:
        """Follow the reasoning text that has gone out: the open code fence, the current line, and the backticks of
        the current paragraph.  Returns the text."""
        # Most token fragments stay on the current line. Fence and paragraph
        # bookkeeping happens only when a newline completes that line.
        if "\n" not in text:
            if text:
                self._line_parts.append(text)
                if len(self._line_parts) == 64:
                    # Release small fragment objects in bounded groups. Completed
                    # blocks are never recopied until the full line is requested.
                    if self._line_blocks is None:
                        self._line_blocks = []
                    self._line_blocks.append("".join(self._line_parts))
                    self._line_parts.clear()
            return text
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k < len(parts) - 1:
                line = self.line + part if self._line_parts or self._line_blocks else part
                self._line_parts.clear()
                self._line_blocks = None
                s = line.lstrip()
                if self.fence:
                    if s.startswith(self.fence * 3):
                        self.fence = ""
                elif s[:3] in ("```", "~~~") and (s[0] * 3) not in s[3:]:
                    self.fence = s[0]
                elif not s:
                    self.ticks = 0
                else:
                    self.ticks += line.count("`")
            else:
                if part:
                    self._line_parts.append(part)
        return text

    def _opener_ok(self) -> bool:
        """The reasoning text so far puts a `<tool_call>` at the start of a line, outside a code fence and outside
        inline code."""
        return not self.fence and not self.line.strip() and self.ticks % 2 == 0

    def _in_code(self) -> bool:
        """The text so far leaves the next character inside a code fence or inline code."""
        return bool(self.fence) or (self.ticks + self.line.count("`")) % 2 == 1

    def _quoted_think_end(self, i: int) -> bool | None:
        """#537: the `</think>` at self.buf[i] is the model quoting the tag while it thinks, not the end of the
        thinking: written right after a quote or backtick and followed by anything but a line break, as in `</think>`
        or "</think>...".  The end of the thinking is followed by a line break.  None while the character after the
        tag has not arrived."""
        before = self.buf[i - 1] if i else self.line[-1:]
        if before not in ('"', "'", "`"):
            return False
        after = self.buf[i + len(THINK_END):]
        if not after:
            return None
        return after[0] not in "\r\n"

    def _release(self, deliver: bool) -> list[Event]:
        """Settle the calls waiting in self.pending: events for real calls, or all of it back as reasoning text."""
        out = []
        for raw, call in self.pending:
            if call is not None and deliver:
                out.append(Event("tool_call", call=call))
                self.rescued += 1
            elif call is None or not deliver:
                if call is not None:
                    self.refused += 1
                if raw:
                    out.append(Event("reasoning", self._track(raw)))
        self.pending = []
        return out

    def _reset_scan(self):
        self._call_end_checked = 0   # closing-tag search progress for the current wrapped call
        self.sp = 0                  # how much of self.buf (the call body) the scanner has consumed
        self.ss = "name"             # name -> between -> str|raw -> ... -> function_done -> done
        self.scall = None            # the ToolCall being streamed (its id is reused by the final event)
        self.sjson = ""              # the JSON of its arguments as streamed: the client's view of them
        self.sstarted = False        # "tool_start" went out: held until the call has a parameter (or is closed whole),
                                     # so a call cut before that never leaves an empty block behind
        self.sfirst = True
        self.sval_started, self.sval_at = False, 0
        self.sdeclared = {}
        self.spname = ""

    def _wrapped_call_end(self) -> int:
        # Without a closing tag the structural walk cannot finish. Search only
        # newly appended text plus enough overlap for a tag split across feeds.
        # A literal closing tag inside a value needs structural disambiguation:
        # once any candidate appears, keep the original full walk for this call.
        # Recovery can rewrite the buffer, so it always uses the full walk.
        if not self.recover and self._call_end_checked >= 0:
            start = max(0, self._call_end_checked - len(CALL_END) + 1)
            if self.buf.find(CALL_END, start) < 0:
                self._call_end_checked = len(self.buf)
                return -1
            self._call_end_checked = -1
        return call_end(self.buf, self.schemas)

    def _scan(self) -> list[Event]:
        """Advance the streaming view of the call body in self.buf (see stream_tools)."""
        out = []

        def args(s):
            if s:
                if not self.sstarted:
                    self.sstarted = True
                    out.append(Event("tool_start", call=self.scall))
                    self.sjson = "{"
                    out.append(Event("tool_args", "{", call=self.scall))
                self.sjson += s
                out.append(Event("tool_args", s, call=self.scall))
        while True:
            rest = self.buf[self.sp:]
            if self.ss == "name":
                a = rest.find("<function=")
                b = rest.find(">", a + 10) if a >= 0 else -1
                if b < 0:
                    return out
                raw_name = rest[a + 10:b]
                name = func_name(raw_name)
                name = self.aliases.get(name, name)
                if raw_name != raw_name.strip() and name not in self.schemas:
                    return out
                self.scall = ToolCall(name=name, arguments={})
                props = ((self.schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
                self.sdeclared = {k: (v or {}).get("type") for k, v in props.items()}
                self.sp += b + 1
                self.ss = "between"
            elif self.ss == "between":
                stripped = rest.lstrip()
                self.sp += len(rest) - len(stripped)
                stub = STUB.match(stripped) if stub_follows(stripped) is not False else None
                if stub:
                    if stub_follows(stripped) is None:
                        return out          # a lone `<` line: what follows it has not arrived
                    self.sp += stub.end()
                elif broken_opener(stripped):
                    self.sp += broken_opener(stripped)
                elif stripped.startswith("<parameter="):
                    b = stripped.find(">")
                    if b < 0:
                        return out
                    pname = param_name(stripped[11:b], self.sdeclared)
                    self.sp += b + 1
                    self.spname = pname
                    if pname is None:
                        self.ss = "drop"        # no parameter of this call: its value is read and left out
                        continue
                    args(("" if self.sfirst else ",") + json.dumps(pname) + ":")
                    self.sfirst = False
                    if self.sdeclared.get(pname) == "string":
                        args('"')
                        self.ss, self.sval_started = "str", False
                    else:
                        self.ss = "raw"
                elif stripped.startswith("</function>"):
                    self.sp += len("</function>")
                    self.ss = "function_done"
                elif any(t.startswith(stripped) for t in (PARAM_START, FUNC_END, PARAM_END, "<")):
                    return out          # a tag still arriving: wait
                else:                   # text no parameter owns (a bare `<NAME>`, `</invoke>`): read over, as parse_tool_call does
                    at = [i for i in (stripped.find(PARAM_START, 1), stripped.find(FUNC_END, 1)) if i >= 0]
                    if not at:
                        return out
                    self.sp += min(at)
            elif self.ss == "str":
                if not self.sval_started:
                    if not rest:
                        return out
                    if rest[0] == "\n":          # the value's leading newline is not part of it
                        self.sp += 1
                        rest = rest[1:]
                    self.sval_started, self.sval_at = True, self.sp
                if self.sp == self.sval_at:          # nothing of the value yet: `</function>` here is the call's closer
                    if empty_value(rest):
                        args('"')
                        self.sp += len(rest) - len(rest.lstrip())
                        self.ss = "between"
                        continue
                    if rest.strip() and FUNC_END.startswith(rest.lstrip()):
                        return out
                end = param_end(rest, names=set(self.sdeclared))
                if end >= 0:
                    value = trim_value(rest[:end], self.spname, self.sdeclared)
                    args(json.dumps(value)[1:-1] + '"')
                    self.sp += end + len(PARAM_END)
                    self.ss = "between"
                    continue
                # an undecided </parameter> (-2) is held from its start, like a tag still arriving
                tags = (PARAM_END, *("</%s>" % n for n in self.sdeclared))
                text = rest.rstrip()        # a `</NAME>` closer may be complete and wait for its `</parameter>`
                done = max((len(t) for t in tags[1:] if text.endswith(t)), default=0)
                safe = rest.find(PARAM_END) if end == -2 else (
                    len(text) - done if done else len(rest) - self._hold(rest, tags))
                if self.recover:                # a second call may begin in the value (split_second_call)
                    safe = min(safe, rest.find(CALL_START) if CALL_START in rest else len(rest) - self._hold(rest, (CALL_START,)))
                if safe > 0 and rest[safe - 1] == "\n":   # may be the trailing newline before </parameter>
                    safe -= 1
                if safe > 0:
                    args(json.dumps(rest[:safe])[1:-1])
                    self.sp += safe
                return out
            elif self.ss == "raw":
                end = param_end(rest, names=set(self.sdeclared))
                if end < 0:
                    return out
                value = trim_value(rest[:end].removeprefix("\n"), self.spname, self.sdeclared)
                v = coerce_param(value, self.sdeclared.get(self.spname))
                args(json.dumps(v, ensure_ascii=False))
                self.sp += end + len(PARAM_END)
                self.ss = "between"
            elif self.ss == "drop":
                end = param_end(rest, names=set(self.sdeclared))
                if end < 0:
                    return out
                self.sp += end + len(PARAM_END)
                self.ss = "between"
            else:
                return out

    def _close_scan(self) -> list[Event]:
        """Close streamed JSON only after its call boundary has been confirmed."""
        out = []
        if self.scall is None or self.ss == "done":
            return out
        if not self.sstarted:                   # nothing of the call went out: there is no block to close
            self.ss = "done"
            return out
        tail = ""
        if self.ss == "str":
            tail += '"'
        elif self.ss == "raw":
            tail += json.dumps(self.buf[self.sp:].strip("\n"))
        tail += "}"
        self.sjson += tail
        out.append(Event("tool_args", tail, call=self.scall))
        self.ss = "done"
        return out

    def _hold(self, text: str, tags: tuple[str, ...]) -> int:
        """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
        best = 0
        size = len(text)
        for tag in tags:
            if len(tag) <= 1:
                continue
            # Only a suffix beginning with the tag's first character can match.
            # Ordinary token deltas contain no '<': one C-level search replaces
            # checking every possible prefix. Earliest candidate is the longest.
            start = text.find(tag[0], max(0, size - len(tag) + 1))
            while start >= 0 and size - start > best:
                if tag.startswith(text[start:]):
                    best = size - start
                    break
                start = text.find(tag[0], start + 1)
        return best

    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
        if self.schemas and "<function=" in self.buf:
            self.buf = fix_fused(self.buf, self.schemas)    # a name fused with its first parameter's opener
        out: list[Event] = []
        while True:
            if self.state == "rcall":
                # #804: a `<tool_call>` inside the reasoning, with tools declared.  It is held whole (never streamed
                # as a call) until it ends: a declared name is then a tool call, anything else stays reasoning text.
                body = self.buf[len(CALL_START):]
                end, think = call_end(body, self.schemas), body.find(THINK_END)
                if end >= 0 and (think < 0 or think >= end):
                    call = None
                    try:
                        name = body[:end].strip()[len("<function="):].split(">", 1)[0].strip()
                        name = self.aliases.get(name, name)
                        if name in self.schemas:
                            call = parse_tool_call(body[:end], self.schemas.get(name))
                            call.name = name
                    except ValueError:
                        pass
                    raw = self.buf[:len(CALL_START) + end + len(CALL_END)]
                    if call is not None or self.pending:
                        self.pending.append([raw, call])    # settled by what follows it (see __init__)
                    else:
                        out.append(Event("reasoning", self._track(raw)))
                    self.buf = body[end + len(CALL_END):]
                    self.state = "reasoning"
                elif think >= 0:                     # the thinking ended inside it: it never was a call
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf[:len(CALL_START) + think])))
                    self.buf = body[think:]
                    self.state = "reasoning"
                elif len(body) > RCALL_MAX:          # a tag in the prose that never closes: stop holding the thinking back
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf)))
                    self.buf = ""
                    self.state = "reasoning"
                else:
                    return out
            elif self.state == "reasoning":
                if self.pending:
                    # calls wait for what follows: more calls or whitespace keep them, `</think>` makes them acts,
                    # any other text means they were quoted
                    stripped = self.buf.lstrip()
                    if len(stripped) < len(self.buf):
                        self.pending[-1][0] += self.buf[:len(self.buf) - len(stripped)]
                        self.buf = stripped
                    if not self.buf:
                        return out
                    if self.buf.startswith(CALL_START):
                        self.state = "rcall"
                    elif self.buf.startswith(THINK_END):
                        out += self._release(True)
                    elif CALL_START.startswith(self.buf) or THINK_END.startswith(self.buf):
                        return out
                    else:
                        out += self._release(False)
                    continue
                i = self.buf.find(THINK_END)
                tool = self.buf.find(CALL_START) if self.schemas else -1
                if tool >= 0 and (i < 0 or tool < i):
                    if tool:
                        out.append(Event("reasoning", self._track(self.buf[:tool])))
                    if self._opener_ok():
                        self.buf = self.buf[tool:]
                        self.state = "rcall"
                    else:                            # mid-sentence, in a fence or in inline code: a quote
                        out.append(Event("reasoning", self._track(CALL_START)))
                        self.buf = self.buf[tool + len(CALL_START):]
                    continue
                if i < 0:
                    keep = self._hold(self.buf, (THINK_END, CALL_START) if self.schemas else (THINK_END,))
                    if len(self.buf) > keep:
                        out.append(Event("reasoning", self._track(self.buf[:len(self.buf) - keep])))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                quoted = self._quoted_think_end(i)
                if quoted is None:                   # held like a partial tag until the character after it is here
                    if i:
                        out.append(Event("reasoning", self._track(self.buf[:i])))
                        self.buf = self.buf[i:]
                    return out
                if quoted:                           # #537: a quoted `</think>` is reasoning text
                    out.append(Event("reasoning", self._track(self.buf[:i + len(THINK_END)])))
                    self.buf = self.buf[i + len(THINK_END):]
                    continue
                if i:
                    out.append(Event("reasoning", self._track(self.buf[:i])))
                self.buf = self.buf[i + len(THINK_END):]
                self.state, self.lead = "content", True
                self.fence, self.line, self.ticks = "", "", 0     # the answer's own text starts here
            elif self.state == "content":
                if self.lead:                                   # newlines right after </think> or a call
                    stripped = self.buf.lstrip("\n")
                    if not stripped:
                        self.buf = ""
                        return out
                    self.buf, self.lead = stripped, False
                i = self.buf.find(CALL_START)
                b, decided = self._bare_opener() if self.recover and self.schemas else (-1, False)
                if b >= 0 and (i < 0 or b < i):
                    # recover: <function=NAME> alone at the start of a line - the call without its <tool_call>.
                    # Until the name has arrived it is held.
                    j = b
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0 and self.buf[:j].strip():
                        out.append(Event("content", self._track(self.buf[:j])))
                    if not decided:
                        self.buf = self.buf[j:]
                        return out
                    self.buf = self.buf[b:]
                    self.state, self.bare = "call", True
                    continue
                if i < 0:
                    # Hold a partial tag AND the newlines before it: if a tool call follows, they are dropped,
                    # so emitting them early would make streamed and whole outputs differ.
                    j = len(self.buf) - self._hold(self.buf, (CALL_START,))
                    if self.recover and self.schemas:            # a line that may be becoming a bare <function=
                        k = self.buf.rfind("\n") + 1
                        tail = self.buf[k:]
                        if tail and FUNC_START.startswith(tail) and self._ok_at(k):
                            j = min(j, k)
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                # #1058: an opener inside a code fence or inline code is text (a quoted example), never a call.  The
                # text before it decides, so streamed and whole outputs agree.  Mid-sentence openers still count.
                snap = (self.fence, self.line, self.ticks)
                self._track(self.buf[:i])
                in_code = self._in_code()
                self.fence, self.line, self.ticks = snap
                if in_code:
                    out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                    self.buf = self.buf[i + len(CALL_START):]
                    continue
                # A call is `<tool_call>` and then (after whitespace) `<function=`; the tag with anything else after
                # it is prose that names the format ("I'll use a <tool_call> block") - content, not a malformed call
                # that ends the request.  Until its follower has arrived it is held, like a partial tag.
                after = self.buf[i + len(CALL_START):].lstrip()
                if self.schemas and not after.startswith(FUNC_START):
                    loose = self._loose_open(after)
                    if loose == "wait":                  # may still become one: held like a partial tag
                        j = i
                        while j > 0 and self.buf[j - 1] == "\n":
                            j -= 1
                        if j > 0:
                            out.append(Event("content", self._track(self.buf[:j])))
                            self.buf = self.buf[j:]
                        return out
                    if loose:                            # #3441: `<=NAME>` etc. is `<function=NAME>`
                        at = len(self.buf) - len(after)
                        if "=" not in after[:loose[1]] and "function" not in after[:loose[1]]:
                            self.alt_end = "</" + loose[0] + ">"    # `<NAME>`: its closer may be `</NAME>`
                        self.buf = self.buf[:at] + FUNC_START + loose[0] + ">" + after[loose[1]:]
                        continue
                kind = self._follower(after) if self.recover else None
                if (after.startswith("<") and kind is None and not FUNC_START.startswith(after)
                        and not after.startswith(FUNC_START)):
                    self.unread += 1
                if kind in ("param", "eq", "name", "json"):
                    if i and self.buf[:i].strip():
                        out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                    rest = self.buf[i + len(CALL_START):]
                    if kind == "param":                  # <parameter=NAME> as the opener: the call's <function=NAME>
                        at = len(rest) - len(after)
                        rest = rest[:at] + FUNC_START + after[len(PARAM_START):]
                    elif kind == "eq":                   # <=NAME> as the opener: likewise
                        at = len(rest) - len(after)
                        rest = rest[:at] + FUNC_START + after[2:]
                    elif kind == "name":                 # <NAME> as the opener: likewise; its closer may be </NAME>
                        at = len(rest) - len(after)
                        self.alt_end = "</" + after[1:after.find(">") + 1]
                        rest = rest[:at] + FUNC_START + after[1:]
                    self.buf = rest
                    self.state = "json" if kind == "json" else "call"
                    continue
                if kind == "wait":
                    after = ""
                is_call = re.match(r"<function=(\s*[^<>\s]+\s*)>", after)
                if is_call:
                    raw_name = is_call.group(1)
                    name = raw_name.strip()
                    name = self.aliases.get(name, name)
                    if raw_name != raw_name.strip() and name not in self.schemas:
                        is_call = None     # only declared names gain whitespace normalization
                is_partial = FUNC_START.startswith(after) or re.fullmatch(r"<function=\s*[^<>\s]*\s*", after)
                fused = re.fullmatch(r"<function=\s*[^<>\s]+\s+(<[^>]*)", after)    # name ended by a line break
                if fused and "<parameter=".startswith(fused.group(1)):
                    is_partial = True               # held until fix_fused can see the whole `<parameter=`
                if not is_call and not is_partial:
                    out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                    self.buf = self.buf[i + len(CALL_START):]
                    continue
                if not is_call:
                    j = i
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                if self.schemas or self.declared_only:
                    # #3075: a call counts only for a declared tool; the name must be complete to tell
                    gt = after.find(">")
                    nl = after.find("\n")
                    if gt < 0 and nl < 0:
                        j = i
                        while j > 0 and self.buf[j - 1] == "\n":
                            j -= 1
                        if j > 0:
                            out.append(Event("content", self._track(self.buf[:j])))
                            self.buf = self.buf[j:]
                        return out
                    if gt >= 0 and (nl < 0 or gt < nl) and not self._callable(after[len(FUNC_START):gt].strip()):
                        self.unread += 1                 # logged by the server: otherwise "the model ignored my tool" is the only clue
                        if len(self.unread_names) < 5:
                            self.unread_names.append(after[len(FUNC_START):gt].strip()[:80])
                        out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                        self.buf = self.buf[i + len(CALL_START):]
                        continue
                if i and self.buf[:i].strip():
                    out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                self.buf = self.buf[i + len(CALL_START):]
                self.state = "call"
            elif self.state == "json":
                # recover: `{"name": ..., "arguments": {...}}` inside the wrapper is the call once its </tool_call> has
                # arrived and it names a declared tool; anything else is the text it is
                i = self.buf.find(CALL_END)
                if i < 0:
                    return out
                call = json_tool_call(self.buf[:i].strip(), self.schemas)
                if call is None:
                    out.append(Event("content", self._track(CALL_START + self.buf[:i + len(CALL_END)])))
                else:
                    out.append(Event("tool_call", call=call))
                self.buf = self.buf[i + len(CALL_END):]
                self.state, self.lead = "content", call is not None
            else:
                drop = None
                if self.recover and ("<=" in self.buf or PARAM_END in self.buf):
                    self.buf = fix_param_openers(self.buf, self.schemas, self.sp if self.stream_tools else 0)
                if self.recover and CALL_START in self.buf[self.sp:]:
                    self.buf = split_second_call(self.buf)
                if self.bare:                       # no wrapper: the call ends at its </function>; a stray
                    i = function_end(self.buf, self.schemas)      # </tool_call> after it is the wrapper's closer, dropped
                    if i >= 0:
                        after = self.buf[i:].lstrip()
                        if after.startswith(CALL_END):
                            drop = len(self.buf[i:]) - len(after) + len(CALL_END)
                        elif after and not CALL_END.startswith(after):
                            drop = 0
                else:
                    if self.alt_end:
                        self.buf = re.sub(r"(</parameter>\s*)" + re.escape(self.alt_end), r"\1" + FUNC_END, self.buf)
                    i = self._wrapped_call_end()
                    if i >= 0:
                        drop = len(CALL_END)
                    # Only a following call opener confirms an omitted wrapper closer.
                    f = function_end(self.buf, self.schemas)
                    nxt = self.buf[f:].lstrip() if f >= 0 else ""
                    if re.match(r"<tool_call>\s*<function=\s*[^<>\s]+\s*>", nxt):
                        i, drop = f, 0
                    if self.recover:                # a batch in one wrapper: the next call follows </function>
                        if nxt.startswith(PARAM_START) and self._follower(nxt) == "param":
                            at = len(self.buf) - len(nxt)
                            self.buf = self.buf[:at] + FUNC_START + nxt[len(PARAM_START):]
                            nxt = self.buf[at:]
                        if nxt.startswith(FUNC_START):
                            i, drop, self.batch = f, 0, True
                if self.stream_tools:
                    if i >= 0:
                        whole, self.buf = self.buf, self.buf[:i]     # scan only the body
                        out += self._scan()
                        out += self._close_scan()
                        self.buf = whole
                    else:
                        out += self._scan()
                if i < 0 or drop is None:
                    return out
                body = self.buf[:i]
                self.buf = self.buf[i + drop:]
                batch, self.batch = self.batch, False
                self.alt_end = ""
                out += self._finish_call(body)
                if batch and out[-1].kind == "tool_call":
                    self.state, self.lead = "call", False

    def finish(self, reason: str | None = None) -> list[Event]:
        """End of generation: flush whatever is held (an unterminated tool call is returned as content; one that was
        already announced stays unfinished: its JSON is not closed and no "tool_call" follows it, #211).  `reason` is
        how the turn ended: calls waiting from the reasoning become real calls only on a natural stop (None = stop);
        a turn cut by max tokens (or cancelled, or failed) keeps them as reasoning text (#1058)."""
        out = []
        if self.pending:
            out += self._release(reason in (None, "stop") and self.state == "reasoning" and not self.buf)
        if self.state == "call" and self.recover and CALL_START in self.buf[self.sp:]:
            split = split_second_call(self.buf, True)       # the output ended: the second call is decided
            if split != self.buf:
                self.buf = split
                out += self.feed("")
        if self.state == "call":                # cut inside a parameter name: the opener is dropped, not the parameters before it
            self.buf = re.sub(r"<parameter=[^>\n]*$", "", self.buf)
        if self.state == "call" and self.bare and function_end(self.buf, self.schemas, True) >= 0:
            i = function_end(self.buf, self.schemas, True)          # recover: a bare call is whole at its </function>
            body, rest = self.buf[:i], self.buf[i:].strip()
            self.buf = ""
            out += self._finish_call(body)
            if rest and not CALL_END.startswith(rest):
                out.append(Event("content", self._track(rest)))
            return out
        if self.state == "call" and self.recover and reason in (None, "stop") and ">" in self.buf[self.buf.find(FUNC_START) + 1:]:
            # the model stopped on its own with the call's closers missing (`true` then the end, no `</parameter>`, or no
            # `</function>` / `</tool_call>`): complete it, so the client gets whole arguments instead of JSON cut
            # after the last key
            text = self.buf.rstrip()
            text = text[:held_from(text)].rstrip("\n")    # a closer cut off by the end is not value
            if text.endswith(CALL_END):
                text = text[:-len(CALL_END)].rstrip()
            if text.endswith("</invoke>"):          # the Anthropic form's closer, written instead of `</function>`
                text = text[:-len("</invoke>")].rstrip()
            bases = ([text[:-len(FUNC_END)].rstrip()] if text.endswith(FUNC_END) else []) + [text]
            head = text[text.find(FUNC_START) + len(FUNC_START):].split(">", 1)[0].strip()
            needs = ((self.schemas.get(func_name(head)) or {}).get("parameters") or {}).get("required")
            if needs and PARAM_START not in text:       # a stub alone is not a call: it would name a tool and give it nothing
                bases = []
            for base, tail in [(b, t) for b in bases for t in ("", "\n" + FUNC_END, "\n" + PARAM_END + "\n" + FUNC_END)]:
                if function_end(base + tail, self.schemas, True) == len(base + tail):
                    body = base + tail
                    if self.stream_tools:
                        whole, self.buf = self.buf, body
                        out += self._scan()
                        out += self._close_scan()
                        self.buf = whole
                    self.buf = ""
                    return out + self._finish_call(body)
        if self.state == "call" and self.stream_tools and self.scall is not None:
            out += self._scan()                 # the output ended inside a call that was already announced
            if self.sstarted and ((self.recover and self.ss == "between" and reason in (None, "stop")) or (
                    self.ss == "function_done" and not self.buf[self.sp:].strip() and reason in (None, "stop"))):
                # every parameter is whole and only junk (`<parameter name=`) follows, or a natural end immediately
                # after </function>: native streaming recovery
                out += self._close_scan()
                self.scall.arguments = json.loads(self.sjson)       # as the client has it
                out.append(Event("tool_call", call=self.scall))
            elif self.ss == "function_done":
                out.append(Event("content", self._track(CALL_START + self.buf)))
            self.buf = ""
            self._reset_scan()
            return out
        if self.state == "reasoning" and self.buf == THINK_END:   # the output ends on a held `</think>`: its end
            self.buf, self.state = "", "content"
        if self.buf:
            # an unfinished call inside the reasoning (#804) is reasoning text, never a call
            kind = {"reasoning": "reasoning", "rcall": "reasoning", "content": "content"}.get(self.state, "content")
            text = CALL_START + self.buf if self.state in ("call", "json") and not self.bare else self.buf
            out.append(Event(kind, text))
            self.buf = ""
        return out

    def _finish_call(self, body: str) -> list[Event]:
        """A whole call body in the template's form -> its tool_call event.  With recover, one that still does not
        parse is the text it is instead of an error that ends the request."""
        raw_name = body.strip()[len(FUNC_START):].split(">", 1)[0]
        name = raw_name.strip()
        name = self.aliases.get(name, name)
        bare, self.bare = self.bare, False
        try:
            if raw_name != raw_name.strip() and name not in self.schemas:
                raise ValueError("whitespace tool name is not declared")
            call = parse_tool_call(body, self.schemas.get(name))
            call.name = name
        except ValueError:
            if not self.recover:
                raise
            self._reset_scan()
            self.state = "content"
            return [Event("content", self._track(body if bare else CALL_START + body + CALL_END))]
        if os.environ.get("STRATA_LOG_TOOL_CALLS"):
            need = (self.schemas.get(name, {}).get("parameters") or {}).get("required") or []
            if [k for k in need if k not in call.arguments]:
                print(f"[strata] tool call missing required parameters: raw={body!r} parsed={call.arguments!r}", flush=True)
        if self.scall is not None:
            call.id = self.scall.id
        self._reset_scan()
        self.state, self.lead = "content", True
        return [Event("tool_call", call=call)]
