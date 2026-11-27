"""Inkling-2-Mini's wire dialect, translated to Strata's OutputParser / JsonOutputParser form.

Inkling-2-Mini renders its assistant output with explicit channel tokens:
    <|message_model|><|content_thinking|> ... <|end_message|>
    <|message_model|><|content_text|> ... <|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
    <|message_model|>NAME<|content_invoke_tool_text|>...<|end_message|>
    <|content_model_end_sampling|>

`InklingDialect.feed()` streams these tokens into:
    <think>...</think>
    plain text
    <tool_call>\n{"name": "NAME", "arguments": {...}}\n</tool_call>

Using JSON inside `<tool_call>...</tool_call>` (read by InklingOutputParser -> JsonOutputParser) preserves
arbitrary XML special characters (`<`, `>`, `&`, `</parameter>`), nested quotes, booleans/numbers/arrays,
and multiline code losslessly while reusing all downstream OpenAI / Anthropic / Responses / MCP streaming!
"""
from __future__ import annotations

import json
import os
import re

MSG_MODEL = "<|message_model|>"
CONTENT_THINK = "<|content_thinking|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_TOOL_ERROR = "<|content_tool_error|>"   # a text block the model can open (id 200022): read as text, not shown as a tag
CONTENT_TOOL_JSON = "<|content_invoke_tool_json|>"
CONTENT_TOOL_TEXT = "<|content_invoke_tool_text|>"
CONTENT_XML = "<|content_xml|>"
END_MSG = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"
ENDOFTEXT = "<|endoftext|>"

ALL_CHANNEL_TAGS = frozenset((
    MSG_MODEL, CONTENT_THINK, CONTENT_TEXT, CONTENT_TOOL_ERROR, CONTENT_TOOL_JSON, CONTENT_TOOL_TEXT,
    CONTENT_XML, END_MSG, END_SAMPLING, ENDOFTEXT,
))
_HEADER_TAGS = (CONTENT_THINK, CONTENT_TEXT, CONTENT_TOOL_ERROR, CONTENT_TOOL_JSON, CONTENT_TOOL_TEXT, CONTENT_XML, END_MSG, END_SAMPLING, ENDOFTEXT, MSG_MODEL)
_IDLE_TAGS = (MSG_MODEL, CONTENT_THINK, CONTENT_TEXT, CONTENT_TOOL_ERROR, CONTENT_TOOL_JSON, CONTENT_TOOL_TEXT, CONTENT_XML, END_MSG, END_SAMPLING, ENDOFTEXT)
_THINK_TAGS = (END_MSG, CONTENT_TEXT, CONTENT_TOOL_ERROR, END_SAMPLING, ENDOFTEXT)
_TEXT_TAGS = (END_MSG, END_SAMPLING, ENDOFTEXT)
_TOOL_TAGS = (END_MSG, END_SAMPLING, ENDOFTEXT)
_AFTER_END_TAGS = (MSG_MODEL, CONTENT_THINK, CONTENT_TEXT, CONTENT_TOOL_ERROR, CONTENT_TOOL_JSON, CONTENT_TOOL_TEXT, CONTENT_XML, END_MSG, END_SAMPLING, ENDOFTEXT)
_MAX_TAG = max(len(t) for t in _HEADER_TAGS)
_MAX_HEADER_CHARS = 128
_MAX_THINK_ACC = 65536


def _safe_json_call(name: str, arguments) -> str:
    """Serialize a tool call as `<tool_call>\\n{...}\\n</tool_call>` with `</` escaped as `<\\/` so literal `</tool_call>` in args never breaks framing."""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False).replace("</", "<\\/")
    return f"<tool_call>\n{payload}\n</tool_call>"


def _parse_tool_body(header_name: str, body: str, kind: str, *, truncated: bool = False) -> str:
    raw = body.strip()
    fallback_name = header_name.strip()
    if kind == "tool_json":
        try:
            obj = json.loads(raw)
        except ValueError:
            # Recover common trailing garbage / unclosed brace if cut
            obj = None
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except ValueError:
                    obj = None
        if isinstance(obj, dict):
            name = str(obj.get("name") or fallback_name).strip()
            args = obj.get("args") if "args" in obj else obj.get("arguments")
            if args is None:
                args = {k: v for k, v in obj.items() if k != "name"} if "name" in obj else {}
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    if isinstance(parsed, dict):
                        args = parsed
                except ValueError:
                    args = {"raw": args}
            if not isinstance(args, dict):
                args = {"value": args}
            if name:
                return _safe_json_call(name, args)
        if truncated and raw and not raw.startswith("{"):
            return raw
        if fallback_name:
            return _safe_json_call(fallback_name, {"raw": raw} if raw else {})
        return raw
    # tool_text / content_xml: check if it is JSON or XML `<function=...>` / raw text
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                name = str(obj.get("name") or fallback_name).strip()
                args = obj.get("args") if "args" in obj else obj.get("arguments", {})
                if name and isinstance(args, dict):
                    return _safe_json_call(name, args)
        except ValueError:
            pass
    if "<function=" in raw:
        return f"<tool_call>\n{raw}\n</tool_call>"
    if "<tool_call" in raw or "<invoke" in raw:
        m_call = re.search(r'<(?:tool_call|invoke)\b[^>]*\bname=["\']([^"\']+)["\'][^>]*>(.*?)</(?:tool_call|invoke)>', raw, re.S)
        if m_call:
            xname = m_call.group(1).strip()
            xbody = m_call.group(2)
            xargs = {
                pm.group(1).strip(): pm.group(2)
                for pm in re.finditer(r'<(?:param|parameter)\b[^>]*\bname=["\']([^"\']+)["\'][^>]*>(.*?)</(?:param|parameter)>', xbody, re.S)
            }
            if xname:
                return _safe_json_call(xname, xargs)
    if fallback_name:
        return _safe_json_call(fallback_name, {"input": raw} if raw else {})
    return raw


class InklingDialect:
    """Stateful streaming translator: one instance per assistant reply."""

    def __init__(self):
        self.state = "idle"          # idle | header | think | think_end | text | text_end | tool_json | tool_text
        self.buf = ""
        self.header_acc = ""
        self.tool_acc = ""
        self._think_acc = ""
        self._end_ws = ""
        self._emitted_visible = False
        self.calls = 0
        self._in_think = False

    def _tags_for(self):
        if self.state == "idle":
            return _IDLE_TAGS
        if self.state == "think":
            return _THINK_TAGS
        if self.state == "header":
            return _HEADER_TAGS
        if self.state == "text":
            return _TEXT_TAGS
        if self.state in ("think_end", "text_end"):
            return _AFTER_END_TAGS
        return _TOOL_TAGS

    def _split(self, tags):
        best, which = -1, None
        for t in tags:
            i = self.buf.find(t)
            if i >= 0 and (best < 0 or i < best):
                best, which = i, t
        if best >= 0:
            return best, which, -1
        start = max(0, len(self.buf) - _MAX_TAG + 1)
        j = self.buf.find("<|", start)
        while j >= 0:
            tail = self.buf[j:]
            if any(t.startswith(tail) for t in tags):
                return -1, None, j
            j = self.buf.find("<|", j + 1)
        if self.buf.endswith("<") and any(t.startswith("<") for t in tags):
            return -1, None, len(self.buf) - 1
        return -1, None, -1

    def feed_channels(self, text: str = "", final: bool = False, reason: str | None = None) -> list[tuple[str, str]]:
        """Stream raw Inkling-2-Mini output into typed `("reasoning" | "content", text)` chunks.
        Tool invocations are emitted in the `"content"` channel as `<tool_call>\\n{...}\\n</tool_call>` JSON frames."""
        self.buf += text
        out: list[tuple[str, str]] = []
        while True:
            tags = self._tags_for()
            i, tag, hold = self._split(tags)
            if tag is None:
                end = len(self.buf) if (hold < 0 or final) else hold
                chunk, self.buf = self.buf[:end], self.buf[end:]
                self._emit_typed(chunk, out)
                break
            chunk, self.buf = self.buf[:i], self.buf[i + len(tag):]
            self._emit_typed(chunk, out)
            if tag in self._tags_for():
                self._handle_tag_typed(tag, out)
            else:
                self._emit_typed(tag, out)
        if final:
            self._finish_typed(out, reason=reason)
        return out

    def feed_token(self, piece: str, *, is_special: bool) -> list[tuple[str, str]]:
        """Token-aware streaming entry point (#6675): ordinary BPE tokens (`is_special=False`) are emitted as
        literal text in the current channel and can never trigger structural channel transitions even when they
        spell `<|message_model|>`, `<|content_thinking|>`, or `<|end_message|>`. Real special channel tokens
        (`is_special=True`) drive the state machine directly."""
        out: list[tuple[str, str]] = []
        if not piece:
            return out
        if self.buf:
            pending, self.buf = self.buf, ""
            self._emit_typed(pending, out)
        if is_special and piece in ALL_CHANNEL_TAGS:
            if piece in self._tags_for():
                self._handle_tag_typed(piece, out)
            else:
                self._emit_typed(piece, out)
            return out
        self._emit_typed(piece, out)
        return out

    def feed(self, text: str = "", final: bool = False, reason: str | None = None) -> str:
        """Stream raw Inkling-2-Mini output into `<think>...</think>` + text + `<tool_call>...</tool_call>` string."""
        parts: list[str] = []
        for kind, chunk in self.feed_channels(text, final=final, reason=reason):
            if kind == "reasoning":
                if not self._in_think:
                    parts.append("<think>")
                    self._in_think = True
                parts.append(chunk)
            else:
                if self._in_think:
                    parts.append("</think>")
                    self._in_think = False
                parts.append(chunk)
        if final and self._in_think:
            parts.append("</think>")
            self._in_think = False
        return "".join(parts)

    def _append_think(self, chunk: str) -> None:
        if chunk and len(self._think_acc) < _MAX_THINK_ACC:
            self._think_acc += chunk[:_MAX_THINK_ACC - len(self._think_acc)]

    def _emit_typed(self, chunk: str, out: list[tuple[str, str]]) -> None:
        if not chunk:
            return
        if self.state == "think":
            # The thinking goes out as it is written (the reasoning channel: reasoning_content, thinking blocks, reasoning items).
            out.append(("reasoning", chunk))
            return
        elif self.state == "think_end":
            # Lookahead after <|end_message|> in thinking (#6675): only another structural tag confirms end-of-thinking;
            # ordinary text means <|end_message|> was quoted inside thinking, so stay in think without leaking!
            if not chunk.strip():
                self._end_ws += chunk
                return
            out.append(("reasoning", END_MSG + self._end_ws + chunk))
            self._end_ws = ""
            self.state = "think"
            return
        elif self.state == "text_end":
            # Lookahead after <|end_message|> in visible text (#6675): ordinary text means <|end_message|> was quoted
            # literally in the answer, so restore it verbatim and continue in text state!
            if not chunk.strip():
                self._end_ws += chunk
                return
            out.append(("content", END_MSG + self._end_ws + chunk))
            self._emitted_visible = True
            self._end_ws = ""
            self.state = "text"
            return
        elif self.state == "idle":
            out.append(("content", chunk))
            if chunk.strip():
                self._emitted_visible = True
                self.state = "text"
        elif self.state == "text":
            out.append(("content", chunk))
            if chunk.strip():
                self._emitted_visible = True
        elif self.state == "header":
            self.header_acc += chunk
            # A real Inkling tool-name header is short and single-line; if newlines or >128 chars arrive without a
            # `<|content_*|>` tag, the model omitted `<|content_text|>` and is writing plain text (#6620).
            if "\n" in self.header_acc or len(self.header_acc) > _MAX_HEADER_CHARS:
                out.append(("content", self.header_acc))
                self._emitted_visible = True
                self.header_acc = ""
                self.state = "text"
        elif self.state in ("tool_json", "tool_text"):
            self.tool_acc += chunk

    def _close_channel_typed(self, out: list[tuple[str, str]]) -> None:
        if self.state in ("tool_json", "tool_text"):
            rendered = _parse_tool_body(self.header_acc, self.tool_acc, self.state)
            if rendered:
                if "<tool_call>" in rendered:
                    self.calls += 1
                out.append(("content", rendered))
                self._emitted_visible = True
            self.tool_acc = ""
            self.header_acc = ""
        elif self.state == "header":
            if self.header_acc:
                out.append(("content", self.header_acc))
                self._emitted_visible = True
            self.header_acc = ""
        elif self.state in ("think", "think_end"):
            self._think_acc = ""
            self._end_ws = ""
        elif self.state == "text_end":
            self._end_ws = ""

    def _handle_tag_typed(self, tag: str, out: list[tuple[str, str]]) -> None:
        if tag in (END_SAMPLING, ENDOFTEXT):
            self._close_channel_typed(out)
            self.state = "idle"
            return
        if tag == END_MSG:
            if self.state == "think":
                self._end_ws = ""
                self.state = "think_end"
                return
            if self.state == "think_end":
                # Second <|end_message|> after a quoted one inside thinking: keep absorbing the first and hold the second
                out.append(("reasoning", END_MSG + self._end_ws))
                self._end_ws = ""
                return
            if self.state == "text":
                self._end_ws = ""
                self.state = "text_end"
                return
            if self.state == "text_end":
                # Adjacent quoted + real <|end_message|> in visible text (#6675): emit the first, hold the second
                out.append(("content", END_MSG + self._end_ws))
                self._emitted_visible = True
                self._end_ws = ""
                return
            self._close_channel_typed(out)
            self.state = "idle"
            return
        if tag == MSG_MODEL:
            self._close_channel_typed(out)
            self.state = "header"
            self.header_acc = ""
            return
        if tag == CONTENT_THINK:
            self._close_channel_typed(out)
            self.header_acc = ""
            self._think_acc = ""
            self.state = "think"
            return
        if tag in (CONTENT_TEXT, CONTENT_TOOL_ERROR):
            if self.state in ("think", "think_end", "text_end"):
                self._close_channel_typed(out)
            self.header_acc = ""
            self.state = "text"
            return
        if tag == CONTENT_TOOL_JSON:
            if self.state in ("think", "think_end", "text_end"):
                self._close_channel_typed(out)
            self.state = "tool_json"
            self.tool_acc = ""
            return
        if tag in (CONTENT_TOOL_TEXT, CONTENT_XML):
            if self.state in ("think", "think_end", "text_end"):
                self._close_channel_typed(out)
            self.state = "tool_text"
            self.tool_acc = ""
            return

    def _finish_typed(self, out: list[tuple[str, str]], reason: str | None = None) -> None:
        if self.state == "header" and self.header_acc:
            out.append(("content", self.header_acc))
            self._emitted_visible = True
        elif self.state in ("tool_json", "tool_text") and (self.tool_acc.strip() or self.header_acc.strip()):
            rendered = _parse_tool_body(self.header_acc, self.tool_acc, self.state, truncated=True)
            if rendered:
                if "<tool_call>" in rendered:
                    self.calls += 1
                out.append(("content", rendered))
                self._emitted_visible = True
        self.header_acc = ""
        self.tool_acc = ""
        self._think_acc = ""
        self._end_ws = ""
        self.state = "idle"


# #6734: what the server writes into the model's hidden thinking when its budget (reasoning_budget_tokens) is spent: a short
# sentence and the block's own end.  The model then opens its next turn itself (`<|message_model|>` and the text, or a call).
# The Qwen wrap-up ends in `</think>`, which Inkling does not use.
REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer.<|end_message|>"


def force_text(force):
    """Translate Qwen-style forced tool call opener into Inkling-2-Mini's native invocation prefix."""
    if not isinstance(force, str):
        return force
    m = re.fullmatch(r"<tool_call>\n(?:<function=([^>\n]+)>\n)?", force)
    if not m:
        return force
    name = m.group(1) or ""
    if name:
        return f'{MSG_MODEL}{name}{CONTENT_TOOL_JSON}{{"name":{json.dumps(name)},"args":'
    return MSG_MODEL


def translate(text: str) -> str:
    return InklingDialect().feed(text, final=True)


# ---- thinking levels: the template's own (chat_template.jinja: "Thinking effort level: <number>") ----
# The model is designed for these levels: 0 = no thinking, then the more the higher.  A request's reasoning_effort / enable_thinking / thinking
# block reaches the template as the word below (None -> the default level).  Like robertkarl/UnbiasedStrata (MIT): none, 0 or enable_thinking
# false is level 0; every other level goes through to the template.
EFFORT_LEVEL = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}
DEFAULT_EFFORT = os.environ.get("STRATA_INKLING_EFFORT", "low").strip().lower() or "low"
EFFORT = {
    "none": "none", "off": "none", "no_think": "none", "disabled": "none", "false": "none", "0": "none", "0.0": "none",
    "minimal": "minimal", "min": "minimal", "low": "low", "medium": "medium", "med": "medium", "high": "high", "auto": "high", "default": "high",
    "xhigh": "xhigh", "max": "max", "maximum": "max",
}


def normalize_effort(value=None):
    """A client's effort -> the template's word (or a number in (0, 0.99]).  None / "" -> the default level.  ValueError for anything else."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFAULT_EFFORT if DEFAULT_EFFORT in EFFORT_LEVEL else "low"
    if value is False:
        return "none"
    if value is True:
        return normalize_effort(None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value <= 0:
            return "none"
        if value > 0.99:
            return "max"
        return float(value)
    key = str(value).strip().lower()
    if key in EFFORT:
        return EFFORT[key]
    try:
        return normalize_effort(float(key))
    except ValueError:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, minimal, low, medium, high or max (or a number from 0 to 0.99)") from None


def effort_of(kwargs: dict | None):
    """The level a request's kwargs ask for: (the template's word or number, its number).  enable_thinking false is level 0."""
    kwargs = kwargs or {}
    word = "none" if kwargs.get("enable_thinking") is False else normalize_effort(kwargs.get("reasoning_effort"))
    return word, (EFFORT_LEVEL[word] if isinstance(word, str) else float(word))


def thinking_on(kwargs: dict | None) -> bool:
    return effort_of(kwargs)[1] > 0.0


def template_kwargs(kwargs: dict) -> dict:
    """Kwargs for Inkling-2-Mini's chat_template.jinja: the request's level (the default when it names none) as reasoning_effort."""
    out = dict(kwargs)
    out.pop("thinking_mode", None)
    word, level = effort_of(out)
    out["reasoning_effort"] = word
    out["enable_thinking"] = level > 0.0
    return out


# ---- #7241 (opt-in, the config's "effort_position": "end"): the thinking level at the tail ----
# The template writes `Thinking effort level: X` right after the tool declarations, so a request at another level starts differently and the engine reads
# the whole conversation again.  With the option the HISTORY is rendered at the one constant level (the default) and a request at another level adds the
# same system turn the template writes, at the END, right before the answer.  Needs an Inkling engine that checkpoints in front of that turn
# (--tail-role-token, as the main engine's #458); without it the option is off.  A later line only BLENDS with the header level (the reporter's
# numbers: header 0.9 + tail 0.2 gave 485 thinking tokens between 253 at 0.2 and 712 at 0.9), so this is not the default and not the same answers.
TAIL_ROLE_TEXT = "<|message_system|>"                                # the token that opens the trailing turn: the engine's --tail-role-token
EFFORT_TAIL_TURN = "<|message_system|><|content_text|>Thinking effort level: {}<|end_message|>"     # the template's own line


def level_text(level: float) -> str:
    """The level as the template prints it: "0" for none, else the number (0.2, 0.9, 0.99)."""
    return "0" if level == 0.0 else str(float(level))


def effort_tail(kwargs: dict | None) -> tuple[str, str]:
    """(the level word the history is rendered at, the trailing turn) for the effort-at-the-end option.  The history is always rendered at the
    default level (STRATA_INKLING_EFFORT, low unless set); a request at that level has no trailing turn and renders exactly as without the option."""
    base = normalize_effort(None)
    _, base_level = effort_of({"reasoning_effort": base})
    _, level = effort_of(kwargs)
    if level == base_level:
        return base, ""
    return base, EFFORT_TAIL_TURN.format(level_text(level))


def prepare_messages(messages: list[dict], *, has_tools: bool = False, thinking: bool = False) -> list[dict]:
    """Canonicalize assistant tool_call arguments from JSON strings to dicts so Inkling-2-Mini's chat_template.jinja
    never raises `tool call arguments must be a parsed object, not a JSON string`.

    With `has_tools` at level 0 (no `thinking`) the history also gets the empty thinking turn `finish_prompt` forces (see
    `_mark_forced_thinking`); pass the same `has_tools` and `thinking` to `finish_prompt`, which takes the marker out again.
    With thinking on nothing is appended to the prompt, so nothing is replayed either."""
    out = []
    for m in messages:
        if not isinstance(m, dict) or not m.get("tool_calls"):
            out.append(m)
            continue
        new_calls = []
        changed = False
        for tc in m["tool_calls"]:
            if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict):
                new_calls.append(tc)
                continue
            fn = dict(tc["function"])
            args = fn.get("arguments")
            if isinstance(args, str):
                changed = True
                try:
                    parsed = json.loads(args) if args.strip() else {}
                    fn["arguments"] = parsed if isinstance(parsed, dict) else {"value": parsed}
                except ValueError:
                    fn["arguments"] = {"raw": args}
            elif args is None:
                changed = True
                fn["arguments"] = {}
            new_calls.append({**tc, "function": fn})
        out.append({**m, "tool_calls": new_calls} if changed else m)
    return _mark_forced_thinking(out) if (has_tools and not thinking) else out


# A private-use marker, put in an assistant message's `reasoning_content` and removed again by `finish_prompt`.
_EMPTY_THINK_MARK = "\ue000strata-inkling-empty-thinking\ue001"


def _mark_forced_thinking(messages: list[dict]) -> list[dict]:
    """With tools, `finish_prompt` ends every prompt with an empty thinking turn and the model answers after it. When
    that answer comes back in the history, the template leaves the turn out (it renders a thinking turn only for a
    non-empty `reasoning_content`), so the next prompt stops matching the previous one right before the answer and the
    engine reads the whole conversation again on every turn of an agent.  Give each earlier assistant turn that has no
    thinking of its own a marker as its `reasoning_content`: the template then writes the thinking turn where the model
    wrote it, from the structure of the messages, and `finish_prompt` empties the marker.  Left alone: the last message
    (a prefilled reply), a named assistant, an assistant turn with neither text nor tool calls, and everything when
    a message already contains the marker."""
    if not messages or _EMPTY_THINK_MARK in json.dumps(messages, ensure_ascii=False, default=str):
        return messages
    last = len(messages) - 1
    out = []
    for i, m in enumerate(messages):
        if (i != last and isinstance(m, dict) and m.get("role") == "assistant" and not m.get("reasoning_content")
                and not m.get("name") and (m.get("tool_calls") or m.get("content"))):
            m = {**m, "reasoning_content": _EMPTY_THINK_MARK}
        out.append(m)
    return out


def finish_prompt(prompt: str, *, has_tools: bool = False, forced: bool = False, thinking: bool = False) -> str:
    """Append an explicit non-thinking turn opener to a rendered Inkling-2-Mini prompt (#6612, #6620, #6627).

    Inkling-2-Mini's `chat_template.jinja` ends at `<|end_message|>` with no assistant opener, so even with
    `Thinking effort level: 0` the model still generates `<|message_model|><|content_thinking|>...<|end_message|>`
    (burning `max_tokens`, causing empty replies, and breaking multi-turn prefix reuse).
    - Without tools: append `<|message_model|><|content_text|>` so generation starts directly in visible text
      and matches the next turn's rendered assistant prefix verbatim.
    - With tools: append an empty thinking turn `<|message_model|><|content_thinking|><|end_message|>` so the
      model immediately selects either `<|message_model|><|content_text|>` or a tool invocation without wasting
      tokens or looping inside hidden thinking.
    With `thinking` (a level above 0) nothing is appended: the model opens its own thinking turn, then its answer or a call.
    Idempotent if the prompt already carries an opener or when a forced tool-call opener will be appended.
    """
    if _EMPTY_THINK_MARK in prompt:
        prompt = prompt.replace(_EMPTY_THINK_MARK, "")
    if not prompt or forced or thinking:
        return prompt
    empty_think = f"{MSG_MODEL}{CONTENT_THINK}{END_MSG}"
    if prompt.endswith((CONTENT_TEXT, CONTENT_TOOL_JSON, CONTENT_TOOL_TEXT, empty_think)):
        return prompt
    if prompt.endswith(MSG_MODEL):
        return prompt + (f"{CONTENT_THINK}{END_MSG}" if has_tools else CONTENT_TEXT)
    return prompt + (empty_think if has_tools else f"{MSG_MODEL}{CONTENT_TEXT}")
