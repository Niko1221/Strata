"""Raw GBNF request contract. Compilation and token legality belong to native code.

This is a Strata extension, not an OpenAI JSON format or custom-tool frontend.
"""
from dataclasses import dataclass, replace
import math
from serve.frontend import Event, OutputParser


CAPABILITY = "gbnf-v3"
JSON_CAPABILITY = "gbnf-v4"
ANSWER_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
THINK_PREFIX = "<|im_start|>assistant\n<think>\n"


@dataclass(frozen=True)
class GrammarConstraint:
    source: str
    thinking: bool = False
    tools: bool = False
    json_schema: bool = False
    reasoning_tokens: int = 0

    @property
    def scoped(self):
        return self.thinking or self.tools

    def with_scope(self, thinking, tools):
        return replace(self, thinking=bool(thinking), tools=bool(tools))

    def __post_init__(self):
        if type(self.json_schema) is not bool or type(self.reasoning_tokens) is not int or not 0 <= self.reasoning_tokens <= 8192:
            raise ValueError("invalid constraint format or reasoning token budget")
        if not isinstance(self.source, str):
            raise ValueError("grammar must be a UTF-8 GBNF source string with a root rule")
        try:
            raw = self.source.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("grammar must be valid UTF-8") from exc
        if not 1 <= len(raw) <= 8192 or b"\0" in raw:
            raise ValueError("grammar must contain 1..8192 UTF-8 bytes, without NUL")

    def frame(self, command):
        # One byte write under the service FIFO; grammar cannot inject commands.
        if "\n" in command or "\r" in command or "\0" in command or not (
                command == "CHECKG" or command.startswith("GEN ")):
            raise ValueError("invalid native grammar command")
        raw = self.source.encode("utf-8")
        if self.json_schema or self.reasoning_tokens:
            head = f"GENG3 {len(raw)} {int(self.json_schema)} {int(self.thinking)} {int(self.tools)} {self.reasoning_tokens}"
        else:
            head = f"GENG2 {len(raw)} {int(self.thinking)} {int(self.tools)}" if self.scoped else f"GENG1 {len(raw)}"
        return head.encode("ascii") + b"\n" + raw + b"\n" + command.encode("ascii") + b"\n"


@dataclass(frozen=True)
class GrammarToken:
    id: int
    channel: str


class GrammarOutput:
    """Render native token channels; never choose or enforce a grammar in Python."""
    def __init__(self, tools):
        self.parser = OutputParser(thinking=False, tools=tools, stream_tools=True)

    def feed(self, channel, delta):
        if channel in ("answer", "reasoning"):
            if self.parser.state == "call" or self.parser.buf:
                raise ValueError("native grammar switched channels inside a tool call")
            return [Event("content" if channel == "answer" else "reasoning", delta)] if delta else []
        if channel == "control":
            return []
        if channel != "tool":
            raise ValueError("native grammar omitted a token channel")
        events = self.parser.feed(delta)
        return self._tools_only(events)

    @staticmethod
    def _tools_only(events):
        if any(not e.kind.startswith("tool_") for e in events):
            raise ValueError("malformed tool protocol in grammar output")
        return events

    def finish(self, outcome):
        if outcome == "length" and self.parser.buf and self.parser.scall is None:
            # Budget ended inside the delimiter/name, before any item was sent.
            # It is incomplete protocol, not an unconstrained assistant answer.
            return []
        return self._tools_only(self.parser.finish())


def validate_grammar_request(req, api):
    """Normalize one optional constraint; reject conflicts before model loading."""
    if "grammar" not in req:
        return None
    constraint = GrammarConstraint(req["grammar"])
    if req.get("functions") or req.get("strata_mcp"):
        raise ValueError("grammar supports client-owned tools, not legacy functions or server MCP execution")
    if req.get("tool_choice", "auto") not in ("auto", "none") or req.get("function_call") is not None:
        raise ValueError("grammar cannot request a tool call")
    if req.get("stop") not in (None, []):
        raise ValueError("grammar cannot be combined with custom stop strings")
    if req.get("response_format") not in (None, {"type": "text"}):
        raise ValueError("grammar cannot be combined with response_format JSON requirements")
    if req.get("text", {"format": {"type": "text"}}) != {"format": {"type": "text"}}:
        raise ValueError("grammar requires plain text.format")
    reasoning = req.get("reasoning", {})
    if not isinstance(reasoning, dict):
        raise ValueError("reasoning must be an object")
    if req.get("reasoning_budget_tokens") is not None:
        raise ValueError("grammar does not support injected reasoning-budget wrap-up")
    kw = req.get("chat_template_kwargs", {})
    if not isinstance(kw, dict):
        raise ValueError("chat_template_kwargs must be an object")
    efforts = [v for v in (reasoning.get("effort"), req.get("reasoning_effort"), kw.get("reasoning_effort"))
               if v is not None]
    if any(v not in ("none", "minimal", "low", "medium", "high", "xhigh", "max") for v in efforts):
        raise ValueError("unsupported reasoning effort")
    if "enable_thinking" in kw and type(kw["enable_thinking"]) is not bool:
        raise ValueError("enable_thinking must be a boolean")
    if api == "chat":
        if req.get("tools") is not None and not isinstance(req["tools"], list):
            raise ValueError("tools must be an array")
        for tool in req.get("tools") or []:
            if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
                raise ValueError("Chat grammar supports function tools only")
            if tool["function"].get("strict") is not False:
                raise ValueError("grammar requires explicit strict:false for client-owned tool parameters")
        # The legacy adapter is permissive. The new constrained profile must not
        # silently ignore behavior-changing fields (e.g. logit_bias or audio).
        allowed = set("model messages stream stream_options max_tokens max_completion_tokens temperature top_p top_k "
                      "min_p seed presence_penalty frequency_penalty repetition_penalty penalty_last_n grammar "
                      "tools tool_choice parallel_tool_calls response_format reasoning reasoning_effort chat_template_kwargs "
                      "stop n logprobs top_logprobs logit_bias user metadata strata_mcp experimental_speed_projection".split())
        unknown = set(req) - allowed
        if unknown:
            raise ValueError("unsupported grammar request fields: " + ", ".join(sorted(unknown)))
        if type(req.get("n", 1)) is not int or req.get("n", 1) != 1 \
                or (req.get("logprobs") is not None and req["logprobs"] is not False) or req.get("top_logprobs") is not None \
                or req.get("logit_bias") not in (None, {}):
            raise ValueError("grammar supports n:1, no logprobs and no logit_bias")
        for key in ("max_tokens", "max_completion_tokens"):
            if key in req and req[key] is not None and type(req[key]) is not int:
                raise ValueError(f"{key} must be an integer")
        if req.get("max_tokens") is not None and req.get("max_completion_tokens") is not None:
            raise ValueError("grammar requests must use only one output-token limit field")
        if "stream" in req and type(req["stream"]) is not bool:
            raise ValueError("stream must be a boolean")
        opts = req.get("stream_options")
        if opts is not None:
            raise ValueError("grammar does not support stream_options; the existing Chat stream includes usage "
                             "in its final choice chunk")
        if set(kw) - {"enable_thinking", "reasoning_effort"}:
            raise ValueError("unsupported chat_template_kwargs for grammar")
        if set(reasoning) - {"effort", "summary"}:
            raise ValueError("unsupported reasoning fields for grammar")
        if reasoning.get("summary") is not None or req.get("reasoning"):
            raise ValueError("Chat grammar uses reasoning_effort; use Responses for reasoning summaries")
    validate_sampling(req)
    return constraint


def validate_sampling(values):
    # These are the existing native sampler's ranges, not a new sampler. A
    # top_k of zero has the existing documented mapping to its 64-candidate cap.
    ranges = {"temperature": (0, 2), "top_p": (0, 1), "min_p": (0, 1),
              "presence_penalty": (-2, 2), "frequency_penalty": (-2, 2), "repetition_penalty": (0, 100)}
    for key, (lo, hi) in ranges.items():
        value = values.get(key)
        if value is None:
            continue
        if type(value) not in (float, int) or not math.isfinite(value) or not lo <= value <= hi \
                or (key in ("top_p", "repetition_penalty") and value == 0):
            raise ValueError(f"unsupported grammar sampling value: {key}")
    for key, lo, hi in (("top_k", 0, 64), ("seed", 0, 2**64 - 1), ("penalty_last_n", 1, 8192)):
        value = values.get(key)
        if value is not None and (type(value) is not int or not lo <= value <= hi):
            raise ValueError(f"unsupported grammar sampling value: {key}")
    if values.get("strata_tune"):
        raise ValueError("grammar requests cannot override native speculation/tuning settings")


def vocabulary_identity(tok, stops, scoped=False):
    """Match the native byte-table diagnostic to catch tokenizer configuration drift.

    This is not authentication. Tokenizer objects and native model files remain
    trusted server configuration; no Python tokenizer or matcher is substituted.
    """
    types = getattr(tok, "token_types", None)
    if not hasattr(tok, "token_bytes") or not types or len(types) > 300000:
        raise ValueError("grammar needs a tokenizer with explicit emitted bytes and token types")
    value = 14695981039346656037

    def feed(raw):
        nonlocal value
        for b in raw:
            value = ((value ^ b) * 1099511628211) & 0xffffffffffffffff

    def number(n):
        feed(n.to_bytes(8, "little"))

    number(len(types))
    for i, kind in enumerate(types):
        raw = tok.token_bytes(i) if kind == 1 and i not in stops else b""
        number(len(raw))
        feed(raw)
    number(len(stops))
    for i in sorted(stops):
        number(i)
    identity = f"bytes-v1-fnv1a64:{value:016x}"
    if scoped:
        controls = []
        for marker in ("</think>", "<tool_call>", "</tool_call>"):
            ids = tok.encode(marker, parse_special=True)
            if len(ids) != 1 or types[ids[0]] == 1 or tok.token_bytes(ids[0]) != marker.encode():
                raise ValueError("grammar scope requires the Qwen reasoning/tool special tokens")
            controls.append(str(ids[0]))
        identity += ":qwen-v1:" + ":".join(controls)
    return identity
