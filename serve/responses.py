"""Native, stateless Responses protocol over Strata's parsed inference events.

Schemas and custom grammars guide the prompt; they do not constrain decoding.
No Chat Completions objects or serializers are used here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field

from serve.frontend import effort_kwargs


class ResponsesError(ValueError):
    http_status = 400
    error_type = "invalid_request_error"

    def __init__(self, message, param=None, code="invalid_request"):
        super().__init__(message)
        self.param, self.code = param, code

    def body(self):
        return {"error": {"type": self.error_type, "message": str(self),
                          "param": self.param, "code": self.code}}


class GenerationError(ResponsesError):
    http_status = 500
    error_type = "server_error"


def _object(value, param):
    if not isinstance(value, dict):
        raise ResponsesError(f"{param} must be an object", param)
    return value


def _string(value, param):
    if not isinstance(value, str):
        raise ResponsesError(f"{param} must be a string", param)
    return value


def _array(value, param):
    if not isinstance(value, list):
        raise ResponsesError(f"{param} must be an array", param)
    return value


def _optional_object(container, key, param=None):
    value = container.get(key)
    return {} if value is None else _object(value, param or key)


def _content(value, param, images=True):
    if isinstance(value, str):
        return value
    parts = []
    for part in _array(value, param):
        _object(part, param)
        kind = part.get("type")
        if kind in ("input_text", "output_text"):
            parts.append({"type": "text", "text": _string(part.get("text"), param + ".text")})
        elif kind == "input_image" and images:
            source = _string(part.get("image_url"), param + ".image_url")
            if not source or part.get("file_id"):
                raise ResponsesError("input_image requires image_url; file IDs are unsupported", param)
            parts.append({"type": "image", "source": source})
        else:
            raise ResponsesError(f"unsupported content type {kind!r}", param, "unsupported_feature")
    if any(p["type"] == "image" for p in parts):
        return parts
    return "".join(p["text"] for p in parts)


@dataclass
class Tool:
    name: str
    namespace: str | None
    kind: str
    internal: str
    template: dict


@dataclass
class Request:
    body: dict
    messages: list[dict]
    tools: list[dict] | None
    kwargs: dict
    registry: dict[str, Tool]
    required: bool = False
    parallel: bool = True
    max_new: int | None = None


def normalize(body, shared=None):
    """Validate Responses input and lower it directly to the model's template form."""
    body = dict(_object(body, "request"))
    for key in ("stream", "store", "background", "parallel_tool_calls"):
        if key in body and not isinstance(body[key], bool):
            raise ResponsesError(f"{key} must be a boolean", key)
    for key in ("store", "background", "previous_response_id", "conversation", "strata_mcp"):
        if body.get(key):
            raise ResponsesError(f"{key} is unsupported; use store=false and resend input history", key,
                                 "unsupported_feature")
    if body.get("truncation", "disabled") != "disabled":
        raise ResponsesError("automatic truncation is unsupported", "truncation", "unsupported_feature")
    text = _optional_object(body, "text")
    fmt = _optional_object(text, "format", "text.format")
    if fmt.get("type", "text") != "text":
        raise ResponsesError("structured output guarantees are unsupported; use text.format.type=text",
                             "text.format", "unsupported_feature")
    reasoning = _optional_object(body, "reasoning")
    if reasoning.get("summary") not in (None, "none"):
        raise ResponsesError("reasoning summaries are unsupported; use reasoning.summary=none", "reasoning.summary",
                             "unsupported_feature")
    include = _array(body.get("include", []) if body.get("include") is not None else [], "include")
    # Codex requests encrypted reasoning even for stateless third-party providers. No encrypted item is emitted.
    if any(x != "reasoning.encrypted_content" for x in include):
        raise ResponsesError("unsupported include field", "include", "unsupported_feature")
    body["store"] = False
    body.setdefault("stream", False)
    body.setdefault("parallel_tool_calls", True)
    body.setdefault("tool_choice", "auto")
    shared = shared or {}
    if "effort" not in reasoning and shared.get("reasoning_effort"):
        reasoning = dict(reasoning, effort=shared["reasoning_effort"])
    body["reasoning"] = reasoning
    kwargs = effort_kwargs(reasoning.get("effort"))
    max_new = body.get("max_output_tokens")
    if max_new is not None and (isinstance(max_new, bool) or not isinstance(max_new, int) or max_new <= 0):
        raise ResponsesError("max_output_tokens must be a positive integer", "max_output_tokens")
    if max_new is None:
        max_new = shared.get("max_tokens")
    for key in ("temperature", "top_p"):
        if key in body and body[key] is not None:
            v = body[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (
                    0 <= v <= 2 if key == "temperature" else 0 < v <= 1):
                raise ResponsesError(f"invalid {key}", key)
    if "model" in body:
        _string(body["model"], "model")
    metadata = _optional_object(body, "metadata")
    body["metadata"] = metadata

    registry, by_key = {}, {}

    def add_tool(spec, namespace=None):
        _object(spec, "tools")
        kind = spec.get("type")
        if kind in ("web_search", "web_search_preview"):
            raise ResponsesError(
                "Strata does not provide hosted web search. Set web_search = \"disabled\" at the top level "
                "of your Codex config.toml (before any [table]), or launch Codex with "
                "-c 'web_search=\"disabled\"'. Restart the Codex session after changing its configuration.",
                "tools", "unsupported_feature")
        if kind not in ("function", "custom", "namespace"):
            raise ResponsesError(f"unsupported tool type {kind!r}; use function or custom tools", "tools",
                                 "unsupported_feature")
        name = _string(spec.get("name"), "tools.name")
        if not name:
            raise ResponsesError("tool names must not be empty", "tools.name")
        if kind == "namespace" and namespace is None:
            for nested in _array(spec.get("tools"), "tools.tools"):
                add_tool(nested, name)
            return
        if kind not in ("function", "custom"):
            raise ResponsesError(f"unsupported tool type {kind!r}; use function or custom tools", "tools",
                                 "unsupported_feature")
        key = (namespace, name)
        if key in by_key:
            raise ResponsesError(f"duplicate tool {key!r}", "tools")
        internal = "strata_tool_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
        description = _string(spec.get("description", ""), "tools.description")
        if kind == "custom":
            parameters = {"type": "object", "properties": {"input": {"type": "string"}},
                          "required": ["input"]}
            description += "\nPass the tool's exact freeform text in the input parameter."
            custom_format = _optional_object(spec, "format", "tools.format")
            if custom_format:
                description += "\nInput format guidance: " + json.dumps(custom_format, ensure_ascii=False)
        else:
            parameters = _optional_object(spec, "parameters", "tools.parameters")
            properties = _object(parameters.get("properties", {}), "tools.parameters.properties")
            for schema in properties.values():
                if not isinstance(schema, (dict, bool)):
                    raise ResponsesError("property schemas must be objects or booleans", "tools.parameters.properties")
        template = {"name": internal, "description": f"Tool {namespace + '.' if namespace else ''}{name}: " + description,
                    "parameters": parameters}
        if kind == "custom":
            template["strata_raw_input"] = True
        tool = Tool(name, namespace, kind, internal, template)
        registry[internal] = by_key[key] = tool

    value = body.get("input")
    if isinstance(value, str):
        items = [{"role": "user", "content": value}]
    else:
        items = _array(value, "input")
    for item in items:
        _object(item, "input")
    specs = list(_array(body.get("tools", []) if body.get("tools") is not None else [], "tools"))
    # Current Codex can attach explicit tool declarations to the input timeline (Responses Lite).
    # They describe client tools, never server-side execution. Replay may include repeated declarations.
    for item in items:
        if item.get("type") == "additional_tools":
            for spec in _array(item.get("tools"), "input.additional_tools.tools"):
                if spec not in specs:
                    specs.append(spec)
    for spec in specs:
        add_tool(spec)
    choice = body["tool_choice"]
    selected = list(registry.values())
    required = choice == "required"
    if isinstance(choice, dict):
        namespace = choice.get("namespace")
        if namespace is not None:
            _string(namespace, "tool_choice.namespace")
        key = (namespace, _string(choice.get("name"), "tool_choice.name"))
        tool = by_key.get(key)
        if not tool or choice.get("type") != tool.kind:
            raise ResponsesError("tool_choice must select a declared tool", "tool_choice")
        selected, required = [tool], True
    elif choice == "none":
        selected = []
    elif choice not in ("auto", "required"):
        raise ResponsesError("unsupported tool_choice", "tool_choice")
    if required and not selected:
        raise ResponsesError("tool_choice requires at least one tool", "tool_choice")

    messages = []
    instructions = body.get("instructions")
    if instructions is not None:
        messages.append({"role": "system", "content": _string(instructions, "instructions")})
    calls, results = {}, set()
    pending_reasoning = ""

    def assistant(new=False):
        nonlocal pending_reasoning
        # Calls can share a preceding assistant message, but a subsequent message/reasoning item
        # starts a new turn segment so it cannot move ahead of an earlier call in the template.
        if new or not messages or messages[-1]["role"] != "assistant" or (
                pending_reasoning and (messages[-1].get("content") or messages[-1].get("tool_calls"))):
            messages.append({"role": "assistant", "content": ""})
        m = messages[-1]
        if pending_reasoning:
            m["reasoning_content"] = m.get("reasoning_content", "") + pending_reasoning
            pending_reasoning = ""
        return m

    for item in items:
        _object(item, "input")
        kind = item.get("type", "message")
        if kind == "additional_tools":
            continue
        if kind == "message":
            role = item.get("role")
            if role not in ("system", "developer", "user", "assistant"):
                raise ResponsesError("unsupported message role", "input.role")
            content = _content(item.get("content"), "input.content", images=role in ("user", "assistant"))
            if role == "assistant":
                assistant(new=True)["content"] = content
            else:
                if pending_reasoning:
                    assistant()
                messages.append({"role": "system" if role == "developer" else role, "content": content})
        elif kind == "reasoning":
            chunks = _array(item.get("content") or [], "input.reasoning.content")
            if item.get("encrypted_content") and not chunks:
                raise ResponsesError("encrypted-only reasoning cannot be replayed; resend plaintext reasoning",
                                     "input", "unsupported_feature")
            for part in chunks:
                _object(part, "input.reasoning.content")
                if part.get("type") != "reasoning_text":
                    raise ResponsesError("unsupported reasoning content", "input")
                pending_reasoning += _string(part.get("text"), "input.reasoning.text")
        elif kind in ("function_call", "custom_tool_call"):
            cid = _string(item.get("call_id"), "input.call_id")
            if not cid or cid in calls:
                raise ResponsesError("call_id must be nonempty and unique", "input.call_id")
            namespace = item.get("namespace")
            if namespace is not None:
                _string(namespace, "input.namespace")
            key = (namespace, _string(item.get("name"), "input.name"))
            tool = by_key.get(key)
            expected = "custom" if kind == "custom_tool_call" else "function"
            if tool and tool.kind != expected:
                raise ResponsesError("call type does not match declared tool", "input")
            # History can reference tools no longer offered on this request.
            internal = tool.internal if tool else "strata_tool_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
            if expected == "custom":
                args = {"input": _string(item.get("input"), "input.input")}
            else:
                raw = _string(item.get("arguments"), "input.arguments")
                try:
                    args = json.loads(raw)
                except ValueError as e:
                    raise ResponsesError("function arguments must be valid JSON", "input.arguments") from e
                _object(args, "input.arguments")
            calls[cid] = (expected, key)
            assistant().setdefault("tool_calls", []).append({"function": {"name": internal, "arguments": args}})
        elif kind in ("function_call_output", "custom_tool_call_output"):
            cid = _string(item.get("call_id"), "input.call_id")
            expected = "custom" if kind == "custom_tool_call_output" else "function"
            if cid not in calls or calls[cid][0] != expected or cid in results:
                raise ResponsesError("tool output must match one preceding call_id and call type", "input.call_id")
            if pending_reasoning:
                assistant()
            results.add(cid)
            content = _content(item.get("output"), "input.output")
            # The template has no call_id field; retain correlation explicitly in the tool result text.
            label = {"type": "text", "text": f"Tool result for {calls[cid][1][1]} (call_id={cid}):\n"}
            content = label["text"] + content if isinstance(content, str) else [label] + content
            messages.append({"role": "tool", "content": content})
        else:
            raise ResponsesError(f"unsupported input item {kind!r}", "input", "unsupported_feature")
    if pending_reasoning:
        assistant()
    if not messages:
        raise ResponsesError("input must contain at least one message or call", "input")
    # Keep system/developer instructions at their actual authority, including later input instructions.
    system = [m["content"] for m in messages if m["role"] == "system"]
    messages = [m for m in messages if m["role"] != "system"]
    guidance = []
    if required:
        guidance.append("You must call an offered tool in this turn.")
    if not body["parallel_tool_calls"]:
        guidance.append("Call at most one tool in this turn.")
    if guidance:
        system.append("\n".join(guidance))
    if system:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system)})
    allowed = {t.internal: t for t in selected}
    return Request(body, messages, [t.template for t in selected] or None, kwargs, allowed, required,
                   body["parallel_tool_calls"], max_new)


@dataclass
class Accumulator:
    request: Request
    model: str
    prompt_tokens: int
    response: dict = field(init=False)
    sequence: int = 0
    active: int | None = None
    call_indexes: dict = field(default_factory=dict)
    closed: set = field(default_factory=set)

    def __post_init__(self):
        b = self.request.body
        self.response = {"id": "resp_" + uuid.uuid4().hex, "object": "response", "created_at": int(time.time()),
                         "status": "in_progress", "model": self.model, "output": [], "error": None,
                         "incomplete_details": None, "usage": None, "store": False, "previous_response_id": None,
                         "instructions": b.get("instructions"), "max_output_tokens": b.get("max_output_tokens"),
                         "parallel_tool_calls": self.request.parallel, "tool_choice": b["tool_choice"],
                         "tools": b.get("tools") or [], "reasoning": b["reasoning"], "text": b.get("text") or {"format": {"type": "text"}},
                         "metadata": b["metadata"], "temperature": b.get("temperature"), "top_p": b.get("top_p"),
                         "truncation": "disabled", "background": False}

    def event(self, kind, **fields):
        obj = {"type": kind, "sequence_number": self.sequence, **fields}
        self.sequence += 1
        # Events must be snapshots: subsequent accumulation must not modify an already yielded event.
        return copy.deepcopy(obj)

    def close_item(self, index, complete=True):
        if index in self.closed:
            return []
        item = self.response["output"][index]
        item["status"] = "completed" if complete else "incomplete"
        fields = {"item_id": item["id"], "output_index": index}
        events = []
        if item["type"] in ("message", "reasoning"):
            part = item["content"][0]
            kind = "output_text" if item["type"] == "message" else "reasoning_text"
            events.append(self.event(f"response.{kind}.done", **fields, content_index=0, text=part["text"]))
            events.append(self.event("response.content_part.done", **fields, content_index=0, part=part))
        elif complete:
            if item["type"] == "function_call":
                events.append(self.event("response.function_call_arguments.done", **fields,
                                         name=item["name"], arguments=item["arguments"]))
            else:
                events.append(self.event("response.custom_tool_call_input.done", **fields, input=item["input"]))
        events.append(self.event("response.output_item.done", output_index=index, item=item))
        self.closed.add(index)
        if self.active == index:
            self.active = None
        return events

    def feed(self, ev):
        events = []
        if ev.kind in ("content", "reasoning") and ev.text:
            kind = "message" if ev.kind == "content" else "reasoning"
            if self.active is None or self.response["output"][self.active]["type"] != kind:
                if self.active is not None:
                    events += self.close_item(self.active)
                part = {"type": "output_text" if kind == "message" else "reasoning_text", "text": ""}
                if kind == "message":
                    part["annotations"] = []
                item = {"type": kind, "id": ("msg_" if kind == "message" else "rs_") + uuid.uuid4().hex,
                        "status": "in_progress", "content": [part]}
                item.update({"role": "assistant"} if kind == "message" else {"summary": []})
                self.active = len(self.response["output"])
                self.response["output"].append(item)
                events.append(self.event("response.output_item.added", output_index=self.active, item=item))
                events.append(self.event("response.content_part.added", output_index=self.active,
                                         item_id=item["id"], content_index=0, part=part))
            index = self.active
            item = self.response["output"][index]
            item["content"][0]["text"] += ev.text
            events.append(self.event("response.output_text.delta" if kind == "message" else "response.reasoning_text.delta",
                                     output_index=index, item_id=item["id"], content_index=0, delta=ev.text))
        elif ev.kind == "tool_start":
            tool = self.request.registry.get(ev.call.name)
            if not tool:
                raise GenerationError("model called a tool that was not offered", "tool_choice", "invalid_tool_call")
            if not self.request.parallel and self.call_indexes:
                raise GenerationError("model exceeded parallel_tool_calls=false", "parallel_tool_calls", "invalid_tool_call")
            if self.active is not None:
                events += self.close_item(self.active)
            index = len(self.response["output"])
            item = {"type": "custom_tool_call" if tool.kind == "custom" else "function_call",
                    "id": ("ctc_" if tool.kind == "custom" else "fc_") + uuid.uuid4().hex,
                    "call_id": ev.call.id, "name": tool.name, "status": "in_progress"}
            if tool.namespace is not None:
                item["namespace"] = tool.namespace
            item["input" if tool.kind == "custom" else "arguments"] = ""
            self.call_indexes[ev.call.id] = index
            self.response["output"].append(item)
            events.append(self.event("response.output_item.added", output_index=index, item=item))
        elif ev.kind in ("tool_args", "tool_input"):
            index = self.call_indexes[ev.call.id]
            item = self.response["output"][index]
            custom = item["type"] == "custom_tool_call"
            if (ev.kind == "tool_input") == custom:
                item["input" if custom else "arguments"] += ev.text
                events.append(self.event("response.custom_tool_call_input.delta" if custom else "response.function_call_arguments.delta",
                                         output_index=index, item_id=item["id"], delta=ev.text))
        elif ev.kind == "tool_call":
            if ev.call.id not in self.call_indexes:
                raise GenerationError("unannounced tool call", "tools", "invalid_tool_call")
            index = self.call_indexes[ev.call.id]
            item = self.response["output"][index]
            if ev.complete:
                if item["type"] == "custom_tool_call":
                    if set(ev.call.arguments) != {"input"} or not isinstance(ev.call.arguments["input"], str):
                        raise GenerationError("custom tool requires exactly one string input", "tools", "invalid_tool_call")
                    if item["input"] != ev.call.arguments["input"]:
                        raise GenerationError("custom tool input stream did not match final input", "tools", "invalid_tool_call")
                else:
                    try:
                        args = json.loads(item["arguments"])
                    except ValueError as e:
                        raise GenerationError("generated function arguments are invalid JSON", "tools", "invalid_tool_call") from e
                    if args != ev.call.arguments:
                        raise GenerationError("function argument stream did not match final arguments", "tools", "invalid_tool_call")
                events += self.close_item(index)
        return events

    def finish(self, done):
        limited = done["finish"] == "length"
        if done["finish"] not in ("stop", "length"):
            raise GenerationError("generation did not finish normally", code="generation_failed")
        if not limited and self.request.required and not self.call_indexes:
            raise GenerationError("model did not make the required tool call", "tool_choice", "invalid_tool_call")
        events = []
        for i, item in enumerate(self.response["output"]):
            if i not in self.closed:
                if item["type"] in ("function_call", "custom_tool_call") and not limited:
                    raise GenerationError("model generated an unterminated tool call", "tools", "invalid_tool_call")
                events += self.close_item(i, complete=not limited)
        self.response["status"] = "incomplete" if limited else "completed"
        if limited:
            self.response["incomplete_details"] = {"reason": "max_output_tokens"}
        else:
            self.response["completed_at"] = int(time.time())
        n = done["completion_tokens"]
        self.response["usage"] = {"input_tokens": self.prompt_tokens, "output_tokens": n,
                                  "total_tokens": self.prompt_tokens + n,
                                  "input_tokens_details": {"cached_tokens": done.get("reused") or 0}}
        events.append(self.event("response.incomplete" if limited else "response.completed", response=self.response))
        return events

    def fail(self, error):
        self.response["status"] = "failed"
        self.response["error"] = {"code": getattr(error, "code", "server_error"), "message": str(error)}
        for item in self.response["output"]:
            if item["status"] == "in_progress":
                item["status"] = "incomplete"
        return self.event("response.failed", response=self.response)


def events(accumulator, run):
    """Consume the shared engine event iterator, closing it on error or disconnect."""
    try:
        yield accumulator.event("response.created", response=accumulator.response)
        yield accumulator.event("response.in_progress", response=accumulator.response)
        for kind, value in run:
            if kind == "ping":
                yield None
            elif kind == "event":
                yield from accumulator.feed(value)
            elif kind == "done":
                yield from accumulator.finish(value)
    finally:
        run.close()
