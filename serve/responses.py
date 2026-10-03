"""Experimental Responses protocol adapter over Service.prepare()/run().

ResponseController is the only response writer. Records are JSON facts; handles
hold iterators, cancellation and incremental buffers. HTTP, SSE and the monitor
observe the same execution. There is no tool executor or second engine queue.
"""
from __future__ import annotations

import copy
import json
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, TypedDict

from serve.structured import jsonschema_modules, prepare_format, validated_json


class Status(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"


TERMINAL = {Status.COMPLETED, Status.FAILED, Status.CANCELLED, Status.INCOMPLETE}


def transition(current, event):
    """Pure lifecycle decision. Phases, I/O and cancellation requests are not states."""
    if current == Status.QUEUED and event == "start":
        return Status.IN_PROGRESS
    if current == Status.IN_PROGRESS:
        if event == "output":
            return current
        if event in ("complete", "exhaust"):
            return Status.COMPLETED if event == "complete" else Status.INCOMPLETE
    if current in (Status.QUEUED, Status.IN_PROGRESS):
        if event in ("fail", "stopped"):
            return Status.FAILED if event == "fail" else Status.CANCELLED
    raise ValueError(f"invalid response transition: {current} / {event}")


class OutputText(TypedDict):
    type: Literal["output_text"]
    text: str
    annotations: list
    logprobs: list


class MessageItem(TypedDict):
    id: str
    type: Literal["message"]
    role: str
    status: str
    content: list


class FunctionCallItem(TypedDict):
    id: str
    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str
    status: str


class FunctionResultItem(TypedDict):
    id: str
    type: Literal["function_call_output"]
    call_id: str
    output: str
    status: str


class ResponseEvent(TypedDict):
    type: str
    sequence_number: int


class RequestError(ValueError):
    def __init__(self, message, param=None, code="invalid_parameter", status=400):
        super().__init__(message)
        self.param, self.code, self.status = param, code, status

    def wire(self):
        return {"error": {"type": "invalid_request_error", "code": self.code,
                          "param": self.param, "message": str(self)}}


def new_id(prefix):
    return prefix + "_" + uuid.uuid4().hex


def strict_json(text):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    # Reject numeric overflow and unpaired surrogates before writing UTF-8 JSON/SSE.
    json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return result


def fields(obj, allowed, param):
    if not isinstance(obj, dict):
        raise RequestError("expected an object", param)
    extra = set(obj) - set(allowed.split())
    if extra:
        key = sorted(extra)[0]
        raise RequestError("unsupported parameter", f"{param}.{key}" if param else key,
                           "unsupported_parameter")


def string(value, param, empty=True):
    if not isinstance(value, str) or (not empty and not value):
        raise RequestError("expected a string" + ("" if empty else " that is not empty"), param)
    return value


def schema_validator(schema, param, strict=False, normalize=False):
    """A deliberately bounded JSON Schema capability; no silent object-only fallback."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise RequestError("schema must have type object at its root", param)
    modules = jsonschema_modules()
    if modules is None:
        raise RequestError("schema validation requires jsonschema>=4.23,<5; install it or use plain text",
                           param, "unsupported_parameter")
    validators, SchemaError, Registry, NoSuchResource = modules
    schema = copy.deepcopy(schema)
    allowed = set("type properties required additionalProperties items $defs $ref anyOf enum const "
                  "description title minimum maximum exclusiveMinimum exclusiveMaximum multipleOf "
                  "minLength maxLength pattern minItems maxItems".split())

    def visit(node):
        if not isinstance(node, dict):
            raise RequestError("schema nodes must be objects", param)
        if set(node) - allowed:
            raise RequestError("unsupported schema keyword: " + sorted(set(node) - allowed)[0], param)
        if "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str) or not (ref == "#" or ref.startswith("#/")):
                raise RequestError("only local JSON pointer schema references are supported", param)
            target = schema
            try:
                for key in ref[2:].split("/") if ref != "#" else []:
                    target = target[key.replace("~1", "/").replace("~0", "~")]
                if not isinstance(target, dict):
                    raise KeyError(ref)
            except (KeyError, TypeError):
                raise RequestError("unresolved schema reference: " + ref, param) from None
        types = node.get("type", [])
        if types == "object" or isinstance(types, list) and "object" in types:
            props = node.get("properties", {})
            if not isinstance(props, dict):
                raise RequestError("properties must be an object", param)
            if normalize:
                node["additionalProperties"] = False
                node["required"] = list(props)
            if strict and (node.get("additionalProperties") is not False or
                           set(node.get("required", [])) != set(props)):
                raise RequestError("strict schemas require all properties and additionalProperties:false", param)
        for key in ("properties", "$defs"):
            if key in node:
                if not isinstance(node[key], dict):
                    raise RequestError(key + " must be an object", param)
                for value in node[key].values():
                    visit(value)
        if "items" in node:
            visit(node["items"])
        if isinstance(node.get("additionalProperties"), dict):
            visit(node["additionalProperties"])
        if "anyOf" in node:
            if not isinstance(node["anyOf"], list):
                raise RequestError("anyOf must be an array", param)
            for value in node["anyOf"]:
                visit(value)

    try:
        visit(schema)
        cls = validators.Draft202012Validator
        cls.check_schema(schema)
    except (SchemaError, TypeError, re.error) as exc:
        raise RequestError(f"invalid schema: {exc}", param) from exc

    def no_remote(uri):
        raise NoSuchResource(ref=uri)

    return schema, cls(schema, registry=Registry(retrieve=no_remote))


def normalize_input(value):
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if not isinstance(value, list):
        raise RequestError("input must be a string or array of items", "input")
    result = []
    for i, item in enumerate(value):
        param = f"input[{i}]"
        if not isinstance(item, dict):
            raise RequestError("expected an input item", param)
        kind = item.get("type", "message")
        if kind == "message":
            fields(item, "type id role content status", param)
            role = item.get("role")
            if role not in ("system", "developer", "user", "assistant"):
                raise RequestError("unsupported message role", param + ".role")
            content = item.get("content")
            if isinstance(content, str):
                content = [{"type": "output_text" if role == "assistant" else "input_text", "text": content}]
            if not isinstance(content, list) or not content:
                raise RequestError("expected nonempty text content", param + ".content")
            parts = []
            for part in content:
                fields(part, "type text annotations logprobs", param + ".content")
                if part.get("type") not in (("input_text", "output_text") if role == "assistant" else ("input_text",)):
                    raise RequestError("only text input is supported", param + ".content", "unsupported_parameter")
                if part.get("annotations", []) != [] or part.get("logprobs", []) != []:
                    raise RequestError("annotated input is not supported", param + ".content")
                text = string(part.get("text"), param + ".content.text")
                if role == "assistant":
                    parts.append({"type": "output_text", "text": text, "annotations": [], "logprobs": []})
                else:
                    parts.append({"type": "input_text", "text": text})
            out = {"type": "message", "role": role, "content": parts}
            prefix = "msg"
        elif kind == "function_call":
            fields(item, "type id call_id name arguments status", param)
            args = string(item.get("arguments"), param + ".arguments")
            try:
                if not isinstance(strict_json(args), dict):
                    raise ValueError("arguments must be a JSON object")
            except ValueError as exc:
                raise RequestError(str(exc), param + ".arguments") from exc
            name = string(item.get("name"), param + ".name", False)
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
                raise RequestError("invalid function name", param + ".name")
            out = {"type": kind, "call_id": string(item.get("call_id"), param + ".call_id", False),
                   "name": name, "arguments": args}
            prefix = "fc"
        elif kind == "function_call_output":
            fields(item, "type id call_id output status", param)
            out = {"type": kind, "call_id": string(item.get("call_id"), param + ".call_id", False),
                   "output": string(item.get("output"), param + ".output")}
            prefix = "fco"
        else:
            raise RequestError("unsupported input item type", param + ".type", "unsupported_parameter")
        if "status" in item:
            if item["status"] != "completed":
                raise RequestError("only completed items can be replayed", param + ".status")
        out["status"] = "completed"
        out["id"] = string(item["id"], param + ".id", False) if "id" in item else new_id(prefix)
        result.append(out)
    return result


def validate_request(request, svc):
    """The capability gate. Unknown fields, including nested fields, are rejected."""
    fields(request, "model input instructions previous_response_id stream store background metadata "
           "max_output_tokens temperature top_p text tools tool_choice parallel_tool_calls truncation include reasoning",
           "")
    req = copy.deepcopy(request)
    if req.get("model") not in svc.model_names():
        raise RequestError("model not found", "model", "model_not_found", 404)
    for key, default in (("stream", False), ("store", True), ("background", False), ("parallel_tool_calls", True)):
        req.setdefault(key, default)
        if type(req[key]) is not bool:
            raise RequestError("expected a boolean", key)
    if req["background"]:
        raise RequestError("background execution and cancellation are not supported", "background", "unsupported_parameter")
    for key in ("instructions", "previous_response_id"):
        if req.get(key) is not None:
            string(req[key], key, empty=key == "instructions")
        req.setdefault(key, None)
    if req.get("truncation", "disabled") != "disabled":
        raise RequestError("only truncation:disabled is supported", "truncation", "unsupported_parameter")
    if req.get("include", []) != []:
        raise RequestError("include representations are not supported", "include", "unsupported_parameter")
    if "reasoning" in req and req["reasoning"] != {"effort": "none"}:
        raise RequestError("only reasoning.effort:none is supported", "reasoning", "unsupported_parameter")
    req["reasoning"] = {"effort": "none", "summary": None}
    cap = req.get("max_output_tokens")
    if cap is not None and (type(cap) is not int or cap < 1):
        raise RequestError("expected a positive integer", "max_output_tokens")
    defaults = {**svc.sampling_defaults, **svc.shared}
    for key, default, maximum in (("temperature", 0.0, 2.0), ("top_p", 1.0, 1.0)):
        value = req.get(key, defaults.get(key, default))
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise RequestError(f"expected a finite number between 0 and {maximum}", key)
        req[key] = value
    metadata = req.setdefault("metadata", {})
    if not isinstance(metadata, dict) or len(metadata) > 16 or any(
            not isinstance(k, str) or len(k) > 64 or not isinstance(v, str) or len(v) > 512
            for k, v in metadata.items()):
        raise RequestError("metadata allows 16 string pairs (64 character keys, 512 character values)", "metadata")
    text = req.setdefault("text", {"format": {"type": "text"}})
    fields(text, "format", "text")
    fmt = text.setdefault("format", {"type": "text"})
    fields(fmt, "type name schema strict description", "text.format")
    kind = fmt.get("type")
    validator = None
    if kind == "json_schema":
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", string(fmt.get("name"), "text.format.name")):
            raise RequestError("invalid schema name", "text.format.name")
        if "strict" in fmt and type(fmt["strict"]) is not bool:
            raise RequestError("expected a boolean", "text.format.strict")
        if "description" in fmt:
            string(fmt["description"], "text.format.description")
        fmt["schema"], validator = schema_validator(fmt.get("schema"), "text.format.schema", fmt.get("strict", False))
    elif kind in ("text", "json_object"):
        fields(fmt, "type", "text.format")
    else:
        raise RequestError("unsupported text format", "text.format.type", "unsupported_parameter")
    tools = req.setdefault("tools", [])
    if not isinstance(tools, list):
        raise RequestError("expected an array", "tools")
    tool_validators = {}
    for i, tool in enumerate(tools):
        param = f"tools[{i}]"
        fields(tool, "type name description parameters strict", param)
        if tool.get("type") != "function":
            raise RequestError("only client-owned function tools are supported", param + ".type", "unsupported_parameter")
        name = string(tool.get("name"), param + ".name")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) or name in tool_validators:
            raise RequestError("function names must be unique and contain 1-64 letters, digits, underscores or dashes", param + ".name")
        if "description" in tool:
            string(tool["description"], param + ".description")
        if "strict" in tool and type(tool["strict"]) is not bool:
            raise RequestError("expected a boolean", param + ".strict")
        implicit = "strict" not in tool
        tool.setdefault("strict", True)
        tool["parameters"], tool_validators[name] = schema_validator(
            tool.get("parameters", {"type": "object", "properties": {}}), param + ".parameters",
            strict=tool["strict"], normalize=implicit)
    req.setdefault("tool_choice", "auto")
    if req["tool_choice"] not in ("auto", "none"):
        raise RequestError("only auto and none tool choices are supported", "tool_choice", "unsupported_parameter")
    if tools and kind != "text":
        raise RequestError("tools with structured text are not supported", "tools", "unsupported_parameter")
    return req, normalize_input(req.get("input", [])), validator, tool_validators


def template_messages(items, instructions):
    """Map typed items straight to the existing template, never to HTTP or chat chunks.

    The native template has positional tool results. Reorder each result batch by
    call_id into call order before rendering, so out-of-order client results bind
    to the right function. Incomplete batches are rejected.
    """
    messages = [{"role": "system", "content": instructions}] if instructions else []
    pending, results, seen = {}, {}, set()

    def flush():
        if pending:
            if set(results) != set(pending):
                raise RequestError("supply one function_call_output for every pending call_id", "input")
            messages.extend({"role": "tool", "content": results[key]} for key in pending)
            pending.clear()
            results.clear()

    for item in items:
        if item["type"] == "function_call":
            if results:
                flush()
            call_id = item["call_id"]
            if call_id in seen:
                raise RequestError("duplicate call_id", "input")
            seen.add(call_id)
            pending[call_id] = item
            if not messages or messages[-1]["role"] != "assistant":
                messages.append({"role": "assistant", "content": ""})
            messages[-1].setdefault("tool_calls", []).append(
                {"function": {"name": item["name"], "arguments": strict_json(item["arguments"])}})
        elif item["type"] == "function_call_output":
            call_id = item["call_id"]
            if call_id not in pending or call_id in results:
                raise RequestError("function_call_output needs a matching, unanswered call_id", "input")
            results[call_id] = item["output"]
        else:
            # A response may contain assistant commentary after a call, before
            # the client supplies its result on the next request.
            if item["role"] != "assistant" or results:
                flush()
            messages.append({"role": item["role"], "content": "".join(p["text"] for p in item["content"])})
    flush()
    return messages


@dataclass
class ResponseRecord:
    response: dict
    input_items: list[MessageItem | FunctionCallItem | FunctionResultItem]

    def stored(self):
        return {"response": copy.deepcopy(self.response), "input_items": copy.deepcopy(self.input_items)}


@dataclass
class RuntimeHandle:
    record: ResponseRecord
    ids: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    max_new: int = 0
    sampling: dict = field(default_factory=dict)
    validator: object = None
    tool_validators: dict = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)
    iterator: object = None
    buffers: dict = field(default_factory=dict)
    current: int | None = None
    calls: dict = field(default_factory=dict)
    sequence: int = 0
    retained: bool = False
    observer: object = None


class ResponseController:
    def __init__(self, service, store):
        self.service, self.store = service, store
        self.lock = threading.RLock()
        self.active = {}
        # Recovery is a controller transition, not a store-invented lifecycle.
        for data in store.records():
            record = ResponseRecord(**data)
            if record.response["status"] not in TERMINAL:
                handle = RuntimeHandle(record, retained=True)
                self.finalize_response(handle, "fail", error={"code": "server_error",
                    "message": "server restarted before generation finished"})

    def _notify(self, handle):
        if handle.observer is not None:
            try:
                handle.observer({"response_id": handle.record.response["id"],
                                 "status": handle.record.response["status"], "at": time.time()})
            except Exception:
                # An optional view cannot change the outcome of generation.
                pass

    def _event(self, handle, kind, **values) -> ResponseEvent:
        event = {"type": kind, "sequence_number": handle.sequence, **values}
        handle.sequence += 1
        return event

    def snapshot(self, handle):
        with self.lock:
            response = copy.deepcopy(handle.record.response)
            for index, pieces in handle.buffers.items():
                item = response["output"][index]
                if item["type"] == "message":
                    item["content"][0]["text"] = "".join(pieces)
                else:
                    item["arguments"] = "".join(pieces)
            return response

    def _lookup(self, response_id):
        try:
            stored = self.store.get(response_id)
        except KeyError:
            raise RequestError("response not found or expired", "response_id", "response_not_found", 404) from None
        handle = self.active.get(response_id)
        if handle is not None:
            return {"response": self.snapshot(handle), "input_items": copy.deepcopy(handle.record.input_items)}
        if stored["response"]["status"] not in TERMINAL:
            # A prior terminal save may have failed. Never return abandoned work
            # as active merely because its creation record is still on disk.
            handle = RuntimeHandle(ResponseRecord(**stored), retained=True)
            self.finalize_response(handle, "fail", error={"code": "server_error",
                "message": "generation ended without a retained terminal record"})
            return handle.record.stored()
        return stored

    def resolve_input(self, req, items):
        with self.lock:
            if req["previous_response_id"] is not None:
                try:
                    parent = self._lookup(req["previous_response_id"])
                except RequestError as exc:
                    exc.param = "previous_response_id"
                    raise
                if parent["response"]["status"] != Status.COMPLETED:
                    raise RequestError("only completed responses can be continued", "previous_response_id")
                items = parent["input_items"] + parent["response"]["output"] + items
            if not items:
                raise RequestError("input context cannot be empty", "input")
            if len({item["id"] for item in items}) != len(items):
                raise RequestError("input item IDs must be unique", "input")
            return items

    def create_response(self, request, observer=None):
        req, items, validator, tool_validators = validate_request(request, self.service)
        items = self.resolve_input(req, items)
        messages = template_messages(items, req["instructions"])
        fmt = req["text"]["format"]
        if fmt["type"] != "text":
            mapped = {"type": fmt["type"]}
            if fmt["type"] == "json_schema":
                mapped["json_schema"] = {k: v for k, v in fmt.items() if k != "type"}
            messages, object_validator = prepare_format(mapped, messages)
            validator = validator or object_validator
        tools = req["tools"] if req["tool_choice"] != "none" else []
        if tools and not req["parallel_tool_calls"]:
            messages.insert(0, {"role": "system", "content": "Call at most one function in this response."})
        self.service.load()
        try:
            ids, _, max_new = self.service.prepare(messages, tools or None, {"enable_thinking": False},
                                                   req.get("max_output_tokens"))
        except ValueError as exc:
            raise RequestError(str(exc), "input") from exc
        # Responses truncation:disabled never inherits the optional chat cap-clamping behavior.
        if req.get("max_output_tokens") is not None and max_new != req["max_output_tokens"]:
            raise RequestError("prompt plus max_output_tokens exceeds context", "max_output_tokens")
        response = {"id": new_id("resp"), "object": "response", "created_at": int(time.time()),
                    "completed_at": None, "status": Status.QUEUED.value, "output": [], "error": None,
                    "incomplete_details": None, "usage": None, "background": False,
                    "max_output_tokens": req.get("max_output_tokens"), "truncation": "disabled",
                    **{key: req[key] for key in ("model", "instructions", "previous_response_id", "store", "metadata",
                        "temperature", "top_p", "text", "tools", "tool_choice", "parallel_tool_calls", "reasoning")}}
        handle = RuntimeHandle(ResponseRecord(response, items), ids=ids, tools=tools, max_new=max_new,
                               sampling={**self.service.sampling_defaults, **self.service.shared,
                                         "temperature": req["temperature"], "top_p": req["top_p"]},
                               validator=validator, tool_validators=tool_validators,
                               retained=req["store"], observer=observer)
        with self.lock:
            if handle.retained:
                self.store.create(handle.record.stored())
            self.active[response["id"]] = handle
            self._notify(handle)
        return handle

    def advance_response(self, handle, event):
        with self.lock:
            if event != "start":
                raise ValueError("advance_response starts work; output and finalization have separate operations")
            response = handle.record.response
            response["status"] = transition(response["status"], event).value
            self._notify(handle)
            return [self._event(handle, "response.in_progress", response=self.snapshot(handle))]

    def _finish_item(self, handle, index, status):
        item = handle.record.response["output"][index]
        if item["status"] != "in_progress":
            return []
        text = "".join(handle.buffers.pop(index, []))
        item["status"] = status
        refs = {"item_id": item["id"], "output_index": index}
        if item["type"] == "message":
            item["content"][0]["text"] = text
            events = [self._event(handle, "response.output_text.done", **refs, content_index=0, text=text, logprobs=[]),
                      self._event(handle, "response.content_part.done", **refs, content_index=0,
                                  part=copy.deepcopy(item["content"][0]))]
        else:
            item["arguments"] = text
            events = [self._event(handle, "response.function_call_arguments.done", **refs, arguments=text)]
        events.append(self._event(handle, "response.output_item.done", output_index=index, item=copy.deepcopy(item)))
        if handle.current == index:
            handle.current = None
        return events

    def append_output(self, handle, event):
        with self.lock:
            response = handle.record.response
            transition(response["status"], "output")  # terminal output is immutable
            events = []
            if event.kind == "content":
                if not event.text:
                    return events
                if handle.current is None:
                    index = len(response["output"])
                    part: OutputText = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
                    item: MessageItem = {"id": new_id("msg"), "type": "message", "role": "assistant",
                                         "status": "in_progress", "content": []}
                    response["output"].append(item)
                    events.append(self._event(handle, "response.output_item.added", output_index=index, item=copy.deepcopy(item)))
                    item["content"].append(part)
                    events.append(self._event(handle, "response.content_part.added", output_index=index,
                                              item_id=item["id"], content_index=0, part=copy.deepcopy(part)))
                    handle.current, handle.buffers[index] = index, []
                index = handle.current
                item = response["output"][index]
                if item["type"] != "message":
                    raise ValueError("text arrived inside a function call")
                handle.buffers[index].append(event.text)
                events.append(self._event(handle, "response.output_text.delta", output_index=index,
                                          item_id=item["id"], content_index=0, delta=event.text, logprobs=[]))
            elif event.kind == "tool_start":
                if response["tool_choice"] == "none" or event.call.name not in handle.tool_validators:
                    raise ValueError("model called a function that was not offered")
                if not response["parallel_tool_calls"] and handle.calls:
                    raise ValueError("model exceeded parallel_tool_calls:false")
                if handle.current is not None:
                    events.extend(self._finish_item(handle, handle.current, "completed"))
                index = len(response["output"])
                item: FunctionCallItem = {"id": new_id("fc"), "type": "function_call", "call_id": event.call.id,
                                           "name": event.call.name, "arguments": "", "status": "in_progress"}
                response["output"].append(item)
                handle.calls[event.call.id] = index
                handle.current, handle.buffers[index] = index, []
                events.append(self._event(handle, "response.output_item.added", output_index=index, item=copy.deepcopy(item)))
            elif event.kind in ("tool_args", "tool_call"):
                index = handle.calls[event.call.id]
                item = response["output"][index]
                if item["status"] != "in_progress":
                    raise ValueError("function arguments changed after completion")
                if event.kind == "tool_args":
                    handle.buffers[index].append(event.text)
                    events.append(self._event(handle, "response.function_call_arguments.delta", output_index=index,
                                              item_id=item["id"], delta=event.text))
                else:
                    args = "".join(handle.buffers[index])
                    strict_json(args)
                    validated_json(args, handle.tool_validators[item["name"]], "stop")
                    events.extend(self._finish_item(handle, index, "completed"))
            else:
                raise ValueError("unsupported model event: " + event.kind)
            return events

    def finalize_response(self, handle, event, error=None):
        with self.lock:
            response = handle.record.response
            if response["status"] in TERMINAL:
                return []
            status = transition(response["status"], event)
            events = []
            for index in list(handle.buffers):
                events.extend(self._finish_item(handle, index, "completed" if status == Status.COMPLETED else "incomplete"))
            changes = {"status": status.value, "error": error,
                       "incomplete_details": {"reason": "max_output_tokens"} if status == Status.INCOMPLETE else None,
                       "completed_at": int(time.time()) if status == Status.COMPLETED else None}
            if handle.retained:
                try:
                    self.store.update({"response": {**response, **changes}, "input_items": handle.record.input_items})
                except OSError:
                    # Persistence is part of finalization. Commit a failure once,
                    # not a completed->failed transition after acknowledging success.
                    status = transition(response["status"], "fail")
                    changes.update(status=status.value, completed_at=None, incomplete_details=None,
                                   error={"code": "server_error", "message": "could not retain the response"})
                    try:
                        self.store.update({"response": {**response, **changes}, "input_items": handle.record.input_items})
                    except OSError:
                        pass  # _lookup/restart resolves the surviving creation record as failed
            response.update(changes)
            self._notify(handle)
            if status != Status.CANCELLED:  # disconnect has no subscriber; no invented response.cancelled event
                events.append(self._event(handle, "response." + status.value, response=self.snapshot(handle)))
            return events

    def events(self, handle):
        """One execution for both transports. Never yield while holding a controller lock."""
        try:
            yield self._event(handle, "response.created", response=self.snapshot(handle))
            handle.iterator = self.service.run(handle.ids, False, handle.tools or None, handle.max_new,
                                               handle.sampling, handle.cancel, lifecycle=True)
            done = None
            for kind, data in handle.iterator:
                if kind == "started":
                    yield from self.advance_response(handle, "start")
                elif kind == "event":
                    yield from self.append_output(handle, data)
                elif kind == "ping":
                    yield None
                elif kind == "done":
                    done = data
                    break
            handle.iterator.close()  # all engine work/draining ends before terminal state
            handle.iterator = None
            if done is None:
                raise ValueError("service ended without a generation result")
            with self.lock:
                n = done["completion_tokens"]
                handle.record.response["usage"] = {"input_tokens": len(handle.ids), "output_tokens": n,
                    "total_tokens": len(handle.ids) + n,
                    "input_tokens_details": {"cached_tokens": done.get("reused", 0), "cache_write_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0}}
            finish = done["finish"]
            if handle.cancel.is_set() or finish == "cancel":
                outcome = "stopped"
            elif finish == "length":
                outcome = "exhaust"
            elif finish == "stop":
                snapshot = self.snapshot(handle)
                if any(item["type"] == "function_call" and item["status"] != "completed" for item in snapshot["output"]):
                    raise ValueError("model stopped inside a function call")
                if handle.validator is not None:
                    text = "".join(part["text"] for item in snapshot["output"] if item["type"] == "message"
                                   for part in item["content"])
                    strict_json(text)
                    validated_json(text, handle.validator, "stop")  # validate; never rewrite streamed content
                outcome = "complete"
            else:
                raise ValueError("unexpected generation finish: " + str(finish))
            yield from self.finalize_response(handle, outcome)
        except GeneratorExit:
            handle.cancel.set()
            raise
        except Exception as exc:
            handle.cancel.set()
            if handle.iterator is not None:
                try:
                    handle.iterator.close()
                except Exception as cleanup_error:
                    exc = cleanup_error
                finally:
                    handle.iterator = None
            if handle.record.response["status"] in TERMINAL:
                raise
            yield from self.finalize_response(handle, "fail", error={"code": "server_error", "message": str(exc)})
        finally:
            self.close_response(handle)

    def close_response(self, handle):
        """Also handles failure before the event generator's first next() (headers/watcher)."""
        if handle.record.response["status"] not in TERMINAL:
            handle.cancel.set()  # requesting a stop does not itself change status
        try:
            if handle.iterator is not None:
                try:
                    handle.iterator.close()  # synchronous STOP + drain via Service.run
                except Exception as exc:
                    self.finalize_response(handle, "fail", error={"code": "server_error", "message": str(exc)})
                    return
                finally:
                    handle.iterator = None
            self.finalize_response(handle, "stopped")
        finally:
            with self.lock:
                self.active.pop(handle.record.response["id"], None)

    def get_response(self, response_id):
        with self.lock:
            return self._lookup(response_id)["response"]

    def list_input_items(self, response_id, *, order="desc", after=None, limit=20):
        if order not in ("asc", "desc"):
            raise RequestError("order must be asc or desc", "order")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise RequestError("limit must be between 1 and 100", "limit")
        with self.lock:
            items = self._lookup(response_id)["input_items"]
        if order == "desc":
            items.reverse()
        if after is not None:
            index = next((i for i, item in enumerate(items) if item["id"] == after), None)
            if index is None:
                raise RequestError("after is not an input item in this response", "after")
            items = items[index + 1:]
        page = items[:limit]
        return {"object": "list", "data": page, "first_id": page[0]["id"] if page else None,
                "last_id": page[-1]["id"] if page else None, "has_more": len(items) > limit}

    def delete_response(self, response_id):
        with self.lock:
            self._lookup(response_id)
            self.store.delete(response_id)
            if response_id in self.active:
                self.active[response_id].retained = False
        return {"id": response_id, "object": "response.deleted", "deleted": True}
