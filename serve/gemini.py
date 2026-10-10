"""serve/gemini.py - the Gemini API (POST /v1beta/models/{model}:generateContent), what Gemini CLI speaks.

Gemini CLI (Google's CLI; its core calls the API through @google/genai 1.30.0) sends the whole conversation in
`contents` on every turn, with `systemInstruction`, `tools[].functionDeclarations` and `generationConfig`, and reads
back `candidates[].content.parts` (text, `thought` parts, `functionCall`), `finishReason` and `usageMetadata`.
A request becomes the same template messages, tools and kwargs an OpenAI or Anthropic request does, runs through
Service.run, and its events come back in Gemini's shapes - so the model, the conversation cache, the thinking levels
and the tool parser are the ones every other dialect already uses.

What the client needs, and what this does (checked against @google/genai 1.30.0's generated converters -
`generateContentParametersToMldev`, `candidateFromMldev`, its SSE reader):
  * `systemInstruction` -> the system message; a turn's `parts[]` text, `inlineData` (a picture as base64),
    `functionCall` (an earlier answer's call) and `functionResponse` (a tool's result) -> the template's messages,
    so a tool result reaches the encoder like the OpenAI and Anthropic paths already put one;
  * a tool's `parametersJsonSchema` / `parameters` -> the template's tools, with the schema's type names lowercased
    (Gemini's own schemas spell them STRING, OBJECT - the template's tool parser reads the JSON Schema names);
  * `generationConfig` (temperature, topP, topK, maxOutputTokens, stopSequences, seed, presencePenalty,
    frequencyPenalty) -> the sampling keys Service.run reads, and `thinkingConfig` -> the thinking level in both
    of Gemini's spellings: `thinkingBudget` (2.5's token count: 0 none, -1 the model's own, a number low under 2K,
    medium under 8K, else high, as Anthropic's budget_tokens is read) and `thinkingLevel` (3's word: minimal, low,
    medium, high - the same words the other routes' reasoning_effort takes, and the budget wins when both come);
    `thinkingConfig.includeThoughts: false` hides the `thought` parts, the model still thinks and
    `thoughtsTokenCount` still counts them;
  * `toolConfig.functionCallingConfig.mode` (AUTO / ANY / NONE) -> the same forced call the other routes force;
  * streaming is `:streamGenerateContent?alt=sse`: every `data:` event is one WHOLE GenerateContentResponse, and the
    SDK parses each one as JSON - it has no "[DONE]" sentinel, and it skips lines that do not start with `data: `,
    so Strata's keep-alive comment stays and nothing else may be written;
  * `usageMetadata` (promptTokenCount, candidatesTokenCount, thoughtsTokenCount) and `finishReason`
    (STOP / MAX_TOKENS), which is what Gemini CLI's context meter and its turn loop read.

Left alone, as the other dialects leave what they cannot run: `safetySettings` (this model has no safety scores to
filter), `cachedContent` (the conversation cache is Strata's own, keyed by the prompt), `responseSchema` /
`responseMimeType` (structured output goes through response_format on the OpenAI and Responses routes),
`logprobs`, `mediaResolution`, and the `thoughtSignature` a Gemini model signs its thoughts with to pair them with a
call - Strata's reasoning has no signature, so its `thought` parts carry only their text.
"""
from __future__ import annotations

import json
import uuid

from serve.frontend import (Event, _late_system_to_user, _object_list, _parts_of, _text_of,
                            budget_effort, effort_kwargs, tool_arguments)

# Gemini's words for the two things an answer can end on; a client that hangs up has no answer to end.
FINISH = {"stop": "STOP", "length": "MAX_TOKENS", "cancel": "STOP"}

# generationConfig's names -> the sampling keys Service.run already reads (the OpenAI ones)
SAMPLING = {"temperature": "temperature", "topP": "top_p", "topK": "top_k", "seed": "seed",
            "presencePenalty": "presence_penalty", "frequencyPenalty": "frequency_penalty",
            "maxOutputTokens": "max_tokens"}


class GeminiError(ValueError):
    """A request error in Gemini's shape: {"error": {"code", "message", "status"}}.  The SDK reads `code` and throws
    an ApiError when it is 400..599, which is also how an error that arrives mid-stream reaches its client."""

    def __init__(self, message, code=400, kind="INVALID_ARGUMENT"):
        super().__init__(message)
        self.code, self.kind = code, kind

    def body(self):
        return error_body(str(self), self.code, self.kind)


def error_body(message, code=400, kind="INVALID_ARGUMENT") -> dict:
    return {"error": {"code": code, "message": message, "status": kind}}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def lower_types(node):
    """A tool schema's type names, lowercased: Gemini spells them STRING, OBJECT (its own Type_ enum) and the
    template's tool parser reads the JSON Schema names.  Only a "type" key holding a name is touched, so a property
    actually named "type" keeps its schema."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "type" and isinstance(value, str):
                out[key] = value.lower()
            elif key == "type" and isinstance(value, list):
                out[key] = [v.lower() if isinstance(v, str) else v for v in value]
            else:
                out[key] = lower_types(value)
        return out
    if isinstance(node, list):
        return [lower_types(v) for v in node]
    return node


# ------------------------------------------------------------------------------------------------ request -> chat
def _part_of(part, param: str) -> dict:
    """One of a turn's parts as the shape the template's helpers read: a text part, an image part in OpenAI's
    image_url form (the part's inlineData as a data: URL), or the part itself (a call, a tool's result).  A bare
    string in the list (what some clients send) is a text part, as the other routes take a bare content string."""
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        raise GeminiError(f"expected an object in {param}", param)
    src = part.get("inlineData") or part.get("inline_data")
    if src:
        if not isinstance(src, dict) or not src.get("data"):
            raise GeminiError("only inlineData (base64 with a mimeType) images are supported; this server keeps no "
                              "files", param)
        mime = src.get("mimeType") or src.get("mime_type") or "image/png"
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{src['data']}"}}
    if part.get("fileData") or part.get("file_data"):
        raise GeminiError("only inlineData (base64) images are supported: uploaded files (fileData) are not kept "
                          "by this server", param)
    return part


def contents_to_messages(req: dict) -> tuple[list[dict], list[dict] | None, dict]:
    """A Gemini request -> (template messages, template tools, template kwargs)."""
    messages = []
    system = req.get("systemInstruction") or req.get("system_instruction")
    if system:
        parts = [_part_of(p, "systemInstruction.parts")
                 for p in _object_list(system.get("parts") if isinstance(system, dict) else system,
                                       "systemInstruction.parts")]
        messages.append({"role": "system", "content": _text_of(parts)})
    for i, content in enumerate(_object_list(req.get("contents"), "contents")):
        if not isinstance(content, dict):
            content = {"parts": content}          # a bare part list, as `contents` sometimes comes
        role = {"model": "assistant"}.get(content.get("role") or "user", content.get("role") or "user")
        text, reasoning, images, calls, results = [], [], [], [], []
        for part in [_part_of(p, f"contents[{i}].parts") for p in _object_list(
                content.get("parts"), f"contents[{i}].parts")]:
            if "functionCall" in part:
                call = part["functionCall"]
                if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                    raise GeminiError(f'contents[{i}].parts functionCall needs a "name"', f"contents[{i}].parts")
                calls.append({"function": {"name": call["name"], "arguments": tool_arguments(call.get("args"))}})
            elif "functionResponse" in part:
                res = part["functionResponse"]
                if not isinstance(res, dict):
                    raise GeminiError("contents[].parts functionResponse needs an object", f"contents[{i}].parts")
                # a tool's result is its own message, as the OpenAI path's tool messages are
                results.append({"role": "tool", "content": _tool_result_of(res.get("response"))})
            elif part.get("thought") is True:
                reasoning.append(part.get("text") or "")
            elif "text" in part:
                if not isinstance(part["text"], str):
                    raise GeminiError(f'contents[{i}].parts text needs a string', f"contents[{i}].parts")
                text.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                images.append(part)
        if text or calls or reasoning or images:
            out = {"role": role, "content": _parts_of(images + text) if images
                   else "".join(t["text"] for t in text)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
        messages.extend(results)
    tools = []
    for tool in _object_list(req.get("tools"), "tools"):
        for d in _object_list(tool.get("functionDeclarations"), "tools[].functionDeclarations"):
            if not isinstance(d.get("name"), str) or not d["name"]:
                raise GeminiError('tools[].functionDeclarations[] needs a "name"', "tools[].functionDeclarations")
            schema = d.get("parametersJsonSchema") or d.get("parameters")
            if schema is not None and not isinstance(schema, dict):
                raise GeminiError("a tool's parameters must be an object (its JSON schema)", "tools[].parameters")
            tools.append({"name": d["name"], "description": d.get("description", ""),
                          "parameters": lower_types(schema) if schema else {}})
    kwargs = {}
    config = req.get("generationConfig") if isinstance(req.get("generationConfig"), dict) else {}
    thinking = config.get("thinkingConfig") if isinstance(config.get("thinkingConfig"), dict) else {}
    # Gemini 3 spells the level as a word; the 2.5 budget below wins when a client sends both, as it is exact
    level = thinking.get("thinkingLevel") or thinking.get("thinking_level")
    if level is not None:
        try:
            kwargs.update(effort_kwargs(level))
        except ValueError:
            raise GeminiError("thinkingConfig.thinkingLevel: minimal, low, medium or high") from None
    budget = thinking.get("thinkingBudget")
    if budget is not None and not isinstance(budget, bool):
        try:
            budget = int(budget)
        except (TypeError, ValueError):
            raise GeminiError("thinkingConfig.thinkingBudget: a whole number of tokens (-1: the model's own)") from None
        if budget <= 0:
            kwargs["enable_thinking"] = False       # 0: no thinking; -1: the model's own default
        else:
            kwargs.update(budget_effort(budget))
    # a client that spells the level out, or the shared Chat settings filled in (Service.with_shared, as for OpenAI)
    kwargs.update(effort_kwargs(req.get("reasoning_effort")))
    return _late_system_to_user(messages), tools or None, kwargs


def _tool_result_of(response):
    """A functionResponse's `response` -> the text a tool message carries.  Gemini sends an object (often
    {"output": ...} or {"result": ...}), sometimes a string or a list of text parts; an object with no obvious text
    is sent back as its JSON, so the model still sees what the tool returned."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        return _text_of([_part_of(x, "functionResponse.response") for x in response])
    for key in ("output", "result", "text", "content"):
        if response.get(key) is not None:
            return _text_of([_part_of(response[key], "functionResponse.response")]) if not isinstance(
                response[key], str) else response[key]
    return json.dumps(response, ensure_ascii=False)


def sampling_of(req: dict) -> dict:
    """generationConfig -> the sampling keys Service.run reads, at the request's top level (where the other routes
    have them already).  A name it does not know is left out, not refused: a client's extra knob must not end its
    request."""
    config = req.get("generationConfig")
    if not isinstance(config, dict):
        return {}
    out = {SAMPLING[k]: v for k, v in config.items() if k in SAMPLING and v is not None}
    if config.get("stopSequences") is not None:
        out["stop"] = config["stopSequences"]
    return out


def tool_choice_of_request(req: dict):
    """toolConfig.functionCallingConfig -> the tool_choice the other routes honour (AUTO: the model decides)."""
    config = req.get("toolConfig") or req.get("tool_config")
    calling = (config.get("functionCallingConfig") or config.get("function_calling_config")
               if isinstance(config, dict) else None)
    if not isinstance(calling, dict):
        return None
    mode = (calling.get("mode") or "AUTO").upper()
    allowed = [n for n in (calling.get("allowedFunctionCalls") or []) if isinstance(n, str)]
    if mode == "NONE":
        return {"type": "none"}
    if mode == "ANY":
        return "any"
    if mode == "VALID_VALUES_ONLY" and len(allowed) == 1:
        return {"type": "tool", "name": allowed[0]}
    return None


# ------------------------------------------------------------------------------------------------ run -> Gemini
def gemini_chunks(svc, req: dict, ids, thinking, tools, max_new, cancel, force=None):
    """One whole GenerateContentResponse per event, as the SDK parses them; None is a heartbeat (an SSE comment).
    A call's arguments come out of the parser piece by piece (stream_tools), and Gemini has no partial form: a
    streamed call is held until its whole tool_call arrives, so the client sees one functionCall part per call.
    `thinkingConfig.includeThoughts: false` (Gemini 3's word) hides the thought parts - the model still thinks and
    thoughtsTokenCount still counts them."""
    rid, model = new_id("strata"), svc.model_for(req)
    config = req.get("generationConfig") if isinstance(req.get("generationConfig"), dict) else {}
    thinking_cfg = config.get("thinkingConfig") if isinstance(config.get("thinkingConfig"), dict) else {}
    shown = thinking_cfg.get("includeThoughts", thinking_cfg.get("include_thoughts")) is not False

    def payload(parts, finish=None, usage=None):
        cand = {"content": {"parts": parts, "role": "model"}, "index": 0, "safetyRatings": []}
        if finish:
            cand["finishReason"] = finish
        out = {"candidates": [cand], "responseId": rid, "modelVersion": model}
        if usage:
            out["usageMetadata"] = usage
        return out

    for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel, force=force):
        if kind == "ping":
            yield None
            continue
        if kind != "event":
            pt = x.get("prompt_tokens", len(ids))     # after MCP rounds: the last round's prompt
            usage = {"promptTokenCount": pt, "candidatesTokenCount": x["completion_tokens"],
                     "totalTokenCount": pt + x["completion_tokens"]}
            if x.get("reasoning_tokens"):
                usage["thoughtsTokenCount"] = x["reasoning_tokens"]
            if x.get("reused"):
                usage["promptTokensDetails"] = [{"modality": "TEXT", "tokenCount": x["reused"]}]
            yield payload([], FINISH.get(x["finish"], "STOP"), usage)
            continue
        ev: Event = x
        if ev.kind == "reasoning" and ev.text and shown:
            yield payload([{"text": ev.text, "thought": True}])
        elif ev.kind == "content" and ev.text:
            yield payload([{"text": ev.text}])
        elif ev.kind in ("tool_start", "tool_args"):
            continue                             # no partial call exists: the whole one arrives on tool_call
        elif ev.kind == "tool_call":
            yield payload([{"functionCall": {"name": ev.call.name, "args": ev.call.arguments}}])


def gemini_collect(chunks) -> dict:
    """The events as one answer: the parts joined, the last event's finish reason and counts."""
    parts, finish, usage, last = [], None, None, None
    for c in chunks:
        if c is None:
            continue
        last = c
        for part in c["candidates"][0]["content"]["parts"]:
            if "functionCall" in part:
                parts.append(part)
            elif part.get("text"):
                # a text part continues the one before it when both are thought or both are answer text
                if parts and "text" in parts[-1] and (parts[-1].get("thought") is True) == (part.get("thought") is True):
                    parts[-1]["text"] += part["text"]
                else:
                    parts.append(dict(part))
        finish = c["candidates"][0].get("finishReason") or finish
        usage = c.get("usageMetadata") or usage
    cand = {"content": {"parts": parts, "role": "model"}, "index": 0, "safetyRatings": []}
    if finish:
        cand["finishReason"] = finish
    out = {"candidates": [cand]}
    if last:
        out["responseId"], out["modelVersion"] = last["responseId"], last["modelVersion"]
    if usage:
        out["usageMetadata"] = usage
    return out
