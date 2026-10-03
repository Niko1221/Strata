"""Generate readable .txt request examples and every nested create-request type in the pinned SDK.

Documentation generation only. This does not introduce a JSON Schema generation
feature, validator or grammar compiler into Strata.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import openai
from pydantic import TypeAdapter
from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming
from cryptography.fernet import Fernet
from serve.response_replay import ReplayCodec


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def example(shape, defs, seen=()):
    """One structural example per pinned type, never a conformance claim."""
    if "$ref" in shape:
        name = shape["$ref"].rsplit("/", 1)[-1]
        return f"<recursive {name}>" if name in seen else example(defs[name], defs, (*seen, name))
    if "const" in shape:
        return shape["const"]
    if "enum" in shape:
        return shape["enum"][0]
    for union in ("anyOf", "oneOf"):
        if union in shape:
            return example(next((x for x in shape[union] if x.get("type") != "null"), shape[union][0]), defs, seen)
    kind = shape.get("type")
    if kind == "object":
        return {k: example(v, defs, seen) for k, v in shape.get("properties", {}).items()}
    if kind == "array":
        return [example(shape.get("items", {}), defs, seen)]
    return {"string": "<example text>", "integer": 1, "number": 0.5, "boolean": False, "null": None}.get(kind, "<value>")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/responses-examples")
    a = ap.parse_args()
    if openai.__version__ != "3.23.0":
        ap.error("use the pinned openai==3.23.0 environment")
    a.out.mkdir(parents=True, exist_ok=True)
    actual = json.loads((ROOT / "docs/responses-evidence/R0/request-fixtures/codex-initial-1.json").read_text(encoding="utf-8"))
    (a.out / "CODEX_ACTUAL_REQUEST.txt").write_text(
        "REAL CAPTURE: Codex CLI 0.160.0, sanitized initial request to loopback recorder.\n"
        "The recorder returned a synthetic error; no model output/tool loop was captured.\n\n"
        "POST " + actual["path"] + "\nContent-Type: application/json\n\n" + pretty(actual["body"]) + "\n", encoding="utf-8")
    parts = ["ACTUAL CODEX REQUEST PARTS\nClient: codex-cli 0.160.0\n"
             "These are field values from the same sanitized request, not invented examples.\n"]
    for key, value in actual["body"].items():
        parts.append(f"\nFIELD: {key}\n" + pretty(value))
    (a.out / "CODEX_ACTUAL_PARTS.txt").write_text("\n".join(parts) + "\n", encoding="utf-8")
    tool = {"type": "function", "name": "read_file", "description": "Read a client-owned file.", "strict": False,
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}
    base = {"model": "qwen3.8-flash-next", "store": False, "input": "Hello."}
    # This public throwaway key and synthetic plaintext demonstrate the real
    # implementation. No deployment key or private reasoning is read here.
    fixture_key = Fernet.generate_key()
    codec = ReplayCodec(fixture_key)
    visible = {"type": "reasoning", "id": "rs_example", "status": "completed", "summary": [],
               "content": [{"type": "reasoning_text", "text": "Synthetic example: compare the two supplied numbers."}]}
    encrypted = {k: v for k, v in visible.items() if k != "content"}
    encrypted["encrypted_content"] = codec.seal(base["model"], visible)
    examples = [
        ("text", "Supported", base),
        ("stream", "Supported; typed SSE response", {**base, "stream": True}),
        ("messages", "Supported; instructions apply to this request only", {**base, "instructions": "Be concise.",
            "input": [{"role": "developer", "content": [{"type": "input_text", "text": "Use plain language."}]},
                      {"role": "user", "content": "Hello."}]}),
        ("functions", "Supported non-strict, client-owned function workflow", {**base, "tools": [tool]}),
        ("function_result_replay", "Supported; item ID and call_id have different roles", {**base, "tools": [tool], "input": [
            {"role": "user", "content": "Read example.txt."},
            {"type": "function_call", "id": "fc_example", "call_id": "call_example", "name": "read_file",
             "arguments": '{"path":"example.txt"}', "status": "completed"},
            {"type": "function_call_output", "call_id": "call_example", "output": "before\n"}]}),
        ("namespace", "Supported non-strict functions; qualified names are preserved", {**base, "tools": [{"type": "namespace", "name": "files",
            "description": "Client-owned file operations.", "tools": [tool]}]}),
        ("reasoning_effort", "Supported native effort mapping and visible reasoning_text", {**base, "reasoning": {"effort": "low"}}),
        ("reasoning_summary", "Supported; a bounded second service pass genuinely generates the summary",
            {**base, "reasoning": {"effort": "low", "summary": "auto"}}),
        ("encrypted_reasoning", "Supported authenticated Strata-issued replay tokens; deployment key survives restarts",
            {**base, "reasoning": {"effort": "low"}, "include": ["reasoning.encrypted_content"]}),
        ("reasoning_replay", "Real encryption of synthetic data under a PUBLIC THROWAWAY EXAMPLE KEY; not accepted by your deployment", {**base, "input": [
            {"role": "user", "content": "Earlier question."},
            encrypted,
            {"type": "message", "id": "msg_example", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "Earlier answer.", "annotations": []}]},
            {"role": "user", "content": "Continue."}]}),
        ("json_schema_EXCLUDED", "Excluded; documentation example is not an implementation", {**base,
            "text": {"format": {"type": "json_schema", "name": "answer", "strict": True,
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}},
                           "required": ["answer"], "additionalProperties": False}}}}),
        ("json_object_EXCLUDED", "Excluded", {**base, "text": {"format": {"type": "json_object"}}}),
        ("strict_function_EXCLUDED", "Excluded; strict:true and omitted strict are rejected", {**base, "tools": [{**tool, "strict": True}]}),
        ("custom_lark_EXCLUDED", "Excluded; raw GBNF does not implement Lark custom tools", {**base, "tools": [{
            "type": "custom", "name": "operation", "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "wait"'}}]}),
        ("storage_EXCLUDED", "Excluded; no response store", {**base, "store": True}),
        ("previous_response_EXCLUDED", "Excluded; supply full input instead", {**base, "previous_response_id": "resp_example"}),
        ("background_EXCLUDED", "Excluded; no background jobs or cancel endpoint", {**base, "background": True}),
        ("image_EXCLUDED", "Excluded input modality", {**base, "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "Describe this."}, {"type": "input_image", "image_url": "<CLIENT_IMAGE_URL>", "detail": "auto"}]}]}),
        ("hosted_tool_EXCLUDED", "Excluded; Strata MCP is not an OpenAI-hosted tool implementation", {**base, "tools": [{"type": "web_search"}]}),
        ("visible_reasoning_replay", "Supported visible content replay without encryption", {**base, "input": [
            {"role": "user", "content": "Compare two and three."}, visible,
            {"role": "assistant", "content": "Three is larger."}, {"role": "user", "content": "Continue."}]}),
        ("diagnostics_and_cache_hint", "Supported diagnostic metadata and routing hint to Strata's sole engine", {**base,
            "client_metadata": {"root_turn_id": "synthetic-turn"}, "prompt_cache_key": "synthetic-session"}),
    ]
    index = ["RESPONSES REQUEST EXAMPLES\nPinned wire inventory: openai-python 3.23.0; Codex capture 0.160.0.\n"
             "Each JSON block is a synthetic documentation example unless explicitly marked REAL CAPTURE.\n"
             "This is a type/field inventory, not every infinite combination of values, and not universal compatibility.\n"
             "Read support labels: requested/in-progress and excluded cases are not certified working.\n"]
    for number, (name, status, body) in enumerate(examples, 1):
        filename = f"{number:02d}_{name}.txt"
        (a.out / filename).write_text(f"SYNTHETIC DOCUMENTATION EXAMPLE\nStatus: {status}\n\nPOST /v1/responses\n"
                                     "Content-Type: application/json\n\n" + pretty(body) + "\n", encoding="utf-8")
        index.append(f"{filename}: {status}")
    index.extend(["", "CODEX_ACTUAL_REQUEST.txt: full sanitized real first request",
                  "CODEX_ACTUAL_PARTS.txt: every field/subpiece in that real request",
                  "ALL_CREATE_FIELDS_AND_PIECES.txt: every top-level field and all 170 nested SDK type definitions",
                  "ALL_CREATE_PIECE_EXAMPLES.txt: a synthetic JSON illustration for every nested type and field",
                  "ENCRYPTION_ROUNDTRIP.txt: public example key, plaintext, real ciphertext and restored data",
                  "REST_AND_TRANSPORTS.txt: method/transport inventory",
                  "REASONING_PIECES.txt: distinctions between raw text, summary and encrypted replay"])
    (a.out / "INDEX.txt").write_text("\n".join(index) + "\n", encoding="utf-8")
    schema = TypeAdapter(ResponseCreateParamsNonStreaming).json_schema()
    defs = schema.pop("$defs", {})
    catalogue = ["COMPLETE PINNED CREATE-REQUEST FIELD/PIECE INVENTORY\n"
                 "Source: openai-python 3.23.0 TypedDict definitions, rendered by Pydantic TypeAdapter.\n"
                 "This describes WIRE TYPES only. It adds no JSON Schema output support to Strata.\n"
                 "anyOf/oneOf list alternative shapes; they must not all be combined into one request.\n"
                 "References resolve to the named pieces below. Required lists and enum choices are included.\n"
                 "The root covers stream:false; stream:true selects SSE without changing these request pieces.\n",
                 "ROOT CREATE REQUEST\n" + pretty(schema)]
    for name, definition in sorted(defs.items()):
        catalogue.append("\nPIECE: " + name + "\n" + pretty(definition))
    (a.out / "ALL_CREATE_FIELDS_AND_PIECES.txt").write_text("\n".join(catalogue) + "\n", encoding="utf-8")
    pieces = ["SYNTHETIC STRUCTURAL EXAMPLES, pinned openai-python 3.23.0\n"
              "One illustration per field and nested type. All optional fields are shown where possible.\n"
              "Union examples select one alternative; the companion field inventory lists EVERY alternative and enum.\n"
              "Placeholder values are explanatory. These are not all valid Strata requests: use INDEX.txt for support.\n"]
    for name, field in schema.get("properties", {}).items():
        pieces.append("CREATE FIELD: " + name + "\n" + pretty({name: example(field, defs)}))
    for name, definition in sorted(defs.items()):
        pieces.append("PIECE: " + name + "\n" + pretty(example(definition, defs, (name,))))
    (a.out / "ALL_CREATE_PIECE_EXAMPLES.txt").write_text("\n\n".join(pieces) + "\n", encoding="utf-8")
    restored = codec.restore(base["model"], encrypted)
    (a.out / "ENCRYPTION_ROUNDTRIP.txt").write_text(
        "EXECUTED ENCRYPTION ROUND TRIP, SYNTHETIC PUBLIC FIXTURE\n"
        "Purpose: opaque reasoning replay state (the model's earlier reasoning, not a hidden instruction).\n"
        "Fernet authenticated encryption using serve/response_replay.py; this file shows the actual payload.\n"
        "This public key is for this example only. A real deployment supplies its private key in the environment.\n"
        "No private payload or another provider's encrypted blob was inspected.\n\n"
        "PUBLIC THROWAWAY KEY\n" + fixture_key.decode() + "\n\nPLAINTEXT ITEM\n" + pretty(visible)
        + "\n\nWIRE ITEM (encrypted-only replay)\n" + pretty(encrypted)
        + "\n\nRESTORED ITEM\n" + pretty(restored) + "\n", encoding="utf-8")
    (a.out / "REST_AND_TRANSPORTS.txt").write_text(
        "PINNED RESPONSES METHOD/TRANSPORT INVENTORY\n\n"
        "POST /v1/responses -- implemented stateless JSON / stream:true SSE.\n"
        "GET /v1/responses/{response_id} -- excluded retrieval/storage.\n"
        "GET /v1/responses/{response_id}/input_items -- excluded retained input pagination.\n"
        "DELETE /v1/responses/{response_id} -- excluded response deletion.\n"
        "POST /v1/responses/{response_id}/cancel -- excluded background cancellation.\n"
        "POST /v1/responses/compact -- excluded compaction.\n"
        "POST /v1/responses/input_tokens -- not implemented in this adapter.\n"
        "WebSocket /v1/responses, response.create -- excluded; HTTP profile only.\n\n"
        "No background worker, conversation DB, hosted tools or JSON-constraint frontend is implied.\n", encoding="utf-8")
    reasoning = {
        "raw_reasoning_piece": {"type": "reasoning_text", "text": "<ACTUAL_MODEL_REASONING_TEXT>"},
        "summary_piece": {"type": "summary_text", "text": "<SEPARATELY_GENERATED_FAITHFUL_SUMMARY>"},
        "reasoning_item": {"type": "reasoning", "id": "rs_example", "status": "completed", "summary": [],
                           "content": [{"type": "reasoning_text", "text": "<ACTUAL_MODEL_REASONING_TEXT>"}],
                           "encrypted_content": "<OPAQUE_REASONING_REPLAY_STATE; see ENCRYPTION_ROUNDTRIP.txt>"},
        "namespaced_function_call": {"type": "function_call", "id": "fc_example", "call_id": "call_example",
                                     "namespace": "files", "name": "read_file", "arguments": '{"path":"example.txt"}', "status": "completed"},
        "function_result": {"type": "function_call_output", "call_id": "call_example", "output": "before\n"}}
    (a.out / "REASONING_PIECES.txt").write_text(
        "SYNTHETIC SHAPE EXAMPLES -- placeholders are not generated text or usable ciphertext.\n"
        "Raw thinking is not a summary. Encrypted replay is not a base64 label or fabricated token.\n"
        "A namespace groups client functions; it does not authorize the server to execute them.\n"
        "The function item's id identifies the output item; call_id connects a client result to its call.\n\n"
        + pretty(reasoning) + "\n", encoding="utf-8")
    print(f"Wrote {len(examples)} request examples, actual Codex fields, and {len(defs)} nested type definitions to {a.out}")


if __name__ == "__main__":
    main()
