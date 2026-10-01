"""Validate the OpenAI response-format contract and completed constrained output."""
import json
import re

from jsonschema import validators
from jsonschema.exceptions import SchemaError
from referencing import Registry
from referencing.exceptions import NoSuchResource


class StructuredOutputError(RuntimeError):
    pass


def check_strict_schema(schema):
    """Reject unsupported strict schemas before loading or generating, as OpenAI does."""
    unsupported = {"allOf", "oneOf", "not", "dependentRequired", "dependentSchemas", "if", "then", "else",
                   "uniqueItems", "contains", "minContains", "maxContains", "unevaluatedProperties",
                   "unevaluatedItems", "patternProperties", "propertyNames", "minProperties", "maxProperties"}
    counts = {"properties": 0, "enum": 0, "strings": 0}

    def visit(node, depth=1):
        if not isinstance(node, dict):
            raise ValueError("strict response_format requires schema objects")
        bad = unsupported.intersection(node)
        if bad:
            raise ValueError("unsupported strict response_format keyword: " + sorted(bad)[0])
        types = node.get("type", [])
        types = [types] if isinstance(types, str) else types
        if "object" in types or "properties" in node:
            props = node.get("properties", {})
            if node.get("additionalProperties") is not False:
                raise ValueError("strict response_format objects require additionalProperties: false")
            if set(node.get("required", [])) != set(props):
                raise ValueError("strict response_format requires every property; use a nullable type for optional values")
            if depth > 10:
                raise ValueError("strict response_format exceeds 10 object nesting levels")
            counts["properties"] += len(props)
            counts["strings"] += sum(map(len, props))
            for child in props.values():
                visit(child, depth + 1)
        values = node.get("enum", [])
        counts["enum"] += len(values)
        counts["strings"] += sum(len(x) for x in values if isinstance(x, str))
        if len(values) > 250 and sum(len(x) for x in values if isinstance(x, str)) > 15000:
            raise ValueError("strict response_format string enum exceeds 15000 characters")
        if isinstance(node.get("const"), str):
            counts["strings"] += len(node["const"])
        for name in ("$defs", "definitions"):
            defs = node.get(name, {})
            counts["strings"] += sum(map(len, defs))
            for child in defs.values():
                visit(child, depth)
        if "items" in node:
            visit(node["items"], depth)
        for child in node.get("anyOf", []):
            visit(child, depth)
    if "anyOf" in schema:
        raise ValueError("strict response_format root cannot use anyOf")
    visit(schema)
    if counts["properties"] > 5000 or counts["enum"] > 1000 or counts["strings"] > 120000:
        raise ValueError("strict response_format exceeds OpenAI schema size limits")


def _no_remote(uri):
    raise NoSuchResource(ref=uri)


def prepare_format(response_format, messages):
    if response_format is None:
        return messages, None
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    kind = response_format.get("type")
    if kind == "text":
        return messages, None
    if kind == "json_object":
        schema = {"type": "object"}
    elif kind == "json_schema":
        spec = response_format.get("json_schema")
        if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
            raise ValueError("response_format.json_schema needs a schema object")
        if not isinstance(spec.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", spec["name"]):
            raise ValueError("response_format.json_schema needs a name")
        if "strict" in spec and not isinstance(spec["strict"], bool):
            raise ValueError("response_format.json_schema.strict must be boolean")
        schema = spec["schema"]
        if schema.get("type") != "object":
            raise ValueError("response_format schema must have type object at its root")
        if spec.get("strict") is True:
            check_strict_schema(schema)
    else:
        raise ValueError("response_format.type must be text, json_object or json_schema")

    def check_refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("$ref", "$dynamicRef") and isinstance(value, str) and not value.startswith("#"):
                    raise ValueError("response_format supports only local schema references (#...)")
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)
    check_refs(schema)
    try:
        cls = validators.validator_for(schema)
        cls.check_schema(schema)
        validator = cls(schema, registry=Registry(retrieve=_no_remote))
    except SchemaError as exc:
        raise ValueError(f"invalid response_format schema: {exc.message}") from exc
    directive = ("OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. "
                 "No Markdown, headings, code fences, commentary, or text outside the JSON. "
                 "Use every required field, correct types, and only allowed fields. "
                 "Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n" +
                 json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    messages = [dict(message) for message in messages]
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
        if isinstance(content, list):
            messages[0]["content"] = content + [{"type": "text", "text": directive}]
        else:
            messages[0]["content"] = content + "\n\n" + directive
    else:
        messages.insert(0, {"role": "system", "content": directive})
    return messages, validator


def validated_json(text, validator, finish):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    if finish != "stop":
        raise StructuredOutputError(f"structured output was incomplete (finish_reason={finish}); increase the output budget")
    try:
        value = json.loads(text or "", object_pairs_hook=pairs, parse_constant=constant)
        canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError) as exc:
        raise StructuredOutputError(f"model did not return valid JSON: {exc}") from exc
    try:
        error = next(validator.iter_errors(value), None)
    except Exception as exc:
        raise StructuredOutputError(f"could not validate structured output: {exc}") from exc
    if error is not None:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise StructuredOutputError(f"model output failed the JSON Schema at {path}: {error.message}")
    return canonical
