"""Responses JSON formats: native schema masks plus mandatory final validation.

No object-only fallback for a missing schema validator; no rewriting streamed text.
The legacy Chat Completions structured-output adapter is independent.
"""
from dataclasses import dataclass
import copy
import json

from serve.grammar import GrammarConstraint


@dataclass
class JsonOutput:
    schema: dict
    validator: object

    def constraint(self, thinking, tools, budget):
        source = json.dumps(native_schema(self.schema), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        return GrammarConstraint(source, bool(thinking), bool(tools), json_schema=True,
                                 reasoning_tokens=min(budget or 0, 8192) if thinking else 0)

    def instruction(self):
        return ("When producing the final assistant answer, return exactly one JSON value matching this schema. "
                "Do not add Markdown fences or text outside the JSON. Function calls may precede the answer. "
                "Schema: " + json.dumps(self.schema, ensure_ascii=False, allow_nan=False))

    def validate(self, text):
        # Import here to reuse the transport's duplicate-key/finite-number/UTF-8
        # rules without introducing a module import cycle.
        from serve.responses import strict_json
        value = strict_json(text)
        error = next(self.validator.iter_errors(value), None)
        if error is not None:
            path = "/" + "/".join(str(part) for part in error.absolute_path)
            raise ValueError("JSON output failed schema validation at " + path + ": " + error.message)


def schema_nodes(schema):
    """Yield schemas, never property names, annotation values or const data."""
    pending = [schema]
    while pending:
        node = pending.pop()
        if not isinstance(node, dict):
            continue
        yield node
        for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
            if isinstance(node.get(key), dict):
                pending.extend(node[key].values())
        for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
            if isinstance(node.get(key), list):
                pending.extend(node[key])
        for key in ("additionalProperties", "items", "contains", "propertyNames", "if", "then", "else", "not",
                    "unevaluatedProperties", "unevaluatedItems", "contentSchema"):
            if key in node:
                pending.append(node[key])


def native_schema(schema):
    """Equivalent simplification for a native compiler's optional-false limitation."""
    result = copy.deepcopy(schema)
    for node in schema_nodes(result):
        if node.get("additionalProperties") is False:
            # A forbidden, non-required property in a closed object is exactly
            # equivalent to omitting it. Required false schemas stay impossible.
            props = node.get("properties", {})
            if isinstance(props, dict):
                for name in list(props):
                    if props[name] is False and name not in node.get("required", []):
                        del props[name]
    return result


def prepare_function_schema(function, param, normalize=False):
    """Normalize omitted strict when possible; validate explicit strict guarantees."""
    from serve.responses import RequestError
    schema = copy.deepcopy(function["parameters"])
    # Check schema shape before operations such as iterating required/properties.
    try:
        prepare_json_output({"type": "json_schema", "name": function["name"], "schema": schema})
    except RequestError as exc:
        exc.param = param + ".parameters"
        raise
    for node in schema_nodes(schema):
        if node.get("type") == "object" or "properties" in node:
            props = node.get("properties", {})
            if not isinstance(props, dict):
                raise RequestError("properties must be an object", param + ".parameters")
            if normalize:
                # An explicitly open/dynamic object cannot be normalized without
                # dropping the caller's behavior. Return the documented non-strict
                # fallback, reflected as strict:false in the response tool.
                if node.get("additionalProperties", False) is not False or node.get("patternProperties"):
                    function["strict"] = False
                    return None
                node["additionalProperties"] = False
                node["required"] = list(props)
            elif node.get("additionalProperties") is not False or set(node.get("required", [])) != set(props):
                raise RequestError("strict functions require additionalProperties:false and all properties required",
                                   param + ".parameters")
    try:
        prepared = prepare_json_output({"type": "json_schema", "name": function["name"], "strict": True, "schema": schema})
    except RequestError as exc:
        exc.param = param + ".parameters"
        raise
    function["parameters"] = schema
    function["strict"] = True
    return prepared


def prepare_json_output(fmt):
    from serve.responses import RequestError, fields, string, unsupported
    kind = fmt.get("type") if isinstance(fmt, dict) else None
    if kind == "text":
        fields(fmt, "type", "text.format")
        return None
    if kind == "json_object":
        fields(fmt, "type", "text.format")
        schema = {"type": "object"}
    elif kind == "json_schema":
        fields(fmt, "type name description schema strict", "text.format")
        name = string(fmt.get("name"), "text.format.name", False)
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise RequestError("expected 1-64 letters/digits/underscores/dashes", "text.format.name")
        if "description" in fmt:
            string(fmt["description"], "text.format.description")
        if fmt.get("strict") is not None and type(fmt["strict"]) is not bool:
            raise RequestError("strict must be boolean or null", "text.format.strict")
        schema = fmt.get("schema")
        if not isinstance(schema, dict):
            raise RequestError("schema must be a JSON Schema object", "text.format.schema")
    else:
        unsupported("supported text.format types: text, json_object, json_schema", "text.format")

    # Bound admission and prohibit external references before model loading.
    raw = json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(raw.encode("utf-8")) > 8192:
        raise RequestError("JSON schema exceeds 8192 UTF-8 bytes", "text.format.schema")
    pending = [(schema, 0)]
    while pending:
        node, depth = pending.pop()
        if depth > 32:
            raise RequestError("JSON schema nesting exceeds 32", "text.format.schema")
        if isinstance(node, dict):
            pending.extend((value, depth + 1) for value in node.values())
        elif isinstance(node, list):
            pending.extend((value, depth + 1) for value in node)
    try:
        from jsonschema import Draft202012Validator, FormatChecker
        from jsonschema.exceptions import SchemaError
        from referencing import Registry
        from referencing.exceptions import NoSuchResource
    except ImportError as exc:
        raise RequestError("JSON output requires requirements-json.txt; no weaker validation is used",
                           "text.format", "unsupported_parameter") from exc
    if schema.get("$schema", "https://json-schema.org/draft/2020-12/schema") not in (
            "https://json-schema.org/draft/2020-12/schema", "https://json-schema.org/draft/2020-12/schema#"):
        raise RequestError("JSON output uses JSON Schema draft 2020-12", "text.format.schema")

    # Walk schemas, not arbitrary object values. A property named "$ref" or
    # JSON data inside const/default must not be mistaken for a reference.
    annotations = set("$schema $id $anchor $dynamicAnchor $defs definitions then else minContains maxContains title description default examples "
                      "deprecated readOnly writeOnly $comment contentEncoding contentMediaType contentSchema".split())
    for node in schema_nodes(schema):
        unknown = set(node) - set(Draft202012Validator.VALIDATORS) - annotations
        if unknown:
            raise RequestError("unknown JSON Schema keyword: " + sorted(unknown)[0], "text.format.schema")
        for key in ("$ref", "$dynamicRef"):
            if key in node and (not isinstance(node[key], str) or not node[key].startswith("#")):
                raise RequestError("only local schema references are supported", "text.format.schema")

    def no_remote(uri):
        raise NoSuchResource(ref=uri)

    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, registry=Registry(retrieve=no_remote), format_checker=FormatChecker())
    except SchemaError as exc:
        raise RequestError("invalid JSON schema: " + exc.message, "text.format.schema") from exc
    return JsonOutput(schema, validator)
