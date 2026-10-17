"""JSON response formats at the HTTP boundary: prompt once, validate before delivery.

The native engine has no grammar decoder. Failed generations are errors, never
silently retried or returned as successful structured output.

`jsonschema` is optional (no new hard dependency of the server): json_object needs only the standard library, and
json_schema is checked against its schema when the package is installed - without it, only that the answer is one
JSON object, and the server says so once.
"""
import json
import threading


class StructuredOutputError(RuntimeError):
    pass


class _ObjectOnly:
    """The validator for json_object, and for json_schema without jsonschema: one JSON object."""

    def iter_errors(self, value):
        if not isinstance(value, dict):
            yield _ObjectError()


class _ObjectError:
    absolute_path = ()
    message = "the answer is not a JSON object"


class _PlainString:
    """A json_schema that is one required string field: the model writes that string as plain text and the server
    builds the object around it.  Measured on a scanned rent roll (Qwen3.8-Flash-Next UD-IQ4_XS, temperature 0): asked
    for {"text": ...}, the model wrote the table inside the JSON string, then the escaped newline 256 times instead of
    the page's last lines and the closing quote; asked for plain text, the same page came out whole and ended on its
    own.  `inner` checks the built object against the schema as usual."""

    def __init__(self, inner, key):
        self.inner, self.key = inner, key

    def iter_errors(self, value):
        return self.inner.iter_errors(value)


_PLAIN_FIELD_KEYS = {"type", "description", "title"}
_PLAIN_ROOT_KEYS = {"type", "properties", "required", "additionalProperties", "description", "title", "$schema"}


def single_string_key(schema):
    """The field name when `schema` is an object with exactly one property, a required plain string (no enum, pattern
    or length limits: anything the model writes is then a valid value), and nothing else allowed; else None."""
    if not isinstance(schema, dict) or schema.get("type") != "object" or set(schema) - _PLAIN_ROOT_KEYS:
        return None
    props = schema.get("properties")
    if not isinstance(props, dict) or len(props) != 1:
        return None
    key, field = next(iter(props.items()))
    if schema.get("required") != [key] or schema.get("additionalProperties", False) is not False:
        return None
    if not isinstance(field, dict) or field.get("type") != "string" or set(field) - _PLAIN_FIELD_KEYS:
        return None
    return key


_jsonschema = None                     # (validators, SchemaError, Registry, NoSuchResource), False when not installed
_jsonschema_lock = threading.Lock()


def jsonschema_modules():
    """jsonschema, imported on first use; None when it is not installed (said once, in the server window)."""
    global _jsonschema
    with _jsonschema_lock:
        if _jsonschema is None:
            try:
                from jsonschema import validators
                from jsonschema.exceptions import SchemaError
                from referencing import Registry
                from referencing.exceptions import NoSuchResource
                _jsonschema = (validators, SchemaError, Registry, NoSuchResource)
            except ImportError:
                _jsonschema = False
                print('[strata] response_format json_schema: the Python package jsonschema is not installed, so '
                      'answers are only checked to be one JSON object (python -m pip install "jsonschema>=4.23,<5")',
                      flush=True)
        return _jsonschema or None


def _only_objects(node, root, refs=()):
    """True when every value `node` accepts is a JSON object, so "return one JSON object" stays true.

    `type: object`, an anyOf/oneOf whose branches all qualify (e.g. a root union of object shapes, which llama.cpp's
    grammar path accepts and apps send), an allOf with a qualifying member, or a local `$ref` to one of these.
    Anything that can also be an array, string, number, boolean or null (including `type: ["object", "null"]`) is
    not, and a `$ref` cycle never qualifies.
    """
    if not isinstance(node, dict):
        return False
    if node.get("type") == "object":
        return True
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#") and ref not in refs:
        target = root
        for part in ref[1:].split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                return False
            target = target[part]
        return _only_objects(target, root, refs + (ref,))
    for key in ("anyOf", "oneOf"):
        branches = node.get(key)
        if isinstance(branches, list) and branches and all(_only_objects(b, root, refs) for b in branches):
            return True
    branches = node.get("allOf")
    return isinstance(branches, list) and any(_only_objects(b, root, refs) for b in branches)


def prepare_format(response_format, messages, with_tools=False, plain_string=False):
    """The messages with the format directive added, and the validator for the answer (None: no format).
    `plain_string` (the config's "structured_plain_string", opt-in): a json_schema that is one required string
    field is answered as plain text and wrapped by the server (see _PlainString); not with tools."""
    if response_format is None:
        return messages, None
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    kind = response_format.get("type")
    if kind == "text":
        return messages, None
    if kind == "json_object":
        schema = {"type": "object"}
        modules = None                 # the standard library is enough for "one JSON object"
    elif kind == "json_schema":
        spec = response_format.get("json_schema")
        if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
            raise ValueError("response_format.json_schema needs a schema object")
        if not isinstance(spec.get("name"), str) or not spec["name"]:
            raise ValueError("response_format.json_schema needs a name")
        if "strict" in spec and not isinstance(spec["strict"], bool):
            raise ValueError("response_format.json_schema.strict must be boolean")
        schema = spec["schema"]
        if not _only_objects(schema, schema):
            raise ValueError("response_format schema must accept only JSON objects at its root "
                             "(type object, or anyOf/oneOf of object schemas)")
        modules = jsonschema_modules()
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
    if modules is None:
        validator = _ObjectOnly()
    else:
        validators, SchemaError, Registry, NoSuchResource = modules

        def no_remote(uri):
            raise NoSuchResource(ref=uri)

        try:
            cls = validators.validator_for(schema)
            cls.check_schema(schema)
            validator = cls(schema, registry=Registry(retrieve=no_remote))
        except SchemaError as exc:
            raise ValueError(f"invalid response_format schema: {exc.message}") from exc
    key = single_string_key(schema) if plain_string and not with_tools and kind == "json_schema" else None
    if key is not None:
        about = schema["properties"][key].get("description")
        directive = ("OUTPUT FORMAT REQUIREMENT: Write the answer as plain text only - no JSON, no braces, no code "
                     "fences, no quotes around it, and nothing before or after it. The server puts this text into "
                     f"the JSON field \"{key}\" of the response itself, so do not write JSON, even where the "
                     "instructions above ask for a JSON object." +
                     (f" The field is described as: {about}" if isinstance(about, str) and about.strip() else ""))
        validator = _PlainString(validator, key)
    else:
        directive = ("OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. "
                     "No Markdown, headings, code fences, commentary, or text outside the JSON. "
                     "Use every required field, correct types, and only allowed fields. "
                     "Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n" +
                     json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    if with_tools:
        # /v1/responses (#782): the schema is for the final answer; a turn that calls a tool is not an answer
        directive += ("\nThis applies only to your final answer. To use a tool, call it as usual; the JSON object is "
                      "what you write once you are done with the tools.")
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


def _extract_json(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if s.startswith("```"):
        first = s.find("\n")
        if first != -1:
            end = s.find("```", first + 1)
            if end != -1:
                s = s[first + 1:end].strip()
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
        else:
            if ch == "\"":
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return s[start:]


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
    if isinstance(validator, _PlainString):
        return _wrapped(text, validator, pairs, constant)
    try:
        value = json.loads(_extract_json(text), object_pairs_hook=pairs, parse_constant=constant)
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


def _wrapped(text, validator, pairs, constant):
    """A plain-string answer (see _PlainString) as the schema's object.  A model that wrote a JSON object anyway is
    taken at its word: that object is the answer, checked against the schema as always (an invalid one is an error,
    never quietly turned into a string).  Any other text, as written, is the field's value."""
    s = (text or "").strip()
    value = None
    if s.startswith("{") or s.startswith("```"):
        try:
            value = json.loads(_extract_json(s), object_pairs_hook=pairs, parse_constant=constant)
        except (ValueError, TypeError):
            value = None
        if not isinstance(value, dict):
            value = None
    if value is not None:
        try:
            error = next(validator.iter_errors(value), None)
        except Exception as exc:
            raise StructuredOutputError(f"could not validate structured output: {exc}") from exc
        if error is not None:
            path = "/" + "/".join(str(part) for part in error.absolute_path)
            raise StructuredOutputError(f"model output failed the JSON Schema at {path}: {error.message}")
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if not s:
        raise StructuredOutputError("model returned no text for the answer")
    value = {validator.key: s}
    try:
        error = next(validator.iter_errors(value), None)
    except Exception as exc:
        raise StructuredOutputError(f"could not validate structured output: {exc}") from exc
    if error is not None:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise StructuredOutputError(f"model output failed the JSON Schema at {path}: {error.message}")
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
