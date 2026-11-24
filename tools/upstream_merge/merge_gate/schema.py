from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

from . import GateError


_JSON_TYPES = {
    "null": type(None),
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "string": str,
    "array": list,
    "object": dict,
}


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, _JSON_TYPES[expected])


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise GateError(f"schema: неподдерживаемый $ref {ref}")
    current: Any = root
    for token in ref[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            raise GateError(f"schema: не найден $ref {ref}")
        current = current[token]
    if not isinstance(current, dict):
        raise GateError(f"schema: $ref {ref} не указывает на объект")
    return current


def _path(parent: str, key: Any) -> str:
    if isinstance(key, int):
        return f"{parent}[{key}]"
    return f"{parent}.{key}" if parent else str(key)


def _validate(value: Any, schema: dict[str, Any], root: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []

    if "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str):
            return [f"{path}: $ref должен быть строкой"]
        errors.extend(_validate(value, _resolve_ref(root, ref), root, path))

    if "type" in schema:
        expected = schema["type"]
        choices = expected if isinstance(expected, list) else [expected]
        if not choices or any(not isinstance(choice, str) or choice not in _JSON_TYPES for choice in choices):
            return [f"{path}: schema содержит неподдерживаемый type"]
        if not any(_matches_type(value, choice) for choice in choices):
            errors.append(f"{path}: ожидается тип {'/'.join(choices)}")

    if "const" in schema and not _equal(value, schema["const"]):
        errors.append(f"{path}: значение должно равняться {schema['const']!r}")

    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, list) or not any(_equal(value, choice) for choice in choices):
            errors.append(f"{path}: значение отсутствует в enum")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: строка короче minLength")
        if "pattern" in schema:
            try:
                matched = re.search(schema["pattern"], value)
            except re.error as exc:
                raise GateError(f"schema: некорректный pattern: {exc}") from exc
            if matched is None:
                errors.append(f"{path}: строка не соответствует pattern")
        if schema.get("format") == "date":
            try:
                dt.date.fromisoformat(value)
            except ValueError:
                errors.append(f"{path}: ожидается дата YYYY-MM-DD")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: значение ниже minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: значение выше maximum")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: элементов меньше minItems")
        if schema.get("uniqueItems") is True:
            seen: set[str] = set()
            for index, item in enumerate(value):
                marker = _canonical(item)
                if marker in seen:
                    errors.append(f"{path}[{index}]: дублирующийся элемент")
                    break
                seen.add(marker)
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(_validate(item, schema["items"], root, _path(path, index)))
        if "contains" in schema:
            if not any(not _validate(item, schema["contains"], root, _path(path, index)) for index, item in enumerate(value)):
                errors.append(f"{path}: ни один элемент не соответствует contains")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: отсутствует обязательное поле {key}")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise GateError(f"schema: properties в {path or '<root>'} должен быть объектом")
        for key, item in value.items():
            if key in properties:
                errors.extend(_validate(item, properties[key], root, _path(path, key)))
                continue
            additional = schema.get("additionalProperties", True)
            if additional is False:
                errors.append(f"{path}: неизвестное поле {key}")
            elif isinstance(additional, dict):
                errors.extend(_validate(item, additional, root, _path(path, key)))

    for subschema in schema.get("allOf", []):
        errors.extend(_validate(value, subschema, root, path))

    if "anyOf" in schema:
        alternatives = schema["anyOf"]
        if not any(not _validate(value, alternative, root, path) for alternative in alternatives):
            errors.append(f"{path}: ни один anyOf-вариант не подходит")

    if "oneOf" in schema:
        alternatives = schema["oneOf"]
        matches = sum(not _validate(value, alternative, root, path) for alternative in alternatives)
        if matches != 1:
            errors.append(f"{path}: oneOf требует ровно один подходящий вариант, получено {matches}")

    if "not" in schema and not _validate(value, schema["not"], root, path):
        errors.append(f"{path}: значение соответствует запрещённой схеме not")

    if "if" in schema:
        condition_matches = not _validate(value, schema["if"], root, path)
        branch = "then" if condition_matches else "else"
        if branch in schema:
            errors.extend(_validate(value, schema[branch], root, path))

    return errors


def validate_schema(instance: Any, schema: dict[str, Any]) -> None:
    if not isinstance(schema, dict):
        raise GateError("schema: корень должен быть объектом")
    errors = _validate(instance, schema, schema, "")
    if errors:
        raise GateError("schema: " + "; ".join(errors[:12]))
