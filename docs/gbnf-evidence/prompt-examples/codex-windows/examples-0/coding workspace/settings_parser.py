def parse_settings(text):
    if not isinstance(text, str):
        raise TypeError("parse_settings() requires a str")
    result = {}
    seen = set()
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(
                "missing '=' in setting on line %d: %r" % (number, line)
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(
                "empty key on line %d: %r" % (number, line)
            )
        if key in seen:
            raise ValueError(
                "duplicate key %r on line %d" % (key, number)
            )
        seen.add(key)
        result[key] = value
    return result
