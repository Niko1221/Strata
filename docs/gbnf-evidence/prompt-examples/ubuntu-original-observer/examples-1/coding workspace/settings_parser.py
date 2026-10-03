def parse_settings(text):
    """Parse simple KEY=VALUE settings text into a dict.

    Raises TypeError for non-str input and ValueError (with the 1-based physical
    line number) for a missing '=', an empty key, or a duplicate key.
    """
    if not isinstance(text, str):
        raise TypeError(
            "parse_settings() requires a str, got %s" % type(text).__name__
        )

    result = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                "line %d has no '=' separator: %r" % (lineno, line)
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(
                "line %d has an empty key: %r" % (lineno, line)
            )
        if key in result:
            raise ValueError(
                "duplicate key %r on line %d" % (key, lineno)
            )
        result[key] = value
    return result
