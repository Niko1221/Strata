def parse_settings(text):
    """Parse simple ``key=value`` settings text into a dict.

    Returns a plain dict in source (insertion) order.  Blank lines and lines
    whose stripped text starts with '#' are ignored.  Lines are split at the
    first '=' only; key and value are whitespace-stripped (an empty value is
    valid).  An empty key, a line without '=', or a duplicate key raises
    ValueError mentioning the 1-based physical line number.  Values are kept
    exactly as written (no unquoting, unescaping, expansion or interpretation).
    """
    if not isinstance(text, str):
        raise TypeError("parse_settings() requires a str, got %s" % type(text).__name__)

    result = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if "=" not in line:
            raise ValueError(
                "missing '=' on line %d: %r" % (lineno, line)
            )

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(
                "empty key on line %d: %r" % (lineno, line)
            )

        if key in result:
            raise ValueError(
                "duplicate key %r on line %d" % (key, lineno)
            )

        result[key] = value

    return result
