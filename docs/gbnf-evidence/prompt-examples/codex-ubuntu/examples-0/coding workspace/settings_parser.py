def parse_settings(text):
    """Parse ``key=value`` settings text into a dict (insertion order preserved).

    Raises TypeError if text is not a str; raises ValueError (with the 1-based
    physical line number) for a missing '=', an empty key, or a duplicate key.
    """
    if not isinstance(text, str):
        raise TypeError("parse_settings requires a str, got %s" % type(text).__name__)

    settings = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("Missing '=' on line %d: %r" % (lineno, line))
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError("Empty key on line %d: %r" % (lineno, line))
        if key in settings:
            raise ValueError("Duplicate key %r on line %d" % (key, lineno))
        settings[key] = value
    return settings
