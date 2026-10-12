def parse_timeout(value, default=30):
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("booleans are not valid timeouts")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("timeout must be nonnegative")
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        if all(ch in "0123456789" for ch in stripped):
            return int(stripped)
        raise ValueError("timeout string must be ASCII decimal digits")
    raise ValueError("unsupported timeout type")
