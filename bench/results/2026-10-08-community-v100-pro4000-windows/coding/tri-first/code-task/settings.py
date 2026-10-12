def parse_timeout(value, default=30):
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("timeout must not be a boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("timeout must be nonnegative")
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        if not all(char in "0123456789" for char in stripped):
            raise ValueError("timeout string must be ASCII decimal digits")
        return int(stripped)
    raise ValueError("timeout must be an int, a digit string, or None")
