def parse_timeout(value, default=30):
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("timeout must be a nonnegative int or an ASCII digit string")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("timeout must be nonnegative")
        return value
    if isinstance(value, str):
        if value.isspace():
            return default
        if value.isascii() and value.isdigit() and value == value.strip():
            return int(value)
        raise ValueError("timeout string must be stripped ASCII decimal digits")
    raise ValueError("timeout must be an int, digit string, None, or whitespace-only string")
