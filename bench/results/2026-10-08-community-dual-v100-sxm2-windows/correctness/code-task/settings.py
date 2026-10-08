def parse_timeout(value, default=30):
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        if all(ch in "0123456789" for ch in stripped):
            return int(stripped)
        raise ValueError("value must be a nonnegative int or an ASCII digit string")
    if isinstance(value, bool):
        raise ValueError("value must be a nonnegative int or an ASCII digit string")
    if isinstance(value, int) and value >= 0:
        return value
    raise ValueError("value must be a nonnegative int or an ASCII digit string")
