def parse_timeout(value, default=30):
    if value is None:
        return default
    if isinstance(value, str):
        if value.isspace():
            return default
        stripped = value.strip()
        if stripped and all("0" <= ch <= "9" for ch in stripped):
            return int(stripped)
        raise ValueError(f"invalid timeout: {value!r}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid timeout: {value!r}")
    if value < 0:
        raise ValueError(f"invalid timeout: {value!r}")
    return value
