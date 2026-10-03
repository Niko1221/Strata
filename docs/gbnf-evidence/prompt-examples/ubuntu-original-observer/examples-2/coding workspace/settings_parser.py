def parse_settings(text):
    result = {}
    for line in text.split("\n"):
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=")
        result[key] = value
    return result
