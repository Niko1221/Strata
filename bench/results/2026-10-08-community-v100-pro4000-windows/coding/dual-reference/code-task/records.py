def stable_unique(records):
    seen = set()
    result = []
    for record in records:
        if "id" not in record:
            result.append(record)
            continue
        record_id = record["id"]
        if record_id in seen:
            continue
        seen.add(record_id)
        result.append(record)
    return result
