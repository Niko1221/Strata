"""csv_stats.py <file.csv> (or the path on stdin): row count, columns, and per column type, empty cells, min/max/mean for numbers."""
import csv
import json
import sys


def stats(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        header = next(reader, [])
        cols = [{"name": h, "empty": 0, "numbers": [], "text": 0} for h in header]
        rows = 0
        for row in reader:
            rows += 1
            for i, col in enumerate(cols):
                value = row[i].strip() if i < len(row) else ""
                if not value:
                    col["empty"] += 1
                    continue
                try:
                    col["numbers"].append(float(value.replace(",", ".")))
                except ValueError:
                    col["text"] += 1
    out = []
    for col in cols:
        nums = col["numbers"]
        entry = {"name": col["name"], "type": "number" if nums and not col["text"] else "text", "empty": col["empty"]}
        if entry["type"] == "number":
            entry.update(min=min(nums), max=max(nums), mean=round(sum(nums) / len(nums), 4))
        out.append(entry)
    return {"file": path, "rows": rows, "columns": out}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()
    if not target:
        sys.exit("usage: csv_stats.py <file.csv>")
    print(json.dumps(stats(target), indent=2, ensure_ascii=False))
