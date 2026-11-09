"""Extract per-case outcomes from JUnit without machine paths or host metadata.

Author: Francisco Angulo de Lafuente.
ORCID: https://orcid.org/0009-0001-1634-7063
The extraction preserves every testcase and outcome; it is not the original XML.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def convert(source, destination, replacements=()):
    raw = Path(source).read_bytes()
    pairs = []
    for old, new in replacements:
        if not old:
            raise ValueError("Replacement paths must be nonempty")
        for variant in (old, old.replace("\\", "/"), old.replace("/", "\\")):
            pairs.append((variant, new))
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    def clean(value):
        for old, new in pairs:
            value = value.replace(old, new)
        return value

    rows = []
    for case in ET.fromstring(raw).iter("testcase"):
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        skipped = case.find("skipped")
        status = "failed" if failure is not None else ("skipped" if skipped is not None else "passed")
        row = {"id": clean(case.get("classname", "") + "::" + case.get("name", "")),
               "status": status, "seconds": case.get("time", "")}
        if failure is not None:
            row["message"] = clean(failure.get("message", ""))
        if skipped is not None:
            row["skip_reason"] = clean(skipped.get("message", ""))
        rows.append(row)
    counts = Counter(row["status"] for row in rows)
    record = {"format": "junit-outcomes-v1", "original_xml_sha256": hashlib.sha256(raw).hexdigest(),
              "extraction": "testcase id, outcome, seconds and failure/skip message; explicit path placeholders; no host metadata or traceback bodies",
              "counts": {key: counts[key] for key in ("passed", "failed", "skipped")}, "cases": rows}
    data = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    Path(destination).write_bytes(gzip.compress(data, mtime=0))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--replace", action="append", default=[], metavar="PATH=LABEL")
    args = parser.parse_args()
    pairs = [value.split("=", 1) for value in args.replace]
    if any(len(pair) != 2 for pair in pairs):
        parser.error("Every replacement must have PATH=LABEL form")
    record = convert(args.source, args.destination, pairs)
    print(json.dumps({"counts": record["counts"], "testcases": len(record["cases"]),
                      "original_xml_sha256": record["original_xml_sha256"]}))


if __name__ == "__main__":
    main()
