"""tools/wex4_check.py - check every expert of a WEX4 container byte by byte (docs/WEX4.md).

    python tools/wex4_check.py FILE [--root HEX]

The engine (`--experts-wex4`) checks a container's header, index and layout at every start, but it does not hash the expert bytes:
that means reading the whole file. This tool does it: it reads the file once, checks each record's SHA-256 and its zero padding, and
names the layer and expert of every damaged record. --root also requires the container's root (the SHA-256 of its header page).
Exit 0: every record is intact. Exit 1: something is damaged or refused (each problem is printed).

The format code is the Wex-WRN2 reference reader, vendored unchanged in engine/third_party/wex (Apache-2.0, see its UPSTREAM.md).
Standard library only.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
EXPERT = re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight#(\d+)$")


def load_wex4():
    """The vendored wex4.py: third_party/wex beside tools/ (an installed Terminal), else the source checkout's engine/third_party/wex."""
    for d in (HERE.parent / "third_party" / "wex", HERE.parent.parent / "engine" / "third_party" / "wex"):
        if (d / "wex4.py").is_file():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            import wex4
            return wex4
    sys.exit("the WEX4 reader was not found: engine/third_party/wex/wex4.py in a source checkout, third_party/wex in the Terminal")


def check(path, root: str | None = None, out=print) -> int:
    """0 when every record of the container at `path` is intact (and its root is `root`, when given); 1 otherwise."""
    w = load_wex4()
    t0, bad, done = time.time(), 0, 0
    with open(path, "rb") as f:
        try:
            rd = w.Reader(f)
        except w.FormatError as e:
            out("%s: refused: %s" % (path, e))
            return 1
        if root and rd.root.hex() != root.lower():
            out("%s: its root is %s, --root asks for %s" % (path, rd.root.hex(), root.lower()))
            return 1
        for r, rec in enumerate(rd.records):
            offset, stored, first, count = rec[0], rec[1], rec[7], rec[8]
            name = rd.tensors[first][0].decode("utf-8", "replace") if count else ""
            m = EXPERT.match(name)
            who = "layer %s expert %s" % m.groups() if m else (name or "record %d" % r)
            try:
                rd.record(r, verify=True)
            except w.FormatError as e:
                out("%s: %s: %s (record %d, %d bytes at byte %d)" % (path, who, e, r, stored, offset))
                bad += 1
            done += stored
    if bad:
        out("%s: %d of %d records damaged" % (path, bad, len(rd.records)))
        return 1
    s = max(time.time() - t0, 1e-9)
    out("%s: ok, %d records intact, root %s (%.2f GB checked in %.1f s, %.2f GB/s)" % (
        path, len(rd.records), rd.root.hex(), done / 1e9, s, done / 1e9 / s))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file")
    ap.add_argument("--root", help="64 hex digits: refuse a container with another root")
    a = ap.parse_args()
    return check(a.file, a.root)


if __name__ == "__main__":
    raise SystemExit(main())
