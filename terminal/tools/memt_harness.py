"""tools/memt_harness.py - the small shared plumbing of the P6/P7 harnesses.

Both `memt_parity.py` (does the memory state match memt_sglang?) and
`memt_bench.py` (how fast is it?) need the same few things: token-id files, a
median over repeated runs, and a JSON report written atomically next to the
snapshots.  Nothing here talks to a server; it stays testable on a CPU box.
"""
from __future__ import annotations

import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Iterable, Optional

__all__ = ["HarnessError", "read_ids", "write_json", "median", "summarize", "prefix_match",
           "GREEDY_SAMPLING"]

#: Greedy sampling for both servers: temperature 0 makes every other field moot,
#: so the reference (which defaults to a non-zero temperature) and the engine
#: (greedy when temperature <= 0) are asked for the same thing.  A token
#: comparison between two sampling runs would be noise.
GREEDY_SAMPLING = {"temperature": 0.0}


class HarnessError(RuntimeError):
    pass


def read_ids(path: str | Path) -> list[int]:
    """A token-id list from a file: JSON array, `{"ids": [...]}`, or comma/space separated text."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HarnessError("cannot read %s: %s" % (path, exc)) from exc
    stripped = text.strip()
    if not stripped:
        raise HarnessError("%s is empty" % path)
    if stripped[0] in "[{":
        try:
            value = json.loads(stripped)
        except ValueError as exc:
            raise HarnessError("%s is not valid JSON: %s" % (path, exc)) from exc
        if isinstance(value, dict):
            if "ids" not in value:
                raise HarnessError("%s has no \"ids\" list" % path)
            value = value["ids"]
        if not isinstance(value, list) or not value:
            raise HarnessError("%s does not carry a nonempty token-id list" % path)
    else:
        value = stripped.replace(",", " ").split()
    ids = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            raise HarnessError("%s: %r is not a token id" % (path, item))
        try:
            number = int(item)
        except ValueError as exc:
            raise HarnessError("%s: %r is not a token id" % (path, item)) from exc
        if number < 0:
            raise HarnessError("%s: token ids must be non-negative" % path)
        ids.append(number)
    return ids


def write_json(path: str | Path, value) -> None:
    """Write the report atomically (a reader never sees a half file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".memt-report-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [float(v) for v in values if v is not None]
    return statistics.median(kept) if kept else None


def summarize(runs: list[dict], keys: Iterable[str]) -> dict:
    """The median of each numeric field over the runs (the P7 reporting rule)."""
    out = {"runs": len(runs)}
    for key in keys:
        out[key] = median(run.get(key) for run in runs)
    return out


def prefix_match(a: Iterable[int], b: Iterable[int]) -> int:
    """How many tokens two generations agree on before the first difference."""
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count
