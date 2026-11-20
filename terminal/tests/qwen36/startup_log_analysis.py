"""Parse strata-q36 stderr into startup attempts.

Historical evidence is counted once per byte-identical upload. A line with no
following result stays unknown. A 0-of-0 hybrid counter has no hit rate.
Times from different generations are not added together.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field


_GENERATION = re.compile(r"strata-q36: startup generation (\S+)")
_CONTEXT = re.compile(r"context (\d+) cells, prompt chunk (\d+)")
_DENSE = re.compile(r"([0-9]+(?:\.[0-9]+)?) GiB on the GPU in ([0-9]+(?:\.[0-9]+)?) s")
_HOST = re.compile(r"no host memory for the experts \((\d+) MiB\)")
_VRAM = re.compile(r"not enough VRAM left|do not fit in VRAM")
_PLACED = re.compile(
    r"hybrid experts: (\d+) of (\d+) per layer in VRAM \(([0-9]+(?:\.[0-9]+)?) GiB\), all ([0-9]+(?:\.[0-9]+)?) GiB in (\w+) host memory.*?([0-9]+(?:\.[0-9]+)?) s"
)
_WARM = re.compile(r"warmed up in ([0-9]+(?:\.[0-9]+)?) s")
_ROUTE = re.compile(r"hybrid experts: (\d+) of (\d+) batch entries")


@dataclass
class Attempt:
    generation: str
    context: int | None = None
    prefill: int | None = None
    dense_seconds: float | None = None
    placement_seconds: float | None = None
    warmup_seconds: float | None = None
    host_mib: int | None = None
    outcome: str = "unknown"
    route_hit_rate: float | None = None
    gpu_route_entries: int | None = None
    total_route_entries: int | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "generation": self.generation,
            "context": self.context,
            "prefill": self.prefill,
            "dense_seconds": self.dense_seconds,
            "placement_seconds": self.placement_seconds,
            "warmup_seconds": self.warmup_seconds,
            "host_mib": self.host_mib,
            "outcome": self.outcome,
            "route_hit_rate": self.route_hit_rate,
            "gpu_route_entries": self.gpu_route_entries,
            "total_route_entries": self.total_route_entries,
        }


def _unique_blobs(blobs: list[bytes]) -> list[bytes]:
    seen: set[str] = set()
    out: list[bytes] = []
    for blob in blobs:
        digest = hashlib.sha256(blob).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append(blob)
    return out


def _apply_route(attempt: Attempt, gpu: int, total: int) -> None:
    attempt.gpu_route_entries = gpu
    attempt.total_route_entries = total
    if total <= 0:
        attempt.route_hit_rate = None
        return
    attempt.route_hit_rate = gpu / total


def parse_startup_log(text: str, generation_prefix: str = "anon") -> list[Attempt]:
    generation: str | None = None
    attempts: list[Attempt] = []
    current: Attempt | None = None
    anon = 0

    def open_attempt() -> Attempt:
        nonlocal anon
        if generation is None:
            anon += 1
            gen = f"{generation_prefix}-{anon}"
        else:
            gen = generation
        attempt = Attempt(generation=gen)
        attempts.append(attempt)
        return attempt

    for raw in text.splitlines():
        line = raw.strip()
        found = _GENERATION.search(line)
        if found:
            generation = found.group(1)
            current = None
            continue
        if _CONTEXT.search(line):
            current = open_attempt()
            ctx, chunk = _CONTEXT.search(line).groups()
            current.context = int(ctx)
            current.prefill = int(chunk)
            continue
        if current is None and (_DENSE.search(line) or _HOST.search(line) or _VRAM.search(line)):
            current = open_attempt()
        if current is None:
            continue
        dense = _DENSE.search(line)
        if dense:
            current.dense_seconds = float(dense.group(2))
        host = _HOST.search(line)
        if host:
            current.host_mib = int(host.group(1))
            current.outcome = "host_allocation_failure"
        if _VRAM.search(line):
            current.outcome = "vram_failure"
        placed = _PLACED.search(line)
        if placed:
            current.placement_seconds = float(placed.group(6))
            if current.outcome == "unknown":
                current.outcome = "expert_placement_complete"
        warm = _WARM.search(line)
        if warm:
            current.warmup_seconds = float(warm.group(1))
            current.outcome = "warmup_complete"
        route = _ROUTE.search(line)
        if route:
            _apply_route(current, int(route.group(1)), int(route.group(2)))
    return attempts


def analyze_startup_logs(blobs: list[bytes]) -> dict:
    unique = _unique_blobs(blobs)
    attempts: list[Attempt] = []
    for index, blob in enumerate(unique):
        text = blob.decode("utf-8", errors="replace")
        attempts.extend(parse_startup_log(text, generation_prefix=f"upload{index}"))
    counts = {
        "engine_start": len(attempts),
        "host_allocation_failure": sum(1 for a in attempts if a.outcome == "host_allocation_failure"),
        "vram_failure": sum(1 for a in attempts if a.outcome == "vram_failure"),
        "expert_placement_complete": sum(1 for a in attempts if a.placement_seconds is not None),
        "warmup_complete": sum(1 for a in attempts if a.outcome == "warmup_complete"),
        "unknown": sum(1 for a in attempts if a.outcome == "unknown"),
    }
    generations = {a.generation for a in attempts if a.placement_seconds is not None}
    cross = None
    if len(generations) == 1:
        only = next(iter(generations))
        cross = sum(a.placement_seconds or 0.0 for a in attempts if a.generation == only)
    return {
        "uploads_counted": len(unique),
        "uploads_received": len(blobs),
        "counts": counts,
        "attempts": [a.as_dict() for a in attempts],
        "cross_generation_placement_seconds": cross if len(generations) <= 1 else None,
    }
