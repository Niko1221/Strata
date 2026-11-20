"""Parse Q36_HOTSPOT stderr lines. Not a speed result; profile-capture helper only."""

from __future__ import annotations

import re
from dataclasses import dataclass


_HOTSPOT = re.compile(
    r"^Q36_HOTSPOT schema=(?P<schema>\d+) phase=(?P<phase>\S+) "
    r"layers_gdn=(?P<layers_gdn>-?\d+) layers_attn=(?P<layers_attn>-?\d+) "
    r"gdn_host_ms=(?P<gdn_host_ms>-?[0-9.]+) attn_host_ms=(?P<attn_host_ms>-?[0-9.]+) "
    r"moe_gpu_host_ms=(?P<moe_gpu_host_ms>-?[0-9.]+) moe_cpu_ms=(?P<moe_cpu_ms>-?[0-9.]+) "
    r"stream_copy_wait_ms=(?P<stream_copy_wait_ms>-?[0-9.]+) stream_mmq_ms=(?P<stream_mmq_ms>-?[0-9.]+) "
    r"stream_h2d_bytes=(?P<stream_h2d_bytes>-?\d+) stream_d2d_bytes=(?P<stream_d2d_bytes>-?\d+) "
    r"hy_cpu=(?P<hy_cpu>-?\d+) hy_gpu=(?P<hy_gpu>-?\d+) hy_swaps=(?P<hy_swaps>-?\d+) "
    r"rebalance_h2d_bytes=(?P<rebalance_h2d_bytes>-?\d+) "
    r"hardware_counters=(?P<hardware_counters>\S+) note=(?P<note>\S+)\s*$"
)


@dataclass(frozen=True)
class HotspotLine:
    schema: int
    phase: str
    layers_gdn: int
    layers_attn: int
    gdn_host_ms: float
    attn_host_ms: float
    moe_gpu_host_ms: float
    moe_cpu_ms: float
    stream_copy_wait_ms: float
    stream_mmq_ms: float
    stream_h2d_bytes: int
    stream_d2d_bytes: int
    hy_cpu: int
    hy_gpu: int
    hy_swaps: int
    rebalance_h2d_bytes: int
    hardware_counters: str
    note: str

    def dominant_host_bucket(self) -> str:
        """Largest of the four host-timed buckets; ties prefer earlier plan candidates."""
        buckets = (
            ("gdn", self.gdn_host_ms),
            ("attn", self.attn_host_ms),
            ("moe_gpu", self.moe_gpu_host_ms + self.stream_mmq_ms),
            ("stream_copy", self.stream_copy_wait_ms),
            ("moe_cpu", self.moe_cpu_ms),
        )
        name, _ = max(buckets, key=lambda kv: kv[1])
        return name


def parse_hotspot_line(line: str) -> HotspotLine | None:
    m = _HOTSPOT.match(line.strip())
    if not m:
        return None
    g = m.groupdict()
    return HotspotLine(
        schema=int(g["schema"]),
        phase=g["phase"],
        layers_gdn=int(g["layers_gdn"]),
        layers_attn=int(g["layers_attn"]),
        gdn_host_ms=float(g["gdn_host_ms"]),
        attn_host_ms=float(g["attn_host_ms"]),
        moe_gpu_host_ms=float(g["moe_gpu_host_ms"]),
        moe_cpu_ms=float(g["moe_cpu_ms"]),
        stream_copy_wait_ms=float(g["stream_copy_wait_ms"]),
        stream_mmq_ms=float(g["stream_mmq_ms"]),
        stream_h2d_bytes=int(g["stream_h2d_bytes"]),
        stream_d2d_bytes=int(g["stream_d2d_bytes"]),
        hy_cpu=int(g["hy_cpu"]),
        hy_gpu=int(g["hy_gpu"]),
        hy_swaps=int(g["hy_swaps"]),
        rebalance_h2d_bytes=int(g["rebalance_h2d_bytes"]),
        hardware_counters=g["hardware_counters"],
        note=g["note"],
    )


def extract_hotspot_lines(text: str) -> list[HotspotLine]:
    out: list[HotspotLine] = []
    for line in text.splitlines():
        parsed = parse_hotspot_line(line)
        if parsed is not None:
            out.append(parsed)
    return out
