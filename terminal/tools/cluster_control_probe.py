#!/usr/bin/env python3
"""Measure the authenticated control-plane round trip on ONE machine (loopback).

This is NOT a two-machine LAN measurement and NOT an inference benchmark. It
only quantifies the overhead of the registry/agent heartbeat design so the
protocol can be sized before anyone claims cluster speed.

Usage: python tools/cluster_control_probe.py [requests]
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.request

from strata_control.cluster import Agent


def ping(port: int, token: str, timeout: float = 2.0) -> float:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/cluster/ping",
                                     headers={"X-Cluster-Token": token})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()
    return (time.perf_counter() - start) * 1000


def main(count: int) -> int:
    token = "probe-token"
    agent = Agent(token)
    port = agent.start()
    try:
        samples = [ping(port, token) for _ in range(count)]
    finally:
        agent.stop()
    samples.sort()
    result = {
        "scope": "single machine, loopback TCP, control plane only",
        "not_a_lan_measurement": True,
        "not_an_inference_benchmark": True,
        "requests": count,
        "min_ms": round(samples[0], 3),
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 3),
        "max_ms": round(samples[-1], 3),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 100))
