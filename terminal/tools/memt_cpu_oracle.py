"""tools/memt_cpu_oracle.py - the Python side of the MemT memory-block parity.

It rebuilds the deterministic fixture of engine/src/qwen35/memt_cpu_test.cpp in
NumPy and compares the C++ vectors with the same math written independently here.
No GPU, no model, no Triton.

    python tools/memt_cpu_oracle.py [--binary PATH] [--json]
    python tools/test_memt_cpu_math.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

H, D, N, R, T = 2, 4, 6, 4, 3
HD = H * D
ROWS = 3 * HD + 2 * HD + 2 * R
TOLERANCE = 2e-5

# filled by compute(): the query rows and the state after each token, for the
# frozen-snapshot test
LAST_TRACE = {}


def value(i: int) -> float:
    return ((i * 37 + 11) % 101 - 50) / 50.0


def fill(count: int, seed: int) -> np.ndarray:
    return np.array([value(seed + i) for i in range(count)], dtype=np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    return x if x > 20.0 else math.log1p(math.exp(x))


def logsigmoid(x):
    return -math.log1p(math.exp(-x)) if x >= 0.0 else x - math.log1p(math.exp(x))


def l2norm_in_place(x: np.ndarray) -> None:
    total = np.float32(0.0)
    for item in x:
        total = np.float32(total + np.float32(item * item))
    inv = np.float32(1.0 / math.sqrt(float(total) + 1e-6))
    for i in range(len(x)):
        x[i] = np.float32(x[i] * inv)


def memory_read(q: np.ndarray, snapshot: np.ndarray, heads: int = H,
                dim: int = D) -> np.ndarray:
    """q [H, D], snapshot [H, D, D] -> [H, D]."""
    scale = 1.0 / math.sqrt(dim)
    out = np.zeros((heads, dim), dtype=np.float32)
    for h in range(heads):
        for v in range(dim):
            acc = np.float32(0.0)
            for k in range(dim):
                acc = np.float32(acc + np.float32(q[h, k] * snapshot[h, k, v]))
            out[h, v] = np.float32(acc * np.float32(scale))
    return out


def gated_rmsnorm(x: np.ndarray, gate: np.ndarray, weight: np.ndarray,
                  eps: float, limit: float, heads: int = H,
                  dim: int = D) -> np.ndarray:
    out = np.zeros_like(x)
    for h in range(heads):
        total = np.float32(0.0)
        for i in range(dim):
            total = np.float32(total + np.float32(x[h, i] * x[h, i]))
        inv = np.float32(1.0 / math.sqrt(float(total) / dim + eps))
        for i in range(dim):
            g = np.float32(limit * math.tanh(silu(float(gate[h, i])) / limit))
            out[h, i] = np.float32(x[h, i] * inv * weight[i] * g)
    return out


def memory_update(state: np.ndarray, k: np.ndarray, v: np.ndarray,
                  log_decay: np.ndarray, beta: np.ndarray,
                  heads: int = H, dim: int = D) -> None:
    for h in range(heads):
        decay = np.exp(log_decay[h].astype(np.float64)).astype(np.float32)
        for i in range(dim):
            for j in range(dim):
                state[h, i, j] = np.float32(state[h, i, j] * decay[i])
        delta = np.zeros(dim, dtype=np.float32)
        for j in range(dim):
            kv = np.float32(0.0)
            for i in range(dim):
                kv = np.float32(kv + np.float32(state[h, i, j] * k[h, i]))
            delta[j] = np.float32(v[h, j] * beta[h, j] - kv)
        for i in range(dim):
            for j in range(dim):
                state[h, i, j] = np.float32(state[h, i, j] + np.float32(k[h, i] * delta[j]))


def compute() -> Dict[str, np.ndarray]:
    in_proj = fill(ROWS * N, 1).reshape(ROWS, N)
    hidden = fill(T * N, 101).reshape(T, N)
    f_down = fill(HD * R, 211).reshape(HD, R).astype(np.float64)
    g_down = fill(HD * R, 307).reshape(HD, R).astype(np.float64)
    decay_scale = fill(H, 401)
    decay_bias = fill(HD, 409)
    o_norm = fill(D, 503)
    state0 = fill(H * D * D, 601).reshape(H, D, D)
    state = state0.copy()

    proj = np.zeros((T, ROWS), dtype=np.float32)
    for t in range(T):
        for row in range(ROWS):
            acc = 0.0
            for j in range(N):
                acc += float(in_proj[row, j]) * float(hidden[t, j])
            proj[t, row] = np.float32(acc)

    read_all = np.zeros((T, H, D), dtype=np.float32)
    q_all = np.zeros((T, H, D), dtype=np.float32)
    states = []
    erase_all = np.zeros((T, H, D), dtype=np.float32)
    write_all = np.zeros((T, H, D), dtype=np.float32)
    decay_all = np.zeros((T, H, D), dtype=np.float32)
    og_all = np.zeros((T, H, D), dtype=np.float32)

    for t in range(T):
        p = proj[t]
        q = p[0:HD].reshape(H, D).copy()
        k = p[HD:2 * HD].reshape(H, D).copy()
        v = p[2 * HD:3 * HD].reshape(H, D).copy()
        erase_raw = p[3 * HD:4 * HD].reshape(H, D)
        write_raw = p[4 * HD:5 * HD].reshape(H, D)
        decay_latent = p[5 * HD:5 * HD + R]
        og_latent = p[5 * HD + R:5 * HD + 2 * R]
        for h in range(H):
            l2norm_in_place(q[h])
            l2norm_in_place(k[h])
        q_all[t] = q

        erase = np.zeros((H, D), dtype=np.float32)
        write = np.zeros((H, D), dtype=np.float32)
        log_decay = np.zeros((H, D), dtype=np.float32)
        out_gate = np.zeros((H, D), dtype=np.float32)
        for h in range(H):
            for d in range(D):
                hd = h * D + d
                decay_value = float(decay_bias[hd])
                for r in range(R):
                    decay_value += float(f_down[hd, r]) * float(decay_latent[r])
                erase[h, d] = np.float32(math.exp(0.5 * logsigmoid(float(erase_raw[h, d]))))
                write[h, d] = np.float32(sigmoid(float(write_raw[h, d])))
                log_decay[h, d] = np.float32(
                    -math.exp(float(decay_scale[h])) * softplus(decay_value)
                )
                og = 0.0
                for r in range(R):
                    og += float(g_down[hd, r]) * float(og_latent[r])
                out_gate[h, d] = np.float32(og)

        k_eff = k.copy()
        for h in range(H):
            for d in range(D):
                k_eff[h, d] = np.float32(k_eff[h, d] * erase[h, d])

        raw = memory_read(q, state0)
        read_all[t] = gated_rmsnorm(raw, out_gate, o_norm, 1e-6, 2.0)
        memory_update(state, k_eff, v, log_decay, write)
        states.append(state.copy())
        erase_all[t], write_all[t] = erase, write
        decay_all[t], og_all[t] = log_decay, out_gate

    global LAST_TRACE
    LAST_TRACE = {"Q": q_all, "STATE0": state0, "STATES": states,
                  "OUTGATE": og_all, "O_NORM": o_norm}
    return {
        "READ": read_all.reshape(-1),
        "STATE": state.reshape(-1),
        "ERASE": erase_all.reshape(-1),
        "WRITE": write_all.reshape(-1),
        "LOGDECAY": decay_all.reshape(-1),
        "OUTGATE": og_all.reshape(-1),
    }


def parse(stdout: str) -> Dict[str, np.ndarray]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    out = {}
    i = 0
    while i < len(lines):
        label, _, count = lines[i].partition(" ")
        if label == "GEOM":
            i += 1
            continue
        if not count.isdigit():
            raise ValueError("unexpected line: %r" % lines[i])
        values = [float(x) for x in lines[i + 1].split()]
        if len(values) != int(count):
            raise ValueError("%s: %d values, expected %s" % (label, len(values), count))
        out[label] = np.array(values, dtype=np.float32)
        i += 2
    return out


def locate_binary(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "engine" / "build" / "cpu-check" / "memt_cpu_test",
        root / "engine" / "build" / "cpu-check-native" / "memt_cpu_test",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def compare(expected: Dict[str, np.ndarray], got: Dict[str, np.ndarray]) -> List[str]:
    problems = []
    for name, want in expected.items():
        if name not in got:
            problems.append("%s: the binary did not print it" % name)
            continue
        if got[name].shape != want.shape:
            problems.append("%s: shape %r, expected %r" % (name, got[name].shape, want.shape))
            continue
        diff = np.max(np.abs(got[name].astype(np.float64) - want.astype(np.float64)))
        if not np.isfinite(diff):
            problems.append("%s: non-finite values" % name)
        elif diff > TOLERANCE:
            problems.append("%s: max |diff| %.3g > %.3g" % (name, diff, TOLERANCE))
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", help="path to memt_cpu_test")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    binary = locate_binary(args.binary)
    if binary is None:
        print("memt_cpu_test not built (run cmake --build --preset cpu-check)", file=sys.stderr)
        return 2
    result = subprocess.run([os.fspath(binary)], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return 1
    expected = compute()
    got = parse(result.stdout)
    problems = compare(expected, got)
    payload = {
        "ok": not problems,
        "binary": os.fspath(binary),
        "tolerance": TOLERANCE,
        "problems": problems,
        "max_diff": {
            name: float(np.max(np.abs(got[name].astype(np.float64) - expected[name].astype(np.float64))))
            for name in expected if name in got and got[name].shape == expected[name].shape
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("memt_cpu_oracle: %s" % ("PASS" if payload["ok"] else "FAIL"))
        for name, diff in sorted(payload["max_diff"].items()):
            print("  %-9s max|diff| %.3g" % (name, diff))
        for problem in problems:
            print("  ERROR: %s" % problem)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
