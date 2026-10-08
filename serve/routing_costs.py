"""Measured request-boundary routing, with no background model calls.

Design references (no source copied): ATSInfer arXiv:2607.10183v2, sections
4.3-4.4; StarPU data-aware performance-model scheduling. This smaller empirical
model uses matched end-to-end samples, which already include exposed transfer,
SSD and synchronization costs. It is not ATSInfer's tensor-placement algorithm.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
from statistics import median


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def runtime_key(engine):
    """Invalidate calibration after engine, launch configuration or asset changes.

    Model assets are identified by path, size and timestamp, not rehashed on every
    startup. The binary is hashed. No paths are exposed in the resulting key.
    """
    exe, args, cwd, _log, env = engine.spawn
    assets = []
    for flag in ("--native", "--ple-gguf", "--expert-profile", "--pack", "--mtp", "--control-vector-scaled"):
        if flag not in args:
            continue
        value = args[args.index(flag) + 1]
        if flag == "--control-vector-scaled":
            value = value.rsplit(":", 1)[0]
        path = Path(value)
        if not path.is_absolute():
            path = Path(cwd or ".") / path
        paths = sorted(path.rglob("*")) if path.is_dir() else [path]
        for p in paths:
            if p.is_file():
                stat = p.stat()
                assets.append((str(p.resolve()), stat.st_size, stat.st_mtime_ns))
    clean = list(args)
    if "--pcie-frac" in clean:
        i = clean.index("--pcie-frac")
        del clean[i:i + 2]
    binary = hashlib.sha256(Path(exe).read_bytes()).hexdigest()
    settings = {k: v for k, v in (env or {}).items() if k.startswith(("STRATA_", "GGML_", "CUDA_"))}
    from serve.telemetry import _Nvml, _cpu_name
    visible = (env or {}).get("CUDA_VISIBLE_DEVICES", "0")
    if not visible.isdecimal():
        raise ValueError("routing calibration currently needs one numeric CUDA device")
    gpu = _Nvml(int(visible)).name()
    if not gpu:
        raise ValueError("routing calibration requires readable GPU identity")
    hardware = [gpu, _cpu_name(), os.cpu_count(), platform.platform()]
    raw = json.dumps([binary, clean, cwd, settings, assets, hardware], sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


class RoutingCosts:
    def __init__(self, profile=None):
        self.profile = profile
        self.reason = "uncalibrated"
        self.last = None
        if profile is None:
            return
        if (not isinstance(profile, dict) or profile.get("schema") != 1
                or not isinstance(profile.get("runtime_key"), str)
                or not isinstance(profile.get("records"), list)
                or not isinstance(profile.get("sampling"), dict)
                or not finite(profile.get("created_unix")) or profile["created_unix"] < 0
                or not finite(profile.get("max_prompt_read")) or profile["max_prompt_read"] < 0
                or not finite(profile.get("max_output_tokens")) or profile["max_output_tokens"] <= 0):
            raise ValueError("invalid routing calibration")
        seen = set()
        for row in profile["records"]:
            required = ("fraction", "prefill_ms_per_token", "decode_ms_per_token", "fixed_ms",
                        "external_cpu", "external_gpu", "resident_mib", "cache_mib", "context_min", "context_max")
            if (not isinstance(row, dict) or not all(finite(row.get(k)) and row[k] >= 0 for k in required)
                    or not 0 <= row["fraction"] <= 1 or row["decode_ms_per_token"] <= 0
                    or row["context_min"] > row["context_max"]
                    or row["external_cpu"] > 100 or row["external_gpu"] > 100
                    or not isinstance(row.get("pair"), str) or row.get("correct") is not True):
                raise ValueError("routing calibration must contain only validated finite measurements")
            identity = (row["pair"], row["fraction"])
            if identity in seen:
                raise ValueError("duplicate matched routing measurement")
            seen.add(identity)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))) if path else cls()

    def choose(self, *, key, baseline, current, snapshot, native, context, prompt_read, output_tokens,
               now, sampling=None, switch_ms=0, allow_more_gpu=True, allow_more_cpu=True):
        self.last = None
        self.reason = "uncalibrated"
        if self.profile is None:
            return baseline
        if key != self.profile["runtime_key"]:
            self.reason = "runtime_changed"
            return baseline
        created = self.profile['created_unix']
        if not finite(now) or not 0 <= now - created <= 86400:
            self.reason = "calibration_expired"
            return baseline
        work = snapshot.get("workload", {})
        stamp = snapshot.get("sampled_at")
        # GPU must be attributable to other work, or a recent idle observation.
        gpu = work.get("gpu_percent")
        cpu = work.get("cpu_percent")
        if (not finite(stamp) or not 0 <= now - stamp <= 5 or work.get("complete") is not True
                or not finite(cpu) or not 0 <= cpu <= 100 or not finite(gpu) or not 0 <= gpu <= 100
                or not isinstance(native, dict)
                or not all(finite(native.get(k)) and native[k] >= 0 for k in
                           ("sampled_at", "resident_mib", "cache_mib"))
                or not 0 <= now - native["sampled_at"] <= 5):
            self.reason = "telemetry_unavailable"
            return baseline
        if not all(finite(v) and v >= 0 for v in (context, prompt_read, output_tokens, switch_ms)):
            self.reason = "request_unavailable"
            return baseline
        if (self.profile['sampling'] != {k: v for k, v in (sampling or {}).items() if k != 'pcie_frac'}
                or prompt_read > self.profile['max_prompt_read']
                or output_tokens > self.profile['max_output_tokens']):
            self.reason = 'outside_request_calibration'
            return baseline
        candidates = {}
        for row in self.profile["records"]:
            if (abs(cpu - row["external_cpu"]) > 12 or abs(gpu - row["external_gpu"]) > 15
                    or abs(native["resident_mib"] - row["resident_mib"]) > 16
                    or abs(native["cache_mib"] - row["cache_mib"]) > 32
                    or not row["context_min"] <= context <= row["context_max"]):
                continue
            f = row["fraction"]
            if (f > baseline and not allow_more_gpu) or (f < baseline and not allow_more_cpu):
                continue
            cost = row["fixed_ms"] + prompt_read * row["prefill_ms_per_token"] + output_tokens * row["decode_ms_per_token"]
            candidates.setdefault(f, {})[row["pair"]] = cost
        base = candidates.get(baseline, {})
        if len(base) < 3:
            self.reason = "outside_calibration"
            return baseline
        # Matched pairs prevent extra easy samples from making a policy look fast.
        # Use a conservative empirical range, not a claimed statistical CI.
        best, best_cost, best_gain = baseline, median(base.values()), 0
        for f, values in candidates.items():
            common = sorted(set(base) & set(values))
            if len(common) < 3 or f == baseline:
                continue
            savings = [base[k] - values[k] for k in common]
            margin = max(10.0, .05 * median(base[k] for k in common))
            penalty = switch_ms if f != current else 0
            if min(savings) <= margin + penalty:
                continue
            # Candidates can cover different matched subsets. Rank their paired
            # advantage, not raw duration on a potentially easier subset.
            gain = min(savings) - penalty
            if gain > best_gain:
                best, best_gain = f, gain
                best_cost = median(base.values()) - median(savings) + penalty
        self.reason = "measured_benefit" if best != baseline else "no_clear_benefit"
        self.last = {"fraction": best, "predicted_ms": round(best_cost, 2), "pairs": len(base)}
        return best

    def status(self):
        return {"calibrated": self.profile is not None, "reason": self.reason, "decision": self.last}
