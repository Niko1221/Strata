"""Compare an experimental Ascend C S2 decoder with Strata's existing CPU reference.

Synthetic Q2_0 blocks exercise the real byte format; no model correctness or
performance claim follows from this probe. Never substitute this for engine tests.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401: registers the NPU backend


def measure(launch, cpu, stream):
    """One 640 x 2560 S2 matrix; host wall times, not isolated device-kernel time."""
    n = 25600
    raw = np.zeros((n, 18), dtype=np.uint8)
    raw[:, 1] = 0x30  # FP16 scale 0.125, code zero -> -0.125
    host_codes = torch.from_numpy(raw[:, 2:].copy())
    host_scales = torch.full((n,), 0.125, dtype=torch.float32)
    codes = torch.empty_like(host_codes, device="npu:0")
    scales = torch.empty_like(host_scales, device="npu:0")
    out = torch.empty(n * 64, dtype=torch.float32, device="npu:0")
    expected = np.empty(n * 64, dtype=np.float32)
    torch.npu.synchronize()
    result = {"blocks": n, "matrix_shape": [640, 2560], "input_bytes": n * 20,
              "output_bytes": n * 64 * 4, "warmups": 2, "repeats": 3, "calls_per_repeat": 5,
              "pageable_host_copy_ms": [], "native_launch_and_sync_ms": [], "cpu_reference_ms": [],
              "rounds": []}
    for repeat in range(-2, 3):
        t = time.perf_counter()
        codes.copy_(host_codes)
        scales.copy_(host_scales)
        torch.npu.synchronize()
        copy_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        for _ in range(5):
            launch(codes.data_ptr(), scales.data_ptr(), out.data_ptr(), n, stream)
        torch.npu.synchronize()
        native_ms = (time.perf_counter() - t) * 1000 / 5
        t = time.perf_counter()
        for _ in range(5):
            cpu(raw.ctypes.data, expected.ctypes.data, n)
        cpu_ms = (time.perf_counter() - t) * 1000 / 5
        result["rounds"].append({"warmup": repeat < 0, "index": repeat,
                                 "pageable_host_copy_ms": copy_ms,
                                 "native_launch_and_sync_ms": native_ms, "cpu_reference_ms": cpu_ms})
        if repeat >= 0:
            result["pageable_host_copy_ms"].append(copy_ms)
            result["native_launch_and_sync_ms"].append(native_ms)
            result["cpu_reference_ms"].append(cpu_ms)
    actual = out.cpu().numpy()
    result["bit_mismatches"] = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    assert result["bit_mismatches"] == 0, result
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", type=Path, default=Path(__file__).parent / "build")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--benchmark", action="store_true", help="Add a synchronized host-wall-time diagnostic")
    args = ap.parse_args()
    if args.report.exists():
        raise SystemExit("Choose a new report path; existing evidence is preserved.")
    native_path = next(args.build.rglob("libstrata_s2.so"))
    reference_path = next(args.build.rglob("libstrata_s2_reference.so"))
    native = ctypes.CDLL(str(native_path.resolve()))
    reference = ctypes.CDLL(str(reference_path.resolve()))
    launch = native.strata_s2_launch
    launch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint64, ctypes.c_void_p]
    launch.restype = None
    cpu = reference.strata_q2_reference
    cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    cpu.restype = None
    torch.npu.set_device(0)
    stream = torch.npu.current_stream().npu_stream
    rng = np.random.default_rng(12345)
    # Include half subnormals, signed zero and both signs of the largest finite
    # half. Widening these to FP32 does not introduce FP32 subnormals.
    scale_bits = np.array([0x3C00, 0x3800, 0x3400, 0xBC00, 0xB800,
                           0x0000, 0x8000, 0x0001, 0x8001, 0x03FF,
                           0x7BFF, 0xFBFF], dtype="<u2")
    report = {"kind": "strata-ascend-s2-correctness-prototype", "machine": platform.machine(),
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "device": torch.npu.get_device_name(0), "seed": 12345,
              "kernel_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": [], "scope": "synthetic blocks; no engine integration or model validation"}
    try:
        for n in [0, 1, 2, 3, 7, 8, 31, 32, 33, 256, 257, 4096]:
            raw = np.empty((n, 18), dtype=np.uint8)
            bits = np.resize(scale_bits, n)
            raw[:, :2] = bits.view(np.uint8).reshape(n, 2)
            raw[:, 2:] = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
            if n >= 256:
                # Every possible byte pattern appears in every byte position.
                raw[:256, 2:] = np.arange(256, dtype=np.uint8)[:, None]
            expected = np.empty(n * 64, dtype=np.float32)
            cpu(raw.ctypes.data, expected.ctypes.data, n)
            codes_cpu = raw[:, 2:].copy()
            scales_cpu = bits.view(np.float16).astype(np.float32)
            codes = torch.from_numpy(codes_cpu).to("npu:0")
            scales = torch.from_numpy(scales_cpu).to("npu:0")
            guard = 32  # 128-byte aligned output with a guard region on each side
            out = torch.full((n * 64 + 2 * guard,), -321.25, device="npu:0", dtype=torch.float32)
            # This ctypes call bypasses torch_npu's host task queue. Drain tensor
            # copies/initialization before submitting directly to its ACL stream.
            # Sharing the stream handle alone does not flush pending host tasks.
            torch.npu.synchronize()
            launch(codes.data_ptr(), scales.data_ptr(), out.data_ptr() + guard * 4, n, stream)
            torch.npu.synchronize()
            got = out.cpu().numpy()
            actual = got[guard:guard + n * 64]
            mismatch = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
            case = {"blocks": n, "elements": n * 64, "bit_mismatches": int(mismatch.size),
                    "guards_ok": bool(np.all(got[:guard] == -321.25) and np.all(got[-guard:] == -321.25)),
                    "codes_unchanged": bool(np.array_equal(codes.cpu().numpy(), codes_cpu)),
                    "scales_unchanged": bool(np.array_equal(scales.cpu().numpy().view(np.uint32), scales_cpu.view(np.uint32)))}
            if mismatch.size:
                i = int(mismatch[0])
                case["first_mismatch"] = {"index": i, "expected_bits": int(expected.view(np.uint32)[i]),
                                           "actual_bits": int(actual.view(np.uint32)[i])}
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
        report["passed"] = all(c["bit_mismatches"] == 0 and c["guards_ok"] and
                               c["codes_unchanged"] and c["scales_unchanged"] for c in report["cases"])
        if args.benchmark and report["passed"]:
            report["timing_diagnostic"] = measure(launch, cpu, stream)
            print(json.dumps(report["timing_diagnostic"]), flush=True)
    except Exception as exc:
        report["passed"] = False
        report["error"] = repr(exc)
        raise
    finally:
        with args.report.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    print("ASCEND_S2_PASS" if report["passed"] else "ASCEND_S2_FAIL", flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
