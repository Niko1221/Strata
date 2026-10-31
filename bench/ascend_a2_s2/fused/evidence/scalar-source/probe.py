"""A2 projection experiment, not a Strata expert or model benchmark."""
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
import torch_npu


def quantize_legacy(x):
    """Portable transcription of expert.cpp's scalar legacy activation quantizer.

    This fixture is not a build/execution of the x86 AVX implementation.
    """
    chunks = x.reshape(x.shape[0], -1, 32)
    scales = np.max(np.abs(chunks), axis=2) / np.float32(127)
    inv = np.divide(np.float32(1), scales, out=np.zeros_like(scales), where=scales > 0)
    scaled = chunks * inv[..., None]
    rounded = scaled + np.where(scaled >= 0, np.float32(.5), np.float32(-.5))
    q = np.clip(rounded.astype(np.int32), -127, 127).astype(np.int8)
    return q.reshape(x.shape), scales


def metrics(actual, expected, magnitude, operations):
    # gamma_n forward bound for normal FP32 arithmetic. Fixtures do not cause
    # FP32 underflow/overflow. Cancellation is judged using sum(abs(products)),
    # rather than a fixed relative tolerance on the potentially tiny result.
    u = np.finfo(np.float32).eps / 2
    bound = operations * u / (1 - operations * u) * magnitude
    error = np.abs(actual.astype(np.float64) - expected)
    return {"max_abs_error": float(error.max(initial=0)),
            "rms_error": float(np.sqrt(np.mean(error**2))) if error.size else 0.,
            "max_error_over_bound": float(np.max(np.divide(error, bound,
                out=np.where(error == 0, 0., np.inf), where=bound > 0), initial=0)),
            "bound_operations": operations,
            "within_bound": bool(np.all(np.isfinite(actual)) and np.all(error <= bound))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--decode-build", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()
    if args.report.exists():
        raise SystemExit("Use a new report name; previous evidence is preserved.")
    native_path = next(args.build.rglob("libstrata_s2_projection.so")).resolve()
    decoder_path = next(args.decode_build.rglob("libstrata_s2.so")).resolve()
    reference_path = next(args.decode_build.rglob("libstrata_s2_reference.so")).resolve()
    native, decoder, reference = [ctypes.CDLL(str(p)) for p in (native_path, decoder_path, reference_path)]
    launch = native.strata_s2_projection_launch
    launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint32] * 3 + [ctypes.c_void_p]
    launch.restype = ctypes.c_int
    decode = decoder.strata_s2_launch
    decode.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint64, ctypes.c_void_p]
    decode.restype = None
    cpu_decode = reference.strata_q2_reference
    cpu_decode.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint64]
    cpu_decode.restype = None
    torch.npu.set_device(0)
    torch_npu.npu.matmul.allow_hf32 = False
    stream = torch.npu.current_stream().npu_stream
    rng = np.random.default_rng(20261021)
    sources = [Path(__file__), Path(__file__).with_name("projection.cpp"),
               Path(__file__).with_name("CMakeLists.txt")]
    report = {"kind": "ascend-a2-s2-quantized-activation-projection", "seed": 20261021,
              "machine": platform.machine(), "device": torch.npu.get_device_name(0),
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "matmul_allow_hf32": torch_npu.npu.matmul.allow_hf32,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "binary_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (native_path, decoder_path, reference_path)},
              "scope": "synthetic single projection; legacy activation fixture; block-order reduction differs from default AVX CPU",
              "cases": [], "timings": []}
    # Half-away ties, independently known scale=1; zero-chunk behavior.
    tie = np.zeros((1, 64), dtype=np.float32)
    tie[0, :7] = [127, .5, -.5, 1.5, -1.5, 126.5, -126.5]
    tq, ts = quantize_legacy(tie)
    assert tq[0, :7].tolist() == [127, 1, -1, 2, -2, 127, -127]
    assert ts.tolist() == [[1., 0.]]
    report["activation_fixture_ties_passed"] = True
    shapes = [(0, 64, 1), (1, 64, 1), (7, 64, 2), (8, 640, 4),
              (9, 2560, 8), (31, 640, 1), (32, 2560, 2), (33, 2560, 4),
              (257, 640, 8)]
    shapes += [(m, k, t) for m, k in ((640, 2560), (2560, 640)) for t in (1, 2, 4, 8)]
    try:
        # Invalid dimensions are rejected on host before submitting a kernel.
        report["invalid_shapes_rejected"] = all(launch(None, None, None, None, None, 8, k, t, stream) == 1
                                                 for k, t in ((0, 1), (63, 1), (64, 0), (64, 9)))
        for index, (m, k, t) in enumerate(shapes):
            blocks, padded = k // 64, (m + 7) // 8 * 8
            raw = rng.integers(0, 256, size=(m * blocks, 18), dtype=np.uint8)
            bits = np.resize(np.array([0x3000, 0xBC00, 0, 0x8000, 1, 0x8001,
                                       0x03FF, 0x7BFF, 0xFBFF, 0x3400], dtype="<u2"), m * blocks)
            # Benchmarks use ordinary finite scales, with a separate stress set above.
            if index >= 9:
                bits = rng.choice(np.array([0x2800, 0x3000, 0xB000, 0x3400], dtype="<u2"), m * blocks)
            raw[:, :2] = bits.view(np.uint8).reshape(-1, 2)
            if m * blocks >= 256:
                raw[:256, 2:] = np.arange(256, dtype=np.uint8)[:, None]
            dense = np.empty(m * k, dtype=np.float32)
            cpu_decode(raw.ctypes.data, dense.ctypes.data, m * blocks)
            dense = dense.reshape(m, k)
            x = rng.normal(size=(t, k)).astype(np.float32)
            # Adjacent activation chunks intentionally have different scales.
            x.reshape(t, -1, 32)[:, ::2] *= np.float32(.03125)
            x[:, 32:64] = 0
            q, a = quantize_legacy(x)
            exact_x = q.astype(np.float64) * np.repeat(a.astype(np.float64), 32, axis=1)
            expected = exact_x @ dense.astype(np.float64).T
            magnitude = np.abs(exact_x) @ np.abs(dense.astype(np.float64)).T
            host = [raw[:, 2:].copy(), bits.view(np.float16).astype(np.float32), q, a]
            tensors = [torch.from_numpy(h).to("npu:0") for h in host]
            out = torch.full((t * padded + 64,), -321.25, dtype=torch.float32, device="npu:0")
            ptrs = [z.data_ptr() for z in tensors]
            def fused():
                assert launch(*ptrs, out.data_ptr() + 128, m, k, t, stream) == 0
            torch.npu.synchronize()  # Drain torch_npu host queue before raw ACL launch.
            fused()
            torch.npu.synchronize()
            got = out.cpu().numpy()
            output = got[32:32+t*padded].reshape(t, padded)
            case = {"shape": [m, k, t], **metrics(output[:, :m], expected, magnitude, 4 * blocks + 4),
                    "guards_ok": bool(np.all(got[:32] == -321.25) and np.all(got[-32:] == -321.25)),
                    "padding_zero": bool(np.all(output[:, m:] == 0)),
                    "inputs_unchanged": all(np.array_equal(z.cpu().numpy().view(np.uint8), h.view(np.uint8))
                                            for z, h in zip(tensors, host))}
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
            if args.benchmark and index >= 9:
                # Same-day split baseline: expand the SAME codes/scales using the
                # previously validated decoder, then FP32 matmul (HF32 disabled).
                expanded = torch.empty((m, k), device="npu:0", dtype=torch.float32)
                xhat = torch.from_numpy(exact_x.astype(np.float32)).to("npu:0")
                baseline_out = torch.empty((t, m), device="npu:0", dtype=torch.float32)
                torch.npu.synchronize()
                def split():
                    decode(ptrs[0], ptrs[1], expanded.data_ptr(), m * blocks, stream)
                    # Order direct ACL submission before the framework's host queue.
                    torch.npu.synchronize()
                    torch.mm(xhat, expanded.T, out=baseline_out)
                timing = {"shape": [m, k, t], "extra_dense_weight_bytes": m * k * 4,
                          "warmups": 2, "repeats": 5, "calls_per_repeat": 3, "rounds": [],
                          "timing": "host wall time, allocated buffers, resident inputs, final sync; split includes interop sync"}
                for repeat in range(-2, 5):
                    round_result = {"index": repeat, "warmup": repeat < 0}
                    # Alternate order to reduce systematic first/second bias.
                    for name, fn in (("fused_ms", fused), ("split_ms", split)) if repeat % 2 else (("split_ms", split), ("fused_ms", fused)):
                        torch.npu.synchronize()
                        start = time.perf_counter()
                        for _ in range(3):
                            fn()
                        torch.npu.synchronize()
                        round_result[name] = (time.perf_counter() - start) * 1000 / 3
                    timing["rounds"].append(round_result)
                timing["split_correctness"] = metrics(baseline_out.cpu().numpy(), expected, magnitude, k + 4)
                for name in ("fused_ms", "split_ms"):
                    timing["median_" + name] = float(np.median([r[name] for r in timing["rounds"] if not r["warmup"]]))
                report["timings"].append(timing)
                print(json.dumps(timing), flush=True)
        report["passed"] = report["invalid_shapes_rejected"] and all(
            c["within_bound"] and c["guards_ok"] and c["padding_zero"] and c["inputs_unchanged"] for c in report["cases"])
        report["passed"] &= all(t["split_correctness"]["within_bound"] for t in report["timings"])
    except Exception as exc:
        report["passed"] = False
        report["error"] = repr(exc)
        raise
    finally:
        with args.report.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    print("ASCEND_PROJECTION_PASS" if report["passed"] else "ASCEND_PROJECTION_FAIL", flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
