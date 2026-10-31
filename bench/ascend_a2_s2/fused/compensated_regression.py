"""Cancellation regressions for the optional compensated S2 dot product."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    assert not args.report.exists()
    torch.npu.set_device(0)
    lib = next(args.build.rglob("libstrata_s2_projection.so")).resolve()
    native = ctypes.CDLL(str(lib))
    launch = native.strata_s2_projection_launch
    launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint32] * 3 + [ctypes.c_void_p]
    launch.restype = ctypes.c_int
    cases = []
    for label in ("lost-sum-bit", "lost-product-bits"):
        # Eight identical rows, K=192, one token. Code 1 means weight zero.
        codes = np.ones((8, 3, 64), dtype=np.uint8)
        scales = np.ones((8, 3), dtype=np.float32)
        q = np.zeros(192, np.int8)
        acts = np.zeros(6, np.float32)
        if label == "lost-sum-bit":
            codes[:, 0, [0, 32]] = 2
            codes[:, 1, 0] = 0
            q[[0, 32, 64]] = 1
            acts[:3] = [1e8, 1, 1e8]
            expected = np.float32(1)
        else:
            codes[:, 0, :9] = 2
            codes[:, 1, 0] = 2
            scales[:, 0] = np.float32(1 + 2**-10)  # exactly an FP16 scale
            q[:8] = 127
            q[8] = 7  # exact integer dot = 1023
            q[64] = 1
            acts[0] = np.float32(1 + 2**-12)
            product = np.float64(scales[0, 0]) * 1023 * np.float64(acts[0])
            acts[2] = -np.float32(product)
            # Both products and their difference are exact in this FP64 oracle.
            expected = np.float32(product + np.float64(acts[2]))
            assert expected == np.float32(-2**-22)
        packed = np.bitwise_or.reduce(codes.reshape(8, 3, 16, 4).astype(np.uint32)
                                     << np.array([0, 2, 4, 6], np.uint32), axis=-1).astype(np.uint8)
        host = [packed, scales, q, acts]
        dev = [torch.from_numpy(v).to("npu:0") for v in host]
        out = torch.full((72,), -321.25, dtype=torch.float32, device="npu:0")
        torch.npu.synchronize()
        assert launch(*(v.data_ptr() for v in dev), out.data_ptr() + 128,
                      8, 192, 1, torch.npu.current_stream().npu_stream) == 0
        torch.npu.synchronize()
        got = out.cpu().numpy()
        result = {"label": label, "expected": float(expected), "output": got[32:40].tolist(),
                  "exact": bool(np.all(got[32:40] == expected)),
                  "guards": bool(np.all(got[:32] == -321.25) and np.all(got[40:] == -321.25)),
                  "inputs_unchanged": all(np.array_equal(v.cpu().numpy(), h) for v, h in zip(dev, host))}
        result["passed"] = result["exact"] and result["guards"] and result["inputs_unchanged"]
        cases.append(result)
    report = {"cases": cases, "passed": all(c["passed"] for c in cases),
              "binary_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "device": torch.npu.get_device_name(0)}
    with args.report.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
