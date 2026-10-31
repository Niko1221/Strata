"""Real Q2_0 layer-zero MoE on A3; synthetic layer inputs, no full-model claim."""
import argparse
import ctypes
import functools
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu

END_MAX_NORMALIZED = 1e-3
END_RMS_NORMALIZED = 5e-4


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def quant_cpu(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    v = x.reshape(len(x), -1, 32)
    s = np.max(np.abs(v), axis=-1) / np.float32(127)
    inv = np.divide(np.float32(1), s, out=np.zeros_like(s), where=s > 0)
    z = v * inv[..., None]
    q = np.trunc(z + np.where(z >= 0, np.float32(.5), np.float32(-.5)))
    return np.clip(q, -127, 127).astype(np.int8).reshape(x.shape), s


def quant_npu(x):
    v = x.reshape(len(x), -1, 32)
    s = v.abs().amax(dim=-1) / 127.0
    inv = torch.where(s > 0, torch.reciprocal(s), torch.zeros_like(s))
    z = v * inv.unsqueeze(-1)
    q = torch.trunc(z + torch.where(z >= 0, .5, -.5)).clamp(-127, 127).to(torch.int8)
    return nd(q.reshape(x.shape)), nd(s)


def nd(x):
    return torch_npu.npu_format_cast(x.contiguous(), 2)


def cpu(a):
    return a.detach().cpu().numpy().copy()


def xhat(q, s):
    return q.astype(np.float64) * np.repeat(s.astype(np.float64), 32, axis=-1)


def sigmoid(x):
    # Stable independent FP64 oracle, also valid for stress inputs.
    y = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + y), y / (1 + y))


def agreement(actual, expected):
    error = np.abs(actual.astype(np.float64) - expected)
    scale = max(float(np.max(np.abs(expected), initial=0)), 1e-12)
    rms_scale = max(float(np.sqrt(np.mean(expected ** 2))), 1e-12)
    return {"max_abs_error": float(error.max(initial=0)),
            "max_normalized_error": float(error.max(initial=0) / scale),
            "normalized_rms_error": float(np.sqrt(np.mean(error ** 2)) / rms_scale),
            "finite": bool(np.all(np.isfinite(actual)))}


def stage_bound(actual, expected, magnitude, k):
    u = np.finfo(np.float32).eps / 2
    n = 4 * (k // 64) + 4
    bound = n * u / (1 - n * u) * magnitude
    error = np.abs(actual.astype(np.float64) - expected)
    ratio = np.divide(error, bound, out=np.where(error == 0, 0., np.inf), where=bound > 0)
    return {"max_abs_error": float(error.max(initial=0)),
            "max_error_over_bound": float(ratio.max(initial=0)),
            "passed": bool(np.all(np.isfinite(actual)) and np.all(error <= bound))}


def route_cpu(x, w):
    logits = x.astype(np.float64) @ w.astype(np.float64).T
    p = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    ids = np.argsort(-p, axis=-1, kind="stable")[:, :10]
    selected = np.take_along_axis(p, ids, axis=-1)
    return logits, ids, selected / selected.sum(axis=-1, keepdims=True)


def route_npu(x, w):
    logits = F.linear(x, w)
    probs = torch.softmax(logits, dim=-1)
    remaining = probs.clone()
    labels = torch.arange(512, device=x.device, dtype=torch.int32).unsqueeze(0)
    indices, scores = [], []
    for _ in range(10):
        best = remaining.amax(dim=-1, keepdim=True)
        # Strata's router breaks equal-score ties by the lower expert ID.
        pick = torch.where(remaining == best, labels, 512).amin(dim=-1)
        indices.append(pick)
        scores.append(best.squeeze(-1))
        remaining = torch.where(labels == pick.unsqueeze(-1), -float("inf"), remaining)
    scores = torch.stack(scores, dim=-1)
    return logits, torch.stack(indices, dim=-1).long(), scores / scores.sum(dim=-1, keepdim=True)


class Layer:
    def __init__(self, args):
        self.manifest = json.loads((args.weights / "WEIGHTS.json").read_text())
        self.entries = {x["name"]: x for x in self.manifest["tensors"]}
        self.raw = {}
        self.dense = {}
        self.packed = {}
        self.device = "npu:0"
        decoder = ctypes.CDLL(str(args.reference.resolve()))
        self.decode = decoder.strata_moe_decode
        self.decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
        self.decode.restype = ctypes.c_int
        native = next(args.build.rglob("libstrata_s2_projection.so")).resolve()
        self.native = ctypes.CDLL(str(native))
        self.launch = self.native.strata_s2_projection_launch
        self.launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint32] * 3 + [ctypes.c_void_p]
        self.launch.restype = ctypes.c_int
        self.binary_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (native, args.reference)}
        self.stream = torch.npu.current_stream().npu_stream
        for name, entry in self.entries.items():
            raw = np.fromfile(args.weights / entry["file"], dtype=np.uint8)
            assert digest(raw) == entry["sha256"]
            self.raw[name] = raw
            shape = tuple(reversed(entry["shape_gguf"]))
            if entry["type_id"] == 42:
                assert shape in ((512, 640, 2560), (512, 2560, 640))
                blocks = raw.reshape(512, -1, 18)
                codes = blocks[:, :, 2:].copy()
                scales = blocks[:, :, :2].copy().view("<f2").astype(np.float32).reshape(512, -1)
                self.packed[name] = (nd(torch.from_numpy(codes).to(self.device)),
                                     nd(torch.from_numpy(scales).to(self.device)))
            else:
                value = np.empty(shape, dtype=np.float32)
                assert self.decode(raw.ctypes.data, value.ctypes.data, value.size, entry["type_id"]) == 0
                assert np.all(np.isfinite(value))
                self.dense[name] = value
        self.dev_dense = {n: nd(torch.from_numpy(v).to(self.device)) for n, v in self.dense.items()}
        torch.npu.synchronize()
        self.guards_passed = True
        self.quant_checks = []
        self.stage_checks = []

    @functools.lru_cache(maxsize=16)
    def expert(self, e, proj):
        name = "blk.0.ffn_" + proj + "_exps.weight"
        shape = tuple(reversed(self.entries[name]["shape_gguf"]))[1:]
        stride = self.raw[name].size // 512
        raw = self.raw[name][e * stride:(e + 1) * stride]
        value = np.empty(shape, dtype=np.float32)
        assert self.decode(raw.ctypes.data, value.ctypes.data, value.size, 42) == 0
        return value

    def project(self, e, proj, q, s):
        name = "blk.0.ffn_" + proj + "_exps.weight"
        shape = tuple(reversed(self.entries[name]["shape_gguf"]))
        _, m, k = shape
        codes, scales = self.packed[name]
        q, s = nd(q), nd(s)
        result = torch.full((len(q) * m + 64,), -321.25, device=self.device, dtype=torch.float32)
        # Required for the framework's asynchronous host queue/raw ACL boundary.
        torch.npu.synchronize()
        rc = self.launch(codes[e].data_ptr(), scales[e].data_ptr(), q.data_ptr(), s.data_ptr(),
                         result.data_ptr() + 128, m, k, len(q), self.stream)
        assert rc == 0
        torch.npu.synchronize()
        assert bool(torch.all(result[:32] == -321.25))
        assert bool(torch.all(result[-32:] == -321.25))
        return result[32:-32].reshape(len(q), m)

    def check_quant(self, x, q, s, label):
        host = cpu(x)
        cq, cs = quant_cpu(host)
        nq, ns = cpu(q), cpu(s)
        result = {"label": label, "codes_equal": bool(np.array_equal(cq, nq)),
                  "scales_equal": bool(np.array_equal(cs, ns)),
                  "max_scale_abs_error": float(np.max(np.abs(cs - ns), initial=0))}
        # CANN's FP32 scalar division by 127 differed by one ULP in the
        # preflight. Keep that difference visible; the full-layer budgets are
        # unchanged. The native projection is checked with the actual scales.
        ulps = np.abs(cs.view(np.int32).astype(np.int64) - ns.view(np.int32).astype(np.int64))
        result["max_scale_ulps"] = int(ulps.max(initial=0))
        result["passed"] = result["codes_equal"] and result["max_scale_ulps"] <= 1
        self.quant_checks.append(result)

    def check_project(self, e, proj, q, s, y):
        w = self.expert(e, proj).astype(np.float64)
        exact = xhat(cpu(q), cpu(s))
        check = {"expert": int(e), "projection": proj,
                 **stage_bound(cpu(y), exact @ w.T, np.abs(exact) @ np.abs(w).T, w.shape[1])}
        self.stage_checks.append(check)

    def expert_npu(self, e, x, diagnostics=True):
        q, s = quant_npu(x)
        if diagnostics:
            self.check_quant(x, q, s, "expert-input")
        gate = self.project(e, "gate", q, s)
        up = self.project(e, "up", q, s)
        if diagnostics:
            self.check_project(e, "gate", q, s, gate)
            self.check_project(e, "up", q, s, up)
        mid = F.silu(gate) * up
        qmid, smid = quant_npu(mid)
        if diagnostics:
            self.check_quant(mid, qmid, smid, "expert-intermediate")
        down = self.project(e, "down", qmid, smid)
        if diagnostics:
            self.check_project(e, "down", qmid, smid, down)
        return down

    def expert_cpu(self, e, x):
        q, s = quant_cpu(x)
        exact = xhat(q, s)
        gate = exact @ self.expert(e, "gate").astype(np.float64).T
        up = exact @ self.expert(e, "up").astype(np.float64).T
        mid = (gate * sigmoid(gate) * up).astype(np.float32)
        q, s = quant_cpu(mid)
        return xhat(q, s) @ self.expert(e, "down").astype(np.float64).T

    def forward_npu(self, x):
        def w(s): return self.dev_dense["blk.0.ffn_" + s + ".weight"]
        logits, ids, weights = route_npu(x, w("gate_inp"))
        shared = F.linear(F.silu(F.linear(x, w("gate_shexp"))) * F.linear(x, w("up_shexp")), w("down_shexp"))
        shared = shared * torch.sigmoid(F.linear(x, w("gate_inp_shexp").reshape(1, -1)))
        host_ids = cpu(ids)
        routed = torch.zeros_like(x)
        for e in np.unique(host_ids):
            ti, rank = np.where(host_ids == e)
            ix = torch.from_numpy(ti).to(self.device)
            rk = torch.from_numpy(rank).to(self.device)
            value = self.expert_npu(int(e), x[ix]) * weights[ix, rk, None]
            routed.index_add_(0, ix, value)
        return logits, ids, weights, shared, routed, routed + shared

    def forward_cpu(self, x):
        def w(s): return self.dense["blk.0.ffn_" + s + ".weight"].astype(np.float64)
        logits, ids, weights = route_cpu(x, w("gate_inp"))
        xd = x.astype(np.float64)
        gate = xd @ w("gate_shexp").T
        shared = (gate * sigmoid(gate) * (xd @ w("up_shexp").T)) @ w("down_shexp").T
        shared *= sigmoid(xd @ w("gate_inp_shexp").reshape(-1, 1))
        routed = np.zeros(x.shape, dtype=np.float64)
        for e in np.unique(ids):
            ti, rank = np.where(ids == e)
            routed[ti] += self.expert_cpu(int(e), x[ti]) * weights[ti, rank, None]
        return logits, ids, weights, shared, routed, routed + shared

    def weights_unchanged(self):
        for name, (c, s) in self.packed.items():
            raw = self.raw[name].reshape(512, -1, 18)
            assert np.array_equal(cpu(c), raw[:, :, 2:])
            scales = raw[:, :, :2].copy().view("<f2").astype(np.float32).reshape(512, -1)
            assert np.array_equal(cpu(s), scales)
        for name, value in self.dev_dense.items():
            assert np.array_equal(cpu(value), self.dense[name])
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    assert not args.report.exists(), "Preserve prior evidence; choose a new report path"
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch_npu.npu.matmul.allow_hf32 = False
    rng = np.random.default_rng(20261021)
    report = {"kind": "real-q2_0-layer0-complete-moe", "seed": 20261021,
              "input_scope": "synthetic FP32 hidden states, not activations captured from a prompt",
              "scope": "router + 10 routed experts + gated shared expert; host expert dispatch; no attention/residual/model integration",
              "device": torch.npu.get_device_name(0), "logical_device": 0,
              "torch": torch.__version__, "torch_npu": torch_npu.__version__, "numpy": np.__version__,
              "hf32": False, "budgets": {"end_max_normalized": END_MAX_NORMALIZED,
              "end_rms_normalized": END_RMS_NORMALIZED, "projection": "gamma(4*K/64+4)*sum(abs(products))"},
              "cases": [], "expert_sweep": [], "passed": False}
    layer = None
    try:
        layer = Layer(args)
        report["weights"] = layer.manifest
        report["binary_sha256"] = layer.binary_hashes
        report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.glob("*.py")}
        fixtures = [("zero-ties", np.zeros((1, 2560), np.float32))]
        fixtures += [(f"normal-t{t}", rng.normal(size=(t, 2560)).astype(np.float32)) for t in (1, 2, 4, 8)]
        fixtures += [("small-t3", rng.normal(size=(3, 2560)).astype(np.float32) * np.float32(.03125)),
                     ("large-t5", rng.normal(size=(5, 2560)).astype(np.float32) * np.float32(4))]
        for label, x in fixtures:
            begin = time.perf_counter()
            xd = nd(torch.from_numpy(x).to("npu:0"))
            n = tuple(cpu(v) for v in layer.forward_npu(xd))
            reference = layer.forward_cpu(x)
            case = {"label": label, "shape": list(x.shape), "input_sha256": digest(x),
                    "output_sha256": digest(n[-1]), "selected_experts": n[1].tolist(),
                    "router_ids_equal": bool(np.array_equal(n[1], reference[1])),
                    "input_unchanged": bool(np.array_equal(cpu(xd), x)),
                    "router_logits": agreement(n[0], reference[0]),
                    "routing_weights": agreement(n[2], reference[2]),
                    "shared": agreement(n[3], reference[3]),
                    "routed": agreement(n[4], reference[4]), "output": agreement(n[5], reference[5]),
                    "validation_wall_seconds": time.perf_counter() - begin}
            case["passed"] = case["router_ids_equal"] and case["input_unchanged"] and all(
                a["finite"] and a["max_normalized_error"] <= END_MAX_NORMALIZED and a["normalized_rms_error"] <= END_RMS_NORMALIZED
                for a in (case["router_logits"], case["routing_weights"], case["shared"], case["routed"], case["output"]))
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
        if args.sweep:
            for e in range(512):
                x = rng.normal(size=(1, 2560)).astype(np.float32)
                actual = cpu(layer.expert_npu(e, nd(torch.from_numpy(x).to("npu:0"))))
                expected = layer.expert_cpu(e, x)
                result = {"expert": e, "input_sha256": digest(x), "output_sha256": digest(actual),
                          **agreement(actual, expected)}
                result["passed"] = result["finite"] and result["max_normalized_error"] <= END_MAX_NORMALIZED and result["normalized_rms_error"] <= END_RMS_NORMALIZED
                report["expert_sweep"].append(result)
                print(json.dumps({"sweep": result}), flush=True)
        report["weights_unchanged"] = layer.weights_unchanged()
        report["execution_complete"] = len(report["cases"]) == 7 and (not args.sweep or len(report["expert_sweep"]) == 512)
        report["stage_contracts_passed"] = all(c["passed"] for c in layer.quant_checks + layer.stage_checks)
        report["passed"] = report["execution_complete"] and report["weights_unchanged"] and report["stage_contracts_passed"] and all(
            c["passed"] for c in report["cases"] + report["expert_sweep"])
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        if layer is not None:
            report["quantization_checks"] = layer.quant_checks
            report["projection_checks"] = layer.stage_checks
        report["npu_peak_allocated_bytes"] = int(torch.npu.max_memory_allocated(0))
        report["npu_peak_reserved_bytes"] = int(torch.npu.max_memory_reserved(0))
        with args.report.open("x") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    print("REAL_MOE_LAYER_PASS" if report["passed"] else "REAL_MOE_LAYER_FAIL", flush=True)


if __name__ == "__main__":
    main()
