"""Locate quantization-boundary divergence without changing pass budgets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu

from probe import Layer, agreement, cpu, digest, nd, quant_cpu, quant_npu, sigmoid, stage_bound, xhat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--expert", type=int, default=423)
    args = ap.parse_args()
    assert not args.report.exists()
    torch.npu.set_device(0)
    torch_npu.npu.matmul.allow_hf32 = False
    torch.set_num_threads(4)
    layer = Layer(args)
    rng = np.random.default_rng(20261021)
    # Match the seven fixtures, then this expert's one-token sweep input.
    for t in (1, 2, 4, 8, 3, 5):
        rng.normal(size=(t, 2560))
    for _ in range(args.expert + 1):
        x = rng.normal(size=(1, 2560)).astype(np.float32)
    e = args.expert
    xd = nd(torch.from_numpy(x).to("npu:0"))
    q, s = quant_npu(xd)
    gate, up = layer.project(e, "gate", q, s), layer.project(e, "up", q, s)
    mid = torch.nn.functional.silu(gate) * up
    qmid, smid = quant_npu(mid)
    down = layer.project(e, "down", qmid, smid)
    cq, cs = quant_cpu(x)
    exact = xhat(cq, cs)
    rg = exact @ layer.expert(e, "gate").astype(np.float64).T
    ru = exact @ layer.expert(e, "up").astype(np.float64).T
    reference_mid = (rg * sigmoid(rg) * ru).astype(np.float32)
    rq, rs = quant_cpu(reference_mid)
    reference_down = xhat(rq, rs) @ layer.expert(e, "down").astype(np.float64).T
    ng, nu, nh, nq, ns = cpu(gate), cpu(up), cpu(mid), cpu(qmid), cpu(smid)
    # Isolate nonlinear differences after holding both native projection outputs.
    held_mid = (ng.astype(np.float64) * sigmoid(ng.astype(np.float64)) * nu.astype(np.float64)).astype(np.float32)
    held_q, held_s = quant_cpu(held_mid)
    actual_mid_cpu_q, actual_mid_cpu_s = quant_cpu(nh)
    w = layer.expert(e, "down").astype(np.float64)
    conditional_x = xhat(nq, ns)
    conditional = conditional_x @ w.T
    counterfactual = cpu(layer.project(e, "down", nd(torch.from_numpy(rq).to("npu:0")),
                                      nd(torch.from_numpy(rs).to("npu:0"))))
    changed = np.argwhere(nq != rq)
    details = []
    for token, column in changed[:32]:
        chunk = column // 32
        def scaled(h, scale):
            return float(np.float32(h * np.float32(1 / scale))) if scale else 0.
        details.append({"token": int(token), "column": int(column),
                        "native_mid": float(nh[token, column]), "fp64_chain_mid_fp32": float(reference_mid[token, column]),
                        "native_code": int(nq[token, column]), "fp64_chain_code": int(rq[token, column]),
                        "native_scale": float(ns[token, chunk]), "fp64_chain_scale": float(rs[token, chunk]),
                        "native_scaled": scaled(nh[token, column], ns[token, chunk]),
                        "fp64_chain_scaled": scaled(reference_mid[token, column], rs[token, chunk])})
    report = {"expert": e, "input_sha256": digest(x), "weights_revision": layer.manifest["revision"],
              "binary_sha256": layer.binary_hashes,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "gate_vs_fp64": agreement(ng, rg), "up_vs_fp64": agreement(nu, ru),
              "mid_vs_fp64_chain": agreement(nh, reference_mid.astype(np.float64)),
              "independent_full_expert": agreement(cpu(down), reference_down),
              "intermediate_code_differences": int(len(changed)), "changed_coordinates": details,
              "nonlinearity_held_projection_code_differences": int(np.sum(nq != held_q)),
              "quantizer_actual_mid_code_differences": int(np.sum(nq != actual_mid_cpu_q)),
              "conditional_down_bound": stage_bound(cpu(down), conditional,
                  np.abs(conditional_x) @ np.abs(w).T, 640),
              "counterfactual_reference_codes_through_native_down": agreement(counterfactual, reference_down),
              "native_output_sha256": digest(cpu(down)), "reference_codes_sha256": digest(rq),
              "native_codes_sha256": digest(nq), "input_unchanged": bool(np.array_equal(cpu(xd), x))}
    with args.report.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
