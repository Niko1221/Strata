# Ornith-1.5 / Qwen35MoE on gfx1101: implementation and measured validation

This directory records the measured evidence for the Ornith-1.5 work. It is deliberately explicit about
what has been run and what has not: **the Qwen35MoE execution backend is not implemented in this
increment**, so there are no decode/prefill throughput numbers here and no `DONE` marker. What is
implemented and measured on the card are the artifact contract, the architecture guard and the routed
expert kernel path Ornith needs.

- Baseline SHA: `7a96f2d3ea6827a4355fea78c8b600c479bbea69` (`feature/rdna3-support`)
- Final SHA: see `shas.txt`
- Branch: `feature/rdna3-support`

## Hardware

See `hardware.txt`. RX 7700 XT (gfx1101, 12,272 MiB), Ryzen 9 7900 (12 physical cores), ~122 GB RAM,
ROCm 7.2.4, `HIP_VISIBLE_DEVICES=0` (the gfx1036 iGPU stays hidden). The engine and tests are built and
run in the `strata-hip-builder:gfx1101` container (`./build.sh`), never on the host, per the project's
build contract.

## Artifacts inspected

| | repository | file | size |
|---|---|---|---|
| main | `AtomicChat/Ornith-1.5-35B-A3B-GGUF` | `Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf` | 20,125,923,520 B |
| MTP | `EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP-GGUF` | `mtpdraft-Q8_0.gguf` | 1,990,649,440 B |

Only the GGUF header (metadata + tensor table, 10,989,248 B) is needed to establish the contract; it is
fetched over HTTP range requests by `tools/ornith_inspect.py`. `gguf-layout.txt` is the full checked-in
report and `gguf-metadata.json` the derived geometry.

Verified geometry (from the artifact, not the task description): `qwen35moe`, 40 layers, 2048 wide,
256 experts top-8, 512-wide routed and shared FF, full attention every 4th layer (30 GDN + 10
full-attention), 16 query / 2 KV heads of 256 with a 64-wide partial RoPE, GDN state 128 with 16 key and
32 value heads, vocab 248,320, RMS eps 1e-6. Routed experts: IQ4_XS gate/up, Q4_K down, in every layer.

## What was validated on gfx1101

`qwen35-layout-test.log` - the Qwen35MoE guard (`src/core/qwen35.cpp`, `qwen35_layout_test`). 16
synthetic mutation cases (wrong architecture, wrong shape, missing tensor, crossing layer families,
Qwen4Exp-only tensors, wrong type, missing metadata, inconsistent lengths) plus the real Ornith header,
which passes and reads back the geometry above. No GPU, no ggml.

`native-expert-ornith-parity.log` - the grouped native expert path at Ornith's 2048/512 geometry with
IQ4_XS gate/up and Q4_K down, against ggml's float reference and ggml-cpu, on the GPU. Q4_K was a
gate/up-only type in the grouped DOWN switch before this change; the test is what admits it.

```
synthetic iq4_xs /q4_K  blob 1703936  cpu rel 2.04e-02  gpu rel 1.24e-02  cpu-gpu 2.28e-02  ok
native_expert_parity: 0 failures
GPU dequant vs ggml to_float: rel 0.00e+00, 0 of 3145728 values differ in any bit
q4_K down rows, AVX2 multi-token vs ggml vec_dot: 0 of 36 token-sets differ in any bit
gpu decode-once vs per-entry kernels: bitwise equal
```

`ctest.log` - the full HIP suite: **67 of 70 pass** in 75 s. The three failures (`ple_parity`,
`expert_parity`, `pool_test`) need a local model pack fixture (`pack/full/experts.bin`) that this
container does not mount, and fail with `cannot read ... from pack/full/experts.bin`; they are
environment failures, not regressions. `hip_prompt_attn_wmma` and `hip_prefill_hipblaslt_gemm` skip as
documented. All 8 `native_expert_parity_*` tests, including the new Ornith one, pass.

`unit-hfmodel.log` - `docker/test_hfmodel_ornith.py`: 6 cases for single-file (Ornith) and two-shard
(Qwen3.8) cache resolution.

`run3-dry-run.txt`, `run3-dry-run-nomtp.txt`, `run3-dry-run-256k.txt` - the Ornith launcher's docker
command for the default (spec 3), `--no-mtp` (spec 0) and 256K context.

## Not measured (and why)

There are **no tok/s, acceptance or VRAM-at-context numbers for Ornith**, because the Qwen35MoE
execution backend (GDN layers, dense attention, MoE block, MTP, session) is future work. Reporting
throughput for a path that does not execute the model would be a fabricated number, which this
repository's evidence rule forbids. The launcher performs the full host-side preparation and artifact
validation; the engine serve step is the remaining work, tracked in `docs/ORNITH_QWEN35MOE.md`.
