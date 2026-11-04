# Community benchmark on RTX 5090 (UD-Q4_K_XL, 262k, k8s, 128 GB RAM)

Measured on 2026-11-03 and 2026-11-04 (two runs) by em3rik. Standard protocol `strata-qwen38-thinking-v1-draft.1`
(draft) on an RTX 5090 box serving the Unsloth UD-Q4_K_XL quant (~111 GB, 4 shards) of
Qwen3.8-Flash-Next inside a Kubernetes pod on a 128 GB DDR4 desktop.

Two runs on one configuration, both published (no cherry-picking):

| Run | When | Server state | Status |
| --- | --- | --- | --- |
| quiet | 2026-11-04 02:16 CET, crew idle | already serving, 925 prior requests, no interleaved traffic (see limitations) | complete, sanity 12/12 |
| under-load | 2026-11-03 17:41 CET | already serving, 689 prior requests, operator interactive-agent traffic interleaving | complete, sanity 12/12, TTFT inflated by queueing |

The quiet run is the primary evidence (root of this folder). The under-load run is the same suite
on the same server under real shared-box traffic, kept in `run-under-load/` because it is honest
data about what a 1-slot server looks like when an interactive agent interleaves: TTFT rises to
19-60 s while engine-side decode stays ~50-57 tok/s.

## Hardware and software

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB dedicated, power limit 575 W, max SM clock 3105 MHz;
  PCIe gen4 x16 verified after the 2026-10-24 CPU swap (operator check)
- CPU: AMD Ryzen 9 5950X (16 cores / 32 threads, Zen 3, AVX2, no AVX-512), `performance` governor
- RAM: 128 GB DDR4-3200, 4x32 GB dual-rank (2 DIMMs per channel; measured STREAM triad ~29 GB/s -
  the Zen 3 IMC loses bandwidth at 2DPC; this is the binding constraint for a 111 GB model)
- Storage of the model files: Samsung 990 PRO 2 TB NVMe (LVM ext4); weights live in RAM/page cache
  during serving, storage matters mostly for model-file hashing and cache refill
- OS: Ubuntu 26.04.1 LTS host, kernel 7.0.0-42-generic. `results.json` reports Ubuntu 24.04.5 LTS
  because the server and the bench runner run inside the official Strata Docker container (Ubuntu 24.04 base).
- Driver 580.178.04; CUDA 13.0; container runtime: microk8s containerd, NVIDIA GPU Operator device plugin;
  pod Burstable (`requests` 60 Gi / `limits` 105 Gi memory) with `IPC_LOCK` for pinned experts
- Strata Terminal 0.2.8 (build 0.1.54), release build, official `cuda13` docker image
  (ghcr `niko1221/strata`, retagged into the cluster-local registry); engine binary `0.1.54-linux-x86_64-cuda-cuda13`
- Protocol/runner commit recorded in the results: `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`

## Model and configuration

- Model repository: `unsloth/Qwen3.8-Flash-Next-GGUF` (repository head `766911a6b7369840a91dbcd95f9f997acaab6cd6`,
  last upload 2026-10-06, i.e. identical to the revision downloaded 2026-10-19)
- Quantization: UD-Q4_K_XL (Unsloth Dynamic 2.0); 4 shards, total 111,289,780,113 bytes:
  - `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` SHA-256 `4448186216b3af4cc558bbce...` (full hashes in `results.json`; all four equal the repo's current LFS objects)
  - `...-00002-of-00004.gguf` `3f342f1c1580473f1ee94ddd...`
  - `...-00003-of-00004.gguf` `56758f40269cad5cd9b0d3d6...`
  - `...-00004-of-00004.gguf` `753bda48b98ba4f1636134a9...`
- Vision encoder: `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, GPU vision enabled
- Context 262,144 tokens; server capacity 1 slot (`PARALLEL=1` - deliberate: 2 slots halved solo
  decode on this box); KV int8 with streaming + `--kv-resident 32768`; MTP draft ON (`--spec 4`)
- **Custom lever:** `--resident-budget-gib 55`. Setup's auto budget at 262k on this box pinned
  ~100 GiB of experts, which OOM-killed the 128 GB host (measured 2026-10-24: 100 GiB pinned is
  not survivable at 262k context). Capping to 55 GiB is what makes this box stable; the cost is
  expert streaming from page cache for the cold tail. Track is therefore, strictly, **custom**;
  the recorded `results.json` says `default` because we passed the release's setup-derived notes -
  flagged here rather than rewriting the recorded file. The quiet-night rerun records `custom`.
- Sampling: model-card defaults (see `results.json`).

```text
# Container entrypoint derives the launch from these deployment env vars (credentials removed):
FAMILY=unsloth MODEL=UD-Q4_K_XL MODEL_PATH=/data/models CONTEXT=262144 VISION=yes KV=int8
PARALLEL=1 STRATA_SET=resident-budget-gib=55 PORT=8080

# bench invocation (already-running server, so no engine log is attached):
strata bench run --url http://127.0.0.1:8080 --suite standard-v1 --credential-env STRATA_API_KEY \
  --track custom --track-notes "release 0.2.8 (build 0.1.54), docker image cuda13, resident-budget 55 GiB, parallel 1, kv int8 + kv-resident 32768, 262k ctx" \
  --background "shared production box; k8s pod; <per-run, see run folders>"
```

## Method

Identical on both runs: frozen `bench/prompts/standard-v1` fixtures (corpus `e75eff8b9feea04f...`),
loopback HTTP, streaming, concurrency 1, protocol order, no retries, no optional stage. Engine-side
`timings` for prompt/decode rates; client monotonic clock for TTFT. Expert cache warm (production
server). Full method text in each run's `RUNNER.md`.

## Results

### Quiet run (primary) - `results.json`, run id `20261104T011658Z-129ef329`

Status: **complete**. 21 of 21 required requests finished, 0 failed, 0 skipped. Sanity **12 of 12**
(instruction 4/4, retrieval 4/4, tool calls 4/4). Time after the model loaded: 2 min 7 s.
Server state: already serving (925 prior requests), crew idle, no interactive traffic interleaving.

| Case        | Prompt tok/s                | Decode tok/s               | TTFT s                    | Total s                    | Capped |
| ----------- | --------------------------- | -------------------------- | ------------------------- | -------------------------- | ------ |
| short       | n/a                         | 61.4 (56.2-63.7) n=3       | 0.55 (0.52-0.66) n=3      | 6.6 (6.4-7.1) n=3          | 0/3    |
| prompt-4k   | 1815.0 (1811.4-1817.0) n=3  | 66.4 (63.7-67.6) n=3       | 2.28 (2.28-2.28) n=3      | 9.8 (9.8-10.3) n=3         | 2/3    |
| prompt-16k  | 3116.2 (3110.8-3152.0) n=3  | 65.0 (61.3-72.4) n=3       | 5.29 (5.23-5.30) n=3      | 13.2 (12.3-13.6) n=3       | 3/3    |

Cells are median (min-max) n=samples. `short` prompt rate is not shown (fixed request cost).
This run recorded `--track custom` (see Model and configuration above).

### Under-load run - `run-under-load/results.json`, run `20261103T164101Z-cb4460b8`

Status complete, 0 failed, 0 skipped, suite 667.4 s after load (queueing inflates the suite wall
time, not the per-request engine rates).

| Case | Prompt tok/s | Decode tok/s | TTFT s | Total s | Capped |
| --- | --- | --- | --- | --- | --- |
| short | n/a | 49.6 (47.7-52.9) n=3 | 31.42 (27.30-48.30) | 37.2 (32.2-55.0) | 0/3 |
| prompt-4k | 1749.6 (1717.5-1776.6) | 56.6 (56.3-60.2) | 21.23 (19.14-51.40) | 30.3 (27.6-60.4) | 3/3 |
| prompt-16k | 3093.8 (3092.2-3121.6) | 56.3 (53.7-58.9) | 58.98 (46.47-60.13) | 68.1 (55.1-69.6) | 3/3 |

Sanity checks: **12 of 12 passed** in this run. TTFT is queue time behind interleaved interactive
traffic on a 1-slot server (19-60 s vs 0.5-5.3 s in the quiet run); engine-side decode drops from
the quiet run's 61-66 tok/s to 50-57 tok/s, while the engine-measured prompt rate is barely moved
(16k: 3094 vs 3116 tok/s) because it excludes queueing. See the quiet run for uncontended numbers.

## Correctness and limitations

- Both runs: 21/21 required requests, 0 failed, 0 skipped, sanity 12/12 each.
- Not cold-model measurements: warm production server, warm expert cache; shared template prefix
  reuse only (allowed by the protocol).
- `--url` mode: engine flags are not readable from a running server, so `results.json` has `null`
  fields with reasons; operator-recorded values above are attributed as such. No engine log attached
  (the runner did not start the engine).
- The 111 GB model on 128 GB RAM cannot fully pin its experts at 262k context; every number here is
  a partially-resident number by hardware necessity, not by tuning taste. A 256 GB version of this
  build would measure differently.
- Server capacity 1 slot: the optional concurrency stage was not run; queueing behaviour under
  interleaved traffic is visible in the under-load run instead of being hidden.
- Cross-machine note: the sister report on an RTX 4090 (i7-7700K DDR4, 62 GB RAM) running the 2-bit
  Gyro-M TQ2_0 quant reaches the same decode band with a fraction of the RAM; the two quants target
  different points on the quality/memory axis and are not a quality comparison.

## Files

- `README.md` - this report
- `results.json`, `outputs.jsonl`, `config.json`, `RUNNER.md`, `provenance/` - primary (quiet) run
- `run-under-load/` - the 17:41 CET run, same layout, untouched runner output
