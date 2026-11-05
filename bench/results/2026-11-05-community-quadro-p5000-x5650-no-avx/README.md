# Community benchmark on Quadro P5000 with a CPU without AVX (2x Xeon X5650)

Measured on 2026-11-04 and 2026-11-05 by @modulismm, with the older-CPU build: an engine compiled
with `STRATA_ISA_FLOOR=none` (SSE4.2 + POPCNT, no AVX at all), on a Pascal card through the CUDA 12 engine. It compares
**IQ3_S and Q2_0**, each with the default CPU pool (5 workers, one socket) and with both sockets (`--pool-workers 11`),
using `strata bench run` (standard-v1). Main limitation: one machine, one run of the protocol per configuration (3 repetitions inside each run).

**Result in one line:** on a CPU without AVX, **Q2_0 decodes about 70% faster than IQ3_S** (17.6 vs 10.2 tok/s on the
short prompt) with the same 12/12 sanity score, and **using both sockets adds another 34-38% on Q2_0** (10-18% on IQ3_S):
Q2_0 with `--pool-workers 11` decodes at about 24 tok/s, 2.3x the IQ3_S default.

## Hardware and software

- GPU: Quadro P5000, 16 GiB, sm_61, power limit 180 W. One card.
- CPU: 2x Intel Xeon X5650 (Westmere-EP), 2.67 GHz, 12 cores / 24 threads. **SSE4.2 only: no AVX, AVX2, FMA or F16C.** The BIOS presents one NUMA node.
- RAM: 110 GiB visible (DDR3; speed not measured, needs root).
- Storage of the model files: WDC WD5000AAKS, a 7200 rpm SATA HDD. Setup chose `--ple-io ram` because of it.
- PCIe: 2.0 x16, limited by the board; the card supports 3.0. The engine measured **6.0 GB/s** host to device.
- OS: Ubuntu 24.04.5 LTS, kernel 6.8.0-94; NVIDIA driver 580.178.04 (CUDA 13.0 driver API).
- Strata: **source build** of commit `ffa80d7`, engine 0.1.56, CUDA toolkit **12.9.1** (userspace runfile install),
  `engine-cuda12/BUILD.json`: `"source": "local", "archs": [61], "toolkit": 12, "isa_floor": "none"`.
  Build options: `STRATA_ISA_FLOOR=none`, `STRATA_EXPERIMENTAL_SM60=ON`, `CMAKE_CUDA_ARCHITECTURES=61`,
  `STRATA_Q6K_EXPERTS=ON`, `STRATA_MMQ_KQUANTS=ON`.
  Benchmark runner: Terminal 0.2.10, `strata bench run --url`, runner commit `974074c`.
  > **Note:** the `results.json` files say `"build_kind": "release"` and `"strata_commit": "974074c…"`. Those describe the
  > runner's install, not the engine under test (`strata_cli/bench_facts.py:433`). The engine is the source build above.
  > Reported separately: https://github.com/Niko1221/Strata/issues/5331.
- Background: SearXNG and Ollama idle on the same host, ComfyUI stopped, nothing else on the GPU.

## Model and configuration

- Model: original Qwen3.8-Flash-Next, `GSQ-RCO` GGUFs fetched by `setup.sh`; repository revision not recorded.
  - IQ3_S: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (sha256 `4c1eb2ceb4915e11…`),
    `-00002-of-00002.gguf` (`316b46f3a2dbd68c…`).
  - Q2_0: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` (`69820c02ec7d0b45…`),
    `-00002-of-00002.gguf` (`316b46f3a2dbd68c…`, the same file as IQ3_S's second shard).
  - Full hashes are in each folder's `results.json`.
- Packs: prepared by setup (`packs/iq3_s`, `packs/q2_0`), shipped expert profile `data/expert-profile.bin`. No custom packs.
- Vision: on, encoder on the CPU (`--vision cpu`; mmproj BF16).
- Context 32,768; KV int8; expert cache auto; prefill auto; PLE table in RAM (`--ple-io ram`, not locked: RLIMIT_MEMLOCK).
- MTP: on (`--spec 4 --spec-min-p 0.5`; the DFlash2 drafter was not downloaded). No calibration, no experimental speed projection.
- Server: 1 slot, loopback only.

```text
setup (once per quant), from the source checkout:
  STRATA_NVCC=<CUDA-12.9>/bin/nvcc STRATA_ISA_FLOOR=none \
  ./setup.sh --family qwen --model Q2_0 --build --no-start --yes --cuda 12 --context 32768 --vision cpu \
             --data-dir <DATA> --host 127.0.0.1 --port 8081

engine args written by setup (strata-q2_0.json; IQ3_S identical except the paths):
  --pack <DATA>/packs/q2_0 --native <DATA>/models/Q2_0/...-00001-of-00002.gguf
  --ple-gguf <DATA>/models/Q2_0/...-00002-of-00002.gguf --expert-profile <SRC>/data/expert-profile.bin
  --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <DATA>/mtp/rt --max-context 32768
  --kv int8 --ple-io ram --vision
  "pw11" variants: the same + --pool-workers 11

start: ./run-q2_0.sh   (serve/server.py --engine strata --config strata-q2_0.json --port 8081)
bench: strata bench run --url http://127.0.0.1:8081 --suite standard-v1 --track custom --source-revision ffa80d7 ...
```

What the engine chose at start (from its log):
- CPU pool: "5 workers by default (12 physical cores, 24 logical processors; bounded by the host's CPU socket)", on logical CPUs 2,4,…,10.
- PCIe share: "the CPU pool takes ~1740 us per missed expert (1.2 GB/s of expert bytes), the link 6.0 GB/s → pcie_frac 0.60".
- "EXPERIMENTAL older-CPU build (ggml-cpu for sse4.2)".
- VRAM free with everything loaded: 766 MiB (IQ3_S). The 512 MiB warning appeared once on IQ3_S during an earlier ad-hoc run; no slowdown was observed.

## Method

`strata bench run --suite standard-v1` (protocol `strata-qwen38-thinking-v1-draft.1`, corpus `e75eff8b9feea04f…`):
- 1 warm-up (excluded);
- 3 repetitions each of short, 4K and 16K prompts (512-token cap, reasoning included);
- the 12 sanity checks.

Thinking on, `reasoning_effort: xhigh`, model-card sampling. Each run starts on a **freshly loaded engine**, which had
answered no request before. The default-configuration runs are `complete` (model files hashed). The two `--pool-workers 11`
runs used `--skip-model-hash` (same files, about 15 minutes of hashing on the HDD saved), so their status is `nonconforming`
for that reason only. Model loading is excluded: cold load took about 8 min (IQ3_S) and about 4 min (Q2_0) from the HDD.

Memory, `free -g` snapshots while serving: IQ3_S about 56 GiB used and 15.5 GiB VRAM; Q2_0 about 40 GiB used and 15.7 GiB VRAM.

## Results

Medians (min-max), n=3; engine timings for prompt/decode, client clock for TTFT.

| Configuration | Result folder | Prompt tok/s 4K | Prompt tok/s 16K | Decode tok/s short | Decode tok/s 4K | Decode tok/s 16K | TTFT s short / 4K / 16K | Sanity | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IQ3_S, default | [`2026-11-04-standard-quadro-p5000-92788cff`](../2026-11-04-standard-quadro-p5000-92788cff/README.md) | 273.7 (267.1-276.8) | 275.9 (275.6-276.3) | 10.2 (9.0-10.6) | 9.6 (9.6-9.8) | 10.4 (9.5-11.0) | 3.25 / 15.08 / 59.57 | 12/12 | complete |
| IQ3_S, `--pool-workers 11` | [`2026-11-04-standard-quadro-p5000-d3ebb648`](../2026-11-04-standard-quadro-p5000-d3ebb648/README.md) | 277.6 (218.5-279.3) | 277.1 (269.2-277.2) | 11.2 (11.1-11.9) | 11.1 (10.0-11.3) | 12.3 (11.1-12.7) | 2.65 / 14.84 / 59.26 | 12/12 | nonconforming (hash skipped) |
| Q2_0, default | [`2026-11-05-standard-quadro-p5000-127743b2`](../2026-11-05-standard-quadro-p5000-127743b2/README.md) | 286.4 (284.6-286.6) | 283.7 (283.5-284.2) | 17.6 (15.6-17.8) | 16.9 (16.4-17.3) | 17.7 (17.4-19.4) | 2.14 / 14.39 / 57.87 | 12/12 | complete |
| **Q2_0, `--pool-workers 11`** | [`2026-11-05-standard-quadro-p5000-73190f07`](../2026-11-05-standard-quadro-p5000-73190f07/README.md) | 287.9 (286.5-288.5) | 284.0 (283.9-284.2) | **23.6 (22.2-24.6)** | **23.4 (21.6-24.6)** | **24.3 (22.4-25.2)** | 1.96 / 14.30 / 57.79 | 12/12 | nonconforming (hash skipped) |

Required duration after load: 15 min 46 s (IQ3_S default), 14 min 38 s (IQ3_S pw11), 11 min 51 s (Q2_0 default), 10 min 24 s (Q2_0 pw11).
Capped at 512 tokens: all 4K/16K performance requests except one 4K repetition in each Q2_0 run; no short request except 2 of 3 in the Q2_0 pw11 run.
No request failed, was retried or skipped in any run.

**Observations**
- Prompt reading is GPU-bound and nearly identical across configurations (274-286 tok/s at 4K and 16K).
- Decode is CPU-bound: the CPU pool's SSE4.2 kernels (about 1.2 GB/s of expert bytes) are slower than the PCIe 2.0 link.
  That is why Q2_0 (less CPU work per token) gains about 70%, and why a second socket helps, more so on Q2_0 (+34-38%) than on IQ3_S (+10-18%).
- The setup tip "try --pool-workers 5 on 2-socket Xeons" is already the engine's default here. On this CPU the opposite (both sockets) is faster for both quants.
- A first ad-hoc request right after load (thinking off) decoded at 2.1 tok/s; the engine re-tunes its PCIe share after the first requests. Judge from the protocol runs, not a first request.

## Correctness and limitations

- All four runs passed **12/12** sanity checks: instruction 4/4, retrieval at 4K/16K 4/4, tool calls 4/4. It is a functional check, not a quality evaluation of Q2_0 against IQ3_S.
- One machine, one protocol run per configuration. The pw11 rows are a same-machine A/B against the default rows, not a standard result.
- Not tested: contexts above 32K, the optional concurrency stage, vision questions, DFlash2, `--vram-reserve-mib`, single-socket operation.
- How to reproduce the build without `strata launch`, which always downloads the AVX2 release engine: see the
  `setup.sh` command above and issue #5114.
