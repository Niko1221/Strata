# Community benchmark: unsloth UD-Q4_K_XL (Qwen3.8-Flash-Next) on one RTX 5060 Ti 16 GB

Measured on 2026-10-03. Scope: **one machine, one quantization tier, one context length** — RTX 5060 Ti 16 GB, AMD EPYC 7B12, 251.5 GiB RAM, the unsloth **UD-Q4_K_XL** GGUF (111.33 GB) at a **262,144-token** context limit, KV `q4_0`. Every number below comes from a measured file in `data/`. Fields that were not measured are marked `unavailable` rather than estimated.

## Hardware and software

- NVIDIA GeForce RTX 5060 Ti, 16311 MiB (15.93 GiB) VRAM, compute capability 12.0.
- Power limit 180 W (`power.limit`; default limit also 180 W). **Actual draw under load: unavailable.**
- PCIe link **Gen4 (16.0 GT/s) x8**; hardware maximum Gen5 x16. Engine transfer probe **14.1 GB/s** host-to-device (Gen4 x8 theoretical is 15.75 GB/s), from which the engine selected `pcie_frac 0.30`.
- AMD EPYC 7B12 64-core, 128 logical CPUs, 1 socket, 1 NUMA node, AVX2 only (no AVX-512).
- 251.5 GiB RAM (`MemTotal 263745772 kB`); 8.0 GiB swap.
- Storage: WD_BLACK SN850X 2000GB NVMe (models and `/tank`), plus a second NVMe drive.
- Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic, NVIDIA driver 595.84.
- CUDA 13.2.86 (`nvcc V13.2.86`), installed into a user prefix. **This exact version matters:** 13.2.78 miscompiles the `IQ*_S` (IQ3_S/IQ2_S) kernels on sm_120; the local `iq_multi_parity` check went from 36 failures to 0, and reported errors fell from 0.67-1.03 to 5e-8-6e-5.
- Strata 0.1.35, source commit `d9ab8435f654c368c586340d490915f6addf56a3`, **plus one local patch** (`src/prefill/prefill.cpp:561`, stager thread clamp). Build: `arch=120a-real`, `CUDA=ON NATIVE_EXPERTS=ON MMQ_KQUANTS=ON BUILD_TESTS=OFF`.
- No background GPU workload; the GPU was dedicated to Strata. The machine was not otherwise isolated.

## Model and configuration

- Tier: unsloth **UD-Q4_K_XL** dynamic quantization (expert tensors Q4_K / Q5_K / Q5_1 / Q8_0).
- Repository and revision: `unavailable` (no download-side revision record was retained on this machine).
- `Qwen3.8-Flash-Next.gguf`, 111,334,654,400 bytes; `mmproj-Qwen3.8-FN.gguf`, 616,703,104 bytes; `mtp-Qwen3.8-FN.gguf`, 1,907,151,936 bytes. SHA-256 values are in the table below.
- 24,576 experts; the expert profile contains 24,576 ranked pairs; pack `native_experts.txt` records `n_expert 512` across 48 layers.
- Pack, expert profile, draft pack and binaries: hashes in the table below. Per-tensor quant-type distribution is `unavailable` (the pack index stores numeric type codes: `gu_type 12 / d_type 7`).

| 产物 | 路径 | 字节 | SHA-256 |
|---|---|---:|---|
| 权重 GGUF | `/tank/models/unsloth/Qwen3.8-Flash-Next.gguf` | 111,334,654,400 | `unavailable` |
| 视觉 mmproj | `/tank/models/unsloth/mmproj-Qwen3.8-FN.gguf` | 616,703,104 | `unavailable` |
| MTP 存根 | `/tank/models/unsloth/mtp-Qwen3.8-FN.gguf` | 1,907,151,936 | `unavailable` |
| pack dense.bin | `work/strata-packs/unsloth-q4/dense.bin` | 1,485,688,320 | `f53174059b262ed342b82e143af62056081948ef5e5f1903a1a70eeb73fa4a23` |
| pack index.txt | `work/strata-packs/unsloth-q4/index.txt` | 93,534 | `ea349751bd7da4b5d9fe03ac3f65f196b6cf3de4728ff11266f5fefe8cb3c20f` |
| pack native_experts.txt | `work/strata-packs/unsloth-q4/native_experts.txt` | 3,278 | `e4ef23629b6071a42b29483e1ce1e9e881cb5d84dabbd7ea6c2b280eac748161` |
| pack compat-bf16.json | `work/strata-packs/unsloth-q4/compat-bf16.json` | 21,024 | `e5080ae8f69c70c8cffa621c01911b4dafd57dc64ae61944c4afaaca4a02c292` |
| pack conversions.json | `work/strata-packs/unsloth-q4/conversions.json` | 155,601 | `cb8de607a3332c2a33f22056b0036bef6ec0bc2f2be2ec876a351e393b4c6c12` |
| expert profile | `strata-study/data/expert-profile.bin` | 196,632 | `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf` |
| draft dense | `work/strata-data/mtp/rt/dense.bin` | 116,099,072 | `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a` |
| draft experts | `work/strata-data/mtp/rt/experts.bin` | 707,788,800 | `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f` |
| draft vocab | `work/strata-data/mtp/rt/draft_vocab.bin` | 425,196 | `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91` |
| 引擎 `strata` | `engine-builds/v-strata-0.1.35-120a-u2-20261003/bin/strata` | 45,369,152 | `cd07d0891aacaed9ad7d95f0e804a46e71ea03379f25211364c88322e05fe082` |
| `strata-vision` | `.../bin/strata-vision` | 56,454,280 | `f6faf82fd4d0a797b62fbfb75fbb55d15a56f803a60a7fbafca0588aa4d440da` |
| `strata-vision-cpu` | `.../bin/strata-vision-cpu` | 7,612,400 | `d1fdd5e28de496c3adb8ffca06390df1422ec4ebb8ac5084e13f188cd2b21bad` |

Launch (recommended configuration from this run):

```text
python3 -m serve.server --engine strata --config work/strata-sweep/S2-256k-best.json --host 127.0.0.1 --port 8087

# engine arguments:
--pack /home/ai-agent/DSHW/work/strata-packs/unsloth-q4
--native /tank/models/unsloth/Qwen3.8-Flash-Next.gguf
--ple-gguf /tank/models/unsloth/Qwen3.8-Flash-Next.gguf
--kv q4_0
--mtp /home/ai-agent/DSHW/work/strata-data/mtp/rt
--expert-profile /home/ai-agent/DSHW/strata-study/data/expert-profile.bin
--spec 4 --spec-min-p 0.5 --prefill auto --spec 2   # the later --spec 2 wins (generate.cpp:1133 assigns per occurrence)
--max-context 262144
```

The two vision arms add `--kv-resident 32768 --resident-budget-gib 71 --vision`; the GPU-vision arm also adds `--vram-reserve-mib 1400` with a CUDA vision helper, the CPU arm uses `strata-vision-cpu` (mmproj in system RAM).

## Method

- Repetitions per fresh request (`cache_n=0`): short x3 on both arms; long x4 (GPU) / x2 (CPU); deep x2 (GPU) / x1 (CPU); image-solve x1. **Only the short cells and the GPU long cell reach three runs.** The CPU long/deep top-up run failed to start (engine exited at `FileExpertSource: allocating 70.19 GiB page-locked cache complement`; see the engine log). One GPU deep run hit a 131,072-token prefix cache (`cache_n=131072`) and is reported separately, not merged into the median. All runs are listed individually in `matrix.md`; medians and ranges are over the engine-reported numbers.
- Cache state: every speed request was a fresh serial request; the engine line reports `0 reused` tokens unless noted; after the first request the OS page cache is warm, which is why load times differ between items.
- **Timing boundaries: model loading is NOT included** in any tok/s or TTFT value; the model-load seconds are reported separately in the sweep table. **Vision encoding IS included** in the image cell's wall clock and in its prompt-processing time; the vision section derives an upper bound for it separately.
- TTFT is **derived** from the engine's non-streaming prompt-processing line (`prompt ... read in N ms`); it is not a streaming first-token measurement.
- `temperature: 0`, `reasoning_effort: none` (the image-solve cell uses `high` with a 1,500-token budget), no calibration and no experimental speed projection.

## Results

### Per-run values

**A1-多模态GPU (GPU vision)**

| Cell | Run | Prompt tok | Reused tok | Generated tok | Prompt tok/s | Decode tok/s | TTFT s | Latency s | Draft accepted | cache_n |
|---|---|---|---|---|---|---|---|---|---|---|
| short | 1 | 25 | 0 | 18 | 33.4 | 20.7 | 0.748 | 1.7 | 13/17 | 0 |
| short | 2 | 25 | 0 | 18 | 39.0 | 28.4 | 0.642 | 1.3 | 11/17 | 0 |
| short | 3 | 25 | 0 | 18 | 43.2 | 26.3 | 0.579 | 1.3 | 13/17 | 0 |
| long | 1 | 10381 | 0 | 62 | 392.6 | 23.2 | 26.441 | 29.5 | 31/50 | 0 |
| long | 2 | 10381 | 0 | 62 | 618.3 | 27.1 | 16.791 | 19.1 | 30/51 | 0 |
| long | 3 | 10383 | 0 | 61 | 555.9 | 22.3 | 18.678 | 21.5 | 33/51 | 0 |
| long | 4 | 10383 | 0 | 62 | 622.9 | 25.8 | 16.669 | 19.1 | 30/53 | 0 |
| deep | 1 | 144610 | 0 | 53 | 795.3 | 25.8 | 181.833 | 184.7 | 27/33 | 0 |
| deep | 2 | 144618 | 0 | 46 | 795.5 | 23.5 | 181.8 | 184.5 | 21/29 | 0 |
| solve | 1 | 369 | 0 | 1502 | 86.9 | 32.7 | 4.246 | 72.2 | 971/1355 | unavailable |


**A2-多模态CPU-系统内存 (CPU vision)**

| Cell | Run | Prompt tok | Reused tok | Generated tok | Prompt tok/s | Decode tok/s | TTFT s | Latency s | Draft accepted | cache_n |
|---|---|---|---|---|---|---|---|---|---|---|
| short | 1 | 25 | 0 | 18 | 38.8 | 22.1 | 0.644 | 1.5 | 10/17 | 0 |
| short | 2 | 25 | 0 | 18 | 39.9 | 30.1 | 0.626 | 1.2 | 11/15 | 0 |
| short | 3 | 25 | 0 | 18 | 43.1 | 26.7 | 0.58 | 1.3 | 13/17 | 0 |
| long | 1 | 10381 | 0 | 62 | 594.6 | 23.6 | 17.458 | 20.1 | 31/51 | 0 |
| long | 2 | 10381 | 0 | 61 | 630.9 | 30.0 | 16.455 | 18.5 | 31/47 | 0 |
| deep | 1 | 144610 | 0 | 58 | 806.6 | 26.9 | 179.279 | 182.1 | 29/33 | 0 |
| solve | 1 | 369 | 0 | 1615 | 88.8 | 34.8 | 4.155 | 51.2 | 1115/1345 | unavailable |


### Engine log counters per run

**A1-多模态GPU (GPU vision)**

| Cell | Run | KV VRAM hit % | KV RAM read MiB | GPU expert hits | RAM blobs | File blobs | File read MB | Resident RAM GiB |
|---|---|---|---|---|---|---|---|---|
| short | 1 | 96.13 | 0.6 | 1436 | 12956 | 0 | 6496.2 | 65.68 |
| short | 2 | 96.28 | 0.5 | 3850 | 23825 | 0 | 6496.2 | 65.68 |
| short | 3 | 96.31 | 0.5 | 4401 | 34201 | 0 | 6496.2 | 65.68 |
| long | 1 | 95.41 | 101.4 | 16973 | 98973 | 1371 | 19291.6 | 65.68 |
| long | 2 | 95.49 | 101.9 | 26064 | 158077 | 2742 | 32163.9 | 65.68 |
| long | 3 | 95.36 | 101.4 | 13522 | 66406 | 1371 | 19250.8 | 65.68 |
| long | 4 | 95.6 | 101.5 | 25417 | 127157 | 2742 | 32123.1 | 65.68 |
| deep | 1 | 87.24 | 212.2 | 13733 | 577431 | 4113 | 113688.5 | 65.68 |
| deep | 2 | 87.7 | 189.3 | 12842 | 545546 | 4113 | 113647.6 | 65.68 |
| solve | 1 | 99.91 | 22.6 | 543750 | 913510 | 4315 | 114794.7 | 65.68 |

**A2-多模态CPU-系统内存 (CPU vision)**

| Cell | Run | KV VRAM hit % | KV RAM read MiB | GPU expert hits | RAM blobs | File blobs | File read MB | Resident RAM GiB |
|---|---|---|---|---|---|---|---|---|
| short | 1 | 96.42 | 0.5 | 1887 | 12434 | 0 | 8279.7 | 64.02 |
| short | 2 | 96.07 | 0.5 | 3870 | 22372 | 0 | 8279.7 | 64.02 |
| short | 3 | 96.31 | 0.5 | 4817 | 32327 | 0 | 8279.7 | 64.02 |
| long | 1 | 95.45 | 101.6 | 18540 | 95073 | 1369 | 21077.1 | 64.02 |
| long | 2 | 95.22 | 100.9 | 26348 | 150506 | 2738 | 33955.2 | 64.02 |
| deep | 1 | 87.53 | 216.6 | 16520 | 558621 | 4107 | 115516.7 | 64.02 |
| solve | 1 | 99.91 | 24.0 | 507523 | 907161 | 4309 | 116631.4 | 64.02 |

### Vision encode A/B (`max_tokens=1`)

| Mode | Image tokens | 1st-call wall s | 1st prompt_n | 1st prefill tok/s | Prompt-prefill part s | Implied encode s (upper bound) | 2nd-call wall s (embedding hit) |
|---|---|---|---|---|---|---|---|
| GPU | 273 | 3.92 | 280 | 74.1 | 3.779 | 0.141 | 0.177 |
| CPU | 273 | 5.398 | 280 | 62.8 | 4.459 | 0.939 | 0.172 |

> The helper's original `encode_s_est = first wall - second wall` is **not** used: the second call hits the prompt cache (`cache_n=273`), so the difference mixes in the first-time prefill of the image tokens. Here: `prefill part = first prompt_n / first prefill tok/s`, `implied encode = first wall - prefill part`. That value is an **upper bound**, not a direct measurement.

### Single-variable sweep (ctx=8192, no vision), sorted by decode

| # | Config | ctx | Load s | Decode tok/s | Prompt tok/s | Accept % | Expert slots | VRAM free MiB | Arena MiB | GPU hits | File blobs | File MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 29 | S2-256k-best | 262144 | 74 | 32.85 | 237.05 | 80.5 | 2042 | 328 | 73450 |  |  |  |
| 3 | S0c-arena | 8192 | 60 | 29.45 | 247.3 | 54.7 | 2809 | 314 | 73450 |  |  |  |
| 4 | S0d-arena-shm | 8192 | 88 | 29.45 | 244.65 | 54.7 | 2809 | 314 | 73450 |  |  |  |
| 28 | S0e-arena-shm-2nd | 8192 | 80 | 29.45 | 248.05 | 54.7 | 2809 | 314 | 73450 |  |  |  |
| 15 | S1-spec2 | 8192 | 76 | 26.05 | 245.6 | 73.25 | 2857 | 202 | 64921 | 20935 | 794 | 13563.4 |
| 18 | S1-shortread0 | 8192 | 77 | 25.8 | 236.05 | 62.05 | 2856 | 190 | 64924 | 20505 | 794 | 13751.3 |
| 5 | S1-base | 8192 | 309 | 25.35 | 241.35 | 62.2 | 2856 | 190 | 64924 | 22068 | 794 | 13560.3 |
| 9 | S1-pool112 | 8192 | 76 | 25.25 | 246.6 | 62.2 | 2856 | 190 | 64924 | 22068 | 794 | 13560.3 |
| 22 | S1-affinity-auto | 8192 | 76 | 25.1 | 247.15 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 12 | S1-ple-mmap | 8192 | 93 | 25.05 | 245.9 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 13 | S1-prefill16k | 8192 | 77 | 25.05 | 246.3 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 14 | S1-prefill32k | 8192 | 76 | 25.05 | 247.3 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 19 | S1-shortread128 | 8192 | 77 | 25.05 | 247.1 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 20 | S1-kvres20480 | 8192 | 76 | 25.05 | 247 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 21 | S1-stager96 | 8192 | 76 | 25.05 | 244 | 58.2 | 2856 | 188 | 64924 | 24024 | 794 | 13560.3 |
| 23 | S1-nohostworker | 8192 | 77 | 25.05 | 247.2 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 11 | S1-ple-ram | 8192 | 181 | 25 | 245.15 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 8 | S1-pool96 | 8192 | 77 | 24.95 | 247.5 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 7 | S1-pool48 | 8192 | 81 | 24.7 | 242.25 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 1 | S0a-budget71 | 8192 | 84 | 24.7 | 241.9 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 17 | S1-suffix0 | 8192 | 76 | 24.6 | 245.85 | 56 | 2857 | 202 | 64921 | 24405 | 794 | 13563.4 |
| 6 | S1-pool32 | 8192 | 101 | 24.55 | 240.8 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 27 | S1-vramres1400 | 8192 | 101 | 22.95 | 240.3 | 57.65 | 2623 | 882 | 65619 | 22931 | 794 | 12871.5 |
| 10 | S1-pool127 | 8192 | 76 | 22.85 | 246.8 | 62.2 | 2856 | 190 | 64924 | 22068 | 794 | 13560.3 |
| 16 | S1-spec8 | 8192 | 77 | 20.85 | 247.95 | 40.3 | 2856 | 178 | 64924 | 32745 | 794 | 13560.3 |
| 2 | S0b-mmap | 8192 | 12 | 7.5 | 116.7 | 58.35 | 2856 | 318 |  | 22883 | 0 | 249029 |

- **FAILED** `S1-expertcache-perlayer-3600` (ctx=8192, load=s): INFEASIBLE: needs 13.39 GiB for 3600 slots x 3993600 B but only 9.29 GiB VRAM free on 16 GB; engine refused to start (work/strata-sweep/S1-expertcache-perlayer-engine.log)
- **FAILED** `S1-expertcache-2400-perlayer` (ctx=8192, load=342s): FAILED at first request, not at startup: engine line 'strata serve: the prompt path borrows 1037 CUDA0 cache slots (3.86 GiB)' then 'strata serve: prefill gemm: cublasCreate: cuBLAS status 3' (CUBLAS_STATUS_ALLOC_FAILED, src/prefill/gemm.cu:310-311). Startup succeeded: 'expert cache 2400 slots, 8.93 GiB of VRAM', 'pre-filled 2400 of 2400 slots from the profile; slot 0 verified'. So this is neither a rejected slot count nor an invalid --expert-cache/--expert-profile combination; the prefill path over-commits VRAM. log work/strata-sweep/S1-expertcache-2400-perlayer-engine.log

## Correctness and limitations

- Image check: a geometry figure was given with four stated conditions and asked for the answer. Both vision arms read all four conditions (4/4) and produced the correct final answer; GPU vision used 2,048 generated tokens (hit the cap), CPU vision stopped at 1,615 and answered in 51.2 s.
- Cells with fewer than three runs: both arms' image-solve (1), the GPU deep cell (2), and the CPU deep and long cells (1 and 2). These are stated rather than padded.
- The engine log counters report 22,500 of 24,576 experts pinned in 65.68 GiB of RAM and 2,076 in the GPU expert cache, but **only the short prompt ran with zero file reads** (`files 0 blobs`); the 10,381-token prompt accumulated `files 1371 / 2742 blobs`, and the 144,610-token prompt `files 4113 blobs / 113,688.5 MB`. Resident-by-profile does not mean zero file reads at long context.
- KV streaming VRAM hit rate varies by request shape: short 96.13-96.31%, long 95.41-95.49%, **deep 144k prompt 87.24%**, and a 1,502-token generation 99.91%.
- The engine log counters (expert tiers, file reads) are cumulative-since-start figures for the file tier; per-request values are listed in the tables where the engine reported them.
- No needle/recall test, no concurrent-request load test, and no power-draw capture was done.
- Actual GPU power draw, memory DIMM topology/bandwidth, streaming TTFT, and the model revision are **unavailable**.

