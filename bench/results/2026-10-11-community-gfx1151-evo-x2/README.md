# Community benchmark on a GMKtec EVO-X2 (Ryzen AI Max+ 395, Radeon 8060S gfx1151, 128 GB)

Measured on 2026-10-11 by @renaudrenaud. Strata 0.1.41 built from source for gfx1151 (docs/STRIX_HALO.md), serving
Unsloth **UD-Q4_K_XL** with every expert in the unified memory, at a **262,144-token context**, plus ISTA-DASLab
**IQ2_XS** on the same box for comparison. Four configurations, 3 runs each on fresh prompts. Main limitation: one
prompt type (a synthetic word list followed by a story request), default sampling, one request at a time.

## Hardware and software

- GMKtec NucBox EVO-X2 (BIOS 1.05): Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB LPDDR5X (124 GiB seen by the
  OS, 512 MiB BIOS carve-out), Samsung 970 EVO Plus 1 TB NVMe. GTT pool 107 GiB (`ttm.pages_limit=28160000
  ttm.page_pool_size=28160000`, `iommu=pt`; not `amd_iommu=off`, and 5 GiB less GTT than the maintainers' 112 GiB).
- Ubuntu 26.04.1, kernel 7.0.0-34, Docker 29.5.2. Nothing installed on the host: build and runtime in containers.
- ROCm 7.14.1 TheRock gfx1151 tarball (SHA-256 `c40e8f2b…afb00`, as in docs/STRIX_HALO.md), hipBLASLt table
  `tools/hip/gfx1151-hipblaslt-100401.txt` (the engine prints `hipBLASLt tuning enabled (90 rows, gfx1151, version
  100401)`).
- Strata **v0.1.41 (`fb58e0d`)**, source build: `-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF
  -DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151`, ggml from `third_party/llama.cpp` at `3cf0325`, gcc 15.
  The image is built on Chainguard's Python images (Wolfi, glibc): [configs/Dockerfile.strix](configs/Dockerfile.strix)
  (comments in French). The image encoder (`strata-vision`) is built for the CPU. `ctest`: 61 of 62 pass, including
  `hip_prefill_hcd_exact_parity` and `hip_prefill_hipblaslt_gemm`; `ple_parity` fails only because its Q2_0 GGUF is
  not present.
- Background: a PostgreSQL database and Grafana stay up on the box (idle, a few GiB). Power: the BIOS profile gives
  the SoC about 120 W (see [Temperature and power](#temperature-and-power)).

## Model and configuration

- **UD-Q4_K_XL**: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), revision
  `38bb39e`, the 4 shards checked against the SHA-256 table of docs/UNSLOTH_Q4.md (all four match). Pack:
  `tools/iq_pack.py --compat-bf16` (1.4 GB; 195 projections rounded to BF16, max |err| 0.0144).
- **IQ2_XS**: ISTA-DASLab GSQ-RCO IQ2_XS, 2 shards (`92cee27a…`, `316b46f3…`), pack without `--compat-bf16`.
- MTP: the base model's draft layer (`mtp_fetch` / `mtp_pack --experts q2_0` / `mtp_rt`), default draft vocabulary.
- Vision: on (`--vision`, CPU encoder, 16 threads, ISTA-DASLab BF16 mmproj), not used by the benchmark.
- **Experts: `--mmap-experts` and no `--resident-budget-gib`.** On unified memory the RAM budget and the GPU expert
  cache are two copies in the same pool (the 2026-10-05 gfx1151 community report shows 31.64 GiB loaded + 31.64 GiB
  cached for Q2_0); with UD-Q4_K_XL's 71.7 GiB of experts both would not fit. With `--mmap-experts` alone the experts
  are read in place from the GGUFs and copied once into the cache: `expert cache auto: 101.74 GiB free … -> 24576
  slots`, `pre-filled 24576 of 24576 slots from the profile in 27.2 s`, start-up ~50 s.

| Config | Engine arguments (besides pack, native, profile, `--expert-cache auto`, `--spec 4 --spec-min-p 0.5`, `--mtp`, `--vision`) | Environment |
|---|---|---|
| **A** UD-Q4_K_XL, defaults | `--prefill auto --max-context 131072 --kv int8` ([json](configs/A-ud-q4_k_xl-defaults.json)) | none (the 18 gfx1151 defaults of section 4 switch themselves on) |
| **D** UD-Q4_K_XL, fast, KV int8 | `--prefill 16384 --max-context 262144 --kv int8 --lookup-chain 3 --mtp-q4 all --mtp-window 8192` ([json](configs/D-ud-q4_k_xl-kv-int8.json)) | [env-fast.env](configs/env-fast.env): section 5 of docs/STRIX_HALO.md + `STRATA_PF_FUSED_KQ=1` + `STRATA_PREFILL_STREAM_MIN=128` |
| **Q** UD-Q4_K_XL, fast, KV fp16 | as D with `--kv fp16` ([json](configs/Q-ud-q4_k_xl-kv-fp16.json)) | as D |
| **D** IQ2_XS, fast, KV int8 | as D without `--mtp-q4 all`, with `--ple-gguf` ([json](configs/D-iq2_xs-kv-int8.json)) | as D |

`--mtp-q4 all` stops the engine on the IQ2_XS pack (issue #1867), hence its absence there. Reasoning: `none`
(`reasoning_effort` in each request). Sampling: the server's defaults. No calibration, no speed projection.

## Method

[bench.py](bench.py) (Python standard library) runs against the server's OpenAI API on the same machine:

- One warm-up request (~1,300 tokens, 64 generated) after each fresh engine start, excluded from the results.
- Then 3 runs of 4 prompt sizes (330 / 2,700 / 8,000 / 30,000 words, i.e. ~1.3K / 10.6K / 31K / 117K tokens; A only
  up to 31K). Each prompt starts with a nonce unique to its run and shifts its words, so **no prefix is reused**
  (`reused 0` in every engine line). Output cap 512 tokens; every request generated 512 (`finish=length`).
- Prompt and decode tok/s are the engine's own timing line (`strata serve: prompt N tokens = R reused + F read in X
  ms (… tok/s), G generated in Y ms (… tok/s), drafts accepted a of b`), read from the engine log after each
  request. TTFT is measured by the client (first non-empty content delta of the stream).
- Each configuration starts on a fresh engine process (`docker rm` + `run`); the expert cache is the profile fill
  (all 24,576 slots resident, so it does not adapt).
- Memory: `mem_info_gtt_used` and `MemAvailable` after start-up. Temperature, power and clock: amdgpu `hwmon` every
  10 s during the whole run ([runs/thermal.txt](runs/thermal.txt): time, temp in m°C, power in µW, sclk in Hz, GPU
  busy %, GTT used in bytes).

Per-run data: [runs/](runs/), one JSON line per request (French field names: `mots` = words, `prompt_tokens`,
`moteur_*` = the engine's line, `brouillons_acceptes / proposes` = drafts accepted / proposed, `ttft_s`,
`total_s`). [summarize.py](summarize.py) builds the table below from them.

## Results

| Configuration | Prompt tokens | Reused | Generated | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) | Drafts accepted |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| A UD-Q4_K_XL defaults | 1,366 | 0 | 512 | 3 | 306 (304-307) | 32.7 (31.9-33.8) | 4.5 (4.5-4.5) | 51% |
| A UD-Q4_K_XL defaults | 10,597-10,599 | 0 | 512 | 3 | 480 (480-482) | 31.6 (31.3-31.6) | 22.1 (22.0-22.1) | 50% |
| A UD-Q4_K_XL defaults | 31,167 | 0 | 512 | 3 | 510 (510-511) | 30.6 (29.0-30.6) | 61.2 (61.0-61.2) | 49% |
| **D UD-Q4_K_XL KV int8** | 1,330-1,332 | 0 | 512 | 3 | 752 (749-771) | 37.8 (35.0-38.3) | 1.8 (1.8-1.8) | 50% |
| **D UD-Q4_K_XL KV int8** | 10,629 | 0 | 512 | 3 | 1,173 (1,171-1,177) | 35.2 (35.0-36.2) | 9.1 (9.1-9.1) | 52% |
| **D UD-Q4_K_XL KV int8** | 31,169 | 0 | 512 | 3 | 1,197 (1,196-1,202) | 34.5 (34.2-34.9) | 26.1 (26.0-26.1) | 49% |
| **D UD-Q4_K_XL KV int8** | 116,750 | 0 | 512 | 3 | 1,151 (1,150-1,153) | 33.9 (33.2-34.5) | 101.5 (101.3-101.6) | 52% |
| Q UD-Q4_K_XL KV fp16 | 1,270-1,274 | 0 | 512 | 3 | 709 (704-720) | 35.2 (34.5-36.2) | 1.8 (1.8-1.8) | 50% |
| Q UD-Q4_K_XL KV fp16 | 10,587-10,589 | 0 | 512 | 3 | 947 (947-952) | 34.0 (33.7-34.5) | 11.2 (11.2-11.2) | 48% |
| Q UD-Q4_K_XL KV fp16 | 31,147 | 0 | 512 | 3 | 940 (940-948) | 33.6 (33.6-35.0) | 33.2 (32.9-33.2) | 50% |
| Q UD-Q4_K_XL KV fp16 | 116,750-116,752 | 0 | 512 | 3 | 900 (899-902) | 33.2 (33.0-33.2) | 129.9 (129.5-130.0) | 51% |
| D IQ2_XS KV int8 | 1,315-1,317 | 0 | 512 | 3 | 631 (630-646) | 41.4 (40.0-43.0) | 2.1 (2.1-2.1) | 51% |
| D IQ2_XS KV int8 | 10,516 | 0 | 512 | 3 | 1,118 (1,118-1,129) | 38.9 (38.7-40.6) | 9.4 (9.4-9.4) | 48% |
| D IQ2_XS KV int8 | 31,166 | 0 | 512 | 3 | 1,174 (1,174-1,177) | 39.4 (39.4-39.6) | 26.6 (26.5-26.6) | 50% |
| D IQ2_XS KV int8 | 116,747 | 0 | 512 | 3 | 1,137 (1,137-1,138) | 38.9 (38.3-39.8) | 102.7 (102.7-102.8) | 53% |

Memory after start-up (GTT used / `MemAvailable`): A 83 / 33 GiB, D 85 / 30 GiB, Q 89 / 27 GiB, IQ2_XS 42 / 75 GiB.
No paging, no out-of-memory, no failed request (45 measured requests + 4 warm-ups).

What the numbers say:

- The section 5 switches + `STRATA_PF_FUSED_KQ=1` are what make UD-Q4_K_XL's prompts fast here: **x2.3-2.5 prompt
  throughput** (A 306-510 -> D 752-1,197 tok/s), decode +10-15%.
- Prompt reading in D (1,151-1,197 tok/s from 10K to 117K) matches docs/STRIX_HALO.md (1,104-1,248). **Decode is
  33.9-37.8 tok/s, below the documented ~50 / ~43 tok/s at 8K / 128K.** The documented runs are greedy on other
  prompts; here sampling is on and about half of the drafts are accepted, which we take to be most of the gap
  (we did not run a greedy A/B).
- **fp16 KV** (Q) costs 19-22% of prompt throughput at 31K-117K against int8 (D), and about 1-3 tok/s of decode.
- **IQ2_XS vs UD-Q4_K_XL** on this box (same fast settings): decode +10-15% (38.9-41.4 vs 33.9-37.8), prompts about
  equal: the 2.4x smaller experts buy little speed on unified memory.

## Temperature and power

226 of the 250 samples have the GPU 90-100% busy. Over them: **86 °C median (55-93), 119 W median (95-131), sclk
2,765 MHz median (2,592-2,900)** of a 2,900 MHz maximum. Clocks stay within ~5% of the maximum most of the time;
93 °C was the peak. Ranges between the 3 runs of a cell stay within a few percent, so we saw no slow-down over the
42 minutes of the run.

## Correctness and limitations

- Checked by hand: an image question (the Firefox logo, recognized) and a tool call on `/v1/messages`
  (`tool_use` with the right argument). No needle test, no quality comparison against llama.cpp on this box.
- Not measured: greedy decoding, other prompt types (code, reasoning), concurrent requests (`parallel`), the
  section 5 switches one by one, `STRATA_HC_Q8` alone, UD-IQ4_XS, the default 112 GiB GTT and `amd_iommu=off`.
- The 4 sizes are synthetic word lists (`mot0 … mot996`) followed by "Écris une longue histoire en français sur un
  phare breton." (write a long story in French about a Breton lighthouse); the output is French prose.
