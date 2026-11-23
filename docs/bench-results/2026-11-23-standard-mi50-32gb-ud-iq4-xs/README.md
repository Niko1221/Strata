# Strata standard benchmark: AMD Instinct MI50 32 GB (gfx906); RX 5700 XT as display

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261123T101310Z-1e0f51fd`, started 2026-11-23T10:13:10Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: **AMD Instinct MI50 32 GB** (Vega 20, gfx906), PCI `0000:2f:00.0` — the compute card;
  AMD Radeon RX 5700 XT, PCI `0000:25:00.0` — display only (the runner lists both, 5700 first)
- CPU: AMD Ryzen 5 5600X 6-Core Processor; RAM: 62.7 GiB
- Storage of the model files: NVMe `XG7000-1TB` (953.9 GB); expert load at 2.89 GiB/s
- OS: Omarchy 7.2.15-1-omarchy; drivers: gpu-driver 7.2.15-1-omarchy; runtime: rocm 7.2.4
- Strata commit of the runner (Terminal CLI): `64524fb9fc82222ddce554c0643b64f056ec8366`; engine 0.1.65 from source at `c6f6427d...` (source build, see `provenance/BUILD.json`)
- Background work: MI50 32GB at 130W via LACT; RX 5700 XT drives the display; no other GPU workload; desktop session running

## Model and configuration

- Model: qwen3.8-flash-next; quantization UD-IQ4_XS; revision f5d175de2cf7f720d45d6064019a81fcf056e986; source revision c6f6427d9de9016fd92d13e29740f65625261df7
- Files: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf` `93ff23a08e5cd750...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf` `13d5dc8b93d039e4...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf` `f6b9b5934908a204...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf` `5de8b5dbf89409ac...`
- Track: **custom** (no notes given by the operator)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 7 min 8 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 41.4 (40.0-42.6) n=3 | 1.42 (1.35-1.47) n=3 | 3.11 (3.02-3.31) n=3 | 10.1 (8.8-10.3) n=3 | 0/3 |
| prompt-4k | 376.1 (374.8-383.9) n=3 | 40.0 (38.9-41.2) n=3 | 10.92 (10.70-10.96) n=3 | 19.09 (17.84-20.34) n=2 | 23.7 (23.1-24.1) n=3 | 3/3 |
| prompt-16k | 367.6 (367.2-370.2) n=3 | 40.7 (38.4-41.0) n=3 | 44.62 (44.31-44.67) n=3 | 52.72 (51.34-54.11) n=2 | 57.2 (56.8-58.0) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

Sanity checks: **12 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | pass | the expected JSON value |
| I2 | pass | the expected JSON value |
| I3 | pass | the expected JSON value |
| I4 | pass | the exact expected text |
| R1 | pass | the exact expected text |
| R2 | pass | the exact expected text |
| R3 | pass | the exact expected text |
| R4 | pass | the expected JSON value |
| T1 | pass | the expected tool call |
| T2 | pass | the expected tool call |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

## Corrections and supplements (added after generation)

**Which GPU did the work.** The compute GPU is an **AMD Instinct MI50 32 GB (Vega 20, gfx906)** at
PCI `0000:2f:00.0`, 60 CUs, HBM2 4096-bit, PCIe 4.0 x16. The RX 5700 XT (`0000:25:00.0`) is the
display adapter only; it drove the desktop during the run and the engine never used it
(`HIP_VISIBLE_DEVICES=1`, so the MI50 was device 0). `strata bench` lists every display adapter and
puts the 5700 XT first, hence the machine line above; that line is not the compute device.

**Which build ran.** The engine is a **source build**, not a release package: source commit
`c6f6427d9de9016fd92d13e29740f65625261df7` (2026-11-22), engine 0.1.65, binary
`sha256 4767692562180e5d865bc1f5311fcdabffccbb72b0d7f45841f61a96ce0e20de`. The
"Strata commit 64524fb9 / release build" line is the Terminal CLI that ran the benchmark, not the
engine. Build flags, the two host patches gfx906 needs on this distribution (GCC 16 / HIP
`__noinline__`, and `STRATA_BUILD_INKLING2=OFF`), and the pack/MTP/profile hashes are in
`provenance/BUILD.json` and `provenance/hashes.txt`.

**Machine facts the runner left null.**
- MI50 VRAM: 34,342,961,152 bytes (32 GiB). Power cap 130 W (LACT; card maximum 225 W), never changed
  during the run. Junction temperature 86-91 °C under sustained load.
- RAM: 62.7 GiB usable, 4 x 16 GiB DDR4-3600 (configured 3600 MT/s, 1.2 V), dual channel with 2 DIMMs
  per channel; measured triad 27.7 GB/s (6 threads, idle).
- Storage: NVMe `XG7000-1TB` (953.9 GB); loading the experts ran at 2.89 GiB/s.
- OS: Omarchy/Arch, kernel 7.2.15-1-omarchy, ROCm 7.2.4, AMD clang 22.0.0. hipBLASLt has no gfx906
  kernels and is not used; the rocBLAS gfx906 kernels are the distribution's.
- Full system facts: `provenance/system.txt` (CPU/cache, RAM, kernel cmdline, IOMMU, PCIe link,
  VBIOS, power/clock state, cooling, storage, packages). Exact reproduction commands:
  `provenance/commands.md`. MTP per-tensor manifest: `provenance/mtp-manifest.json`. Power-cap
  configuration: `provenance/lact-config.yaml`.

**Settings that ran** (`provenance/run-config.json`, engine argv):
`--max-context 131072 --kv int8 --kv-resident 32768 --expert-cache auto --vram-reserve-mib 1024
--prefill 4096 --spec 3 --spec-min-p 0.5 --resident-budget-gib 44`, UD-IQ4_XS in RAM-budget mode
(10,814 GPU expert slots, 24.5 GiB VRAM cache, 33.55 GiB page-locked complement, 96.8-97.8 % decode
hit rate), MTP draft layer with the Latin draft vocabulary, vision off.

The config asks for `--kv-resident 32768`, but the server's VRAM fit reduced it to **4,096 cells**
at start (the engine argv in `provenance/engine-log-excerpt.txt` shows it; 1,118 MiB stayed free).
With the 4-bit experts the KV window is the knob that gives: a variant with
`--vram-reserve-mib 512` kept 8,192 cells and measured 39.6 tok/s at a 32K prompt against 44.3 here,
so the 4,096-cell window is the faster setting, not a loss. Track is `custom` because the
engine was built from source; the operator notes are the power cap and the source build above.

**Supplementary long-context measurements** (not part of standard-v1; community script
`benchmark_mi50.py` and `needle_bench.py`, same machine/engine/quant, 256-token outputs):
- 128K config: decode **43.9 / 44.3 / 36.3** tok/s at 4K / 32K / 120K prompts; prefill
  **375.9 / 360.6 / 336.7** tok/s. Needle in a 107,568-token document at depths 10/50/90: **3/3**.
- 256K config (`--max-context 262144`, KV resident forced to 4,096 cells): 240,000-token prompt
  decode **35.63** tok/s, prefill **310.1** tok/s, TTFT **774 s**; needle at 203,262 tokens, depths
  50/90: **2/2**. At 32K this config decodes 39.1 tok/s (−12 % against 128K).
- Vision: tested separately (not in this run), GPU reader `strata-vision-hip`, OCR exact on a
  text image, 2.3-9.7 s per picture, ~10 % decode cost at 32K when enabled.

Raw extra data: `provenance/long-context.json`.
