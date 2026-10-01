# Community benchmark on RTX 5090 + Ryzen 9 5950X (AVX2), Unsloth UD-Q4_K_XL

Measured on 2026-10-01 by [brenoperucchi](https://github.com/brenoperucchi). Engines 0.1.31, 0.1.32 and 0.1.33 on
the same machine, model and prompts, with UD-Q4_K_XL and two RAM budgets. The main finding: with every expert in
VRAM + RAM, 0.1.32 and 0.1.33 read prompts 4-8% slower than 0.1.31 here, and setting the prompt stager back to
0.1.31's values (`STRATA_STAGER_THREADS=4 STRATA_STAGER_RING=16`) recovers most of it. Main limitation: one machine,
3 runs per cell, and the 40 GiB budget does not reproduce a 64 GB PC (see below).

## Hardware and software

- NVIDIA RTX 5090, 32 GB, 600 W limit, PCIe Gen 4 x16 (the CPU's maximum); the engine's startup probe read
  28.3 GB/s host-to-device. Single GPU, which also drives the display.
- AMD Ryzen 9 5950X, 16 cores / 32 threads, AVX2 only (no AVX-512); the engine chose AVX2 and 15 expert-pool workers.
- 96 GB DDR4-3200 (2x32 + 2x16, dual channel); models on an NVMe SSD.
- Windows 11 (build 26200), NVIDIA driver 616.64.
- Engines 0.1.32 and 0.1.33: release binaries ([BUILD-0.1.32.json](BUILD-0.1.32.json),
  [BUILD-0.1.33.json](BUILD-0.1.33.json)), each with `serve/server.py` from its own tag's checkout. Engine 0.1.31:
  the release binary that setup installed, run with the 0.1.32 `server.py`.
- Background: nothing else on the GPU; normal desktop use. No power limit changes.

## Model and configuration

- Unsloth UD-Q4_K_XL, the four `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf` shards (revision not recorded),
  packed by hand on 0.1.31 following `docs/UNSLOTH_Q4.md`; the same pack for every engine. No vision encoder.
- Context 32,768; KV int8, no KV streaming; expert cache auto (7,808 slots, 22.79 GiB of VRAM; 374-385 MiB of VRAM
  free with everything loaded); prefill auto; MTP on (`--spec 4 --spec-min-p 0.5`); no calibration, no speed
  projection. Reasoning left at the model default.
- Budget 72 GiB (setup's default for 96 GB). The engine clamped it to 58.31 GiB ("62.31 GiB available minus 4 GiB
  headroom"), which still holds all 48.94 GiB of experts the GPU cache does not hold, so nothing is read from the files.
- Budget 40 GiB (setup's default for 64 GB): 40 GiB in RAM, the rest read from the GGUF. With 96 GB installed the OS
  file cache holds those reads, so this does not reproduce a 64 GB PC's SSD traffic.

```text
strata.exe --serve --pack <packs>\ud-q4_k_xl --native <UD-Q4_K_XL shard 1> --resident-budget-gib 72|40
  --expert-profile data\expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
  --mtp <mtp>\rt --max-context 32768 --kv int8
```

Every server config is in [configs/](configs/) (paths as on this PC). The `stager4` configs add
`STRATA_STAGER_THREADS=4` and `STRATA_STAGER_RING=16` to `env`.

## Method

- [run_bench.py](run_bench.py) sends `/v1/chat/completions` requests (temperature 0, seed 42, `cache_prompt: false`,
  256-token output cap), one at a time: 1 warm-up and 3 measured runs per prompt, one server process per
  configuration. Throughput is the engine's own `timings` (`prompt_per_second`, `predicted_per_second`).
- The server reuses a matching prefix even with `cache_prompt: false`, so every request starts with a unique first
  line and each run reads its whole prompt fresh (`cache_n` is 0 in every measured run). Model loading is not timed.
  The expert cache warms up across the warm-up and measured runs of each configuration.
- [prompts/](prompts/) are built from this repo at tag `v0.1.32`, with hashes in `SHA256SUMS`: short (one sentence),
  medium (a review instruction plus the first 9,000 bytes of `src/core/expert_source.cpp`) and long (a summary
  instruction plus the first 50,000 bytes of the same file). Token counts below include the chat template and the
  unique first line.
- Memory: peak VRAM 31.4 GB in every configuration (`nvidia-smi`, sampled every second). Host RAM was not measured
  usefully (the sampler counted the OS file cache).
- [runs/](runs/) has one JSON per configuration: `sizes.<prompt>.runs[]` holds each measured request (`timings` as
  returned by the server, `wall_s` in seconds at the client, `content_sha256` of the answer), `warmup[]` the warm-up
  requests, and `prompt_tps` / `decode_tps` the median, min and max in tokens/s. [logs/](logs/) has the engine log of
  each configuration. `logs/0132-b72.log` also holds an earlier, discarded pass where the prompts were reused.

## Results

| Configuration | Prompt | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median [range] | Decode tok/s median [range] | TTFT seconds |
| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| 0.1.31, 72 GiB | short | 103 | 0 | 227, 109, 237 | 3 | 88.0 [88.0-94.9] | 71.0 [61.8-71.8] | not measured |
| 0.1.31, 72 GiB | medium | 2,684 | 0 | 256, 256, 256 | 3 | 961 [960-982] | 75.2 [62.7-76.2] | not measured |
| 0.1.31, 72 GiB | long | 14,689 | 0 | 256, 256, 256 | 3 | 2,161 [2,156-2,167] | 88.3 [81.8-93.8] | not measured |
| 0.1.32, 72 GiB | short | 103 | 0 | 168, 156, 237 | 3 | 103.3 [99.5-109.7] | 68.9 [67.8-76.4] | not measured |
| 0.1.32, 72 GiB | medium | 2,684 | 0 | 256, 256, 256 | 3 | 902 [890-912] | 64.8 [61.4-66.0] | not measured |
| 0.1.32, 72 GiB | long | 14,689 | 0 | 256, 256, 256 | 3 | 1,992 [1,980-2,036] | 91.6 [87.5-92.0] | not measured |
| 0.1.32, 72 GiB, second server run | short | 105 | 0 | 227, 154, 159 | 3 | 86.6 [81.1-91.9] | 61.3 [59.5-72.6] | not measured |
| 0.1.32, 72 GiB, second server run | medium | 2,686 | 0 | 256, 256, 256 | 3 | 884 [853-887] | 72.9 [64.0-76.6] | not measured |
| 0.1.32, 72 GiB, second server run | long | 14,691 | 0 | 256, 256, 256 | 3 | 1,997 [1,972-1,998] | 87.3 [72.4-95.3] | not measured |
| 0.1.32, 72 GiB, old stager values | medium | 2,687 | 0 | 256, 256, 256 | 3 | 942 [931-945] | 70.7 [60.1-71.3] | not measured |
| 0.1.32, 72 GiB, old stager values | long | 14,692 | 0 | 256, 256, 256 | 3 | 2,116 [2,059-2,137] | 79.6 [72.0-81.0] | not measured |
| 0.1.32, 40 GiB | short | 103 | 0 | 149, 160, 231 | 3 | 95.6 [83.1-100.3] | 54.8 [50.2-65.1] | not measured |
| 0.1.32, 40 GiB | medium | 2,684 | 0 | 256, 256, 256 | 3 | 797 [785-799] | 54.4 [54.1-59.8] | not measured |
| 0.1.32, 40 GiB | long | 14,689 | 0 | 256, 256, 256 | 3 | 1,839 [1,810-1,856] | 73.5 [72.3-79.5] | not measured |
| 0.1.33, 72 GiB | short | 103 | 0 | 237, 148, 157 | 3 | 92.3 [85.5-94.0] | 66.0 [55.9-69.9] | not measured |
| 0.1.33, 72 GiB | medium | 2,684 | 0 | 256, 256, 256 | 3 | 923 [914-932] | 74.8 [60.3-80.4] | not measured |
| 0.1.33, 72 GiB | long | 14,689 | 0 | 256, 256, 256 | 3 | 2,021 [1,962-2,029] | 89.2 [83.5-90.1] | not measured |
| 0.1.33, 72 GiB, old stager values | medium | 2,687 | 0 | 256, 256, 256 | 3 | 947 [943-947] | 69.5 [57.5-72.7] | not measured |
| 0.1.33, 72 GiB, old stager values | long | 14,692 | 0 | 256, 256, 256 | 3 | 2,139 [2,122-2,141] | 87.5 [83.0-96.1] | not measured |
| 0.1.33, 40 GiB | short | 103 | 0 | 148, 228, 105 | 3 | 94.8 [82.4-95.1] | 48.7 [43.8-70.9] | not measured |
| 0.1.33, 40 GiB | medium | 2,684 | 0 | 256, 256, 256 | 3 | 796 [794-808] | 62.8 [62.0-68.5] | not measured |
| 0.1.33, 40 GiB | long | 14,689 | 0 | 256, 256, 256 | 3 | 1,803 [1,797-1,804] | 75.6 [67.2-79.3] | not measured |

The short prompt's output length varied between runs, so its decode numbers are less comparable. Decode expert-cache
hit rate on the first requests: 81-96%.

Prompt reading, all experts in memory (72 GiB), relative to 0.1.31: 0.1.32 is 6-8% slower on both the medium and the
long prompt (the second 0.1.32 server run reproduced it), and 0.1.33 is 4-7% slower. With the stager at 0.1.31's values
(4 threads, 16 in flight), both are within 2% of 0.1.31, and 0.1.33 matches it on the long prompt (2,139 vs 2,161).
The 0.1.32 notes describe the larger stager (32 threads, 128 in flight) for the GGUF-in-place mode, measured on a
64 GB PC where a third of the experts come from the SSD. When nothing comes from the SSD it seems to cost a little.
Choosing it by whether the budget holds every expert might keep both cases fast; we have not tried that.

Decode does not show a consistent difference between the three engines. No failed or cancelled requests.

## Correctness and limitations

No quality checks: answers were only hashed. One machine, one request at a time, greedy decoding, 3 runs per cell, one
72 GiB configuration for 0.1.31, Windows with the GPU also driving the display. Time to first token was not measured
separately; the engine's prompt time (`prompt_ms`) is in the run JSON.

Measurements and this write-up were put together with Claude Code on the owner's machine and checked by the owner.
