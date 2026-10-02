# Community benchmark on RTX 5090 + Ryzen 9 5950X (AVX2), Unsloth UD-Q4_K_XL and IQ3_S

Measured on 2026-10-01 and 2026-10-02 by [brenoperucchi](https://github.com/brenoperucchi). Engines 0.1.31 to 0.1.34
on the same machine and prompts, with UD-Q4_K_XL at two RAM budgets, plus a `--prefill auto:32768` A/B on UD-Q4_K_XL
and IQ3_S. Two findings:

- With every expert in VRAM + RAM, 0.1.32 to 0.1.34 read UD-Q4_K_XL prompts 4-10% slower than 0.1.31 here. Setting
  the prompt stager back to 0.1.31's values (`STRATA_STAGER_THREADS=4 STRATA_STAGER_RING=16`) recovers most of it.
- `--prefill auto:32768` reads the 14.7K-token prompt 35% faster than the default on 0.1.34 (2,623 vs 1,949 tok/s),
  which is also faster than 0.1.31. On IQ3_S it gives +17% at 14.7K and +31% at 28.9K tokens. Prompts of 2.7K do not
  change, and decode does not either.

Main limitation: one machine, 3 runs per cell, and the 40 GiB budget does not reproduce a 64 GB PC (see below).

## Hardware and software

- NVIDIA RTX 5090, 32 GB, 600 W limit, PCIe Gen 4 x16 (the CPU's maximum); the engine's startup probe read
  28.3 GB/s host-to-device. Single GPU, which also drives the display (the desktop takes about 320-360 MiB of VRAM).
- AMD Ryzen 9 5950X, 16 cores / 32 threads, AVX2 only (no AVX-512); the engine chose AVX2 and 15 expert-pool workers.
- 96 GB DDR4-3200 (2x32 + 2x16, dual channel); models on an NVMe SSD.
- Windows 11 (build 26200), NVIDIA driver 616.64.
- Engines 0.1.32, 0.1.33 and 0.1.34: release binaries ([BUILD-0.1.32.json](BUILD-0.1.32.json),
  [BUILD-0.1.33.json](BUILD-0.1.33.json), [BUILD-0.1.34.json](BUILD-0.1.34.json)), each with `serve/server.py`
  from its own tag's checkout. Engine 0.1.31: the release binary that setup installed, run with the 0.1.32
  `server.py`.
- Background: nothing else on the GPU; normal desktop use. No power limit changes.

## Model and configuration

UD-Q4_K_XL:
- The four `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf` shards from Unsloth (revision not recorded), packed by
  hand on 0.1.31 following `docs/UNSLOTH_Q4.md`; the same pack for every engine. No vision encoder.
- Context 32,768; KV int8, no KV streaming; expert cache auto (7,808 slots, 22.79 GiB of VRAM; 374-385 MiB of VRAM
  free with everything loaded); prefill auto unless noted; MTP on (`--spec 4 --spec-min-p 0.5`); `--pcie-frac` left
  to the engine (0.55); no calibration, no speed projection. Reasoning left at the model default.
- Budget 72 GiB (setup's default for 96 GB). The engine clamped it to 58.31 GiB ("62.31 GiB available minus 4 GiB
  headroom"), which still holds all 48.94 GiB of experts the GPU cache does not hold, so nothing is read from the files.
- Budget 40 GiB (setup's default for 64 GB): 40 GiB in RAM, the rest read from the GGUF. With 96 GB installed the OS
  file cache holds those reads, so this does not reproduce a 64 GB PC's SSD traffic.

```text
strata.exe --serve --pack <packs>\ud-q4_k_xl --native <UD-Q4_K_XL shard 1> --resident-budget-gib 72|40
  --expert-profile data\expert-profile.bin --expert-cache auto --prefill auto|auto:32768 --spec 4
  --spec-min-p 0.5 --mtp <mtp>\rt --max-context 32768 --kv int8
```

IQ3_S (0.1.34 only): ISTA-DASLab Flash-Next GSQ-RCO IQ3_S (2 shards + PLE), with the config this PC uses in
production: `--expert-cache auto --prefill auto|auto:32768 --spec 4 --spec-min-p 0.70 --mtp <mtp>\rt
--max-context 65536 --kv int8 --pcie-frac 0.55` and `STRATA_IQ_MT_MIN=1` in `env`.

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
  medium (a review instruction plus the first 9,000 bytes of `src/core/expert_source.cpp`), long (a summary
  instruction plus the first 50,000 bytes of the same file) and xlong (a summary instruction plus the first
  100,000 bytes of `src/program/generate.cpp`, IQ3_S only). Token counts below include the chat template and the
  unique first line.
- Memory: peak VRAM 31.4 GB in every UD-Q4_K_XL configuration (`nvidia-smi`, sampled every second on 0.1.32 and
  0.1.33). Host RAM was not measured usefully (the sampler counted the OS file cache).
- [runs/](runs/) has one JSON per configuration: `sizes.<prompt>.runs[]` holds each measured request (`timings` as
  returned by the server, `wall_s` in seconds at the client, `content_sha256` of the answer), `warmup[]` the warm-up
  requests, and `prompt_tps` / `decode_tps` the median, min and max in tokens/s. [logs/](logs/) has the engine log of
  each configuration. `logs/0132-b72.log` also holds an earlier, discarded pass where the prompts were reused.

## Results

UD-Q4_K_XL:

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
| 0.1.33, 72 GiB | short | 103 | 0 | 237, 148, 157 | 3 | 92.3 [85.5-94.0] | 66.0 [55.9-69.9] | not measured |
| 0.1.33, 72 GiB | medium | 2,684 | 0 | 256, 256, 256 | 3 | 923 [914-932] | 74.8 [60.3-80.4] | not measured |
| 0.1.33, 72 GiB | long | 14,689 | 0 | 256, 256, 256 | 3 | 2,021 [1,962-2,029] | 89.2 [83.5-90.1] | not measured |
| 0.1.34, 72 GiB | short | 99 | 0 | 155, 235, 235 | 3 | 86.2 [70.6-86.7] | 68.2 [55.1-72.7] | not measured |
| 0.1.34, 72 GiB | medium | 2,680 | 0 | 256, 256, 256 | 3 | 890 [862-894] | 73.7 [54.4-75.9] | not measured |
| 0.1.34, 72 GiB | long | 14,685 | 0 | 256, 256, 256 | 3 | 1,949 [1,942-2,005] | 74.6 [72.7-80.5] | not measured |
| 0.1.32, 72 GiB, old stager values | medium | 2,687 | 0 | 256, 256, 256 | 3 | 942 [931-945] | 70.7 [60.1-71.3] | not measured |
| 0.1.32, 72 GiB, old stager values | long | 14,692 | 0 | 256, 256, 256 | 3 | 2,116 [2,059-2,137] | 79.6 [72.0-81.0] | not measured |
| 0.1.33, 72 GiB, old stager values | medium | 2,687 | 0 | 256, 256, 256 | 3 | 947 [943-947] | 69.5 [57.5-72.7] | not measured |
| 0.1.33, 72 GiB, old stager values | long | 14,692 | 0 | 256, 256, 256 | 3 | 2,139 [2,122-2,141] | 87.5 [83.0-96.1] | not measured |
| 0.1.34, 72 GiB, old stager values | medium | 2,683 | 0 | 256, 256, 256 | 3 | 944 [941-952] | 70.5 [68.7-77.3] | not measured |
| 0.1.34, 72 GiB, old stager values | long | 14,688 | 0 | 256, 256, 256 | 3 | 2,133 [2,133-2,148] | 79.9 [78.5-81.8] | not measured |
| 0.1.34, 72 GiB, `--prefill auto:32768` | short | 103 | 0 | 210, 225, 229 | 3 | 85.5 [85.0-88.0] | 66.2 [62.0-71.8] | not measured |
| 0.1.34, 72 GiB, `--prefill auto:32768` | medium | 2,684 | 0 | 256, 256, 256 | 3 | 900 [881-912] | 67.0 [64.6-79.3] | not measured |
| 0.1.34, 72 GiB, `--prefill auto:32768` | long | 14,689 | 0 | 256, 256, 256 | 3 | 2,623 [2,527-2,680] | 79.9 [71.2-80.9] | not measured |
| 0.1.32, 40 GiB | short | 103 | 0 | 149, 160, 231 | 3 | 95.6 [83.1-100.3] | 54.8 [50.2-65.1] | not measured |
| 0.1.32, 40 GiB | medium | 2,684 | 0 | 256, 256, 256 | 3 | 797 [785-799] | 54.4 [54.1-59.8] | not measured |
| 0.1.32, 40 GiB | long | 14,689 | 0 | 256, 256, 256 | 3 | 1,839 [1,810-1,856] | 73.5 [72.3-79.5] | not measured |
| 0.1.33, 40 GiB | short | 103 | 0 | 148, 228, 105 | 3 | 94.8 [82.4-95.1] | 48.7 [43.8-70.9] | not measured |
| 0.1.33, 40 GiB | medium | 2,684 | 0 | 256, 256, 256 | 3 | 796 [794-808] | 62.8 [62.0-68.5] | not measured |
| 0.1.33, 40 GiB | long | 14,689 | 0 | 256, 256, 256 | 3 | 1,803 [1,797-1,804] | 75.6 [67.2-79.3] | not measured |
| 0.1.34, 40 GiB | short | 99 | 0 | 204, 228, 212 | 3 | 90.2 [83.7-91.0] | 55.2 [51.4-58.7] | not measured |
| 0.1.34, 40 GiB | medium | 2,680 | 0 | 256, 256, 256 | 3 | 803 [787-816] | 67.3 [63.4-67.5] | not measured |
| 0.1.34, 40 GiB | long | 14,685 | 0 | 256, 256, 256 | 3 | 1,844 [1,823-1,848] | 69.8 [64.3-74.3] | not measured |


IQ3_S:

| Configuration | Prompt | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median [range] | Decode tok/s median [range] | TTFT seconds |
| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| 0.1.34, IQ3_S, `--prefill auto` | medium | 2,680 | 0 | 256, 256, 256 | 3 | 2,331 [2,288-2,333] | 141.6 [140.1-157.1] | not measured |
| 0.1.34, IQ3_S, `--prefill auto` | long | 14,685 | 0 | 256, 256, 256 | 3 | 4,610 [4,604-4,612] | 156.0 [153.0-168.7] | not measured |
| 0.1.34, IQ3_S, `--prefill auto` | xlong | 28,883 | 0 | 256, 256, 256 | 3 | 4,682 [4,665-4,694] | 148.6 [144.0-148.8] | not measured |
| 0.1.34, IQ3_S, `--prefill auto:32768` | medium | 2,683 | 0 | 256, 256, 256 | 3 | 2,309 [2,285-2,338] | 140.4 [132.0-168.2] | not measured |
| 0.1.34, IQ3_S, `--prefill auto:32768` | long | 14,688 | 0 | 256, 256, 256 | 3 | 5,406 [5,381-5,421] | 148.5 [148.1-165.5] | not measured |
| 0.1.34, IQ3_S, `--prefill auto:32768` | xlong | 28,886 | 0 | 256, 256, 256 | 3 | 6,150 [6,149-6,151] | 142.0 [139.7-144.2] | not measured |

The short prompt's output length varied between runs, so its decode numbers are less comparable. Decode expert-cache
hit rate on the first UD-Q4_K_XL requests: 81-96%.

Prompt reading, UD-Q4_K_XL with all experts in memory (72 GiB), relative to 0.1.31: 0.1.32 is 6-8% slower on both
prompts (the second 0.1.32 server run reproduced it), 0.1.33 4-7% and 0.1.34 7-10%. With the stager at 0.1.31's
values (4 threads, 16 in flight), 0.1.32 to 0.1.34 stay within 2% of 0.1.31 on both prompts. The 0.1.32 notes describe
the larger stager (32 threads, 128 in flight) for the GGUF-in-place mode, measured on a 64 GB PC where a third of the
experts come from the SSD. When nothing comes from the SSD it seems to cost a little. Choosing it by whether the
budget holds every expert might keep both cases fast; we have not tried that.

`--prefill auto:32768` on 0.1.34: UD-Q4_K_XL reads the 14.7K prompt at 2,623 tok/s against 1,949 with `auto` (+35%)
and 2,161 on 0.1.31. IQ3_S goes from 4,610 to 5,406 at 14.7K (+17%) and from 4,682 to 6,150 at 28.9K (+31%), in line
with #440 on a 9950X3D (+21% at 32K, +35% at 128K). The 2.7K prompt is the same with either setting on both models.

Decode does not show a consistent difference between the engines or prefill settings; its run-to-run range is wide
(for example 72.4-95.3 tok/s on the long prompt in the second 0.1.32 run). No failed or cancelled requests.

## Correctness and limitations

No quality checks: answers were only hashed. One machine, one request at a time, greedy decoding, 3 runs per cell, one
72 GiB configuration for 0.1.31, `auto:32768` and IQ3_S on 0.1.34 only, Windows with the GPU also driving the display.
Time to first token was not measured separately; the engine's prompt time (`prompt_ms`) is in the run JSON.

Measurements and this write-up were put together with Claude Code on the owner's machine and checked by the owner.
