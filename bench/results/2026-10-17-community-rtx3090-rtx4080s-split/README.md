# Community benchmark on RTX 3090 + RTX 4080 SUPER (layer split): what the second card is doing decides whether the split pays

Measured on 2026-10-17 with `tools/strata_bench.py --long` (3 runs per test, medians with
ranges). Speed only: no correctness check was run.

The same two cards, the same engine and the same quantization are **6% slower** than the
3090 alone in one arm and **17-38% faster** in the other. The only differences between the
arms are whether another CUDA process held memory on the second card, and the
`--vram-reserve-later-mib` that left it room.

## Hardware and software

- GPUs: NVIDIA GeForce RTX 3090 (24 GiB) + NVIDIA GeForce RTX 4080 SUPER (16 GiB), 40 GiB total
  - both links run at **PCIe 4.0 x8** (the board splits x16 between the two slots)
  - **the 4080 SUPER drives the display** (Xorg, Cinnamon, browsers: ~2.3 GiB at idle).
    This cannot be removed without ending the desktop session, so it is present in every arm.
- CPU: Intel Core i9-14900KF (8P + 16E, 32 threads, **no AVX-512**); RAM 123.4 GiB; NVMe SSD
- OS: Linux 7.0.0-31-generic (Linux Mint 22.3); driver 595.84; CUDA 13.2
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
  - compiled from source: the releases carry `strata-windows-x64*.zip` only, so `setup.sh`
    on Linux 404s on `strata-linux-x64.zip` and falls back to compiling. CUDA 13.0 toolkit
    from conda-forge (this is Linux Mint, which `setup.sh` does not install the toolkit on).
- `ulimit -l unlimited`; everything ran in a bubblewrap sandbox (no effect on speed measured)

## Model and configuration

- Model and quantization: Qwen3.8-Flash-Next **Q2_0** (`qwen3.8-flash-next-q2_0`)
- Context 131072, KV int8, KV streaming on (`--kv-resident 32768`), `--expert-cache auto`,
  `--spec 4` with the MTP draft layer, `--pool-workers 15`, one request at a time
- Otherwise setup's own choices, unchanged; thinking off for the suite; temperature 0
- Layer split: `"gpu": [0, 1]`, `"layer_split": "auto"`, `"gpu_order": "as_given"`
  (3090 first, 4080 last)

### The four arms

| Arm | Cards | `--vram-reserve-mib` | `--vram-reserve-later-mib` | Other CUDA work on the 4080 |
| --- | --- | ---: | ---: | --- |
| 3090 alone | 3090 | 700 | - | n/a |
| 4080 alone | 4080 SUPER | 3072 | - | display only (~2.3 GiB) |
| **split, second card busy** | both | 700 | 3072 | display + an embedding server holding ~3.2 GiB |
| **split, second card free** | both | 700 | 1536 | display only (~2.3 GiB) |

The embedding server is a local `sentence-transformers` HTTP service (google/embeddinggemma-300m).
It was stopped for the "second card free" arm and for both single-card arms. No other GPU
work ran in any arm.

## Method

`./bench.sh --url http://127.0.0.1:8095 --long` against an already-running server; model
loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt
starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.
VRAM and host RAM were sampled every 5 s from outside the sandbox.

`--url` is used because `bench.sh` cannot start the server itself on current main
(`NameError: name 'port' is not defined`, reported as #2872).

## Results

Medians over 3 runs; ranges in `results.json`.

| Measure | 3090 alone | 4080 alone | split, **busy** 4080 | split, **free** 4080 | free split vs 3090 alone |
| --- | ---: | ---: | ---: | ---: | ---: |
| chat decode tok/s | 113.2 | 84.3 | 106.3 | **132.9** | **+17%** |
| chat first token (s) | 0.26 | 0.36 | 0.32 | **0.22** | **-15%** |
| 4K prompt tok/s | **2100** | 1379 | 1795 | 1852 | -12% |
| 16K prompt tok/s | 2460 | 2461 | 2228 | **2825** | **+15%** |
| 32K prompt tok/s | 2424 | 2427 | 2189 | **3347** | **+38%** |
| agent-session decode tok/s | 125.1 | 93.4 | 108.8 | **132.2** | +6% |
| agent turn first token (s) | 1.98 | 3.00 | 3.25 | **1.93** | -3% |
| peak VRAM, card 0 / card 1 (MiB) | 23783 / - | - / 13270 | 23751 / 13912 | 23751 / 14822 | |

### What the engine logged

| Arm | Split | Expert residency on the second card | Decode expert-cache hit rate (median of 40 log lines) |
| --- | --- | --- | ---: |
| split, busy 4080 | layers 0-31 (CUDA0), 32-47 (CUDA1) | expert cache 959 slots (1.23 GiB), **959 of its 8192 profiled pairs** | **93.0%** (81.9-97.3) |
| split, free 4080 | layers 0-28 (CUDA0), 29-47 (CUDA1) | **"82% of the experts resident"**, streamed ring 96 slots | **99.7%** (96.4-99.9) |

That is the mechanism: with 3.2 GiB of someone else's memory on the second card and a 3 GiB
reserve on top, the fixed costs of a split stage (its own dense weights, a ~1.5 GiB prompt
buffer, and on the last card the head and the MTP draft layer) leave almost nothing for
experts, so a third of the model falls back to the CPU pool. Free the card and `auto` also
moves the boundary (K=32 -> K=29).

### Notes on specific numbers

- **4K prompts are slower on the pair in both arms** (1852 and 1795 against 2100 single).
  A 4K prompt does not fill the pipeline, so the hand-off is overhead rather than overlap.
  The crossover is between 4K and 16K; not measured.
- **16K and 32K prompt reading is identical on the 3090 alone and the 4080 alone**
  (2460/2461 and 2424/2427) despite a 34% decode gap. Long prefill on one card appears
  bound by something both share here - both are on x8 links, same CPU and RAM. Not isolated.
- The 4080-alone arm carries a 3 GiB reserve for the display, so it is not a like-for-like
  single-card comparison with the 3090's 700 MiB.
- 32K prompt reading in the free split (3347 tok/s) is above the project's published
  RTX 5070 figure for Q2_0 (2650 tok/s).

## Files in this folder

| File | What it is |
| --- | --- |
| `results-3090-alone.json` | per-request data, 3090 alone |
| `results-4080-alone.json` | per-request data, 4080 SUPER alone |
| `results-split-second-card-busy.json` | per-request data, split with the embedding server on the 4080 |
| `results-split-second-card-free.json` | per-request data, split with the 4080 free |
| `engine-log-excerpt.txt` | the `layer split` and `decode expert cache hit rate` lines of both split arms; absolute paths replaced with `<strata>` / `<data>`, nothing else changed |

The JSON files are `tools/strata_bench.py`'s own output format, unchanged, one per arm.

## Correctness and limitations

Not measured: this is a speed suite. Background load is not otherwise recorded; power limits
are stock. The agent session uses the model's own replies, so token counts differ a little
between runs. Each arm is one suite run of 3 repetitions per test, not repeated installs, so
the ranges capture run-to-run spread, not restart-to-restart spread.

The attached `results.json` of each split arm reports `"gpu": ["NVIDIA GeForce RTX 3090"]`
and 24576 MiB of VRAM: the bench tool drops the second card when the server is reached with
`--url` (see the separate issue). The hardware above is what actually ran, and the engine
log excerpt shows the split.
