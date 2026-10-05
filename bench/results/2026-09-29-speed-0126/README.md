# Speed with engine 0.1.26

RTX 5070 12 GB (PCIe 5.0 x16), Ryzen 5 7600, 64 GB DDR5-5200, Windows 10. The ready-made 0.1.26 engine, one-shot runs
with what setup writes: `--prefill auto`, the shipped expert profile, `--expert-cache auto`, 8-bit KV above 4K, KV
streaming (`--kv-resident 32768`) from 64K, MTP (`--spec 4 --spec-min-p 0.5`), greedy, 256 generated tokens, images
off, no timing marks. The same code-agent prompts as `2026-09-29-speed-0122`. Every run: [`matrix.json`](matrix.json).

## Prompt (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 536 | 1,299 | 2,171 | 2,126 | 2,107 |
| **IQ2_XS** | 534 | 1,256 | 2,092 | 1,754 | 1,752 |
| **IQ3_XXS** | 482 | 1,007 | 1,745 | 1,609 | 1,602 |
| **IQ3_S** | 427 | 913 | 1,624 | 1,640 | 1,443 |
| **Coder** | 656 | 1,583 | 2,177 | 2,236 | 2,208 |

## Output (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 87.3 | 93.0 | 81.8 | 76.2 | 73.7 |
| **IQ2_XS** | 79.6 | 78.6 | 76.3 | 63.7 | 62.7 |
| **IQ3_XXS** | 61.9 | 61.6 | 58.5 | 57.2 | 49.0 |
| **IQ3_S** | 52.4 | 53.3 | 48.3 | 46.3 | 45.5 |
| **Coder** | 58.9 | 55.1 | 54.9 | 53.2 | 43.0 |

- Against 0.1.22 (`2026-09-29-speed-0122`), prompts at 32K-128K are 8-28% faster: Q2_0 1,844 -> 2,171 at 32K and
  1,682 -> 2,107 at 128K, the Coder 1,779 -> 2,208 at 128K. The gains come from:
  - 0.1.24's QSA selection on tensor cores;
  - 0.1.25's grouping tables in mapped memory and fused norms;
  - 0.1.26's batched draft-layer pass.
- 262K was not run again (0.1.22: Q2_0 1,304, IQ2_XS 1,181 tokens/s).
- Output speed is the same decode as before. One run per cell moves several percent with the text (the share of
  accepted drafts, `spec_accept` in `matrix.json`). For example, the Coder's 128K answer accepted 55% of its drafts
  and its 64K answer 71%.


## KV per cell and True-Prefill (host extension)

Engine `0.1.26` ships with `KV=int8` chosen by `setup.py` whenever the
context fits 8-bit comfortably; on this README's host (RTX 5070 12 GB) the
baseline matrix above used `int8` for every cell. KV-streaming (`--kv-resident
32768`) moves the KV table to RAM from 64 K up, but does not change the kv
schema value — it's still `int8`.

Two distinct prefill metrics appear in the schema:
  - **`prefill_tok_s`** — engine-reported `timings.prompt_per_second` during a
    streamed call. Because `--prefill auto` reads prompts in chunks of up to
    8 192 tokens and overlaps decode with the next chunk, this measures
    *first-chunk-completion throughput, not wall-clock end-of-prompt.*
  - **`prefill_cold_tok_s`** — engine-reported `timings.prompt_per_second`
    during a non-stream `max_tokens=1` call after `POST /unload`. This is the
    true end-of-prompt throughput — the number the 2 000+ t/s README headlines
    measure. This row is `null` for all 25 baseline rows above (we did not
    re-measure on the RTX 5070 host); populated for the 8 new rows below.

The 8 new rows below were measured 2026-10-05 on a different host:

  - **host**: `RTX 4070 Ti SUPER 16 GiB / sm_89 / Linux 6.8.0-146 / CUDA 13.2 / NVIDIA 595.91.07`
  - **engine boot**: same `--prefill auto`, same `--expert-cache auto`, same
    `--prefill auto` defaults from setup.py.
  - **bottleneck**: piper-mamba's KV-streaming keeps int8 viable at all 4
    contexts for IQ2_XS; IQ3_XXS residents 47 GB and falls back to
    `kv=q4_0` at 128 K, `kv=k8v4` at 256 K to keep VRAM under 16 GiB.

Source-of-truth JSON dumps live at `bench/results/2026-10-01-ctx-ladder/<file>.json`,
plus per-row audit-trail markdown at `bench/results/2026-10-01-ctx-ladder/logs/`.

| Model | Context | KV | Prompt tokens | Decode TPS | Prefill (SSE) | Prefill (cold) | TTFT | Draft accept |
|---|:---:|---|---:|---:|---:|---:|---:|---:|
| IQ2_XS | 32k | int8 |  29,966 |     95.62 |         141.3 |        2694.8 |   183 ms |  75.8 % |
| IQ2_XS | 64k | int8 |  60,270 |     91.80 |         140.9 |        2678.0 |   271 ms |  74.5 % |
| IQ2_XS | 128k | int8 | 120,748 |     87.95 |         132.6 |        2670.4 |   422 ms |  76.9 % |
| IQ2_XS | 256k | int8 | 240,912 |     79.63 |         127.3 |        2483.9 |   726 ms |  69.8 % |
| IQ3_XXS | 32k | int8 |  29,966 |     73.32 |         116.5 |        2747.4 |   217 ms |  69.8 % |
| IQ3_XXS | 64k | int8 |  60,270 |     69.16 |         107.0 |        2544.8 |   295 ms |  70.3 % |
| IQ3_XXS | 128k | q4_0 | 120,748 |     65.51 |         105.2 |        2477.9 |   461 ms |  67.7 % |
| IQ3_XXS | 256k | k8v4 | 240,912 |     53.70 |          91.9 |        2403.9 |   768 ms |  66.6 % |

## Reproducing the per-host extension

```bash
# from /home/lfontanez/dev/strata (your working tree)
docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .          # sm_89 fat-binary
docker volume create strata-data
docker run -d --name strata --network host --gpus all --ulimit memlock=-1   --shm-size=4g -v strata-data:/data   -e FAMILY=qwen -e MODEL=<see src row> -e CONTEXT=<see tier>   -e VISION=no -e KV=<see kv> -e REINSTALL=1 -e HOST=127.0.0.1 -e PORT=8090   --restart unless-stopped strata

# 3 SSE runs + 1 cold-prefill probe per row -> matrix.json
python3 bench/results/2026-10-01-ctx-ladder/run.py \
  --model qwen3.8-flash-next-<model> \
  --contexts <tier-as-int> \
  --out bench/results/2026-10-01-ctx-ladder/<model>_<tier>.json
```

## Why the two prefill columns disagree

`-prefill auto` chooses chunk size up to 8 192 tokens; chunks are streamed
in  parallel, but multi-chunk overlap with decode means `prefill_tok_s`
(measured mid-stream) saturates early at ~100-150 t/s on this card — it
reflects the rate at which the first chunk completes, not the rate at
which the whole prompt lands in the engine's KV store.

The `prefill_cold_tok_s` column is what every prior 2026-09-29 entry is
*actually* measuring, if you read the engine README carefully. Setup runs
it once and divides `prompt_ms / prompt_n`; the 2026-10-05 Shin-BlackMamba
extension deliberately re-runs the probe with `max_tokens=1` after a `/
unload` so the answer cannot be confounded with a previous decode's KV reuse.


## KV per cell and True-Prefill (host extension)

Engine `0.1.26` ships with `KV=int8` chosen by `setup.py` whenever the
context fits 8-bit comfortably; on this README's host (RTX 5070 12 GB) the
baseline matrix above used `int8` for every cell. KV-streaming (`--kv-resident
32768`) moves the KV table to RAM from 64 K up, but does not change the kv
schema value — it's still `int8`.

Two distinct prefill metrics appear in the schema:
  - **`prefill_tok_s`** — engine-reported `timings.prompt_per_second` during a
    streamed call. Because `--prefill auto` reads prompts in chunks of up to
    8 192 tokens and overlaps decode with the next chunk, this measures
    *first-chunk-completion throughput, not wall-clock end-of-prompt.*
  - **`prefill_cold_tok_s`** — engine-reported `timings.prompt_per_second`
    during a non-stream `max_tokens=1` call after `POST /unload`. This is the
    true end-of-prompt throughput — the number the 2 000+ t/s README headlines
    measure. This row is `null` for all 25 baseline rows above (we did not
    re-measure on the RTX 5070 host); populated for the 8 new rows below.

The 8 new rows below were measured 2026-10-05 on a different host:

  - **host**: `RTX 4070 Ti SUPER 16 GiB / sm_89 / Linux 6.8.0-146 / CUDA 13.2 / NVIDIA 595.91.07`
  - **engine boot**: same `--prefill auto`, same `--expert-cache auto`, same
    `--prefill auto` defaults from setup.py.
  - **bottleneck**: piper-mamba's KV-streaming keeps int8 viable at all 4
    contexts for IQ2_XS; IQ3_XXS residents 47 GB and falls back to
    `kv=q4_0` at 128 K, `kv=k8v4` at 256 K to keep VRAM under 16 GiB.

Source-of-truth JSON dumps live at `bench/results/2026-10-01-ctx-ladder/<file>.json`,
plus per-row audit-trail markdown at `bench/results/2026-10-01-ctx-ladder/logs/`.

| Model | Context | KV | Prompt tokens | Decode TPS | Prefill (SSE) | Prefill (cold) | TTFT | Draft accept |
|---|:---:|---|---:|---:|---:|---:|---:|---:|
| IQ2_XS | 32k | int8 |  29,966 |     95.62 |         141.3 |        2694.8 |   183 ms |  75.8 % |
| IQ2_XS | 64k | int8 |  60,270 |     91.80 |         140.9 |        2678.0 |   271 ms |  74.5 % |
| IQ2_XS | 128k | int8 | 120,748 |     87.95 |         132.6 |        2670.4 |   422 ms |  76.9 % |
| IQ2_XS | 256k | int8 | 240,912 |     79.63 |         127.3 |        2483.9 |   726 ms |  69.8 % |
| IQ3_XXS | 32k | int8 |  29,966 |     73.32 |         116.5 |        2747.4 |   217 ms |  69.8 % |
| IQ3_XXS | 64k | int8 |  60,270 |     69.16 |         107.0 |        2544.8 |   295 ms |  70.3 % |
| IQ3_XXS | 128k | q4_0 | 120,748 |     65.51 |         105.2 |        2477.9 |   461 ms |  67.7 % |
| IQ3_XXS | 256k | k8v4 | 240,912 |     53.70 |          91.9 |        2403.9 |   768 ms |  66.6 % |

## Reproducing the per-host extension

```bash
# from /home/lfontanez/dev/strata (your working tree)
docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .          # sm_89 fat-binary
docker volume create strata-data
docker run -d --name strata --network host --gpus all --ulimit memlock=-1   --shm-size=4g -v strata-data:/data   -e FAMILY=qwen -e MODEL=<see src row> -e CONTEXT=<see tier>   -e VISION=no -e KV=<see kv> -e REINSTALL=1 -e HOST=127.0.0.1 -e PORT=8090   --restart unless-stopped strata

# 3 SSE runs + 1 cold-prefill probe per row -> matrix.json
python3 bench/results/2026-10-01-ctx-ladder/run.py \
  --model qwen3.8-flash-next-<model> \
  --contexts <tier-as-int> \
  --out bench/results/2026-10-01-ctx-ladder/<model>_<tier>.json
```

## Why the two prefill columns disagree

`-prefill auto` chooses chunk size up to 8 192 tokens; chunks are streamed
in  parallel, but multi-chunk overlap with decode means `prefill_tok_s`
(measured mid-stream) saturates early at ~100-150 t/s on this card — it
reflects the rate at which the first chunk completes, not the rate at
which the whole prompt lands in the engine's KV store.

The `prefill_cold_tok_s` column is what every prior 2026-09-29 entry is
*actually* measuring, if you read the engine README carefully. Setup runs
it once and divides `prompt_ms / prompt_n`; the 2026-10-05 Shin-BlackMamba
extension deliberately re-runs the probe with `max_tokens=1` after a `/
unload` so the answer cannot be confounded with a previous decode's KV reuse.
