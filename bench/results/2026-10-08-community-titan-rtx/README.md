# Community benchmark: NVIDIA TITAN RTX 24 GB, Xeon E5-2660 v3, 126 GiB RAM — IQ3_S at 262K (first single-TITAN-RTX row)

Measured on 2026-10-08 by victorgabr. One card, Strata engine 0.1.40.2, Flash-Next IQ3_S,
262,144-token context. A speed sweep at 3.6K/28.6K/115K prompt tokens with a fixed
256-token output cap, plus nine needle-in-a-haystack recall checks up to 250K prompt
tokens. The existing TITAN RTX entry (#389) is a 2-card NVLink box; this is the first
single-card measurement. Main limitation: one server session, expert cache already warm
(88-91% hit rate) during the speed sweep.

## Hardware and software

- NVIDIA TITAN RTX, 24,576 MiB VRAM, driver 580.178.04, CUDA 13.3 toolkit.
  PCIe link not measured (Haswell platform, nominally PCIe 3.0 x16).
- Intel Xeon E5-2660 v3 @ 2.60 GHz (10 cores / 20 threads, AVX2, no AVX-512), 1 NUMA node.
- 125.7 GiB RAM; about 67 GiB stayed available during the sweep.
- Models, packs and OS on a Sabrent 3.7 TB NVMe. Four HDDs present, unused by Strata.
- Manjaro Linux, kernel 6.1.183-1-MANJARO.
- Strata source checkout `5b9d3dd` (fork working branch), engine binary 0.1.40.2
  (installed by `setup.py`; upstream was already at 0.1.40.4 on the report date).
- Idle desktop; no other GPU users during the runs. Power limits not measured.

## Model and configuration

ISTA-DASLab `Qwen3.8-Flash-Next-GSQ-RCO-GGUF` **IQ3_S**, two shards; vision projector
enabled but not exercised. Full run configuration: [strata-iq3_s.json](strata-iq3_s.json)
(absolute paths replaced with `<STRATA>` / `<STRATA_DATA>` placeholders; nothing else changed).

Key settings as run: `--max-context 262144 --kv int8 --kv-resident 32768 --expert-cache auto
--prefill auto --spec 4 --spec-min-p 0.70 --pcie-frac 0.33 --vram-reserve-mib 700`.
Expert cache: 6,667 slots (17.43 GiB free at load minus 700 MiB reserve and 218 MiB draft
head). Steady-state VRAM: 23,659 of 24,576 MiB used. Server sampling defaults:
temperature 1.0, top_p 0.95, top_k 20; reasoning left at the server default, so generated
tokens include reasoning tokens.

## Method

- **Speed sweep** ([bench.py](bench.py), results in [runs.json](runs.json)): streamed
  `/v1/chat/completions`, `max_tokens=256`, three prompt sizes (3.6K / 28.6K / 115.4K
  actual tokens), three runs each. Prompts are built from a 12-block corpus in a seed-shuffled
  order with a unique header, so **no run reuses another's prefix** — every timed prefill is
  fully fresh (the engine line shows `0 reused` on all nine). One untimed 32K warm-up
  request preceded the sweep; the expert cache was already warm from the needle run below
  (hit rate 88-91% across the sweep). All `tok/s` figures are the engine's own
  `strata serve: prompt ...` lines; client-side TTFT and total latency are recorded separately.
  Engine lines in [log-excerpt.txt](log-excerpt.txt).
- **Recall** (`tools/needle_bench.py --lengths 32k,128k,256k --depths 10,50,90`, results in
  [needles.json](needles.json)), run on the same session before the speed sweep.

## Speed results

| Configuration | Actual prompt tokens | Reused | Generated | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) | Total s median |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| ~4K | 3,651 | 0 | 256 (cap hit) | 3 | 1,006 (1,001-1,007) | 61.4 (60.2-62.1) | 3.68 (3.68-3.70) | 7.9 |
| ~32K | 28,647 | 0 | 256 (cap hit) | 3 | 1,213 (1,210-1,224) | 56.1 (54.7-56.7) | 23.83 (23.62-23.91) | 28.4 |
| ~128K | 115,378 | 0 | 256 (cap hit) | 3 | 1,061 (1,055-1,064) | 53.4 (51.9-54.3) | 109.5 (109.3-110.1) | 114.4 |

Draft acceptance 71-100% (e.g. 144/184 at 32K, 123/163 at 128K). Every run finished at the
256-token cap (`finish_reason: length`). VRAM held at 23,659 MiB through all runs.

Contrast with the 2x TITAN RTX NVLink entry (#389, engine 0.1.31): 865 / 1,612 / 1,785
prompt and 68 / 60 / 62 decode at 4K/32K/128K. On different prompt lengths and a newer
engine, single-card decode sits in the same 52-62 band; one card reaches 1,061 tok/s at
128K versus 1,785 on the NVLink pair.

## Recall results

9 of 9 needles found (32K, 128K, 256K at depths 10/50/90). Fresh-prefill rates from the
engine lines: 1,275 tok/s at 32K (fully fresh), 1,052-1,061 at 128K (fully fresh), and
**862-865 at 250K** (234K fresh tokens, only the shared 16K first chunk reused). Short
9-10-token needle answers are not decode-throughput rows. 250K prompts fit the 262,144
context without paging.

## Failures and notes

- The first 32K/depth-10 needle attempt was cancelled mid-prefill
  (`prompt 32340 ... 1 checkpoints (cancelled)`, 0 generated); the successful retry reused
  that attempt's 16,384-token prefix, so the 32K/10 row is not a fresh prefill. The fresh
  32K rate is taken from the 32K/50 row (1,275 tok/s).
- Expert cache warm-up is visible and matters: hit rate rose 47% -> 85% over the needle
  run; a cold short-lived server will sit at the low end of the ranges above.
- `--pcie-frac 0.33 --spec-min-p 0.70` came from the installer's calibration; no manual
  sweep was run on this box.
- The server ran bound to `0.0.0.0` with no API key as recorded in the config — reproduced
  as measured, but anyone repeating this should bind `127.0.0.1` or set a key.
- Engine 0.1.40.2, not the then-current 0.1.40.4.

## Reproduce

```bash
./setup.sh   # Linux, picks IQ3_S for 126 GiB RAM; start the server
python tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k,256k --depths 10,50,90 --out needles.json
python bench/results/2026-10-08-community-titan-rtx/bench.py --warm --out runs.json
# tok/s come from the "strata serve: prompt ..." lines in the serve log
```
