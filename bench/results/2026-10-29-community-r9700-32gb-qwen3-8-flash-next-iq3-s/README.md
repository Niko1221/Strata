# Community benchmark on Radeon AI PRO R9700 32GB (gfx1201)

Measured on 2026-10-29 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon AI PRO R9700 32GB (gfx1201), 32 GiB VRAM, PCIe Gen4 x16
- CPU: AMD Ryzen Threadripper PRO 3975WX 32-Cores (32 cores, 64 threads); installed RAM: 247.3 GiB; storage: nvme0n1 Lexar SSD NM620 1TB        nvme; nvme1n1 SAMSUNG MZVLB1T0HBLR-000L7 nvme
- OS: Linux 7.0.0-42-generic; GPU driver: 7.1.3.31500000; CUDA (driver supports up to): not read
- Strata: project 0.1.53, engine 0.1.53; host checkout `46c152a1` (upstream main)
- Background workloads: a second GPU (Radeon Instinct MI50 32GB) is resident in a separate container running a coder model; it was idle during this suite. The engine under test was pinned to the R9700 with `HIP_VISIBLE_DEVICES=1`.

Note: the bench tool initially stamped this report as "Radeon Instinct MI50 32GB" at commit `14b121b5` — both values come from the engine-side hardware probe / bench checkout and are wrong on this dual-GPU host (see issue #4640). The hardware and commit lines above were corrected by hand against `rocm-smi`.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 88.9 (76.4-112.2) | 0.31 (0.28-0.34) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1407.9 (1405.4-1409.2) | not measured | 2.88 (2.88-2.88) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2406.7 (2404.9-2410.4) | not measured | 6.71 (6.70-6.71) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2415.7 (2412.3-2416.1) | not measured | 13.31 (13.31-13.33) |
| agent session, follow-up turns | 12937 (12926-12938) at the last turn | see below | up to 64 | 3 | see below | 93.4 (83.3-104.8) | 2.59 (2.34-2.81) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.5); earlier-turn tokens read again per session: 73 (4-74) (earlier turns that the cache did not hold; the 12063 (12062-12063) new tokens of the files are read once); first turn (nothing cached) first token: 1.80 s (1.75-1.81).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 353.4 | 1.747 | 59.9 |
| 1 | 3362 | 634 | 2728 | 1066.2 | 2.585 | 95.8 |
| 2 | 4895 | 3425 | 1470 | 635.5 | 2.34 | 94.5 |
| 3 | 8134 | 4954 | 3180 | 1181.8 | 2.719 | 85.7 |
| 4 | 10192 | 8187 | 2005 | 819.7 | 2.474 | 96.8 |
| 5 | 12926 | 10243 | 2683 | 1029.4 | 2.636 | 93.4 |

Per-request data: `results.json`.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server probe (see the correction note above), `rocm-smi` and the OS; background load is not recorded. Total time of the suite: 149 s.
