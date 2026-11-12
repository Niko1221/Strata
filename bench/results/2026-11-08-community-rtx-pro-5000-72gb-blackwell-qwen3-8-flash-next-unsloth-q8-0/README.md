# Community benchmark on NVIDIA RTX PRO 5000 72GB Blackwell

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. The suite itself is speed only; a separate needle recall check was run afterwards (see "Correctness" at the end).

## Hardware and software

- GPU and VRAM: NVIDIA RTX PRO 5000 72GB Blackwell, 72 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 89.7 GiB; storage: sda     PSSD T9                  usb; nvme1n1 Samsung SSD 9100 PRO 4TB nvme; nvme0n1 Samsung SSD 9100 PRO 4TB nvme
- OS: Linux 6.8.0-147-generic; GPU driver: 595.99.02; CUDA (driver supports up to): 13.2
- Strata: project 0.1.58, commit 146df3c4; engine 0.1.58
- GPU detail (after the run, near idle): power_limit_w=300.00, power_draw_w=128.57, pcie=gen5 x8, pstate=P1, temp_c=76
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-q8_0
- Model repository and revision: `unsloth/Qwen3.8-Flash-Next-GGUF` @ `084c631ac221016019d6e0a863afe92d90edd3b0` (HEAD at run time; files downloaded ~2026-11-07); source `Qwen/Qwen3.8-Flash-Next` @ `812d96e10a0dd82b69d749e68693698cf6c36daa`
- GGUF filenames: `Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf` … `-00006-of-00006.gguf` (SHA-256 in the standard report's `results.json`); MTP draft `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`; no vision encoder
- Context: 262144 tokens; parallel slots: 2
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0
- Launch command (paths shortened; the same saved configuration the standard run used): `strata --serve --pack <models>/packs/unsloth-q8_0 --native <models>/unsloth-Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf --expert-profile <terminal>/data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <models>/mtp/rt --max-context 262144 --kv int8 --resident-budget-gib 66 --batch 2`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 54.8 (41.4-60.7) | 0.44 (0.33-0.57) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1162.8 (1153.5-1189.7) | not measured | 3.47 (3.40-3.50) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2297.2 (2287.8-2320.0) | not measured | 7.01 (6.94-7.04) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2452.7 (2443.1-2460.2) | not measured | 13.09 (13.05-13.14) |
| agent session, follow-up turns | 13591 (13540-13602) at the last turn | see below | up to 64 | 3 | see below | 97.1 (52.4-178.0) | 3.21 (3.03-3.39) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.8 % (69.7-69.8); earlier-turn tokens read again per session: 4 (2-74) (earlier turns that the cache did not hold; the 12724 (12723-12725) new tokens of the files are read once); first turn (nothing cached) first token: 2.42 s (2.34-2.42).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 262.7 | 2.338 | 39.2 |
| 1 | 3823 | 621 | 3202 | 972.5 | 3.315 | 52.5 |
| 2 | 5356 | 3886 | 1470 | 460.6 | 3.21 | 52.4 |
| 3 | 8600 | 5419 | 3181 | 943.9 | 3.39 | 73.0 |
| 4 | 10828 | 8663 | 2165 | 680.9 | 3.199 | 90.0 |
| 5 | 13591 | 10881 | 2710 | 837.3 | 3.256 | 68.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 166 s.

## Correctness (added by hand)

Needle recall with `tools/needle_bench.py` against the same server: 6 of 6 found at 32k and 128k prompt tokens, depths 10/50/90 percent (actual prompts 31,634 and 123,370 tokens; 13-52 s per request). Per-request data: `needles.json`. A needle test measures recall on those inputs, not overall model quality.

The suite ran against the production server (the same saved configuration) freshly loaded ten minutes earlier; the GPU embedding/reranking servers were stopped while measuring.
