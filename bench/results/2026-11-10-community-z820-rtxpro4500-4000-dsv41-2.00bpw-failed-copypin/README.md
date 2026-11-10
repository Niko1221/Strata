> **INVALID ARM — not a performance result.** The forced `DS4_COPY_PIN=1` made every
> long prompt fail with `cudaMemcpyAsync(... lay.trellis[2] ...): invalid argument`.
> See `FAILED-ARM.md`. The clean results are in
> `../2026-11-10-community-z820-rtxpro4500-4000-dsv41-2.00bpw-fill-6exp-0995-0512/`.

# Community benchmark on RTX PRO 4500 32 GB + RTX PRO 4000 24 GB in a dual Xeon E5-2697 v2 (HP Z820)

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX PRO 4500 Blackwell, 32 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2697 v2 @ 2.70GHz (24 cores, 48 threads); installed RAM: 251.8 GiB; storage: sda     SanDisk 3.2Gen1                 usb; sdb     CT1000BX500SSD1                 sas; sdc     CT2000BX500SSD1                 sas; sdd     Compact Flash                   usb; sde     SM/xD-Picture                   usb; sdf     SD/MMC                          usb; sdg     M.S./M.S.Pro/HG                 usb; sdh     SD/MMC/M.S.PRO                  usb; sr0     DVD A DH16ACSHR                 sas; zram0; nvme0n1 WD PC SN5000S SDEPNSJ-1T00-1006 nvme
- OS: Linux 7.0.14-101.fc43.x86_64; GPU driver: 580.178.04; CUDA (driver supports up to): 13.0
- Strata: project 0.1.59.1, commit 628f7495; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=200.00, power_draw_w=47.20, pcie=gen3 x16, pstate=P1, temp_c=45
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 65536 tokens; parallel slots: 1
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
| short chat | 32 (29-38) | 0 | 100 (100-100) | 9 | not measured | 28.7 (17.2-33.4) | 0.73 (0.53-1.41) |
| 4K prompt, fresh | not measured | not measured | 24 cap | 0 | not measured | not measured | not measured |
| 16K prompt, fresh | not measured | not measured | 24 cap | 0 | not measured | not measured | not measured |
| 32K prompt, fresh | not measured | not measured | 24 cap | 0 | not measured | not measured | not measured |
| agent session, follow-up turns | not measured at the last turn | see below | up to 64 | 0 | see below | not measured | not measured |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: not measured; earlier-turn tokens read again per session: 0 (0-0) (earlier turns that the cache did not hold; the 0 (0-0) new tokens of the files are read once); first turn (nothing cached) first token: not measured.

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |

### Skipped and failed

- failed 4K: {'type': 'server_error', 'message': 'cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs): invalid argument'}
- failed 4K: {'type': 'server_error', 'message': 'cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs): invalid argument'}
- failed 4K: {'type': 'server_error', 'message': 'cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs): invalid argument'}
- failed 16K: {'type': 'server_error', 'message': 'cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs): invalid argument'}
- failed 16K: {'type': 'server_error', 'message': 'cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs): invalid argument'}
- failed 16K: {'type': 'server_error', 'message': 'cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs): invalid argument'}
- failed agent: {'type': 'server_error', 'message': 'cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs): invalid argument'}
- failed agent: {'type': 'server_error', 'message': 'cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs): invalid argument'}
- failed agent: {'type': 'server_error', 'message': 'cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs): invalid argument'}
- failed 32K: {'type': 'server_error', 'message': 'cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs): invalid argument'}
- failed 32K: {'type': 'server_error', 'message': 'cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs): invalid argument'}
- failed 32K: {'type': 'server_error', 'message': 'cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs): invalid argument'}

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 48 s.
