# Community benchmark on AMD Radeon RX 6600M

Measured on 2026-10-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 6600M, 8 GiB VRAM
- CPU: AMD Ryzen 7 5700G with Radeon Graphics (8 cores, 16 threads); installed RAM: 31.8 GiB; storage: not measured
- OS: Windows 10 (10.0.19045); GPU driver: 32.0.21045.1000; CUDA: not read
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
- Background workloads: none; the desktop runs on the integrated Radeon, the RX 6600M is dedicated to Strata. Power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --resident-experts --vram-reserve-mib 500
```

Environment of the config: `STRATA_RESIDENT_PIN=0`, `STRATA_PREFILL_CPU_SHARE=auto`, `STRATA_WDDM_BUDGET=0`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 17.0 (10.6-24.2) | 1.81 (1.63-2.83) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 216.8 (172.5-219.4) | not measured | 18.59 (18.38-23.38) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 219.9 (219.7-223.4) | not measured | 73.17 (71.95-73.22) |
| agent session, follow-up turns | 14567 (14540-14597) at the last turn | see below | up to 64 | 3 | see below | 16.3 (9.7-21.5) | 21.99 (11.70-30.69) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 75 (4-75) (earlier turns that the cache did not hold; the 13718 (13718-13719) new tokens of the files are read once); first turn (nothing cached) first token: 9.26 s (6.52-9.71).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 63.7 | 9.709 | 12.8 |
| 1 | 3758 | 635 | 3123 | 122.9 | 25.666 | 14.5 |
| 2 | 5291 | 3751 | 1540 | 130.4 | 11.944 | 16.0 |
| 3 | 8723 | 5354 | 3369 | 147.0 | 23.016 | 17.6 |
| 4 | 11642 | 8780 | 2862 | 133.3 | 21.649 | 19.3 |
| 5 | 14597 | 11698 | 2899 | 145.5 | 20.048 | 21.5 |

Per-request data: `results.json`. The suite ran against an already-running server, so there is no `engine-log-excerpt.txt`.

## Appendix: the WDDM budget and the two unusual settings

`STRATA_WDDM_BUDGET=0` and `--vram-reserve-mib 500` are not defaults; a same-day A/B on this machine chose them before this run.

HIP reports 8,176 MiB total for the card, while the WDDM budget DXGI reports for the process is 7,378 MiB, so by default every free figure is lowered by 798 MiB (the #380/#377 workaround; the engine logs it once per boot). On this machine the card drives no display (the desktop runs on the integrated Radeon), so the 798 MiB turned out to be spendable. Three arms, same model and context, one boot each:

| Arm | WDDM budget | `--vram-reserve-mib` | Expert cache | 3.2K prompt tok/s | 16-20K prompt tok/s | Decode tok/s | Task Manager dedicated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| counted (default) | 798 MiB | 350 | 546 slots (1.06 GiB) | 97.8 (a shorter prompt, earlier session) | 125.7 (20.5K) | 17.4 | 7.1 GB |
| off | 0 | 350 | 949 slots (1.83 GiB) | 135.6 | not run | 17.1 | 7.9 GB |
| off (this run) | 0 | 500 | 874 slots (1.69 GiB) | 136.7 | 168.5 (16.2K, single run) | 18.3-19.5 | 7.78 GB |

No stalled or failed request and no paging in any arm; Task Manager's dedicated figure stays flat after requests (shared ~2.7 GB is the pinned host tiers by design). This is the same situation as [#1846](https://github.com/Niko1221/Strata/issues/1846) (a Windows dGPU that drives no display): the withholding is real on a card that does drive a display — #380 measured decode 41 -> 30 tok/s when it was ignored — so `STRATA_WDDM_BUDGET=0` is machine-specific here, not a recommendation. 350 and 500 were close on speed; 500 keeps ~320 MiB of VRAM free instead of ~170, above the engine's low-VRAM warning. The appendix's prompt numbers are single runs and are not directly comparable with the 3-run medians above.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 780 s.
