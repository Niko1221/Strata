# Community benchmark on NVIDIA GeForce RTX 5070 Ti

Measured on 2026-11-02 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti, 16 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 9 285K (24 cores, 24 threads); installed RAM: 127.5 GiB; storage: Samsung SSD 9100 PRO with Heatsink 4TB SSD/NVMe
- OS: Windows 11 (10.0.26200); GPU driver: 617.42; CUDA (driver supports up to): 13.4; engine built for CUDA 13.0
- Strata: project 0.1.55, commit fad04925 (source build for `sm_120` with PR #4951 AVX-VNNI `vpdpbusd` and PR #4952 Windows EcoQoS unthrottling); engine 0.1.55
- GPU detail (after the run, near idle): power_limit_w=300.00, power_draw_w=161.67, pcie=gen5 x16, pstate=P1, temp_c=57
- Background workloads: headless RTX 5070 Ti (`Disp.A: Off`, both displays connected to the motherboard Intel iGPU so `dwm.exe`/browser framebuffers use 0 MiB on the 5070 Ti); NVIDIA driver `CUDA - Sysmem Fallback Policy` set to `Prefer No Sysmem Fallback`; Windows `SeLockMemoryPrivilege` enabled (`2 MiB` large pages for the 63.28 GiB pinned expert arena)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-nvfp4-nvfp4-BF16
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack pack --native qwen-nvfp4-gptq-dense.gguf --native-dense-gguf qwen-nvfp4-gptq-dense.gguf --ple-gguf ple-fp8.gguf --embd-gguf token-embd-bf16.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:32768 --spec 5 --spec-min-p 0.70 --mtp rt --mtp-draft-vocab draft_vocab_en.bin --max-context 262144 --kv int8 --kv-resident 32768 --vram-reserve-mib 500 --adapt-every 1 --adapt-swaps 240 --adapt-decay 0.97 --adapt-async 1 --host-core last --vision --pool-workers 15 --pcie-frac 0.25 --expert-profile-save expert-profile.bin
```

Environment of the config: `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=7`, `STRATA_ADAPT_OVERLAP=1`, `STRATA_EXCHANGE_ROTATE=1`, `STRATA_SPEC_COUPLED=1`, `STRATA_PREFILL_NVFP4=w4a4`, `STRATA_PCIE_BALANCE=1`, `STRATA_PREFILL_ROUTED_STREAM=1`, `STRATA_SELECT_TWO_PASS=1`, `STRATA_CKPT_PARK_REUSE=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 87.2 (80.6-124.2) | 0.42 (0.39-0.50) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2717.0 (2667.6-2767.3) | not measured | 1.51 (1.48-1.55) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3309.3 (3285.8-3369.0) | not measured | 4.89 (4.80-4.92) |
| agent session, follow-up turns | 12939 (12930-12941) at the last turn | see below | up to 64 | 3 | see below | 111.0 (88.1-122.2) | 1.64 (1.48-1.73) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.5); earlier-turn tokens read again per session: 74 (3-74) (earlier turns that the cache did not hold; the 12062 (12061-12065) new tokens of the files are read once); first turn (nothing cached) first token: 1.20 s (1.13-1.25).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 548.0 | 1.133 | 68.4 |
| 1 | 3355 | 625 | 2730 | 1669.3 | 1.668 | 88.1 |
| 2 | 4888 | 3348 | 1540 | 961.7 | 1.634 | 95.8 |
| 3 | 8132 | 4952 | 3180 | 1864.3 | 1.733 | 114.4 |
| 4 | 10201 | 8195 | 2006 | 1345.4 | 1.527 | 102.1 |
| 5 | 12939 | 10256 | 2683 | 1672.7 | 1.639 | 122.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 77 s.
