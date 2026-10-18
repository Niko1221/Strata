# Community benchmark on NVIDIA GeForce RTX 4060 Laptop GPU

Measured on 2026-10-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB VRAM
- CPU: Intel(R) Core(TM) i7-14700HX (20 cores, 28 threads); installed RAM: 31.7 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 580.97; CUDA: 13.0
- Strata: project 0.1.45, commit 45af0dac; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-q2_0
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack q2_0 --native Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --prompt-cache-every 8192 --kv q4_0 --kv-resident 20480 --vram-reserve-mib 200 --vision --resident-experts --pool-workers 19
```

Environment of the config: `STRATA_PCIE_AUTO_CANON=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 30.5 (22.9-44.1) | 1.10 (0.92-1.87) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 220.1 (217.7-220.6) | not measured | 18.33 (18.30-18.54) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 220.4 (215.8-225.6) | not measured | 72.97 (71.28-74.52) |
| agent session, follow-up turns | 14530 (14491-14547) at the last turn | see below | up to 64 | 3 | see below | 23.1 (7.3-33.7) | 16.21 (13.63-17.67) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.5-68.8); earlier-turn tokens read again per session: 73 (3-74) (earlier turns that the cache did not hold; the 13719 (13717-13719) new tokens of the files are read once); first turn (nothing cached) first token: 10.60 s (10.10-11.09).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 61.7 | 10.104 | 20.1 |
| 1 | 3748 | 626 | 3122 | 196.0 | 16.167 | 19.0 |
| 2 | 5281 | 3811 | 1470 | 106.8 | 13.957 | 17.3 |
| 3 | 8714 | 5345 | 3369 | 193.3 | 17.674 | 27.2 |
| 4 | 11634 | 8772 | 2862 | 175.2 | 16.471 | 33.7 |
| 5 | 14547 | 11648 | 2899 | 182.7 | 15.951 | 17.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 636 s.

## Notes from the author

**Build.** 0.1.45 with two local patches that are not on upstream: three SSE data-event heartbeats in
`serve/server.py` (a long agent turn otherwise hits a client's five-minute idle timeout; server side only,
the engine's own timings are unaffected), and `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared` in `setup.py` (build
time only). The commit above, `45af0dac`, is that local merge, so it is not an upstream SHA.
`tools/strata_bench.py` also needed two local fixes to run here (the `port` NameError, and the
`CMakeLists.txt` encoding on a Windows whose ANSI code page is not UTF-8); those are filed as a separate
pull request and neither touches a measurement.

**Not tested in the suite.** Long prompts above 16K (`tools/strata_bench.py` runs 4K and 16K), `--clients`
concurrency, vision, a layer split, a long soak, and any non-Windows host.

**Long context is exercised in daily use, though.** This install also runs an agent client against the
same Q2_0 server, and its engine log (ten days, 1,932 requests, thinking on and the agent client's own
sampling, so not the suite's protocol) goes far past 16K:

- the longest prompt carried **113,621 tokens** (112,781 reused from the conversation cache + 840 read)
  at 99.5 prompt tok/s, and generated 3,175 tokens at 24.5 tok/s decode
- 88 requests with a 64K-120K prompt: prompt median 47.0 tok/s, decode median 24.0 tok/s
- 27 requests above 96K: decode median 23.9 tok/s - no decay against the 21.2 tok/s median of the 1,072
  requests above 32K
- the longest single generation was 64,879 tokens (from a 649-token prompt) at 16.2 tok/s

So with 8 GiB of VRAM and 31.7 GiB of RAM the context is not what limits this machine: the KV cache keeps
20,480 of 131,072 cells per QSA layer in VRAM with about 0.84 GiB of it in pinned RAM, and the
conversation cache reuses most of a long prompt. What does limit decode is how much of the expert set fits
in RAM when the server starts - see the previous paragraph.

**One observation outside the suite.** Decode here is very sensitive to how much of the expert set fits in
RAM when the server starts, because the rest is read from the model files during decode. Same machine,
same model (Flash-Next IQ2_XS), same settings, two runs on the same day: RAM complement 10.53 GiB of the
31.03 GiB the VRAM cache does not hold (34%) gave a decode median of 15.5 tok/s (13.7-18.5 over 4
requests); a complement of 16.93 GiB (55%) gave 21.6 tok/s (17.4-24.8). The only difference was how much
RAM was free at startup (a clean boot instead of a session with other work loaded). `--max-context`,
`--kv-resident` and the expert cache were identical.
