# Community benchmark on Tesla V100-PCIE-32GB + Tesla V100-PCIE-32GB

Measured on 2026-11-08 by thegliffy with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla V100-PCIE-32GB + Tesla V100-PCIE-32GB, 64 GiB VRAM
- CPU: Intel(R) Xeon(R) Platinum 8168 CPU @ 2.70GHz (48 cores, 96 threads); installed RAM: 251.5 GiB; storage: sda  PERC H740P Adp
- OS: Linux 6.8.0-147-generic; GPU driver: 580.178.04; CUDA (driver supports up to): 13.0
- Strata: project 0.1.58, commit 146df3c4; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=45.40, pcie=gen3 x16, pstate=P0, temp_c=37
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=44.72, pcie=gen3 x16, pstate=P0, temp_c=34
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-dsv41 --pack deepseek-v4.1-flash-ist-das-2.00bpw --max-context 131072 --eos 1 --vision on --dspark auto --dspark-conf 0.65 --tp 2 --reserve-mib 900
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 9 | not measured | 4.8 (3.0-30.3) | 1.05 (0.67-5.47) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 333.2 (10.0-340.0) | not measured | 12.40 (12.15-414.32) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 3 | 525.6 (14.9-536.4) | not measured | 31.55 (30.91-1109.66) |
| agent session, follow-up turns | 5318 (5318-5318) at the last turn | see below | up to 64 | 3 | see below | 3.5 (3.3-3.6) | 226.69 (154.43-298.96) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 49.4 % (49.4-49.4); earlier-turn tokens read again per session: 0 (0-2) (earlier turns that the cache did not hold; the 0 (0-4614) new tokens of the files are read once); first turn (nothing cached) first token: 647.47 s (121.23-1921.74).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 5.0 | 121.231 | 3.1 |

### Skipped and failed

- failed agent: {'type': 'server_error', 'message': 'the engine said nothing for 376 s during the request; the server ended the engine; the next request restarts it'}
- failed agent: {'type': 'server_error', 'message': 'the engine said nothing for 376 s during the request; the server ended the engine; the next request restarts it'}
- failed agent: {'type': 'server_error', 'message': 'the engine said nothing for 469 s during the request; the server ended the engine; the next request restarts it'}

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured by the suite: this is a speed suite. Informally, one multi-step reasoning task
(two-robot BFS pathfinding over a walled grid, plus complexity and Dijkstra questions) was
answered fully correct at 11.7 tok/s decode, drafter idle. The whole 257 GB weight set,
n-gram table and drafter sit inside 251 GB of RAM, so disk reads are rare in steady state;
the first cold load from the SAS spindle took about 50 minutes. The engine built for sm_70
is the community CUDA 12 path (STRATA_EXPERIMENTAL_SM60, CUDA 12.0 toolkit, gcc-12 host,
`-DCMAKE_CUDA_ARCHITECTURES=70-real`); the server ran it with `--tp 2` as setup planned it. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 6477 s.
