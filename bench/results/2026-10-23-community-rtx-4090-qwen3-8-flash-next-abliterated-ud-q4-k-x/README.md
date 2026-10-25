# Community benchmark on NVIDIA GeForce RTX 4090

Measured on 2026-10-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090, 24 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 125.4 GiB; storage: not measured
- OS: Linux 7.3.0-4-cachyos; GPU driver: 615.78.08; CUDA: not read; Python 3.14.8
- Strata: the **Strata Mnemosyne** fork of upstream 0.1.48.1 (the memory system lives in the server), commit 62299039; engine 0.1.48
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-abliterated-ud-q4_k_xl
- Context: 393216 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack huihui-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --native-dense-gguf Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --native-dense-gguf Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf --native-dense-gguf Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf --native-dense-gguf Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf --ple-gguf Qwen3.8-Flash-Next-Q8PLETable.gguf --ple-io direct --expert-profile expert-profile.bin --expert-cache auto --prefill auto:32768 --spec 4 --mtp rt --max-context 393216 --rope-scaling yarn --rope-scale 1.5 --kv int8 --kv-resident 32768 --conversation-cache-mib 16384 --vision --vram-reserve-mib 1000 --pcie-frac 0.35 --spec-min-p 0.70 --adapt-async 1 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

### What this build adds to every prompt, and what it does to the numbers

Memory was **on** for this run (the engine log carries the `memory core on:` banner), and this fork's server
prepends a standing text plus a recall block to every request; the fork records 7,009 tokens for the standing
text of that era. `Actual prompt tokens` is the server's own count, which is why the short-chat row shows
7,204 tokens for three questions and why the "4K" arm landed at 7,323 instead of near 4,000: the tool sizes its
word lists with a calibration request, and on this build that request already carries the prefix. Read the 4K
and 16K rows as measured prompt throughput at the token counts shown rather than as a clean 4K/16K sweep, and
compare with another report by the token column, not by the arm name. The conversation-cache rows carry the
same stamp: the 7,785-7,819 reused tokens per follow-up turn are largely this standing prefix, which the fork's
own design puts in front of every conversation on purpose.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 7204 (7029-7391) | 0 | 100 (34-100) | 9 | not measured | 80.3 (73.7-97.9) | 3.38 (0.52-3.47) |
| 4K prompt, fresh | 7323 (7109-7385) | 0 (0-0) | 24 cap | 3 | 2171.4 (2120.7-2172.3) | not measured | 3.46 (3.44-3.49) |
| 16K prompt, fresh | 16297 (16272-16315) | 0 (0-0) | 24 cap | 3 | 2636.0 (2612.3-2648.7) | not measured | 6.22 (6.20-6.30) |
| agent session, follow-up turns | 22302 (22234-22362) at the last turn | see below | up to 64 | 3 | see below | 97.8 (58.8-126.9) | 3.32 (2.90-3.82) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token. The short-chat `Reused` column is 0 for eight of the nine requests; the ninth hit the conversation cache for 6,986 tokens, and that is the request whose first token came in 0.52 s, which is where that row's TTFT range floor comes from.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 46.6 % (46.5-46.7); earlier-turn tokens read again per session: 30344 (30195-30433) (earlier turns that the cache did not hold; the 14301 (14300-14302) new tokens of the files are read once); first turn (nothing cached) first token: 3.35 s (3.33-3.38).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7792 | 0 | 7792 | 2366.6 | 3.335 | 82.1 |
| 1 | 11549 | 7819 | 3730 | 1260.9 | 3.062 | 88.5 |
| 2 | 13121 | 7785 | 5336 | 1725.5 | 3.139 | 97.8 |
| 3 | 17033 | 7785 | 9248 | 2729.9 | 3.496 | 91.1 |
| 4 | 19599 | 7785 | 11814 | 3351.6 | 3.581 | 117.3 |
| 5 | 22302 | 7785 | 14517 | 3910.0 | 3.816 | 102.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 150 s.
