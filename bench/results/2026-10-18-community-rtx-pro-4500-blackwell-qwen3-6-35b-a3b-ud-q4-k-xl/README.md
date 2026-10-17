# Community benchmark on NVIDIA RTX PRO 4500 Blackwell

Measured on 2026-10-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX PRO 4500 Blackwell, 32 GiB VRAM
- CPU: AMD Ryzen Threadripper 7960X 24-Cores (24 cores, 48 threads); installed RAM: 61.0 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.91.07; CUDA: 13.2
- Strata: project 0.1.45, commit 477ac13c; engine not read
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q4_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: server defaults from the config's `sampling` block: temperature 0, presence_penalty 0, repetition_penalty 1.0, frequency_penalty 0 (needed for MTP drafts, see "MTP drafts" below); thinking off for the suite

```text
strata --native Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --max-context 65536 --prefill auto --mtp 2 --vision
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 259.5 (236.7-314.7) | 0.03 (0.03-0.03) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 6967.8 (6962.9-6984.7) | not measured | 0.58 (0.58-0.58) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 6123.2 (6112.9-6135.1) | not measured | 2.63 (2.63-2.64) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 4855.2 (4845.6-4863.9) | not measured | 6.62 (6.61-6.63) |
| agent session, follow-up turns | 14520 (14516-14547) at the last turn | see below | up to 64 | 3 | see below | 161.7 (125.1-208.9) | 1.50 (0.62-2.69) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 0.0 % (0.0-0.0); earlier-turn tokens read again per session: 30094 (30093-30199) (earlier turns that the cache did not hold; the 13697 (13697-13697) new tokens of the files are read once); first turn (nothing cached) first token: 0.12 s (0.12-0.12).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 5254.5 | 0.119 | 227.5 |
| 1 | 3749 | 0 | 3749 | 6096.9 | 0.629 | 208.9 |
| 2 | 5263 | 0 | 5263 | 5958.3 | 0.888 | 183.2 |
| 3 | 8679 | 0 | 8679 | 5826.8 | 1.496 | 159.4 |
| 4 | 11579 | 0 | 11579 | 5639.8 | 2.06 | 161.7 |
| 5 | 14520 | 0 | 14520 | 5489.2 | 2.652 | 129.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 65 s.

## MTP drafts

The Qwen3.6 engine path (`src/qwen35`) runs the GGUF's own MTP block only for a greedy request that is alone and carries no repetition, frequency or presence penalty (`Sampling::needs_logits()`). With the server's default sampling the first run of this suite had no drafts at all (engine log: `drafts accepted 0 of 0`; short chat decode 183.6 tok/s, agent decode 137.3 tok/s, same build, same machine). This entry has the `sampling` block above in the config, so drafts run: `drafts accepted 1207 of 1786` over the suite (68 %). Decode-over-context check with `--mtp 0` / `2` / `4` (temperature 0, penalties off, 3 runs, medians): 1K prompt 166.5 / 246.9 / 240.2 tok/s, 16K prompt 113.1 / 125.0 / 109.9 tok/s, so `--mtp 2` is the setting used here. Prompt speed is the same with and without drafts.
