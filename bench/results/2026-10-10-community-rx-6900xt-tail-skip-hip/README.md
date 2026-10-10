# Community benchmark: RX 6900 XT 16 GB (gfx1030), Ryzen 9 5950X, 128 GB RAM - `STRATA_ROUTE_TAIL_SKIP=7` on HIP

Measured on 2026-10-10 by elightcap. 0.1.42 turns the route tail skip on by default on CUDA only ("HIP and SYCL stay
as they were"). The skip itself is host code in `expert_pool_dispatch_multi` (`src/core/expert_source.cpp`) and reads
`STRATA_ROUTE_TAIL_SKIP` on any build, so this report sets it by hand on a HIP build and compares decode against
leaving it unset. Result on this machine: **decode x1.24 (median of 15 pairs), faster in 14 of 15**, needles 10/10
in both arms. Main limitations: one machine, one model, 300-token greedy replies, no KL or task check.

## Hardware and software

- GPU: AMD Radeon RX 6900 XT 16 GB (gfx1030, Navi 21), PCIe 4.0 x16 (16 GT/s, width 16), power cap 332 W (stock).
- CPU: AMD Ryzen 9 5950X, 16 cores / 32 threads; the engine's CPU pool ran 15 workers.
- RAM: 128 GB installed (125.7 GiB `MemTotal`). Storage: Samsung 980 PRO 2 TB NVMe (the GGUF files).
- OS: Arch Linux, kernel 7.2.8-arch1-2, ROCm 7.2.4.
- Strata `61b3fb5` (engine 0.1.42), source build by setup (`build-hip`: Release, `CMAKE_HIP_ARCHITECTURES=gfx1030`,
  `STRATA_ENABLE_HIP=ON`, `STRATA_PREFILL_MMQ=ON`).
- Background: a Linux desktop session on the same card (about 0.5 GB of VRAM with Strata stopped); no other GPU work.

## Model and configuration

- Model: [unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
  the four shards `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf`, run in place (`--native`) with setup's
  pack, setup's expert profile (`data/expert-profile.bin`) and setup's MTP draft layer. No vision.
- Context 131,072, `--kv int8 --kv-resident 32768`, `--expert-cache auto` (2,831 slots, 8.25 GiB of VRAM),
  `--resident-budget-gib 71` (63.47 GiB of experts pinned in RAM, the rest read from the GGUF), `--prefill auto`,
  `--spec 4 --spec-min-p 0.5`, `--prompt-cache-tail`, no conversation cache.
- Environment in both arms: `STRATA_SH_STREAM=1 STRATA_HIP_PROMPT_F16=1`. The only difference between arms:
  `STRATA_ROUTE_TAIL_SKIP=7` set (`tail7`) or unset (`off`).
- Sampling: greedy (`temperature 0`), thinking off. No calibration, no experimental speed projection.

```text
serve/server.py --engine strata --config config.json --port 8091
```

[config.json](config.json) is the config both arms ran (paths replaced by `<repo>`, `<strata-data>`, `<models>`).

## Method

[ab_tail.py](ab_tail.py) runs the whole A/B. Five rounds; each round starts a fresh server once per arm, order
`off`, `tail7` in even rounds and `tail7`, `off` in odd ones. Each start:

1. "Say hi." (8 tokens, warm-up, not recorded). Model loading is not timed.
2. Three decode requests, 300 tokens each, all 300 generated in every run: a story (44-46 prompt tokens), a code
   answer (57-59) and a summary of the first 24,000 characters of `docs/DETAILS.md` (7,913-7,915 prompt tokens,
   "doc6k" in the data).
3. Two needle prompts, about 8K and 32K tokens: a code word hidden at 30-70% depth in a list of random words, the
   same texts in both arms of a round. A row is correct when the answer contains the code word.

Decode tok/s is the server's `timings.predicted_per_second` (the engine's own decode clock, drafts included).
Every prompt was read fresh (the decode prompts are unique per start; nothing reused). Each decode prompt starts with
a tag naming the round and the arm (`[0-off]`, `[0-tail7]`), so the two arms' prompts differ by about two tokens: the
greedy texts are not comparable token for token, the speeds are.

## Results

Decode tok/s, five starts per arm. Ratio is `tail7 / off` within the same round.

| Request | Prompt tokens | Generated | `off` median (range) | `tail7` median (range) | Ratio, median (range) | `tail7` faster |
| --- | ---: | ---: | --- | --- | --- | ---: |
| story | 44 / 46 | 300 | 28.3 (26.9-28.7) | 31.8 (26.7-34.3) | 1.116 (0.993-1.195) | 4 of 5 |
| code | 57 / 59 | 300 | 31.9 (31.4-32.4) | 39.6 (39.1-39.8) | 1.238 (1.207-1.264) | 5 of 5 |
| doc6k | 7,913 / 7,915 | 300 | 28.2 (27.5-29.0) | 35.2 (34.1-36.0) | 1.248 (1.176-1.277) | 5 of 5 |

All 15 pairs pooled: median ratio **1.237**, `tail7` faster in **14 of 15**. Per-round ratios: story 1.116 / 0.993 /
1.143 / 1.046 / 1.195, code 1.207 / 1.237 / 1.264 / 1.248 / 1.238, doc6k 1.176 / 1.265 / 1.258 / 1.277 / 1.248.

Engine log, every `tail7` start: `STRATA_ROUTE_TAIL_SKIP=7 is ON`, and 77,508-81,145 missed experts skipped per start
by the end of its requests (the cumulative `route tail skip:` line). No `off` start printed either line.

Draft acceptance (accepted / offered drafts, median of five): story 0.597 `off` / 0.545 `tail7`, code 0.832 / 0.806,
doc6k 0.725 / 0.740. The story's lower acceptance is the effect #1884 reports on a V100; here the time saved on
missed experts outweighs it.

Decode expert-cache hit rate as the engine logs it (story / code / doc6k, the five starts of each within 3 points):
about 74 / 60 / 60% `off` and 80 / 67 / 68% `tail7`. That is far below #1884's 99.9%, which is where the skip has
the most to remove. Why the logged rate is higher with the skip on was not checked in the code.

Per-run data: [runs.jsonl](runs.jsonl) (fields: `round`, `arm`, `req`, `decode_tps`, `n` generated tokens,
`prompt_tps`, `prompt_n`, `drafts`, `accepted`, `correct` for needles, `answer` = first 80 characters). Engine log
lines per start (start-up, cache, skip, hit rate): [engine-lines.txt](engine-lines.txt).

No request failed, hung or was cancelled.

## Correctness and limitations

- Needles: 10 of 10 in both arms (five at 8,037-8,103 tokens, five at 31,978-32,018 tokens).
- Not measured: KL against the unskipped path, coding or reasoning task scores, long replies, sampled decoding,
  prompt throughput with the skip (it does not touch prompt reading, per docs/DETAILS.md), other models or cards,
  Windows HIP, `--batch` slots, layer splits.
- One machine and one model; the answers differ from the unskipped path (the docs give KL ~0.003-0.006 on CUDA).
- `STRATA_HIP_PROMPT_F16=1` and `STRATA_SH_STREAM=1` were on in both arms (this machine's daily configuration).
