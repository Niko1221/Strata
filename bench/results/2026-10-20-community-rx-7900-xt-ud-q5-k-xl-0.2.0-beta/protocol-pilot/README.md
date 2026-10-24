# Hand-run pilot of the draft benchmark protocol, v0.2.0-beta on an RX 7900 XT 20 GB

Measured on 2026-10-20 by loeeeee. Strata v0.2.0-beta (engine 0.1.45.3), Unsloth UD-Q5_K_XL, 262,144-token context.

**Status: nonconforming draft pilot, not a standard result.** [BENCHMARK_PROTOCOL.md](../../../../docs/BENCHMARK_PROTOCOL.md)
(`strata-qwen38-thinking-v1-draft.1`) says no current tool implements it and that the runner, frozen fixtures and an
approved manifest do not exist yet. I followed its schedule, request settings, fixture construction rules and scoring rules by
hand with the small script in `scripts/pilot.py`, and list every deviation below. It is submitted as draft-phase pilot data;
it does not claim a place in any approved list. The result file is **not** in the draft's JSON schema.

## What ran (two complete runs, each on a freshly restarted engine)

Per run, in order, at concurrency 1 on loopback: 1 warm-up (excluded), performance `short` / `prompt-4k` / `prompt-16k` x 3
repetitions at a 512-token cap, 12 sanity cases (I1-I4, R1-R4, T1-T4) at a 2,048-token cap, then the **optional** concurrency
stage (4 short requests released together, 3 rounds, 512-token cap). Afterwards `tools/needle_bench.py` ran
32K and 128K prompts at depths 10/50/90 (outside the draft; the tool uses thinking off and greedy decoding).

Request settings as the draft requires: `enable_thinking` and `preserve_thinking` true, `reasoning_effort` `"xhigh"`,
temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1.0, seeds per the draft,
streaming with usage, no system prompt, no extra stop strings, `[case <id>]` marker first in every user message.
The server accepted all of them without error.

## Results

Medians, with (min-max) in parentheses; 3 requests per cell. Engine timings (`prompt_n`, `prompt_per_second`,
`predicted_per_second`); client time to first reasoning/answer delta. Run 1 and run 2 are not pooled.

| Case (actual prompt tokens) | Run | Prompt tok/s | Decode tok/s | TTFT s (first delta is reasoning in every request) | First answer s | Finish |
| --- | --- | --- | --- | --- | --- | --- |
| short (77) | 1 | 39.8 (35.9-41.7) | 18.5 (17.2-19.9) | 2.07 (1.90-2.21) | 6.56 (6.01-9.59) | 3 stop |
| short (77) | 2 | 44.0 (42.0-45.0) | 19.0 (17.2-19.7) | 1.83 (1.77-1.91) | 7.54 (6.81-11.72) | 3 stop |
| prompt-4k (4,093) | 1 | 511.4 (451.5-528.8) | 17.7 (17.6-19.4) | 8.14 (7.85-9.17) | 24.39 (21.09-33.07) | 3 length |
| prompt-4k (4,093) | 2 | 506.3 (498.5-663.8) | 19.7 (18.2-22.3) | 8.16 (6.29-8.29) | 23.02 (22.42-26.18) | 3 length |
| prompt-16k (16,387) | 1 | 994.6 (817.9-1021.4) | 18.9 (17.8-21.5) | 16.56 (16.14-20.18) | 39.29 (38.81-47.57) | 3 length |
| prompt-16k (16,387) | 2 | 934.2 (710.2-956.3) | 18.7 (17.9-19.2) | 17.63 (17.25-23.16) | 40.89 (40.20-41.58), n=2 | 3 length |

- The thinking profile spends most of the 512-token cap on reasoning: all 4K and 16K requests ended at the cap (`length`),
  and in run 2 one 16K request produced no answer text before the cap (first-answer n=2). Per the draft, caps are
  expected performance boundaries, not failures. Prompt rates for the 77-token prompt are a few dozen tokens and not meaningful.
- Zero tokens were reused in every performance request (`cache_n` 0); the engine's decode token count equalled the API
  `completion_tokens` in all of them. No request failed or errored in either run.
- The 4K and 16K prompt rates are lower than the same server measured with `strata_bench.py`
  (828 / 1,153 tok/s, `../v0.2.0-beta/`). The prompt content differs (repeated synthetic records vs. a word list) and the
  settings differ, so they are not comparable and I do not draw a conclusion from the gap.

**Sanity: 12/12 in both runs** (I 4/4, R 4/4, T 4/4), scored automatically by strict JSON / exact text / one structured tool call.
Every case finished with `stop` or `tool_calls` well under the 2,048-token cap (44-152 output tokens). Raw outputs, including
reasoning text, are in `run*/outputs.jsonl`; scoring per case in `run*/results.json`.

**Needle recall (`tools/needle_bench.py`), both runs: 6/6** (32,045-token prompts and 123,966-123,968-token prompts at depths 10, 50
and 90 %). Prompt processing took 31-42 s at 32K and 79-139 s at 124K. 262K was not tested.

**Optional concurrency.** The server has one slot (capacity 1), so the four clients queued; this is not four-way
parallel decoding. Round aggregate output rates, tok/s: run 1 18.3 / 19.9 / 20.5, run 2 18.4 / 18.1 / 19.0 - the same as a
single stream's decode rate. First-token times ranged 1.5-85 s because of the queue. Finish reasons: 8 length / 4 stop
(run 1) and 9 length / 3 stop (run 2). Three rounds are too few for a percentile.

**Duration.** Required stages: 420 s (run 1) and 419 s (run 2), about 7 minutes including warm-up; the draft's 20-30 minute
target is unmeasured and was not reached on this machine. Optional concurrency: 284 s and 320 s.

## Memory and clocks (run 2 only)

`scripts/sample.sh` sampled sysfs every ~2 s from 05:04:08 to 05:23:43 UTC (574 rows, covering the pilot, concurrency and the needle
runs): `run2/telemetry.csv`. Run 1 has no telemetry: the sampler's PATH was wrong in that run and wrote empty rows (fixed
before run 2; the shipped scripts are the fixed ones).

- VRAM in use 19.27-19.31 GiB of 20.0 GiB, steady (a start-up level, no growth during the run).
- Host RAM in use 97.2-98.5 GiB of 128 GiB, minimum available 29.5 GiB (swap was not sampled; no request failed).
- GPU shader clock median 2,559 MHz, max 2,974 MHz; memory clock 1,249 MHz when busy. GPU power sensor (average) median
  210 W, 95th percentile 323 W, maximum 419 W; the card's reported power cap is 290 W, so the maximum is a transient sample above it. Temperature median 53 C, max 60 C.
- PCIe link Gen4 x16 in all samples. 1-minute load average peaked at 45.1 on the 48 CPUs given to the container.

## Hardware, software and deviations from the draft

- GPU AMD Radeon RX 7900 XT (gfx1100) 20 GiB; CPU AMD EPYC 7K62, 48 CPUs given to an LXC container; 128 GiB RAM; storage a ZFS dataset on the Proxmox host
  (physical devices not recorded); Linux 7.0.14 (Proxmox) host kernel; host amdgpu driver; nixpkgs ROCm 7.2.4. Source build of tag `v0.2.0-beta`.
- Configuration track would be `custom`: setup's output plus `--vram-reserve-mib 1024 --prefill auto:32768`, `--kv int8`, MTP speculation as setup wrote it.
- Background work: zero client connections at the start of both runs and after the pilot stage (`context.txt`; run 1's file is blank
  because `ss` was not on the PATH, so that run's client count is unverified); the sampler and the coding-agent server itself were the only other activity. Fixture sizing requests ran on the previous engine, before the restart.

Deviations that make this run nonconforming (not exhaustive of the draft's activation checklist):

1. No approved, hashed fixture bundle exists. I built fixtures by the draft's construction rules; `fixtures.json` and the script are hashed in `SHA256SUMS`, but this is not `protocol.corpus_sha256`.
2. Tokenizer and chat-template revisions were not pinned or verified against upstream; record counts were chosen from the server's `prompt_tokens` (4,091 and 16,384 at calibration; 4,093 and 16,387 in the runs, within 2 %).
3. I did not verify the server's resolved sampling/effort configuration, only that the requests were accepted. Whether the `seed` field is honored was not checked.
4. Model file SHA-256 values, source model revision and quantization repository revision are not recorded here.
5. The result file is a flat hand-made structure, with no `missing` JSON-pointer explanations, and `outputs.jsonl` does not follow the draft's record layout.
6. v0.1.44.1 was not run through this protocol, so this pilot makes no improvement claim. The before/after comparison is the separate `strata_bench.py` one in `../README.md`.
7. Two runs on the same machine are replicates of each other; they are not a second hardware configuration.
