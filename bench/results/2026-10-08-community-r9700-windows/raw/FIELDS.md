# Raw data format

Every `*.json` file in this folder is one **arm** (one engine configuration, started fresh, measured three
times). They were produced verbatim by [`../scripts/bench_decode.py`](../scripts/bench_decode.py); nothing in
this folder was edited by hand. The sibling `<arm>.md` file of the same base name is the human-readable summary
the harness wrote for that same run from the same data — it is derived, not independent, and is included only
so a reviewer can read the per-window split without running Python.

Three `ab-text-*.txt` files are the greedy model output of single 512-token requests, kept as evidence for the
non-determinism finding in the README.

## Units

- Times: **milliseconds** unless the name ends in `_ms` or the value is explicitly seconds.
- Throughput: **tokens per second**, one token counted per generated token.
- `GIB`: gibibytes (2^30 bytes). `MiB`: mebibytes. `B`: bytes.
- Percentages: `pct` fields are already in percent, not fractions.
- All clocks are the engine's own, taken from its summary lines. See "Where the numbers come from" below.

## Top level

| Field | Type | Meaning |
| --- | --- | --- |
| `run` | object | How this arm was requested. Constant across a file. |
| `stats` | object | Aggregates over the usable repeats. |
| `requests` | array | One entry per repeat, in the order the engine logged them. |

## `run`

| Field | Meaning |
| --- | --- |
| `timestamp` | Arm start, `YYYYMMDD-HHMMSS` local time. |
| `base_url` | The server the harness talked to (always loopback). |
| `model` | `model_name` from the engine config. |
| `requested_prompt_tokens` | Size knob passed to the harness's synthetic prompt generator, **not** a token count. See the note below. |
| `requested_max_tokens` | Output cap per request (128 in every arm here). |
| `repeats_requested` | Repeats requested (3 in every arm here). |
| `seed` | Seed passed to the request builder (1234). Sampling was greedy, so the seed does not change the text. |
| `arm_label` | Human label of the arm. |
| `warm_cold` | Free-text statement of the cache condition, copied from the driver. |
| `log_path` | Engine log this arm's timing lines were parsed out of. Not included in this PR (see README). |
| `prompt_file` | `null` for every arm here: all prompts were generated, none read from a file. |

**Requested vs actual prompt tokens.** The harness's generator expands each requested unit into several real
tokens, so `requested_prompt_tokens` is only a size selector. The mapping actually used was:

| `--prompt-tokens` | actual `prompt_tokens` | ratio |
| ---: | ---: | ---: |
| 4,096 | 17,154 | 4.19 |
| 24,576 | 99,696 | 4.06 |
| 32,768 | 132,886 | 4.05 |
| 49,152 | 199,316 | 4.05 |

Always report `stats.prompt_tokens.median`, never `run.requested_prompt_tokens`.

## `stats`

| Field | Meaning |
| --- | --- |
| `n_total` | Repeats attempted. |
| `n_usable` | Repeats that entered the aggregates (parsed, not cancelled, not truncated). |
| `n_rejected` | Repeats excluded. `3 of 3` in every arm here. |
| `rejected` | Per-repeat exclusion reasons; empty in every arm here. |
| `decode_tps` | `{n, median, min, max, range}` of generated tokens/s over **all** usable repeats (cold rep 1 included). |
| `decode_tps_warm` | The same over **repeats 2..n only** (warm expert cache, reused prompt). This is the steady state. `n` is 2, not 3. |
| `prefill_tps` | Prompt tokens/s per the engine, over all usable repeats. **See the warning below — its median is not a prompt-throughput number.** |
| `ms_per_token` | Milliseconds per generated token (median of all usable repeats). |
| `hit_pct` | Decode expert-cache hit rate, percent, medians. |
| `accept_pct` | Draft acceptance, percent of drafted tokens accepted, medians. |
| `prompt_tokens` | Actual prompt tokens after the tokenizer. |
| `context` | `prompt_tokens + generated`, i.e. the context the engine actually held. |
| `generated` | Generated tokens; 128 in every request here (the cap was always reached). |

> **`prefill_tps` median is not prompt throughput.** In every arm, repeat 1 read the whole prompt
> (tens of thousands of tokens) while repeats 2 and 3 reused it and re-read only 5 tokens. Taking a median
> over the two states mixes a ~90 s operation with a ~50 ms one. The honest prompt-throughput figure for an
> arm is repeat 1's `prefill_tps` (in `requests[0].prefill_tps`), which is what the README's Results table
> reports, marked `n=1 (cold repeat only)`. `stats.prefill_tps.max` is usually that same cold value.

## `requests[]`

| Field | Meaning |
| --- | --- |
| `label` | Arm label. |
| `repeat` | 1-based repeat index. Repeat 1 = cold; 2..n = warm. |
| `line_no`, `byte_offset` | Position of this request's summary line in the engine log, so a reviewer can find it again. |
| `raw_line` | The engine's verbatim timing line. This is the primary evidence; every field below is parsed out of it. |
| `prompt_tokens` | Actual prompt tokens. |
| `reused` | Prompt tokens served from the conversation cache. 0 on repeat 1. |
| `read` | Prompt tokens actually read this request (`prompt_tokens - reused`). |
| `prefill_ms` | Engine-side prompt-processing time. |
| `prefill_tps` | `read / prefill_ms`. Meaningful only on repeat 1. |
| `generated` | Generated tokens. |
| `decode_ms` | Engine-side decode time for the generated tokens. |
| `decode_tps` | `generated / decode_ms`. |
| `drafts_accepted`, `drafts_total` | Speculative draft tokens accepted / drafted. |
| `checkpoints` | Prefix checkpoints reported by the engine for this request. |
| `usable` | Whether this repeat entered the aggregates. |
| `reject_reason` | Why not, when `usable` is false. |
| `hit_pct`, `hit_hits`, `hit_lookups` | Decode expert-cache hits and lookups; `hit_pct = 100 * hits / lookups`. |
| `hit_rate_attribution` | How the engine attributed a hit to its layer. `adjacent` in every arm here. |
| `ram_gib` | Expert data resident in system RAM, GiB, as the engine reported for this request. |
| `exchanged` | Expert blobs moved between the RAM tier and the VRAM cache during this request. |
| `blob_reads` | Expert blobs read back from the model file. 0 in every arm here. |
| `suffix_windows`, `suffix_accepted`, `suffix_total` | Suffix-drafting windows and acceptance. |
| `decode_timing` | Per-window split, see below. |
| `stage_profile` | Per-GPU-stage split; `null` in every arm here because no reported arm ran with `STRATA_VERIFY_PROFILE=1`. |
| `warm_cold` | Cache condition for this repeat. |

### `requests[].decode_timing`

Means over the windows of this one request. All `*_ms` values are **milliseconds per window**, not totals.

| Field | Meaning |
| --- | --- |
| `windows` | Decode windows in this request (count). |
| `avg_t` | Mean window size T, i.e. mean tokens submitted per window. |
| `tokens_per_window` | Mean tokens **emitted** per window. |
| `ms_per_window` | Mean wall time per window. |
| `verify_ms` | GPU verification work per window. |
| `wait_ms` | "GPU-reach wait": host time per window blocked waiting for the GPU. This is the field the README's mechanism argument turns on. |
| `host_ms` | Per-layer host planning cost per window. |
| `plan_ms`, `actq_ms`, `jobs_ms`, `cpu_ms` | Components of `host_ms`: plan, action queue, job dispatch, CPU expert work. |
| `stage_ms` | Activation staging. |
| `commit_ms` | Commit / emit. |
| `draft_ms` | MTP draft work per window. |
| `cpu_experts_per_layer` | Mean experts computed on the CPU per layer-window (count, not ms). |
| `entries_per_layer` | Mean cache entries touched per layer-window (count). |
| `vram_hits_per_layer` | Mean expert hits served from VRAM per layer-window (count). |
| `pcie_per_layer` | Mean expert fetches over PCIe per layer-window (count). 0 in every window of every arm here. |

The engine prints these as a partial decomposition:

```
verify = wait + host + stage ;  ms/window ~= verify + commit + draft
```

**The decomposition is not exhaustive.** In this dataset `verify_ms` exceeds
`wait_ms + host_ms + stage_ms`, and `ms_per_window` exceeds `verify_ms + commit_ms + draft_ms`. The
unaccounted remainder is real time the engine did not itemise. Do not treat the components as a closed
budget.

## Where the numbers come from

Every timing field is parsed from the engine's own unbuffered stderr summary lines, not from the HTTP round
trip. The two line kinds are:

```
strata serve: prompt 132886 tokens = 0 reused + 132886 read in 121182 ms (1096.6 tok/s), 128 generated in 2984 ms (42.9 tok/s), drafts accepted 62 of 80, 6 checkpoints
strata decode timing: 66 windows, avg T 2.14, 1.94 tokens/window, 40.17 ms/window = verify 33.61 (GPU-reach wait 27.70 + per-layer host 2.60 [plan 0.09 actq 0.22 jobs 0.01 CPU 2.26] + stage 0.01) + commit/emit 0.25 + draft 3.07; per layer-window: CPU experts 0.98 (1.11 entries), VRAM hits 20.25, PCIe 0.00
```

Consequences:

- `decode_tps` is `generated / decode_ms` from the engine line, so it **excludes** prompt processing. It was
  never computed as `generated / total request time`.
- Decode throughput is **not measured** client-side, so there is no HTTP or queueing overhead in it.
- **TTFT is not in these files at all.** There is no field for it. See the README's limitations.
- Total latency is likewise absent; `prefill_ms + decode_ms` is engine-side only and excludes HTTP, server
  dispatch and tokenisation.

## Text files

`ab-text-base.txt`, `ab-text-base2.txt` and `ab-text-shstream0.txt` are the complete greedy output of a single
512-token request each, written by `tools/opt/cfg-ab/token_check.py`. They are evidence, not input: they are
not parsed by the harness and nothing in this report depends on their content beyond counting characters and
words.