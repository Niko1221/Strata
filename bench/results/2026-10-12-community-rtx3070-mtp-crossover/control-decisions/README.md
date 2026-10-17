# Control decisions: MTP on or off for a known output length?

This follow-up measured **exactly 1,000 generated tokens** at four fresh input
lengths on the same RTX 3070 / Ryzen 7 5800X / nominal 64 GB RAM machine and
pinned `61b3fb5d` binary as the [earlier report](../README.md). Model:
Qwen3.8-Flash-Next **GSQ-RCO IQ3_S**, INT8 KV. It is not a Q4 result.

MTP off finished sooner in all four measured pairs. The main rule for this
specific source-code workload is: **short control outputs favor off; the
projected crossing is approximately one-sixth of the fresh input tokens**.
Long-output crossings are estimates, not new measurements.

![Measured 1,000-token completion times, projected break-even output lengths and a capacity-aware decision map.](control-decisions.png)

[SVG](control-decisions.svg) · [Per-run metadata](data/results.json) ·
[Decision model](data/decision-model.json) · [Chart source](plot.py)

## What changed for this cohort

All eight arms used a fixed `--max-context 133120` and `--kv-resident 32768`,
with `--kv int8`. The earlier cohort used different capacities per input
length and no explicit KV residency. Do not merge their thresholds into one
curve or attribute the difference to prompt length alone.

The new prompts contain varied public C++ source from the pinned Strata checkout,
starting with `src/program/generate.cpp`, followed by a request for five Python
components and extensive unit tests. There is no repeated ` x` padding. The
same leading source corpus is truncated to fit each exact input length; each
on/off pair has the same full prompt hash. Template-tokenized input lengths are
49,152 / 65,536 / 98,304 / 131,072 tokens. Here input labels K mean 1,024 tokens;
output counts such as 1,000 and 10,000 are literal token counts.

Each arm used a newly loaded engine and the same small warm-up. Model loading
and warm-up are excluded from request timings. Order was 48K off/on, 64K on/off,
96K off/on, 128K on/off; one measured request per arm, sequential execution.
All requests emitted exactly 1,000 tokens, with `reused=0`. MTP off used
`--spec 2` and offered no drafts. On used `--spec 4 --mtp <MTP_RT>
--mtp-window 32768`. Both used `--prefill auto:8192`, automatic expert cache,
`--spec-min-p 0.5`, the same expert profile, disabled suffix/prompt/conversation
reuse, temperature 0 and thinking disabled in the chat template.

Hardware and pinned binary details are in [the parent provenance](../provenance.json).
Full sanitized arguments, phase timings and exact prompts are in `data/`.
The original measurement script is [recorded-harness.py](recorded-harness.py).
Its host-directory layout must be adapted before rerunning it.

## Measured 1,000-token request times

| Fresh input | MTP off, s | MTP on, s | Seconds saved by off | Reduction versus on |
| ---: | ---: | ---: | ---: | ---: |
| 49,152 (48K) | 95.26 | 158.09 | 62.83 | 39.7% |
| 65,536 (64K) | 118.15 | 207.88 | 89.73 | 43.2% |
| 98,304 (96K) | 163.99 | 305.27 | 141.27 | 46.3% |
| 131,072 (128K) | 212.70 | 405.41 | 192.71 | 47.5% |

MTP saved only 8.77 / 8.99 / 9.41 / 9.20 seconds of decode time for 1,000
tokens. Its additional prefill cost was 71.60 / 98.72 / 150.68 / 201.92 seconds.
These are single observations, not medians or confidence intervals.

## Rule of thumb for a controller

For a **fresh, uncached input** in this measured 48K–128K range:

1. For expected outputs of 10, 100 or 1,000 tokens, choose **off** for request
   completion latency. Off also produced the first token sooner in every pair.
2. Estimate the output crossover from a matched timing profile:

   ```text
   A = prefill_on_seconds - prefill_off_seconds
   s = 1/decode_off_tps - 1/decode_on_tps
   estimated_on_time_saved(N) = N*s - A
   projected_output_crossing = A/s
   ```

   In this cohort `A > 0` and `s > 0`. If `s <= 0` and `A >= 0`, enabling MTP
   cannot repay the penalty in this model. This is a mean-latency estimate, not
   a bound or a p95 policy.
3. Here `A/s` is about **16.3–16.7% of fresh input length**, or approximately
   `input_tokens / 6`. Off is the candidate well below that point; on is the
   candidate well above it. Measure around the boundary rather than switch
   solely because a weak projection changes sign. First-token latency remains
   a separate objective: on's faster decode does not undo its later start.
4. Check `input_tokens + expected_output_tokens <= configured_context` before
   making a mode recommendation. A timing projection does not establish fit.
5. Use expected **actual generated tokens**, including any reasoning tokens,
   not `max_tokens` unless that cap is expected to be reached. These tests had
   thinking disabled. For uncertain output lengths, use expected savings over
   the output-length distribution instead of its upper limit.

This is a workload-specific latency heuristic, not an engine default or a
correctness guarantee for a real controller. Calibrate the actual model, KV
layout, expert-cache state and control prompts. The four-point profile does not
establish behavior outside this input range.

## Where does MTP on start to pay off? Estimates only

| Fresh input | Additional prefill, s | Decode saving per output token, ms | Projected crossing | Output slots remaining in tested setup |
| ---: | ---: | ---: | ---: | ---: |
| 49,152 | 71.6020 | 8.7719 | approximately 8,163 | 83,968 |
| 65,536 | 98.7238 | 8.9944 | approximately 10,976 | 67,584 |
| 98,304 | 150.6833 | 9.4110 | approximately 16,011 | 34,816 |
| 131,072 | 201.9175 | 9.2035 | approximately 21,939 | **2,048** |

All crossings extrapolate the observed **1,000-token** decode rates. Acceptance,
output content, attention/KV work and expert placement may change as the output
grows, so these crossings have not been experimentally confirmed.

The 128K projection is outside this setup's remaining capacity. **No fitting
output length reaches its projected MTP benefit in the tested 133,120-token
configuration.** A larger capacity and a fresh matched measurement are required
before using that projected crossing for a decision.

## 10 / 100 / 1,000 / 10,000 / 100,000 output tokens

The signed values below are **seconds saved by enabling MTP**: negative means
on is slower. The 10/100 columns are recorded arrival prefixes of the 1,000-token
requests, not separate requests capped at 10/100; final-window overhead can differ
under a smaller cap. The 1,000 column is the measured full request. Higher columns
are constant-rate extrapolations, not new runs.

| Fresh input | 10, prefix | 100, prefix | 1,000, measured | 10,000, projected | 100,000, hypothetical |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 48K | -71.60 s | -71.07 s | -62.83 s | **+16.12 s** | +805.59 s, needs larger context |
| 64K | -98.69 s | -98.17 s | -89.73 s | -8.78 s | +800.72 s, needs larger context |
| 96K | -150.64 s | -150.01 s | -141.27 s | -56.57 s | +790.42 s, needs larger context |
| 128K | -201.87 s | -201.32 s | -192.71 s | -109.88 s, needs larger context | +718.43 s, needs larger context |

A single 10,000-token answer is therefore an **on candidate only at 48K input**
in this arithmetic; the projected gain is small and unverified. At 64K and
96K it still favors off. At 128K it does not fit the tested capacity.

A single 100,000-token answer is a hypothetical exercise: **none of these
input/output combinations fits the tested context capacity**. Changing capacity
can change allocation and throughput, so the positive numbers are not deployment
recommendations. The extrapolation also extends far beyond the recorded output.

## If “1,000 ×10 / ×100” means separate control decisions

Ten or 100 separate uncached 1,000-token calls each pay a fresh prefill penalty.
Adding their outputs together does not turn them into one long request. Assuming
the same per-call latencies and no inter-call changes, the arithmetic savings
from keeping MTP off are:

| Fresh input per call | 10 calls: off saves | 100 calls: off saves |
| ---: | ---: | ---: |
| 48K | 628.30 s (10.47 min) | 6,282.98 s (104.72 min) |
| 64K | 897.29 s (14.95 min) | 8,972.85 s (149.55 min) |
| 96K | 1,412.74 s (23.55 min) | 14,127.36 s (235.46 min) |
| 128K | 1,927.14 s (32.12 min) | 19,271.38 s (321.19 min) |

These multi-call totals are estimates obtained by multiplying one measured
paired latency, not tests of 10/100 calls or a concurrent queue.

If decisions reuse a cached prefix, measure the **incremental prefill and draft
state cost** instead. The full-context penalty above no longer represents the
request. In the idealized limit of zero extra prefill cost and unchanged decode
rates, MTP's decode advantage could help even a short answer. That cached case
was not measured; nominal context length alone is not enough to select a mode.

## Evidence and reproduction

The source corpus is from the same pinned public Strata source used to run the
engine. Full prompts, prompt IDs, generated IDs, outputs, arrivals and engine
logs are preserved. Every input count and output count was checked, pair input
hashes match, and all prompt reuse counts are zero. Outputs diverged at one-based
positions 65 / 65 / 125 / 68. Generated code was not executed or quality-scored.
No three-repeat statistics, cached-turn experiment, 10K/100K generation or
long-output correctness qualification is claimed.

The measured original harness is [recorded-harness.py](recorded-harness.py).
Model paths and hardware/user identifiers in published data have been sanitized.
The harness used its existing model config and pinned build; its directories
must be adapted before running it on another machine. No new inference run was
made while preparing this chart.

To check and rebuild the figure using the published data:

```bash
python analyse.py data
python build_decision_model.py
python verify_decisions.py
python -m pip install -r ../requirements-plot.txt
python plot.py
```

`analyse.py` verifies eight 1,000-token runs, exact prompt lengths, pair prompt
hashes, output hashes, arrival counts and phase/wall timing arithmetic.
`build_decision_model.py` derives both prefix observations and longer-output
projections; `verify_decisions.py` checks the model and context flags.
