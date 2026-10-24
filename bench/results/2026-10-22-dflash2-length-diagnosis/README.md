# Why the DFlash2 advantage shrinks — 2026-10-22

The original comparison mixes task and context length. It also shows a genuine loss
of DFlash2's advantage in speculative verification. A new controlled comparison
separates the two tasks from the two lengths. No new optimization, training, weight
change or head change is used in the main comparison.

## What the original +32.23% measures

The [original evidence](../2026-10-20-dflash2-long-tuned/README.md) compares a
27-token request for interval-merging code with 2425 tokens of repeated cache notes
followed by a summarization/example request. Both emit 96 tokens.

| Original task | MTP tok/s | DFlash2 tok/s | MTP accepted/offered | DFlash2 accepted/offered | Target positions evaluated, MTP / DFlash2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code, 27 input tokens | 38.46 | 50.86 | 60/103 | 70/84 | 139 / 110 |
| Cache notes, 2425 input tokens | 55.31 | 56.77 | 64/68 | 68/71 | 100 / 99 |

These are the recorded three-repeat medians and deterministic counters, not new
runs. The derived counters are in [original-counters.json](original-counters.json).
Each evaluated target position accounts for 480 expert lookups in this target;
the original raw lookup totals independently match these position counts.

MTP wastes 43 evaluated positions on rejected suffixes in the code task, versus
14 for DFlash2. In the cache task that becomes just 4 versus 3. DFlash2 therefore
loses almost all of its advantage in avoided target evaluations. Its throughput
actually increases on the original long fixture; MTP increases much more.

Expert-cache locality also changes with the task. The original GPU hit fraction
rises from 35.4% to 51.2% for MTP and from 36.7% to 51.0% for DFlash2. The raw miss
counts fall from 43087 to 23411 for MTP, versus 33403 to 23290 for DFlash2. Misses
are lookup counters, not a count of independent CPU jobs.

## Controlled task × length comparison

Same interval-merging instruction with either no notes or the same repeated notes
prepended; same cache instruction with one copy or all copies of those notes.
The original 27-token code and 2425-token cache prompts are retained byte for byte.
[Prompts and lengths](prompt-provenance.json), [prompt text](prompts-text.json).

RTX 4070 Ti SUPER 16 GiB, Ryzen 9 5900X, Linux, target NVFP4 Flash-Next, INT8 KV,
BF16 embeddings and unchanged shared Q8_0 head. All arms use context 8192, actual
2900 expert slots, 11 CPU workers, verifier capacity 8, prefill 512, short-read 64,
prompt cache 0, PCIe fraction 0 and no adaptive swaps or suffix/lookup drafting.
MTP is capped at a four-token verifier window: three proposals. DFlash2 Q8_0 uses
block 7 and floor 0.7; MTP uses floor 0.5. Verifier splitting is off.

One warmup per case, then three measured repeats. Prefill/startup are excluded.
These settings isolate placement/capacity; they are not an exact rerun of the older
auto-cache/prefill-256/short-read-4096 configuration. The same executable is used;
its experimental execution flags are unset. Shared head interleaving remains on
for both arms. Commands, binary hashes and native INFO are retained per arm.
The tested source changes relative to PR head `5c25ecdc4760fd3650d3d76b437535ce28084527`
are preserved in [tested-engine.patch](tested-engine.patch).

| Task | Input tokens | MTP tok/s | DFlash2 tok/s | DFlash2 vs MTP |
| --- | ---: | ---: | ---: | ---: |
| Code, short | 27 | 37.89 | 49.06 | +29.48% |
| Code, padded | 2427 | 36.92 | 40.72 | +10.30% |
| Cache, short | 41 | 44.95 | 48.58 | +8.07% |
| Cache, repeated | 2425 | 51.12 | 50.64 | −0.93% |

All speculative outputs match a common genuine target-only control, for every
96-token response. The controls run without a drafter and offer zero proposals.
No output mismatch or near-tie exemption is accepted. This does not mean the
short and long variants produce identical answers to each other.

The short-code percentage is workload-specific. Even the short cache task has
only an 8% advantage. Holding the code instruction fixed also reveals a real
context effect: DFlash2 acceptance falls from 70/85 to 58/87, and verifier windows
increase from 26 to 38. MTP's windows increase from 36 to 41.

## Where the time goes

Native `STRATA_DECODE_TIMING=1` counters; no CUDA event profiling is enabled in
the throughput runs. Values are medians over measured requests.

| Task | MTP / DFlash2 windows | MTP / DFlash2 draft ms per window | MTP / DFlash2 CPU-expert ms per emitted token |
| --- | ---: | ---: | ---: |
| Code, short | 36 / 26 | 2.65 / 3.14 | 20.36 / 15.59 |
| Code, padded | 41 / 38 | 2.53 / 4.07 | 20.24 / 17.43 |
| Cache, short | 35 / 27 | 2.49 / 3.22 | 16.49 / 15.46 |
| Cache, repeated | 32 / 30 | 2.38 / 4.12 | 14.33 / 14.06 |

DFlash2 needs more windows for the padded code task, and each draft portion is
more expensive. Its total draft time per emitted token rises from 0.85 to 1.61 ms.
MTP's rises only from 0.99 to 1.08 ms. In the long cache task DFlash2 saves little
CPU work and spends about 0.49 ms/token more on drafting.

The implementation performs five draft attention layers over the visible context;
its local window is capped at 2048, so this is not unbounded growth with prompt
length. The observed draft-time increase is consistent with this extra context
work. These wall counters do not isolate an individual attention kernel.

Drafting accounts for approximately 6.5% of DFlash2 decode time in these long cases.
With unchanged acceptance and verification costs, even eliminating that entire
portion would improve its throughput by only about 7%. Small selector/convolution
kernel wins cannot by themselves restore a 32% end-to-end advantage here.

## Checks of alternative explanations

Exploratory one-repeat ablations, with warmup and output controls; not performance
wins inferred from a single sample:

- Full draft context instead of the 2048 local window: padded-code acceptance is
  58/86, versus baseline 58/87. It does not restore short-code accuracy. The short
  control retains 70/85. The earlier diagnostic binary's hash is retained.
- Original BF16 backbone instead of Q8_0: padded code remains 58/87; short code is
  71/82. Backbone quantization does not explain the main long-code loss in this test.
- Short-read 4096 instead of 64: the padded-code response changes at output index
  30, including in its target-only control. Its 64/93 acceptance is therefore not
  a clean same-output improvement over 58/87. [Comparison](readpath-comparison.json).
  Both reading modes individually pass their target-only controls. This does not
  establish a DFlash verification bug or identify a faulty feature tap.

The initial MTP max-T=3 attempt offered only two proposals and was interrupted.
It is retained under `mtp-maxT3-interrupted`, excluded from every table, and
explained in [EXCLUDED.md](EXCLUDED.md). The corrected study uses max-T=4.

## Interpretation and limits

The measured causes are reduced savings in target verification, task-dependent MTP
accuracy/cache locality, and higher DFlash2 draft cost per window. Context padding
also reduces the drafter's accuracy on the code task. Full context and BF16 do not
remove that reduction. Its exact internal model/feature-level cause remains
unresolved; these tests do not justify claiming a specific implementation bug.

The next optimization target should be accepted tokens per verification and the
resulting CPU work, alongside draft-context cost. No default or universal
length-based setting follows from these four cases. There is no +32% long-context
claim, no other-model performance claim, and no HIP/SYCL validation in this study.

Raw summaries: [summary.json](summary.json), [MTP](mtp/mtp.json),
[DFlash2](dflash2/dflash2-il.json), [target-only](control/mtp.json).
`prepare.py`, `run.py`, `ablations.py`, `readpath.py` and `analyze.py` preserve the
local drivers. Run from the repository root with the recorded model files;
their working evidence directory is `../Strata-data/dflash2-diagnosis-20261022`.
They refuse overwriting completed arms through the shared benchmark runner.
