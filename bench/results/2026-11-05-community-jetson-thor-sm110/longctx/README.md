# Long context on the Thor: 128K and 256K, measured (Qwen3.8-Flash-Next IQ3_S)

Runs 2026-11-09 (02:23Z onward) on the same Jetson AGX Thor and the same fat `sm_121 + sm_110` engine
package as the rest of this report. **Power mode for every run here: MAXN** (`nvpmodel -q` at the start of
the session) - these rows sit alongside the MAXN 64K cells above, and alongside them decode barely moves.

Method: `tools/dgx_bench.py` (fresh nonce per request, no cache reuse, the engine's own `timings`, TTFT the
client's), cells `16384/65536/131072 : T=0 : 256`, medians of n=2. `iq3s-128k-*` ran 2 slots,
`iq3s-128k-b4-*` 4 slots, `iq3s-256k-*` 1 slot; every arm's own `*.config.json` and raw `*.jsonl` are in the
folders here.

## IQ3_S at 128K and 256K (default settings, MAXN)

| context | slots | cell | prompt tok/s | decode tok/s | TTFT s |
|---|---|---|---:|---:|---:|
| 128K | 2 | 16K | 1117.2 | 37.7 | 14.8 |
| 128K | 2 | 64K | 1066.7 | 38.0 | 62.9 |
| 128K | 4 | 16K | 1128.8 | 35.3 | 14.6 |
| 128K | 4 | 64K | 1155.4 | 36.0 | 57.7 |
| 256K | 1 | 16K | 1160.8 | 37.5 | 14.2 |
| 256K | 1 | 64K | 1171.5 | 36.3 | 56.3 |
| 256K | 1 | 128K | 1117.8 | 38.2 | 119.3 |

The shape: **decode is flat at 36-38 tok/s from 64K through 256K context** (the same 37-38 the 64K MAXN cells
give), prefill stays ~1.1K tok/s all the way to a 131K-token prompt, and what grows is TTFT - 128K prompt
means ~2 min before the first token, which is simply 131K tokens / ~1.1K tok/s. Four slots at 128K cost ~2
tok/s of decode and nothing else.

## The switches at 128K: none of them is a lever here

Same cells, paired arms, `iq3s-128k-maxn/`:

| arm | 16K prompt / decode | 64K prompt / decode |
|---|---:|---:|
| base (default: 5 sm_110 switches on, spec 4 + cost rule) | 1117.2 / 37.7 | 1066.7 / 38.0 |
| `STRATA_SPEC_COST=0` (cost rule off) | 1153.8 / 37.5 | 1184.2 / 36.6 |
| `STRATA_SM110_PERF=0` (whole table off) | 1052.2 / 36.6 | 1082.0 / 36.6 |
| spec 8 | 1091.5 / 36.1 | 1182.4 / 37.6 |
| `--kv-resident 131072` | 1123.2 / 37.5 | 1195.5 / 38.5 |
| spec 0 (drafting off) | refused: the native IQ pack requires `--spec T` with T >= 2 | - |

All of these land inside the session noise (~±8% prefill, ~±2 tok/s decode) - unlike the 32K thinking-on
suite, where the table's win was +20%/+10% prompt, the T=0 long-prompt cells are prefill-bandwidth-bound and
the switches do not separate. **The defaults are the setting at long context; leave them alone.**

## The cold-pool trap, caught the second time

Two early arms printed prefill of 504-844 tok/s and two later ones 508-961: those are first-round cells
after ~30 GB of previous-run page cache sat on the pack, caught by the same Tegra accounting as
`expert_cache_segmented_test`. Every table above is the second, warmed pair (dgx_bench writes both rounds;
the first-round rows are marked in the raw jsonl). The rule of thumb this machine teaches: **warm the pack
(64-token warm cell) before reading a number, and treat a sudden -30-50% prefill as cache state, not as the
switch under test.**
