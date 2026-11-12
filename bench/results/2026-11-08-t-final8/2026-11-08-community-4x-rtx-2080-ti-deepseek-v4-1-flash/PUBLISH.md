# [Benchmark] DeepSeek-V4.1-Flash on 4x RTX 2080 Ti (sm_75): 39.1 tok/s chat, 539/905/1035 tok/s prompt

Issue or pull-request body for this report. It follows CONTRIBUTING.md: the problem and the result come first, then the
approach, then the validation split into passed / failed / not run, then the limits. One problem: publish a community
measurement, with no engine change in the same pull request.

## Problem

A user with four RTX 2080 Ti cards (22 GB each, sm_75, no AVX-512 on the CPU) asked what DeepSeek-V4.1-Flash does on this
class of machine, and the existing community reports cover RTX 5090, TITAN RTX, MI50, B60, RTX 4000, V100, R9700, Strix
Halo, 3060, RX 9070, RX 9060 XT, RTX 4070 and 780M — not four Turing cards with 88 GiB of VRAM between them.

## Result

Same box, ISTA-DASLab 2.00 bpw pack, DSpark `Q8_0` drafter, 131,072-token context, `--tp 4 --experts 5 --threads on`,
5 `--long` runs per test, `tools/strata_bench.py`:

| test | median | range |
| --- | --- | --- |
| short chat decode | 39.1 tok/s | 26.4-51.8 |
| agent-session decode | 31.4 tok/s | 25.5-39.3 |
| agent turn TTFT | 11.93 s | 8.26-15.27 |
| 4K prompt | 539.4 tok/s | 467.5-556.4 |
| 16K prompt | 905.4 tok/s | 871.3-917.3 |
| 32K prompt | 1035.0 tok/s | 984.6-1052.0 |

A 128-token decode at 32K runs at 34.96 ms a token (28.4 tok/s), and the engine's own split of that token is in the report:
12.5-15.3 ms of CPU lane, 7.7 ms of GPU expert matvec, 26.25 ms of final device sync that overlaps the rest.

For context, the same engine commit with the FAT/slab path off and no 075 profile, measured the day before: chat 31.0,
agent 28.3, prompts 426.0 / 673.2 / 764.8.

## Approach

Model identity: ISTA-DASLab 2.00 bpw (GSQ-RCO, IQ2_XXS routed experts) of DeepSeek-V4.1-Flash-1017, pinned by the four
`...-IQ2_XXS-0000N-of-00004.gguf` shards, the pack's `ds4.txt` (sha256 `d10cf9e3…62ff1`) and `tensors.txt` (sha256
`46197620…97395`), plus the DSpark `Q8_0` drafter `6ba6ac1b…3927a`. The report's Model identity section lists all of them.

Nothing in the engine changed for these numbers. The report is the install's own arguments plus three engine environment
variables, all recorded in the folder (`config.json`, `env.txt`):

- `DS4_ASYNC_T=8` — the drafter's verify window is T=6, so anything below 6 runs every verify round synchronously. This is
  also the engine's default; the explicit value only overrides an older preset.
- `DS4_DEFAULTS=075` — the measured 2.00 bpw speed profile of the expert path (opt-in, changes numerics).
- `DS4_ATTN_SPLIT=0` — the default attention split, pinned so the run cannot inherit an experimental value.

## Validation

Passed:

- The suite ran to completion on this configuration with no failed requests and no skipped tests (`results.json` records
  `failures: []` and no skipped test).
- Every number above is a median in the folder's `results.json`; the arms are in `bench/results/2026-11-08-t-*` and
  `bench/results/2026-11-07-t-075base`.
- The adopted configuration was measured in **four separate runs**: `t-async8a`, `t-async8b`, `t-async8c` (chat and agent
  only) and `t-final8` (the full suite published here). Agent-session decode medians: 32.0 / 31.0 / 31.9 / 31.4 tok/s,
  i.e. the number repeats inside the range quoted above. `t-final` is the same engine with `DS4_ASYNC_T=4` and is a
  different configuration, not a repeat of this one.

Failed: nothing in this work; no test failed.

Not run:

- HIP/ROCm build of this engine: no HIP toolchain on this machine, so the AMD path is unmeasured here.
- Windows packaged build: no Windows machine available.
- The standard protocol (`strata bench run`, BENCHMARK_PROTOCOL.md): this is a `tools/strata_bench.py` report, which the
  guide allows for measurements made outside the protocol. It is labelled as such in the README.
- Correctness or quality of the answers: this is a speed suite. The 075 profile's numerics are stated separately in the
  README (top-1 87.07 %, mean KL 0.0371 against the reference profile on `data/quant_quality/v1`, and a needle test that
  found 6 of 6 in both profiles at 32K and 128K).

## Limits

This box's chat decode drifts with page-cache warmth: identical baselines read 34.0, 35.5, 37.5 and 36.3 chat in one
afternoon, which is why chat is quoted with its full range and each comparison is against an arm measured within the hour.
Agent decode is the stabler metric (28.3-30.5 in every baseline). Prompt medians moved under 1 % between arms of the same
length. The machine is one unit of this hardware: another 4x 2080 Ti machine will differ in clock and cooling, and four
cards on PCIe switches (two NVLink pairs, no cross-pair link) is a specific topology.
