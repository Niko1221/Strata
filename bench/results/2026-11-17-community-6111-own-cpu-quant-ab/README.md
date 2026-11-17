# Community A/B: STRATA_OWN_CPU_QUANT on a single-socket Broadwell (issue #6111)

Measured on 2026-11-17 on a 10-core Broadwell box running 3x Tesla V100-SXM2-16GB.
Two engine binaries built from the **same commit (`c775728`, v0.2.15)** with only
`-DSTRATA_OWN_CPU_QUANT=ON` vs `=OFF` differing, run interleaved ON, OFF, ON, OFF
against `tools/strata_bench.py --long --runs 3`. Speed only: this A/B ran no
correctness pass (the machine's earlier report,
2026-11-16 "3x V100-SXM2-16GB power-limit sweep" — merged into the
Strata-V100 fork as PR jmnargi/Strata-V100#108 — carried needles 9/9 at
32k/128k/256k on the sibling model).

## What we measured, and why it is a different datapoint

Issue #6111 reports `STRATA_OWN_CPU_QUANT=ON` (the 0.2.15 x86_64 default) being
~4% slower than `ggml-cpu` for CPU-expert decode on AVX2, on a dual-socket
32-core host. This machine is the other corner of the design space: a **single
Xeon E5-2630 v4 (10 cores / 20 threads)** with AVX2 and no AVX-512, where the
expert pool sizes itself at **9 workers** and the decode hit rate sits at
99.6–99.9%, so the CPU quantization path is rarely on the critical path at all.

**Result: no regression here.** ON is +0.4% to +3.1% over OFF depending on the
test (within or near the arm-to-arm noise), never slower. Whole-run CPU
utilisation was ~5–6% median across all 20 logical cores in every arm
(`mpstat -P ALL 1`), with exactly one core above 50%.

## Hardware and software

- GPU: 3x Tesla V100-SXM2-16GB, power limit 300 W (measured draw ~51 W after
  the run), SXM2, driver 580.178.04, CUDA toolkit 12.6 used for the build
- CPU: Intel Xeon E5-2630 v4 @ 2.20GHz, 10 cores / 20 threads, AVX2, no AVX-512;
  125.7 GiB RAM; Linux 6.8.0-147-generic
- Strata commit `c7757280` (the 0.2.15 release test); engine self-reported
  version 0.1.61; both binaries built with `cmake -DSTRATA_OWN_CPU_QUANT=ON|OFF`
  and `CMAKE_CUDA_ARCHITECTURES=70` (CMakeCache values re-checked before each run)
- Model: `unsloth/Qwen3.8-Flash-Next-UD-IQ3_XXS-GGUF`, all 4 shards, sha256
  matching the `setup.py` pins; pack built with
  `setup.py --prepare-pack --family unsloth --model UD-IQ3_XXS` (1079 tensors,
  303 served natively); MTP runtime from `strata-data-v31/mtp/rt`
- Production llama-swap was stopped for the whole window; nothing else ran on
  the CPU or GPUs during any arm (checked: no compute apps before each start)

## Configuration (identical in both arms)

```text
python serve/server.py --engine strata --config <arm>.json --port 18211
  exe = strata/build-(on|off)/strata
  --pack pack --native <shard1> --ple-gguf <shard4>
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5 --mtp <mtp/rt>
  --max-context 262144 --kv int8 --ple-io ram --kv-resident 32768
  --conversation-cache-mib 24576 --conversation-cache-slots 8
  --conversation-cache-min-tokens 4096 --vram-reserve-mib 1500
```

Both engine logs confirm the same code path: `CPU expert kernels: AVX-2`,
`9 pool workers on logical processors 1..9, the host thread on 0`, expert cache
33,895 MiB / 17,453 slots. Note the 4-shard Unsloth UD-IQ3_XXS carries its PLE
n-gram table in shard 4, so `--ple-gguf` is required; without it the engine
silently skips PLE and the decode numbers move.

## Method

Model load is not timed; `/health` gate, then one 64-token warm-up chat, then
`strata_bench.py --long --runs 3` (suite: short chat, fresh 4K/16K/32K prompts,
agent sessions; every fresh prompt carries a unique tag). Arms interleaved
ON, OFF, ON, OFF on the same day; after each arm the engine was stopped and the
bench waited for the port to be released before the next start. Each
`results.json` below is the untouched tool output.

## Results (decode tok/s for chat/agent, prompt tok/s for fresh prompts; medians, range in brackets)

| Test | ON #1 | OFF #2 | ON #3 | OFF #4 | ON med | OFF med | ON vs OFF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chat decode | 84.2 (68.8–112.2) | 83.6 (69.1–115.1) | 84.4 (70.4–114.8) | 81.4 (71.9–113.4) | 84.3 | 82.5 | **+2.2%** |
| agent decode | 85.0 (62.3–112.9) | 84.4 (40.9–109.6) | 87.2 (56.9–103.6) | 81.5 (54.1–102.0) | 85.5 | 83.0 | **+3.1%** |
| 4K prompt | 716.3 | 716.2 | 716.6 | 715.3 | 716.5 | 715.8 | +0.1% |
| 16K prompt | 1345.6 | 1343.4 | 1346.3 | 1340.9 | 1346.0 | 1342.2 | +0.3% |
| 32K prompt | 1731.0 | 1743.7 | 1739.3 | 1742.2 | 1735.2 | 1743.0 | -0.4% |
| 32K decode (n=3/arm) | 73.2 | 64.1 | 70.0 | 70.0 | 71.6 | 67.1 | +6.7% (noise) |

- chat TTFT identical to two decimals in all four arms (0.39 s); fresh-prompt
  TTFT 5.6 / 12.0 / 18.5 s in every arm — prefill is GPU-side and untouched by
  the flag, as expected.
- agent KV hit rate: 69.13–69.19% in all four arms; decode expert hit 99.6–99.9%
  in both binaries (engine log).
- mpstat (whole arm, 1 s samples, 20 logical cores): median "all" 5.5–5.9%,
  mean per-core busy 2.1–5.4%, cores >50% busy: 0–1 in every arm.

## Reading

On this machine the 300 W / 3-card / 10-core combination leaves the CPU expert
path cold (≈5% CPU), so the flag barely matters, and the small ON-side tilt
(+2–3% on the decode-heavy tests, repeated in both ON arms) is the opposite
sign of #6111's -4%. We do not claim a win — the deltas sit next to the
arm-to-arm noise (OFF #4 is 2–3% below OFF #2 with no flag change) — only that
**the reported regression did not reproduce on a single-socket 10-core Broadwell
at 99.6–99.9% hit rates**, which is consistent with the hypothesis that the
#6111 effect is pool-size-sensitive (30 workers there, 9 here).

## Reproduction notes (pitfalls that cost us two failed runs)

- The serve wrapper needs the install's venv python (needs `regex`/`jinja2`);
  the bare system python fails at import.
- After `SIGTERM` the engine keeps the API port for tens of seconds while it
  drains; a next arm started on the same port fails with "port already in use".
  Wait for the listen socket to close, don't sleep a fixed 5 s.
- Ports: this box proxies production llama-swap on a fixed engine port; pick a
  bench port away from the production proxy port (the engine on the prod port
  answers `/health` and looks "ours" if you probe blindly).

## Files

- `arm-on-1/results.json`, `arm-off-2/results.json`, `arm-on-3/results.json`,
  `arm-off-4/results.json` — untouched `strata_bench.py --long` output
- `arms-summary.json` — the medians above, machine-computed from the four files
