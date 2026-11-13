# Live validation of two generated profiles

Author: **Francisco Angulo de Lafuente** · [ORCID](https://orcid.org/0009-0001-1634-7063)

The existing Strata release loaded both generated profiles successfully, with the
same **1,587 effective cache slots and 2,794,931,200 bytes**. All **nine generated
answers** passed the fixed Python/SQLite correctness checks. The two global-profile
controls differed by 0.0793% in total request time.

The order was A1 global, B code-domain, A2 global. The profiles were frozen from the
original 37 training requests, and the three coding prompts were new. The code
domain was selected after its positive exploratory signal. This is a small,
targeted compatibility/measurement pilot.

| Request | A1 global, seconds | B code, seconds | A2 global, seconds | Generated tokens A1/B/A2 |
| --- | ---: | ---: | ---: | ---: |
| clamp | 16.42 | 15.61 | 16.66 | 21/26/21 |
| interval overlap | 53.90 | 53.19 | 53.75 | 43/43/43 |
| positive-amount SQL | 12.87 | 14.69 | 12.71 | 20/20/20 |
| Total request wall time | 83.19 | 83.49 | 83.12 | |
| Startup and warmup, separately | 84.31 | 81.10 | 84.33 | |

The same engine binary, model files, sequence and settings were used. Relative to
the trace-collection configuration, all legs set prompt cache0, PCIe fraction0,
route prior0, ADAPT_NOWAIT0 and ADAPT_LAG1. Settings are visible in each configuration.
Profiles guide both the GPU cache and resident RAM tier; this experiment does not
isolate their effects.

## Reproduction files

- [result.json](result.json) retains every raw answer, request timing, usage count
  and the original gate assessment. Individual leg files and sanitized logs are
  included too.
- [LIVE_PROTOCOL.md](LIVE_PROTOCOL.md) and [PREREGISTRATION.json](PREREGISTRATION.json)
  were fixed before the pilot. The original execution-source hash and the export
  adjustment are recorded in [execution-receipt.json](execution-receipt.json).
- [capacity-plan.json](capacity-plan.json) and [native_experts.txt](native_experts.txt)
  explain the nominal1200-to-effective1587 allocation. The engine sizes native slots
  against a byte budget. The layer quotas and byte totals match in both profiles;
  the same calculation predicts the original shipped profile's1582 slots.
- `global.bin` and `code.bin` contain rankings, not model weights.
  [public-files.json](public-files.json) hashes the exported files.
- [runner.py](runner.py) is the runnable harness, with the lab directory taken from
  `baseline.json` rather than the original local hard-coded path. Replace `<LAB>`
  in the baseline with a prepared installation directory before running on Linux
  with the existing model and engine. The harness writes a new `out/`, runs one
  GPU and applies a24min inner cutoff. The managed execution used a25min outer
  cutoff and finished in513.18s, including startup and cleanup.

## Observations

The frozen speed-comparison gate requires identical text and token counts, all
quality checks, matched capacity, and at most10% difference between the controls.
The code profile generated a different correct clamp answer with26tokens instead
of21. **Output parity fails, so the speed comparison is invalid and the improvement
gate fails. No acceleration is established by this pilot.** The five-percent
improvement gate was retained. Raw times and both control repeats are preserved.

One B run, three short tasks, one model and one PCIe x1 RTX3090 do not estimate
performance or quality on other workloads or12GB cards. The main report's static
coverage result remains a separate observation on45profiling requests.
