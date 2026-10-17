# Methods and data fields

## Comparisons

The complete schedule contains A/A pair 0, A/B pairs 1–12 and A/A pair 13. `arm` is a logical comparison label. **For A/A, both logical labels (`on` and `auto`) run with `engine_arm=on`.** Figures normalize A/A differences to the second run relative to the first; raw pair columns retain logical ON/AUTO labels and `order` identifies chronology.

A measured English/code pair is included when `run_ab.compare` accepts output IDs, full 600-token length, finish reason, prompt reuse, positive decode timing and zero expert-file reads; MTP accepted/offered counts and every preceding request's output/work must also match. Eligibility does not depend on the speed result. Tier-counter equality is reported as an additional diagnostic, not an extra post-run filter.

The primary throughput estimate is `100 * (exp(mean(log(ON_decode_ms / AUTO_decode_ms))) - 1)`. CPU reduction is `100 * (1 - exp(mean(log(AUTO_cpu_s / ON_cpu_s))))`. Whole-request latency uses the same reduction form. The percentile bootstrap resamples complete engine pairs 50,000 times with seed 1922476. Tokens and requests within an engine are not independent resampling units. The primary workload is English; fewer than eight matched pairs is exploratory. The protocol's practical throughput band is ±2%.

The recorded schedule, warmup length, workload order and sample counts were fixed before timed results. One startup was rejected for partial residency before measurements. The subsequent memory-admission amendment added a bounded wait for released memory; the planned comparisons stayed unchanged.

## Fields

| Location | Meaning and units |
|---|---|
| `results.arms` | Chronological process starts; pair, logical arm, actual mode, INFO and memory-admission samples |
| `requests.output_ids` | Generated token IDs in order; `produced` matches their length |
| `prompt_ms`, `decode_ms` | Engine-reported milliseconds from `DONE` |
| `ttft_s`, `wall_s` | Client monotonic seconds from writing GEN to first token / DONE; excludes loading |
| `process_cpu_s` | Process user+system CPU delta across the whole request, including prefill |
| `draft_accepted`, `draft_offered` | MTP work counters used for comparability |
| `reused`, `prompt_read` | Reused and freshly processed prompt tokens |
| `decode_file_blobs` | Expert-provider file reads during decode; not all OS disk activity |
| `raw_done` | Exact DONE line, including GPU hits, expert lookups, RAM blobs and offload counts |
| `memory_before`, `memory_after` | psutil process memory snapshots; Windows page-fault count includes soft faults |
| `system_available_*` | Available host RAM in bytes at request boundaries |
| `host_cpu_before`, `host_cpu_after` | Cumulative host CPU seconds; their difference minus process CPU is not a precise attribution to other applications |
| `comparisons` | Matched eligibility and every exclusion reason |
| `restored` | Recorded controller restored its original service; the portable runner leaves this null because it manages no existing service |

`requests.csv` supplies derived token rates and hashes for all 84 streams. Full output IDs remain in JSON; `outputs.json` deduplicates the decoded text. `pairs.csv` contains every measured English/code comparison, including A/A and unmatched work.

## Earlier check

`earlier-results.json` preserves the previous five ON/AUTO pairs on the same executable with a 150-token warmup and English only. Its throughput median was -0.22%; the geometric mean was +0.66%, with an approximate paired-bootstrap interval of -1.59% to +3.38%. It is not pooled with the 12-pair run because the warmup and request history differ. The earlier and current intervals overlap.

## Input provenance

The expert profile has SHA256 `a514661124d5305effe9c1bcecff7dc069af398b2b19ff9d11628a2cad0a8915`. It and the model/draft pack are external inputs. The largest aligned expert blob is 2662400 bytes. Budget 4416 gives 11757158400 bytes and fits the first 5759 profile-ranked slots (11756288000 bytes). Startup INFO is checked rather than assuming that the CLI budget equals the packed slot count.

The publication copies the numeric request records and token arrays exactly. It removes original service IDs/configuration metadata and replaces local launch paths. The source controller is a path-sanitized record, separate from the portable rerun script. Driver/OS/SDK metadata was read after the measurement window and is timestamped in provenance.json.

## Additional diagnostic

The retrospective leave-one-pair-out English throughput estimates range from +1.34% to +1.95%; [sensitivity.json](data/sensitivity.json) retains each value. This diagnostic does not change the primary cohort or estimate.
