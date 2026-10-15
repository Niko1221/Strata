# Data format and units

`requests.json` and `requests.csv` contain the same 360 records in execution order. JSON retains nulls/booleans; CSV serializes null as an empty cell and booleans as True/False.

| Fields | Meaning / unit |
|---|---|
| block, arm | Reversed execution block (1/2) and recorded configuration name |
| workload | chat, 4K, 16K, 32K, agent_initial, agent, or calibration; headline agent medians exclude agent_initial |
| run, prompt, turn | Zero-based repetition, prompt index, or agent turn as applicable; calibration uses run for repeat |
| ok, err | Request success and error; all measured records succeeded |
| prompt_tokens, reused_tokens, read_tokens | Effective prompt tokens, cached prefix tokens, newly read tokens; integer counts reported by API/engine |
| generated_tokens | Actual completion token count, including early stops; not the requested cap |
| prompt_tps, decode_tps | Separate engine prompt/decode throughput in tokens/s |
| prompt_tps_source, decode_tps_source | engine for all measured requests; no client total-time rate substituted |
| prompt_ms, decode_ms | Corresponding engine durations in milliseconds |
| ttft_s | Client request send to first nonempty streamed token, seconds |
| total_s | Client full request duration, seconds; excludes service loading |
| drafts_offered, drafts_accepted | Engine speculative token counts; not an answer-correctness score |

Per-service `official.json` preserves the sanitized upstream benchmark output, including individual requests, suite parameters, durations, engine info and configuration. Its machine GPU inventory lists both physically installed cards, including in SINGLE arms; use the explicit config/README GPU selection and numeric per-GPU telemetry to identify inference devices. Do not interpret the host VRAM sum as single-card inference VRAM. This is the inventory caveat already tracked in upstream #2062, not a new measurement bug report.

Per-service `calibration-probes.json` retains public prompts, repeat/index, complete streamed text, SHA-256 text hash, API timings/usage and digested metrics. Official suite output texts were not retained by the upstream result writer. Agent follow-up prompts depended on its generated answers during the original run.

`summary.json` contains pooled n/median/min/max statistics by arm and workload, token-weighted decode rates, generated-token totals, percent comparisons and per-service evidence. `runs/*/summary.json` separates load_wall_s, official suite_wall_s and full monitored service_wall_s. RAM uses GiB (2^30 bytes), telemetry VRAM uses MiB (2^20 bytes), power W, clocks MHz, temperature degrees C. Global swap summaries retain byte precision.

`telemetry.jsonl.gz` is a lossless gzip encoding of the sanitized numeric JSONL export (decompress with Python gzip or `gzip -dc`). Each row records a Unix epoch time, CPU utilization percent/temperature list, global available RAM and swap bytes, per-device disk counters, cgroup memory/current/peak/swap/events and GPU CSV. Raw cgroup files remain strings with Linux field names; memory values are bytes, event values are counts. Disk read/write counters are cumulative bytes; read_count/write_count are operations and read_time/write_time/busy_time milliseconds. Read deltas use last minus first row for nvme0n1, the parent of the measured model/root mount.

GPU CSV columns, in order: index, temperature.gpu (C), memory.used (MiB), memory.free (MiB), utilization.gpu (%), power.draw (W), clocks.sm (MHz), clocks.mem (MHz), pcie.link.gen.current, pcie.link.width.current, clocks_event_reasons.sw_thermal_slowdown. Index 0 is the 5070 Ti; index 1 is the 3060. Engine logical CUDA IDs can be reordered for the dual pipeline; the startup log and report identify the physical order.

`oc-receipt.json` records the unchanged LACT SHA-256 and before/after driver limits, not the owner's complete LACT configuration. `kernel-error-scan.json` return code 1 with stdout '-- No entries --' and empty stderr means no matching journal entries; kernel readability was checked separately in audits/kernel-log-access.json. Absence of error matches is bounded by these scans and sampling, not proof of universal stability.

`provenance.json` hashes the original operational binaries, weights, pack metadata, tokenizer/MTP assets and pinned upstream scripts. Redaction changes metadata text bytes, so package SHA256SUMS separately identifies every published file. No model or generated pack binary is included. Telemetry compression and this documentation were prepared after measurement; no timing number was changed and no extra GPU run was made.
