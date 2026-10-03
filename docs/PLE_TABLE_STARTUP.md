# Parallel table loading candidate

Review: [upstream PR #651](https://github.com/Niko1221/Strata/pull/651),
head `02406df1111759ba1ccc841001f540ded9dad24f`, read 2026-10-03.

The useful new performance idea is concurrent page faulting before `mlock`.
The PR uses up to 16 workers and 64 MiB chunks; its reported cold-read throughput
is about 0.47 to 2 GiB/s on ZFS. This changes model startup. Once the same table
is resident, the read/dequantization path has not become a new GPU kernel.

Its Q8_0 reader overlaps our existing Q8 support. Q5_1 support is additional.
The table's quantization is independent of the rest of the model: a Q6 model can
use a Q8 table. Their RAM-vs-direct decode measurements use a different Q5_K_M
model and two RTX 3090s; they are not predictions for full Q8 on llm-60.

## Proposed isolated experiment

- Use the current adaptive base, preserving the enum-based format handling and
  existing malformed-metadata, page-boundary and original-byte oracle checks.
- Add only parallel page faulting, with a worker-count control (1, 4, 8, 16).
  Keep the locked pages, bytes, format, inference configuration and model fixed.
- Measure model load time separately from prompt processing and generation.
  Record page residency before/after, major faults, CPU time, file-read bytes,
  storage throughput, table-lock time, model-ready time and available RAM.
- Test warm and cold file-cache conditions. Evict only the table's file range
  with `posix_fadvise` while no job uses it; verify residency with `mincore`.
  Do not globally drop the host's caches. If eviction fails, label that run warm
  or partially resident instead of claiming a cold comparison.
- After load, compare a matched inference request and table-row oracle. Startup
  placement may change timing but must preserve table contents and outputs.
- Include lock failure and worker-count clamping. Already-touched pages must not
  be touched a second time after a failed lock; record that they are reclaimable.

This experiment is planned, not measured on llm-60. It is behind the current
ESP and adaptive Q4/Q8 KV matrix, so the model tests keep their binary unchanged.
If the NVMe/warm-cache measurements show no useful load-time gain, stop here and
prioritize the Q8 expert miss path rather than attributing a decode gain to it.
