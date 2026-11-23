# Provenance (additions by the submitter, 2026-11-23)

- Engine: release tarball `strata-engine-0.1.66-linux-x86_64-cuda-cuda13` (not a git checkout).
  Terminal project at tag v0.2.20. Engine 0.1.62 (git `58ee322`) was installed on the same
  machine until the morning of 2026-11-23; both A/B directions are switchable and verified
  with `./strata --version`.
- `STRATA_HC_Q8=0` is set in the server env as the documented workaround for the 0.1.66
  start crash on sm_120 (see issues #6712 / #6704 / #6767: `q8_ld16()` `__ldg()` on shared
  memory -> "verify: layer 0 never rang" during window warm-up). On 0.1.62 the same config
  ran with HC_Q8 enabled (the default for UD packs). So every 0.1.62 -> 0.1.66 delta below
  includes the cost of the workaround, not just the engine change.
- Server flags (unchanged across both engines): `--mmap-experts --kv-resident 32768
  --max-context 262144 --kv int8 --vision --vram-reserve-mib 700 --draft dflash2
  --draft-conf 0.5 --spec 8 --vram-elastic`, expert-profile + expert-cache auto.
- State before the measured runs: server up 40 min (started 10:56, suite at 11:26, stbench
  at 11:34), page cache warm from real chat traffic. Background was audited and cleaned
  immediately before the runs (one 70%-CPU swiftshader gpu-process from an abandoned
  headless Chrome was killed; a dev server and two orphan browser daemons were stopped).
  No chat or agent traffic during either measured run.
- `stbench-*` files: the same machine's own long-context matrix script (engine-internal
  timings, temp 0, 2 runs per point, 1K..262K), used for the 0.1.62 <-> 0.1.66 A/B because
  the 06-11 baseline was measured with exactly this script (see root README).
- Read the numbers with two eyes open: the first sample of every point after an idle gap
  rides the GPU clock ramp (visible as low `#1` values), so use the `#2` samples as the
  steady state; the `n-a-cache` prefill rows are conversation-cache hits and carry no
  prefill rate by definition.
