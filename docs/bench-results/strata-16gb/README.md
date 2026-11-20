# Strata 16GB (RTX 5060 Ti) measurement harness

PR-A diagnostic capture for Windows 11 / RTX 5060 Ti 16GB.

## Rules

- Dedicated VRAM and Windows shared GPU memory are **separate fields**. Never merge them.
- `requested.context` vs `effective.context` must be compared explicitly. A silent shrink is a failed / non-comparable run.
- Do not commit prompt text, API keys, or private absolute paths.
- Attention KV theory (e.g. Hikari 131072 Q8_0 = 4.25 GiB) is **not** a full GPU fit claim.

## Capture baseline

```powershell
# From repo root
powershell -NoProfile -File .\bench\strata-16gb\run-windows.ps1
```

Writes `results/baseline-<timestamp>.json` with git SHAs, nvidia-smi dedicated VRAM, Windows shared usage, CPU/RAM, and a RunObservation shell.

## Related modules

- `apps/host/server/run-observation.mjs`
- `apps/host/server/vram-observation.mjs`
- `apps/host/server/model-memory.mjs`

## Rollback

Remove or ignore `bench/strata-16gb/results/*` local captures. The baseline capture remains a separate, explicit command.

## Opt-in lifecycle and attach-only capture

See [LIFECYCLE.md](LIFECYCLE.md) for startup capture enabled with
`STRATA_RUN_OBSERVATION=1` and the attach-only synthetic runner `run.mjs`.
Those observations use [run-observation.schema.json](run-observation.schema.json).
The baseline command, `measurement-schema.json` and existing results retain
their original contract. Unset the flag and restart the source host to disable
lifecycle capture.

## Phase 0 qualification ladder

`L1-near-full-ladder` in `workload-manifest.json` fixes the qualification
ladder at 32768 → 65536 → 98304 → 131072 with near-full input (>= 99% of
capacity), 3 repetitions per step, judged by the median run. Do not advance
to the next step until the current step completes all repetitions.

`measurement-schema.json` additionally accepts an optional `ram` summary
(free-RAM minimum, commit-headroom minimum, sample interval), an `outcome`
field (`completed` / `oom` / `watchdog_stop` / `manual_stop` /
`unresponsive`), and disk read/write byte counts in `metrics`. Runs that
end in `watchdog_stop` or `unresponsive` are not comparable.

The watchdog free-RAM floor and per-step timeouts live in the manifest's
`safety` block. The floor is provisional; verify the watchdog and the
manual stop with a dummy load before any run at 98304 or above.
