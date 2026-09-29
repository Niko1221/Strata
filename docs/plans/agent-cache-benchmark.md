# Controlled local Pi cache comparison

`tools/agent_cache_benchmark.py` runs a bounded real coding task through the Pi
SDK with parking disabled and enabled. It defaults to a dry run and never uses
an existing server. `--run` requires an exclusive GPU window, the configured
model assets, Node.js, and an installed `@earendil-works/pi-coding-agent` package.

Both runs start from identical files and use the same working-directory path,
prompts, model configuration and tool limits. Each loads a fresh private server.
The workflow is controller A implementing a CSV-report library, worker B adding
its CLI, then A returning to verify/fix the integration. This exercises real file
edits and shell checks rather than replaying prerecorded model responses. The
input corpus is generated public test data, not a user conversation.

The task uses the Python standard library. Ten independent checks cover UTC
date grouping, decimal sums, sorting, invalid input, last-valid duplicate IDs,
quoted commas, file/stdin CLI behavior and missing-file status. These validators
run outside the agent workspace after the workflow finishes. A model's assertion
that the task succeeded is not accepted as validation.

Each phase allows 12 executed tools, 16 model turns and 240 seconds. Bash calls
are limited to 15 seconds and returned text to 16,000 characters per tool result.
Requests use greedy sampling, disabled thinking, and at most 2,048 output tokens.
No compaction, skills, discovered extensions, inherited context files or remote
model catalogs are loaded. Generated event logs remain local; review them before
publishing any benchmark artifact.

Example (paths to the local model configuration and Pi installation are supplied
by the operator):

```sh
python tools/agent_cache_benchmark.py \
  --config /path/to/engine-config.json --engine build-validation/strata \
  --pi-package /path/to/node_modules/@earendil-works/pi-coding-agent \
  --output /path/to/new-results --run
```

Use a fixed expert allocation small enough that neither engine trims it for
VRAM headroom. The harness rejects mismatched engine settings. `--order on,off`
supports a reverse-order repetition; changing the order is not a substitute for
reporting repeat variability.

Results include task wall time (excluding model startup), per-phase/tool times,
prompt/decode times, token reuse and generation volume, engine configuration,
independent success status, generated files, and the engine's Linux kernel
high-water RSS. RSS includes startup and model mappings; it is not a system-wide
or GPU peak-memory measurement. Normal speculative inference can produce
different tool paths, so report output/task differences and do not infer a
general speedup from a single pair. This local workload does not reproduce the
earlier remote Windows game-building workflow.
