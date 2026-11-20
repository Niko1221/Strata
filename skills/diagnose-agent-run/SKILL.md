---
name: diagnose-agent-run
description: "Diagnose a stalled or repetitive long-running model or agent session from its actual progress."
---

# Diagnose Agent Run

Compare two recent observations of the same run. Identify whether the engine is reading fresh tokens, decoding, waiting for a tool, blocked on another writer, or waiting for a helper. Inspect the last completed action, artifact and timestamps. Treat elapsed time as a symptom. Repeated calls with identical verified outcomes are a stronger loop signal than changing timestamps.

Choose the smallest recovery for the measured phase. Preserve the active task and unaffected workers. Report the observed blocker, last useful result, and the next verification.
