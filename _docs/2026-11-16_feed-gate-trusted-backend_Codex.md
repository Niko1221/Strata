# Feed install gate: validate the selected backend

The synthetic feed-install gate now selects its backend contract from the
caller's backend argument. Previously a run config labelled Metal could bypass
the CUDA resident-budget check even when the caller selected CUDA. The existing
cross-backend regression failed before the correction and passes afterwards.

The change is one condition in tools/feed_model_install_gate.py. The caller
derives its backend from the platform and passes it explicitly. Actual Metal
validation still requires the Metal label, --mmap-experts and matching --native
and --ple-gguf. CUDA still requires --resident-budget-gib. Component resolution,
no draft/no vision and complete neutral expert-profile checks are unchanged.
The existing test and all its assertions are unchanged; it already covers the
CUDA/Metal distinction and the malformed Metal/profile cases.

Work started from 8bedc28f660d8c76f865bc2ed214ba25ca3bd7ad, following the
c77572806996349b49995e4f98d41b5294d36a21 intake. The earlier hosted Checks
failure at 75ab9a7bb3dcb8d75dc9ff047e481daafc5d3f80 remains failed evidence.
PC/Codex/project AGENTS, the canonical prompt and common, implementation-start,
skill-selection, security, application and Python SOPs were applied. Skills:
milspec-codex-standard, debug-failing-test and review-only
agent-self-review-evolution. No governance or memory file was changed.

Local validation used the existing offline uv/Python 3.12 guarded runner, one
serial slot, unique attempts and explicit --allow-low-ram. Commit/disk floors,
owned Windows Job containment/drain, deadline and output limits remained.
The unchanged failing module produced 1 failed and 4 passed in 30.70s before
the fix. After the fix, it and adjacent signed-feed model/engine setup modules
produced 46 passed, no skips, in 61.34s. AST parsing, Ruff fatal checks
(E9/F63/F7/F82), diff whitespace and unchanged regression/resource-guard source
checks passed. Each Job finished, recorded its native exit and drained to zero
active processes. Exact commands, raw receipts, logs, XML, source bytes and
hashes are retained in the scoped review evidence.

Self-review-event: qa_defect. The existing cross-backend fixture detected the
wrong contract selector. Weakening its assertion or adding a duplicate test
was rejected; no instruction update is needed. Rollback is a reviewed
restoration of this single condition. Independent review and final exact-source
hosted CI remain required. This helper correction does not qualify native
engines, model quality, full-window inference or GPU concurrency. No product
build, real-model inference, installation, signing or publication was run;
releaseQualified=false remains unchanged.
