# Delivery workflow

1. Inspect nested instructions, manifests, lockfiles and existing CI. Identify the
   actual package manager and supported runtime versions; do not install guessed tools.
2. Establish the relevant baseline. For small edits use targeted checks; run the
   configured complete verification before delivery when feasible. Report pre-existing failures.
3. State acceptance criteria and the smallest useful plan for complex tasks.
4. Implement the scoped change. Add meaningful tests for changed behavior or risk;
   do not add tests that merely duplicate implementation.
5. Review the diff for regressions, security issues and boundary violations.
6. Run verification and report evidence. Missing checks remain missing, never passed.
7. For long work, record decisions, outstanding work and reproduction commands in
   a project handoff; avoid duplicating the entire conversation.

## Verification configuration
Edit `.agentic/checks.json`. Every check has `name`, `cwd` (relative to repo),
`argv` (argument array) and optional `timeout` seconds (default 900).
Commands run sequentially from the specified directory, without a shell.
Use non-mutating checks, e.g. format-check rather than format-write.
An empty list exits 2. A command failure retains its failure status in the log.
Missing executables, timeout, bad configuration and failing checks exit nonzero.
Use separate checks for each package in a monorepo and for architecture invariants.
A successful run means only the configured commands passed, not zero regressions.

Reuse existing project commands, including scripts/verify.sh if appropriate.
Never configure a command that calls this verifier recursively.
For Node select the repo's existing lockfile/packageManager and non-watch scripts.
For Python use the repo's existing environment or explicit runner (uv, tox, etc.).
For C/C++ reuse its configured build and CTest/Make targets; do not guess them.
Boundary checking requires actual repo-specific rules and tools, not a placeholder.

## CI and agent integration
Keep existing CI runtime setup, dependency installation and caching. Add
`python3 .agentic/verify.py` after those steps when checks are configured. Do not
replace existing workflows or hooks. Local hooks are optional, not enforcement.
AGENTS.md is the canonical instruction entrypoint when supported. If an existing
AGENTS.md or CLAUDE.md was preserved, incorporate the workflow reference deliberately.
For other agents, verify the installed client's documented instruction-loading
mechanism; arbitrary DSH.md or ZCODE.md names are not assumed to work.
