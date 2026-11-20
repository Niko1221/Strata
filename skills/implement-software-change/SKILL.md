---
name: implement-software-change
description: "Implement a scoped software change against a current worktree and verify its normal entry point."
---

# Implement Software Change

Identify the caller, intended behavior, and state that must survive retries, cancellation, and upgrades. Reproduce an observed failure at its real entry point when feasible. Locate the owning module and reuse the repository's existing flow. Inspect dirty changes and edit the smallest complete scope.

Exercise the regression and relevant existing contracts. For concurrent jobs, verify isolation of state and ownership. For edits, verify the resulting diff. Record the exact revision and checks, with separate results for any deployment or runtime work.
