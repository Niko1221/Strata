---
name: qualify-build-and-release
description: "Qualify a build, installer, or deployed revision and prepare a source-backed handoff or pull request."
---

# Qualify Build And Release

Pin the clean source revision and reproduce the supported build. Match packaged payloads to the source or expected digests. Before an authorized install, check for live work that would be interrupted. After startup, verify the installed revision and a representative operation on the installed path.

Record source, build, installer, installed state, and observed behavior with their own receipts. For a Strata PR, use its area prefix and template, with exact commands under Passed, Failed, and Not run. Leave product or device acceptance to the recorded acceptance decision.
