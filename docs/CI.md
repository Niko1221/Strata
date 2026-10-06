# Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main` and the live-memory contribution branch, pull requests to
`main` (including Drafts), and manual dispatch. It uses GitHub-hosted Ubuntu 24.04 and Windows 2022 runners
with read-only repository permission and no deployment step.

The Python jobs install `requirements.txt` and the optional JSON-schema validator, then run all server
unit tests and the parallel setup tests. The tests use mock engines, synthetic protocol processes and local
HTTP services; they do not download or load model weights. Pack-tokenizer tests skip without a tokenizer,
and Windows job-object tests skip on Linux. These skips remain visible in the test output.

The native jobs configure with CUDA, HIP, native experts and the general test suite disabled. The standalone
conversation tests are enabled, and only these five targets are built and run:

- `conversation_cache_test`
- `conversation_memory_test`
- `conv_cache_test`
- `serve_window_test`
- `coupled_draft_test`

They check cache ownership/budget arithmetic, memory telemetry and serving-prefix/speculative-state rules
without a GPU, model files, GGML downloads or ISA-specific expert kernels. Native CTest logs are saved as
run artifacts; Python test output is in the job log. The same commands are listed in the workflow.

This CI does not test CUDA VMM release/regrowth, live RAM allocation, GPU graphs, model quality or inference
throughput. `live_memory_test --protocol-only` still needs the CUDA-linked target, so it is not part of these
hosted jobs. Keep local GPU/native and real-model acceptance separate, as described in `LIVE_MEMORY.md`.

Fork push runs can execute in the contributor repository. A run in the upstream repository may require a
maintainer's approval under its fork-workflow policy. A passing fork run does not imply upstream approval,
required branch protection, merge or local runtime deployment.
