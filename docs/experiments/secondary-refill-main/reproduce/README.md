# Recorded reproduction plan

Paths are recorded llm-60 examples. Build this branch, replace the binary, native GGUF, pack, MTP, Python and client-source paths in plan.json, then run python case.py LABEL for each listed case. Keep the configuration and stopping policy in the parent report. The client uses project Python dependencies and engine stdin/stdout.

For a default-off comparison, build the pinned upstream main and point a matching copied case at it. Component test/benchmark commands are in the parent report. Ordinary timing and profiling remain separate. These scripts do not provide full-model sanitizer or state-hash tests.
