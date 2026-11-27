# Provenance (Inkling-2-Mini via the Strata inkling2 engine, 2026-11-23)

- Engine: component `shapes/inkling2/engine/0.1.66-linux-x86_64-cuda-cuda13` (strata-inkling 38.7 MB) on
  terminal v0.2.20, on top of the 0.1.66 A/B of the same day (see the sibling 0.1.66 folder).
- Model: Inkling-2-Mini SQ-20 `.strata` (19.35 GB) + Vision shard (0.64 GB), all experts pinned to VRAM
  (`--vram-experts-gb 15.68 --ram-experts-gb 0.00`), `--kv fp8`, context 262144 (daily config, parallel 1).
- DSpark drafter: `dspark-Inkling-2-Mini-Q8_0.gguf` (402278560 B, SHA 2ec78510...f3773, from the official
  unsloth GGUF repo) was passed (`--dspark <file> --draft dspark`) and the engine accepted it with no
  "Skipping DSpark drafter" line - **but it changed nothing measurable** (chat decode 161.3 vs 155.0
  without, agent 141.8 vs 143.0; within noise). On this `.strata` native build the drafter appears to be a
  no-op; the companion GGUF drafter is documented for Unsloth GGUF builds. Left in the config for future
  engine versions that honour it.
- Conversation cache: this engine reports 0% cache reuse on the agent session (every turn re-reads the
  full conversation: 37k re-read tokens/session vs 5 for the Qwen UD pack on the same suite). No
  correctness impact, latency impact on multi-turn agents.
- Farm runs (`farm-results-N.json`, `farm-report.md`): same farm config at context 32768, `--parallel`
  4/8/16/32, one official suite each with `--clients N` (single burst round, 512 tokens per round),
  servers started by the harness; background was audited and idle; no chat traffic during any run.
- State: box idle, page cache warm from real use; suite run at 15:58 and farm sweep at 16:12 CET.
