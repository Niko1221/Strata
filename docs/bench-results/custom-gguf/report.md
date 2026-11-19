# Strata Custom GGUF & Empirical Model Qualification Report (Protocol v2)

> [!IMPORTANT]
> **AUDITED EMPIRICAL STATUS (2026-10-25 JST - Tasks R01 & R16)**:
> In strict compliance with the corrective repair plan (`Strata_Antigravity_修復実装計画_20261025.md`), all previous synthetic dry-run entries have been revoked. Under **Protocol v2** (`STRATA-CUSTOM-GGUF-EVAL-V2`), qualification requires physical weight presence, bounded container inspection, memory headroom validation against live hardware, and cryptographic log verification of real multi-turn token generation.

- **Protocol**: `STRATA-CUSTOM-GGUF-EVAL-V2` (Active)
- **Date**: 2026-10-25 JST
- **Verifier**: Antigravity / MILSPEC-Codex Standard
- **Execution Target**: Windows 11 x64, 32 GB RAM, NVIDIA GeForce RTX 5060 Ti 16 GB (`sm_120`), AVX2

---

## 1. Executive Summary

Strata v0.3.0 establishes an arbitrary GGUF container inspection pipeline, atomic model registry transactions, memory admission gates, and refusal-direction steering isolation.

Empirical evaluation of candidate models on local storage was conducted under the four tiers of Protocol v2:

| Candidate Model | Physical File Presence | Architecture & Container | 32 GB Memory Budget | Qualification Status |
| :--- | :--- | :--- | :--- | :--- |
| **DeepSeek-V4.1-Flash** (Standard ISTA) | Shard 1 of 6 present (5.7 MB) | `dsv41` MLA detected | Shards 2–6 missing | **BLOCKED** (`real_model_qualified: false`) |
| **DeepSeek-Maya-S** (Kiwimaker Abliterated) | Target slot 1 recipe only | `dsv41` recipe | Base shards missing | **BLOCKED** (`real_model_qualified: false`) |
| **DeepSeek-V4.1** (AtomicChat Uncensored) | Not present on disk | `dsv41` | N/A | **BLOCKED** (`real_model_qualified: false`) |
| **Swift-1.5-Qwen3.8-Flash-Next** (Abliterated) | Complete (63.4 GiB across 2 shards) | `qwen4exp` (mapped to `swift`) | Exceeds 32 GB Host RAM | **BLOCKED_BY_MEMORY** (`real_model_qualified: false`) |
| **Qwen3.8-Flash-Next** (`Q2_0`) | Complete (21.2 GiB across 2 shards) | `qwen` / `standard` | Fits static 32 GB budget | **READY_FOR_INFERENCE_ON_HEADROOM** |

---

## 2. Hardware Environment Snapshot

Measurements captured on live host via `strata_control.model_memory.capture_live_hardware_snapshot`:

| Parameter | Measured Value | Constraint / Limit | Status |
| :--- | :--- | :--- | :--- |
| **Physical Host RAM** | 34.22 GB (~31.87 GiB) | 32.0 GiB Target Baseline | **NOMINAL** |
| **Available Host RAM** | ~10.16–13.0 GB | >= Planned RAM + 4.0 GiB Headroom Required | **DYNAMIC_CHECK** |
| **System Commit Limit** | 68.58 GB | > 48.0 GiB Required | **PASS** |
| **GPU Model** | NVIDIA GeForce RTX 5060 Ti | Blackwell Class (`sm_120`) | **CONFIRMED** |
| **Total VRAM** | 16,311 MiB (15.93 GiB) | 16.0 GiB Target Baseline | **NOMINAL** |
| **Free VRAM** | ~14.0 GiB | >= 10.0 GiB Required | **PASS** |

---

## 3. Detailed Audit Findings

### DeepSeek-V4.1 Derivatives (Candidates 1–3)
- Physical disk inspection reveals that only `DeepSeek-V4.1-Flash-1017-Uncensored-2.67bpw-IQ1_M-00001-of-00006.gguf` (5.7 MB) exists in `H:\Strata\models`. Shards 2 through 6 are not present on local storage.
- In strict adherence to MILSPEC rules, zero synthetic tokens or mock passes are issued. The candidates are truthfully designated as **BLOCKED** pending acquisition of the complete 6-part shard set.

### Swift-1.5 Abliterated (`H:\Strata\models\SC117`)
- Full model weights (Shard 1: 39.79 GB, Shard 2: 28.36 GB) and official projector `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907.5 MB) are verified on disk.
- Container inspection successfully resolves the architecture as `swift` / `IQ2_XS`.
- Because the uncompressed weight footprint of 63.4 GiB exceeds the 32 GB physical host RAM budget, single-host launch without offloading is safely rejected with `BLOCKED_BY_MEMORY`.

### Qwen3.8-Flash-Next Q2_0 (`H:\Strata\models\q2_0`)
- Complete model shards (21.2 GiB) verified on disk.
- The model architecture and quantization dialect (`standard` / `Q2_0`) are qualifiable within the static 32 GB budget (`planned_host_bytes: 18.79 GB`).
- Launch readiness dynamically verifies that sufficient free physical RAM (18.79 GB + 4 GB OS headroom = 22.79 GB) is available at runtime before binding server resources.

---

## 4. Verification Evidence & Ledger Links

- **Protocol v2 Specification**: [bench/custom-gguf/protocol-v2.json](file:///C:/Users/downl/Documents/ChatGPT/Strata/bench/custom-gguf/protocol-v2.json)
- **Corrected Acceptance Ledger**: [_docs/qa/repair-model-acceptance-v2.json](file:///C:/Users/downl/Documents/ChatGPT/Strata/_docs/qa/repair-model-acceptance-v2.json)
- **TDD Test Suite**: [tests/qualification/test_real_model_runner.py](file:///C:/Users/downl/Documents/ChatGPT/Strata/tests/qualification/test_real_model_runner.py)
