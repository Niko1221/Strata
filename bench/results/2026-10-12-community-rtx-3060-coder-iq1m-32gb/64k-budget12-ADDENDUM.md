# 64K Budget12: configuration evidence, detailed interpretation, and limitations

**Scope:** one Windows 11 RTX 3060 12 GB / ~32 GB RAM host with stock, unmodified Strata v0.1.43. 65,536-context experimental config with `--max-context 65536`, `--kv int8`, `--expert-cache auto`, `--prefill auto`, `--spec 4`, `--spec-min-p 0.5`, and `--resident-budget-gib 12` (substituted for `--resident-experts`). No source-code changes. Original 32K generated config preserved.

**Pin and export:** The locally verified original Budget12 JSON SHA-256 is `9F5DB7BC7065A18E8870DC503B7F8F60D695C24FF50883FDCF9AE2A3FF6C55EC` (1,753 bytes). The exported, path-redacted `budget12-sanitized-config.json` has SHA-256 `3716B75CF4B77D7ED5857595DE3B4A0293B5F3F536FC5411D8F3501CE75B9BDE` (785 bytes). The exact original flags and Strata config schema are preserved; local paths are replaced by placeholders. `budget12-export-receipt.json` records the local hash check and unchanged-source status. **Neither the original absolute paths nor the original file bytes are published.** Restoring environment-specific paths and other host conditions is required; equal performance or bit-identical output is not guaranteed.

**The pivotal results:** Full-resident 64A finished with only 451 MiB system RAM free; 64B safety-aborted. Budget12 64C–64J showed much healthier sampled available RAM in completed runs, despite slowdowns versus full residency. A 64I preflight with ~154–156 MiB free VRAM was refused safely; later same sealed case resumed after resources were freed. The model's actual 64I/J inference did not start without two passing headroom checks.

**Reasoning:** 64H spent all 8,192 output tokens thinking and returned no content. Strata supports per-request `reasoning_budget_tokens=6144`; 64I and 64J both completed with normal `stop` and valid exact-schema JSON. Reserve after thinking is approximate and does not guarantee content correctness.

**Accuracy:** 64G 12/13 (refund omission); 64H not evaluable; 64I 12/18 (Invoice C incomplete reconciliation, downstream queue error); 64J 24/24 (exact canonical/replay record arrays plus all prior answer fields). Each case was fresh and one-shot; the strongest demonstration remains **one synthetic success**, not proof of general 100% reliability.

**Speed:** 64J unexpectedly had only 349.2 prefill tok/s versus prior ~833–863 tok/s Budget12 runs. Store this discrepancy; no controlled attribution to prompt, disk, load or setup was made.

**Production:** do not rely on unchecked model-only financial reconciliation; use deterministic independent arithmetic and provenance checks.

**Reproduction:** see `budget12-REPRODUCE.md`. The memory gates are external experiment safeguards rather than built-in Strata settings; the exact model files, profile, pack, draft layer, driver, free headroom, storage and workload may affect results.
