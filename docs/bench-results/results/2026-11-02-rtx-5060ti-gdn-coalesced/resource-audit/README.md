# GDN resource audit, 2026-11-02

The qualified compiler configuration here uses CUDA 13.3, sm_120 and Windows Release /MT. These are standalone translation units and synthetic executables, not a rebuilt engine. The prototype is not applied to production sources.

Read the accompanying `_docs/2026-11-02_gdn-resource-audit_Codex.md` report. `summary.json` contains all resource comparisons and all per-run timing medians. `validation.json` distinguishes correctness passes from unqualified performance and unrun model/backend checks. `prototype-only.patch` is for review; applying it would change normal dispatch and is not recommended by this experiment.

`binding.json` records compiler inputs. `runtime-binding.json` records the existing reference kernel library used for non-GDN symbols. The raw compiler `receipt.json` retains hashes for objects kept under build/, even though binaries are omitted from this evidence folder. `evidence-receipt.json` hashes every curated file in this directory except itself.

`native-mmvq-compiler-only/` is the initial /MD compiler-only exploration; it was not the source of the runtime GDN claims. `diagnostics/` preserves the failed /MD-to-/MT reference-library link attempt; the main runtime was rebuilt consistently with /MT. `source/` preserves the exact tools and benchmark source used for the final runs.

Performance is not qualified. The GPU was shared, clocks were not fixed, and the 144 MiB timing varied substantially. No E2E tok/s or DRAM traffic was measured. No failed or slow timing pair was dropped.

Final timing: runtime-r2/summary.json and its raw logs supersede the initial timing summary. The benchmark now explicitly waits for default-stream initialization before using its nonblocking timing stream, and is formatted. Compiler objects are unchanged; both parity executables passed again. The initial 960 samples remain here; the final 960 are under runtime-r2/. source/ at the root preserves the initial source; runtime-r2/source/ preserves the final benchmark. Sanitizer passes refer to the initial parity executable with identical CUDA objects, not to a second sanitizer run.