# Correctness fingerprints, six prompts at `temperature=0`

Arithmetic, date, capital, code, and sum prompts; the answer text and its reasoning are hashed separately so a change in wording is visible even when the answer is right, and vice versa.

| configuration | file | passed | content fingerprints |
| --- | --- | --- | --- |
| stock IQ3_S baseline (before promotion) | `qa-stock-baseline.json` | 6/6/0 | `` |
| v0.2.12 single-slot, fine-tune weights | `qa-v0212-single.json` | 6/6/0 | `` |
| STRATA_HIP_WMMA=1 | `qa-wmma.json` | 6/6/0 | `` |
| DFlash2 drafter | `qa-dflash2.json` | 6/6/0 | `` |
| --kv k8v4 | `qa-k8v4.json` | 6/6/0 | `` |
| --mtp-q4 | `qa-mtp-q4.json` | 6/6/0 | `` |
| promoted configuration (images on) | `QA-promoted-vision.json` | 6/6/0 | `` |

The promoted configuration's fingerprints are identical to the pre-promotion stock baseline, which is the point: the promotion changed no answer.

The first A/B rounds of WMMA, DFlash2, `--kv k8v4` and `--mtp-q4` each passed 6/6, so the non-bitwise arms (`HIP_WMMA`, `k8v4`) were behaviourally equivalent on these prompts even though their speeds were not.
