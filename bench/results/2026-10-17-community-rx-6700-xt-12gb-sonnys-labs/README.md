# Radeon RX 6700 XT 12 GB / 32 GB RAM — official Strata speed results

Strata v0.1.45, source `477ac13cbe8e582c20484871ce459d20d82df09d`, native HIP gfx1031. Ryzen 5 3600 (6 cores / 12 threads), 32 GB DDR4-3200 (2 × 16 GB), RX 6700 XT 12 GB, Intel 1 TB NVMe. Runs collected October 17–18, 2026 UTC on Proxmox / Debian 13 in an LXC guest using isolated Docker inference. Guest memory limit 28 GiB; inference memory limit 26 GiB and CPU quota 9. Reported installed/usable RAM and GPU identity in original tool output reflect what the container could discover; see PROVENANCE.json for exact hardware and runtime pins.

Only the unmodified upstream `tools/strata_bench.py` measurements are submitted. This is a results-only contribution. No custom quality, coding, retrieval or dossier scores are included. No engine/server source changes are included. Speed measurements do not establish correctness.

## Official measurements

Three repetitions per workload; table shows medians. Context is allocated capacity, not the length of every request. Each linked original README and JSON includes prompt throughput, TTFT, agent/cache measurements, sample ranges and any optional client results.

| Model | Allocated tokens | Chat tok/s | Chat TTFT seconds | Original evidence |
|---|---:|---:|---:|---|
| Flash-Next Coder IQ1_M | 32768 | 22.4 | 1.499 | [original output](coder-iq1_m-ctx32768/README.md) |
| Flash-Next Coder IQ1_M | 65536 | 21.2 | 1.693 | [original output](coder-iq1_m-ctx65536/README.md) |
| Qwen3.6 UD-Q4_K_XL | 32768 | 33.2 | 0.667 | [original output](qwen36-ud-q4_k_xl-ctx32768/README.md) |
| Qwen3.6 UD-Q4_K_XL | 65536 | 24.6 | 0.865 | [original output](qwen36-ud-q4_k_xl-ctx65536/README.md) |
| Qwen3.6 UD-Q3_K_M (no MTP) | 65536 | 27.4 | 0.673 | [original output](qwen36-ud-q3_k_m-zero-mtp-ctx65536/README.md) |
| Qwen3.6 UD-Q3_K_M (no MTP) | 81920 | 23.5 | 0.834 | [original output](qwen36-ud-q3_k_m-zero-mtp-ctx81920/README.md) |
| Qwen3.6 UD-Q4_K_M (MTP file) | 65536 | 27.5 | 0.815 | [original output](qwen36-ud-q4_k_m-mtp-ctx65536/README.md) |
| Qwen3.6 UD-Q4_K_M (MTP file) | 81920 | 24.8 | 0.879 | [original output](qwen36-ud-q4_k_m-mtp-ctx81920/README.md) |
| Qwen3.6 UD-Q3_K_XL (no MTP) | 65536 | 26.4 | 0.706 | [original output](qwen36-ud-q3_k_xl-zero-mtp-ctx65536/README.md) |
| Qwen3.6 UD-Q3_K_XL (no MTP) | 81920 | 22.6 | 0.831 | [original output](qwen36-ud-q3_k_xl-zero-mtp-ctx81920/README.md) |

The standard primary runs allocate 32,768 tokens. Extended primary runs allocate 65,536 and use `--long --clients 2`. The three supplemental quant variants use `--long` at 65,536 and 81,920; they have no two-client measurements. All runs use three repetitions and thinking off. Two-client tests queue on one engine slot; this does not demonstrate parallel engine slots or batching. Requested Qwen MTP produced zero offered/accepted drafts; Coder produced drafts, without an MTP-on/off ablation. Do not attribute speed differences solely to quantization: these are different models/configurations. No hardware-only comparison is established.

## Reproduction and model compatibility

Check out the pinned source commit and follow its AMD HIP build documentation with gfx1031 and `STRATA_Q6K_EXPERTS=ON`. Use the matching TheRock libraries recorded in PROVENANCE.json. Model repositories, revisions and SHA-256 values are pinned there; model weights and pack data are intentionally excluded. Serving configuration and native engine arguments for each run are retained in that run's original results.json `machine.config` / engine information. Recreate those settings and start the server locally on loopback, then run:

```sh
python tools/strata_bench.py --url http://127.0.0.1:8080 --runs 3
# Primary extended runs:
python tools/strata_bench.py --url http://127.0.0.1:8080 --runs 3 --long --clients 2
# Supplemental quant runs:
python tools/strata_bench.py --url http://127.0.0.1:8080 --runs 3 --long
```

Keep the matching allocation and configuration for each run. Use the exact original filenames/configuration, not a default model inferred from a quant label. Local pack/profile inputs must be regenerated using the pinned upstream tools. The original tool's abbreviated command field omits some wrapper CLI arguments; the commands above state the actual repetition/extended/client settings. Server launch and Linux library-path setup were orchestration adjustments; no benchmark calculations or output files were edited.

The two Q3 non-MTP source GGUFs omit `qwen35moe.nextn_predict_layers`. Their separately named compatibility copies add that metadata key with value zero. All other metadata, tensor descriptors and complete tensor-data bytes were verified unchanged. This is a model metadata compatibility adjustment, not an unmodified original GGUF or an engine modification. Exact source/runtime hashes and byte-identity receipts are in PROVENANCE.json. Recreate with the supplied helper and pinned upstream `tools/gguf_reader.py`:

```sh
python normalize_zero_mtp.py ORIGINAL.gguf COMPATIBILITY.gguf receipt.json /path/to/Strata/tools
```

The original model is preserved. This adjustment is specific to the verified no-MTP Q3 files, not a general fix for arbitrary models. No unmeasured compatibility claim is made.

Original tool README.md/results.json pairs are copied byte-for-byte. ORIGINAL_OUTPUT_VERIFICATION.json records all 20 original hashes; SHA256SUMS covers this report package. Operational logs, private addresses, telemetry archives, weights and custom benchmarks are excluded.
