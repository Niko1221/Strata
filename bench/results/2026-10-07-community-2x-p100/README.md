# Community benchmark: 2x Tesla P100 16GB, 2-socket Xeon

Measured on 2026-10-07 by [touchtop](https://github.com/touchtop). This is the
first reported P100 result for Strata (docs/OLDER_GPUS.md lists P100 as
"not measured"). Tests Strata at commit `82f46a8` with Qwen3.8-Flash-Next
IQ3_S, two P100s in layer-split, and a 131,072-token context limit.

Decode throughput was **31.1 tok/s at short prompts, 28.0 tok/s at
4,096 prompt tokens, and 33.3 tok/s at 32,768**. Prompt (prefill) throughput
was **42 tok/s (short), 282 tok/s (4K), and 454 tok/s (32K)**. These are
synthetic repeated-text prompts with greedy decoding (temperature 0) and a
100-token output cap. They do not establish general answer quality or
performance on other workloads.

For comparison, community P40 reports cite 217-374 tok/s prompt and 30-33
tok/s decode. The 2x P100 exceeds the P40 prompt ceiling by ~21% at 32K
tokens, with comparable decode.

## Hardware and software

- 2x NVIDIA Tesla P100-PCIE-16GB; 16,384 MiB each; PCIe Gen 3 x16.
  Used GPUs 2 and 3 (same NUMA node, NODE interconnect). GPUs 0 and 1 were
  idle during the test (GPU 0 kept free, GPU 1 runs unrelated services).
- 2-socket Xeon, 18 cores per socket (72 logical CPUs total).
- 251 GB installed RAM. Model weights on rotational HDD (/home/images,
  1.8 TB); 55 GB expert data loaded into RAM at startup.
- Ubuntu 20.04.6 LTS, NVIDIA driver 535.247.01 (CUDA 12.2).
- Strata source commit `82f46a8`; engine built locally with CUDA 12.9.1
  and `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=60`
  (sm_60 experimental path for Pascal). GCC 13.4 (conda-forge).
  A local patch to `src/core/vmm.cpp` forces the legacy
  `cudaGetDriverEntryPoint` for driver 535 compatibility.
  Runtime uses CUDA 12.1 libcudart via LD_LIBRARY_PATH.
- Python 3.12.15 (uv), venv at `.venv`.

## Model and configuration

Model: Qwen3.8-Flash-Next IQ3_S (GGUF), 87 GB total:
- `models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`
- `models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf`
- `mtp/rt` (MTP draft)
- `packs/iq3_s/` (native pack, tokenizer)

Settings (strata-iq3_s.json):
- Context 131,072; INT8 KV; 20,480 KV cells resident on GPU (minimum),
  remainder streams from RAM.
- Expert cache 2,000 slots (manual; `auto` OOMs on 16 GB).
- `--vram-reserve-mib 2000` (prevents prefill-to-decode OOM on 16 GB cards).
- MTP `--spec 4 --spec-min-p 0.5`.
- Vision disabled (text-only for this report).
- API on 0.0.0.0:8080 with API key.

## Method

Three runs each at short (~10 tokens), 4,096, and 32,768 prompt tokens,
100 generated tokens, temperature 0. Only the first run of each size is
reported below because Strata reuses conversation prefixes: runs 2-3 hit
the prefix cache (e.g. 32K run 2-3 completed in ~3s vs 74s fresh).

Prompt construction: short = "Explain quantum computing in simple terms.";
4K/32K = "The quick brown fox jumps over the lazy dog. " repeated 400 /
3,200 times. Throughput figures are from the engine timing output
(`prompt_per_second`, `predicted_per_second`), not derived from wall time.

## Results

| Prompt tokens | Prefill (tok/s) | Decode (tok/s) | Wall time (s) |
|--------------|-----------------|----------------|---------------|
| ~10          | 42.2            | 31.1           | 4.6           |
| 4,052        | 281.8           | 28.0           | 18.0          |
| 32,052       | 454.0           | 33.3           | 73.8          |

## Notes and limits

- The 16 GB P100 requires conservative settings: setup's `auto` expert
  cache (6,045 slots) OOMs at verify; 2,000 slots is stable. A 15K-token
  prompt OOMed at the prefill-to-decode transition until
  `--vram-reserve-mib 2000` was added.
- KV streaming works: 32K prompt (exceeding the 20,480 resident window)
  ran without OOM, with the bulk of KV in RAM.
- These are single fresh-prompt runs, not medians of three; repeat runs
  would require distinct prompt texts to avoid prefix-cache hits.
