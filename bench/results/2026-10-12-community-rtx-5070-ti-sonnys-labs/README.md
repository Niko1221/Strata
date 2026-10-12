# RTX 5070 Ti 16 GB / 64 GB RAM — Strata v0.1.43

Measured on 2026-10-12 by Sonny's Labs (@firetussin). Four reports from the unmodified upstream `bench.sh` / `tools/strata_bench.py`: IQ2_XS and IQ3_XXS, each at 32,768 context (standard suite) and 65,536 context (`--long --clients 2`). Each report retains the original README and all per-request JSON byte-for-byte; hashes are in ORIGINAL_OUTPUT_VERIFICATION.json. The separately labeled supplement supplies preparation, hardware and runtime details the tool cannot discover. No engine or scoring changes are proposed.

| Quantization | Standard chat decode median tok/s | Standard 4K fresh prefill tok/s | Standard 16K fresh prefill tok/s | Optional 32K fresh prefill tok/s | Optional two-client combined tok/s |
|---|---:|---:|---:|---:|---:|
| IQ2_XS | 81.9 | 1459.5 | 2176.8 | 2183.4 | 65.93 |
| IQ3_XXS | 66.4 | 1210.0 | 1945.4 | 1939.8 | 48.71 |

## Reports

- [IQ2_XS standard / 32,768 allocation](iq2_xs-standard-32k/README.md)
- [IQ2_XS optional / 65,536 allocation](iq2_xs-optional-64k/README.md)
- [IQ3_XXS standard / 32,768 allocation](iq3_xxs-standard-32k/README.md)
- [IQ3_XXS optional / 65,536 allocation](iq3_xxs-optional-64k/README.md)

## Provenance and measurement boundaries

Strata tag v0.1.43 resolves to `7695859aa13afb248bdb4d36f9b2e95707db079d`; llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Locally compiled CUDA 13.0.0 / SM120 engine, vision off, precompiled Docker serving image. RTX 5070 Ti (16,303 MiB reported), Ryzen 5 5600G, PCIe Gen3 x16, NVMe storage, 64 GB installed RAM / about 58.8 GiB OS-usable. The generated README calls its detected 58.8 GiB field installed RAM; the independently confirmed installed capacity is 64 GB. See supplements for OS/driver/container versions, image/engine/model hashes, power limits and exact model filenames/revisions. Container memory limits differ from installed host RAM.

LOW_RAM resident experts, auto expert cache/prefill, int8 KV, requested MTP spec 4 / minimum probability 0.5, upstream route-tail-skip 7. At 64K allocation, resident KV window is 32,768 tokens. No vision or server-side MCP execution. Serving uses an isolated Docker network namespace; the benchmark client shares loopback. Other LXC workloads were stopped; one-second telemetry ran during measurements. No downloads, packing or vulnerability scans ran during timed official tests.

Commands, per quantization/context (QUANT and CONTEXT select the corresponding staged profile):

```sh
./bench.sh --url http://127.0.0.1:8080 --config /data/config/strata-QUANT-CONTEXT.json --runs 3 --out /output
# Optional 65,536 allocation: same command with --long --clients 2
```

Three measured repetitions, with upstream warmup discarded and thinking off. The process/expert cache and filesystem were warmed; these are not storage-cold measurements. Model loading is excluded. Fresh prompts have unique tags; conversation-prefix reuse is separate from expert cache state. Throughput uses engine timings; streaming TTFT uses client timing. Original results include ranges, actual token counts and per-request timing/errors. No startup engine excerpt is emitted by the upstream tool in --url mode.

The optional two-client test queues through one serial engine slot (`--batch` off); combined throughput is not concurrent GPU batching. The official suite measures speed, not answer correctness. Separately scored Sonny's Labs coding/practical/dossier and needle evaluations are documented in the [full technical report](https://report5.firetussin.com/); they are not substituted for upstream methodology or represented as upstream quality scores. These runs do not establish full 128K task quality, a universal quantization ranking, or a security certification. Model files, generated packs, image archives and large internal telemetry are excluded from this results-only contribution.
