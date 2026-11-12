# Strata Benchmark - Environment

Date: 2026-11-05 13:05:16
Server: http://127.0.0.1:8070
OS: Microsoft Windows 11 Pro 10.0.26300
CPU: AMD Ryzen 7 9700X 8-Core Processor             
RAM: 125.61 GiB
GPU: NVIDIA GeForce RTX 5070, 12227 MiB, 621.14, 250.00 W, 4, 16
Storage: Samsung SSD 990 EVO 1TB (931.5 GiB); Samsung SSD 990 PRO 2TB (1863 GiB)
Strata Git commit: ffa80d73133ea5c548aefe57889a16e8085a938f
Working tree modified: True

## Official tools
tools/strata_bench.py
tools/needle_bench.py

## Conditions
Existing local server, model loading excluded.
Three runs per performance scenario.
Thinking OFF and ON measured separately.
Four simultaneous clients (not necessarily four parallel engine slots).
Recall targets: 32K, 64K, 128K; depths 10%, 50%, 90%.
The recall test records actual input token counts.
Any skips, errors or misses must remain in the published report.

Model revision, exact GGUF hashes, background load and
effective engine settings must be checked before publication.
