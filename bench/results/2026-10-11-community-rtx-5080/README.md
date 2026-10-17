# Community benchmark on NVIDIA GeForce RTX 5080 (Windows)

Measured on 2026-10-11 by [Hotziplotzi](https://github.com/Hotziplotzi), on a Windows desktop called MyPC. This tests Strata 0.1.42 (`61b3fb5dd3f1`, locally built source, CUDA 13.4) with the original Flash-Next IQ3_XXS, one GPU, and a 262,144-token context limit.

Median decode throughput was **91.9 tok/s at 4,096 prompt tokens, 94.8 tok/s at 32,768 prompt tokens and 95.5 tok/s at 128,000 prompt tokens**; median prompt throughput was 3,060.9 / 3,461.9 / 3,607.0 tok/s. These are synthetic code-explanation requests with greedy decoding and a 256-token output cap. They do not establish general answer quality or performance on other workloads.

## Hardware and software

- NVIDIA GeForce RTX 5080; 16,303 MiB reported VRAM; 360 W power limit (default 360 W). PCIe capability: Gen 5, x16 (current link at idle: Gen 5 x16); GPU clocks were not fixed.
- AMD Ryzen 7 7800X3D 8-Core Processor; 16 logical CPUs (8 physical).
- 63.38 GiB RAM reported by Windows; 4.00 GiB system swap. Storage: Samsung SSD 980 PRO 2TB (NVMe, local; models on `D:`).
- Windows 11 (build 26200) Professional; NVIDIA driver 616.64.
- Source: git `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406` at `D:\8_KI\Strata`; engine 0.1.42, locally built (BUILD.json: source `local`, src hash `ce8fa084659f667d`), CUDA architecture sm_120. See [BUILD.json](BUILD.json).
- CUDA 13.4; Python 3.13.15. Installed Python dependencies are listed in [python-packages.txt](python-packages.txt).
- Background workloads at the start of the run are listed in [hardware-windows.json](hardware-windows.json) (processes by working set) and [environment.json](environment.json). Visual Studio Code, Bitdefender services, Signal, Everything and Cryptomator were open during the run; no deliberate additional GPU workload was started.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (the installer's pinned revision):

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (47,039,860,096 bytes; sha256 `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`)
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` (28,800,138,432 bytes; sha256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`)
- `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,543,008 bytes; sha256 `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0`)

Local SHA-256 hashes were computed on this machine; all artifacts are in [model-provenance.json](model-provenance.json).
MTP experts were fetched by the installer; per-tensor hashes are in [mtp-manifest.json](mtp-manifest.json).

Configuration: the machine's daily-use IQ3_XXS config ([strata-iq3_xxs.json](strata-iq3_xxs.json)):

```text
--pack D:\8_KI\Strata-data\packs\iq3_xxs --native D:\8_KI\Strata-data\models\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf D:\8_KI\Strata-data\models\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile D:\8_KI\Strata\data\expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp D:\8_KI\Strata-data\mtp\rt --max-context 262144 --kv int8 --vision --vram-reserve-mib 700 --pcie-frac 1.00 --spec-min-p 0.70 --pool-workers 5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

Reasoning disabled, temperature 0, maximum 256 generated tokens per speed run. There was no calibration and no experimental speed projection; the bundled expert profile was used as-is.

```text
D:\8_KI\Strata\.venv\Scripts\python.exe serve/server.py --engine strata --config D:\8_KI\Strata\strata-iq3_xxs.json --host 127.0.0.1 --port 18080
```

## Method

The attached [strata_bench_rtx5080.py](strata_bench_rtx5080.py) ran the whole sequence; per-run data is in [results.json](results.json), aggregates in [summary.json](summary.json). Server state snapshots before and after the sweep are in [initial-status.json](initial-status.json) and [final-status.json](final-status.json); the effective runtime properties are in [props.json](props.json).

It generates a deterministic synthetic Python module, adds a different nonce near the start of every request so no prompt can reuse a prefix (nonces are before the filler, so earlier long prompts cannot be reused), and counts the complete rendered chat prompt with Strata's own tokenizer plus chat template. It adjusts the filler by binary search until the count lands in [target − 20, target] and verifies the count against the engine's record afterwards. One warm-up request (`Reply with exactly the word READY.`, 16 tokens) preceded the sweep and is excluded from the tables.

All speed runs of one engine load were executed serially in increasing-length order on the same loaded engine; no engine restart or OS page-cache reset separated them. The expert cache was filled during model load and adapted during the run (auto policy); it was not reset between runs. Loading time is excluded; the model load took 20 seconds.

All 9 speed requests processed their entire prompt: **zero reused tokens**.

TTFT is measured client-side from immediately before the HTTP request until the first nonempty text delta, ignoring empty role chunks and keep-alives. With reasoning off, that delta contains answer text. Total seconds end at stream completion; both include HTTP and frontend tokenization over loopback. Engine prompt throughput uses freshly read tokens divided by `prompt_ms`; decode throughput uses `engine_generated / decode_ms`. The JSON preserves the engine timing values and output text per run. Host RAM, GPU memory, temperature and board power were sampled every two seconds into [telemetry.jsonl](telemetry.jsonl); aggregates are in [resource-summary.json](resource-summary.json).

## Results

Each cell below is the median **[minimum–maximum]** across the successful runs. Every request was capped at 256 generated tokens.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range | Total seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| IQ3_XXS, 4,096-token prompt | 4,086 [4,086–4,086] | 0 [0–0] | 256 [256–256] | 3 | 3,060.9 [2,728.9–3,126.0] | 91.9 [90.9–95.2] | 1.379 [1.348–1.532] | 4.146 [4.052–4.302] |
| IQ3_XXS, 32,768-token prompt | 32,764 [32,764–32,764] | 0 [0–0] | 256 [256–256] | 3 | 3,461.9 [3,428.7–3,482.2] | 94.8 [94.8–96.7] | 9.517 [9.460–9.594] | 12.198 [12.090–12.277] |
| IQ3_XXS, 128,000-token prompt | 127,990 [127,990–127,990] | 0 [0–0] | 256 [256–256] | 3 | 3,607.0 [3,605.4–3,610.8] | 95.5 [89.0–96.8] | 35.574 [35.550–35.587] | 38.235 [38.178–38.443] |

Raw per-run records: [results.json](results.json); the complete request bodies are the `tokens-*-request.json` files. Calculated aggregates: [summary.json](summary.json).

MTP draft acceptance across the speed runs: 1,278 of 1,518 proposed draft tokens accepted (84.2%).

Decode expert-cache hit rates across the speed requests: 98.1%–99.2%.

[telemetry.jsonl](telemetry.jsonl) contains 150 samples over 309 seconds, starting before model load:

- GPU memory (system-wide on the selected GPU): peak **15351 MiB**.
- Host RAM: minimum available **6.09 GiB** (sampled; other workloads included).
- Swap in use: 0.0 MiB at the first sample, peak 25.0 MiB.
- GPU temperature peak: 70 °C; board power peak: 297 W.

## Correctness and limitations

The repository's unchanged `tools/needle_bench.py` found **6 of 6 needles** across depths 10%, 50%, 90% at lengths 32k and 128k. Actual prompt lengths and per-case results are in [needles.json](needles.json).

This is one machine, one quantization, one configuration, and a small synthetic workload. Long output, sampled decoding, thinking, coding-task correctness, vision (the vision helper was enabled but these requests contained no images), tool use, multi-request concurrency, and a sustained thermal run were not evaluated. The built-in needle check measures recall on these inputs, not general model accuracy. No same-workload baseline on another engine or GPU was run; other reports' numbers are not a controlled comparison. The longest speed prompt does not fill the full 262,144-token context.
