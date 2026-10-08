# Community benchmark: 2× Tesla V100-SXM2-16GB, Windows

Measured **2026-10-08**: Strata **0.1.40**, Flash-Next **IQ2_XS**, **262144 context, Vision enabled**. One existing workstation configuration; no tuning or version comparison.

## Hardware and software

- **GPUs:** two V100-SXM2-16GB, physical indices 1/2, TCC, PCIe Gen3 ×8 each, six active NVLinks between the cards, 300 W power limit each. The RTX PRO 4000 Blackwell desktop GPU was **excluded by UUID**; selected UUIDs are in [config.json](config.json).
- **CPU/RAM/storage:** AMD EPYC 7A23, 48 cores/96 threads; nominal 64 GB RAM, Windows-visible **63.86 GiB**; KINGSTON SNV3S1000G NVMe. Desktop/Codex/browser activity remained present; background CPU contribution was not measured.
- **Software:** Windows 10 Home 22H2 **19045.6466**, NVIDIA **582.16**, Python **3.12.10**, CUDA release build **12.9** / runtime **12.9.79**.
- **Strata:** frontend commit `1735d6471df29b42c26170efaac1f1446a58640f`; engine **0.1.40**, CUDA 12 portable experimental release including **sm_70**, PTX enabled. Executable and pack hashes: [provenance.json](provenance.json). Exact release CMake invocation was not recorded.

## Model and configuration

- `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, **IQ2_XS**. Exact two GGUF filenames and sizes are in [provenance.json](provenance.json).
- Vision: `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, same revision, first V100, max **1024 image tokens**, 16 threads. MTP: setup-built **Q2_0 draft**, source `Qwen/Qwen3.8-Flash-Next` revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`; CJK draft vocabulary **106299 tokens**. Profile/pack/tokenizer hashes are included.
- Context **262144**, **INT8 KV**, **32768** resident window; layer split auto → **23/25 layers**; prefill auto → **8192**, CPU pool **47 workers + host**. Expert cache auto → **7561 + 7315 slots**, profile-ranked without eviction. Expert-arena pin limit **8 GiB**. Low-RAM mode off.
- MTP **spec=4, min-p=0.5**; concurrency **1**. Speed requests: **temperature=0**, reasoning **none**, output cap **256**, default penalties. No calibration override or experimental speed projection. Full requested and resolved settings: [config.json](config.json), [resolved.json](resolved.json).

From the checkout, with installation paths substituted in a copied config:

```powershell
.\.venv\Scripts\python.exe -u -m serve.server --engine strata --config <CONFIG_PATH> --host 127.0.0.1 --port 8080
```

The config binds both V100 UUIDs and the Vision GPU. [REPRODUCE.md](REPRODUCE.md) supplies the benchmark commands and path conventions.

## Method

One loaded server, an excluded **3401-input/256-output warm-up**, then ascending input lengths, three serial trials each. [benchmark.py](scripts/benchmark.py) generates shareable synthetic Python-module prompts, counts the rendered template with the model tokenizer, and changes an early nonce per trial. **All measured prefixes reused zero tokens**; warm expert residency was retained. The 128K label means **128000 actual tokens**. The 258000 extension plus 256 output fits the context budget.

Prefill and decode rates use their **separate engine durations**. TTFT is client send to first nonempty streaming delta, ignoring keep-alives/empty deltas; all first tokens were **answer** text. Total latency includes frontend/possible queueing, prefill and decode, excluding model loading. Speed requests contain no images, so Vision is loaded but image encoding is not timed. RAM and VRAM are approximately one-second sampled **whole-system/whole-device** observations.

## Results

**Median [min–max]**; 12 successful trials, each 256 output tokens, zero reuse, no failed/cancelled speed requests. [results.csv](results.csv) retains every trial, durations, resource peaks, draft/cache counters; [run-details.json](run-details.json) retains engine timing and generated text; [engine.log](engine.log) records startup and timing lines.

| Configuration | Actual prompt tokens | Reused | Generated | Runs | Prefill tok/s | Decode tok/s | TTFT seconds |
|---|---:|---:|---:|---:|---|---|---|
| IQ2_XS / Vision | 4,096 | 0 | 256 | 3 | 879.39 [877.80–880.65] | 93.05 [88.56–101.63] | 4.700 [4.699–4.700] |
| IQ2_XS / Vision | 32,768 | 0 | 256 | 3 | 1804.62 [1803.78–1808.78] | 91.66 [88.17–92.47] | 18.278 [18.244–18.285] |
| IQ2_XS / Vision | 128,000 | 0 | 256 | 3 | 1962.85 [1961.57–1969.93] | 87.02 [83.19–89.22] | 65.566 [65.322–65.611] |
| IQ2_XS / Vision | 258,000 | 0 | 256 | 3 | 1630.94 [1613.28–1694.60] | 78.99 [75.64–79.53] | 158.877 [152.925–160.595] |

| Actual prompt tokens | Total client seconds, median [min–max] | Peak system RAM GiB |
|---|---|---:|
| 4,096 | 7.435 [7.201–7.574] | 52.31 |
| 32,768 | 21.055 [21.036–21.131] | 51.75 |
| 128,000 | 68.463 [68.382–68.490] | 52.09 |
| 258,000 | 162.077 [156.147–163.960] | 52.16 |

Sampled VRAM peaked at **15874 / 15882 MiB**, GPU temperatures **63 / 77 °C** across the speed sweep. Minimum available system RAM was **11.55 GiB**. No OOM occurred; per-request paging was not measured. Per-trial power/clocks and cache/draft fractions are in the CSV. Draft acceptance combines engine speculative counters; pure MTP acceptance and a separate internal reasoning-token count are **not measured**.

## Correctness and limitations

- Unmodified official `tools/needle_bench.py`, lengths **32k/128k/250k**, depths **10/50/90%**: **9/9 found**, zero misses/errors/skips. Actual inputs were **32308–32309 / 125703–125705 / 243257–243258** tokens. All verdicts and answers: [needles.json](needles.json).
- Anthropic tool roundtrip: `add_numbers(19,23)` → `42` → final `42`, passed. [Synthetic image](correctness/vision-card.png): code `VX-4827`, total `37`, three blue squares, all correct. [Check results](correctness/checks.json) and [generator/check script](scripts/correctness.py) included.
- One Claude Code repair task passed **29/29** contract checks; [task/repaired code](correctness/code-task/), [test results](correctness/code-quality.json) and [test script](scripts/check_code_task.py) included. These small checks and needle recall are not overall model-quality scores.

Limits: one prompt family, three trials per size, ascending order, 256-token output cap, sampled peaks, and an older installed release. This does not isolate Vision's speed cost or establish image-heavy/long-session reliability. A local needle preflight initially failed on Windows text decoding, before sending requests; explicit UTF-8 fixed it, with the official scorer unchanged.
