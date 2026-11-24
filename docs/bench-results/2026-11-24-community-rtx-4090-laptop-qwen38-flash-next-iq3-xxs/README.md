# Community benchmark: Qwen3.8-Flash-Next IQ3_XXS on an RTX 4090 Laptop

Measured on 2026-11-24 by [andyholst](https://github.com/andyholst). One model only:
Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, served as `qwen3.8-flash-next-iq3_xxs`, on Strata 0.1.67.
This is a community speed log from a custom harness. It is not a
`strata bench run` standard-protocol result, and it is not an engine change.

The rates below are client-derived. The harness did not keep the engine's
`prompt_per_second` or `predicted_per_second` lines. Prompt tok/s is
`prompt_tokens / time-to-first-token`. Decode tok/s is
`(completion_tokens - 1) / (stream end - first token)`. Both clocks are the
client's. Thinking tokens are included when the server streams them.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090 Laptop, 16376 MiB, sm 8.9, driver 615.78.08, power cap 175 W. PCIe link at measurement time: gen 4 x8. One GPU.
- CPU and RAM: Intel Core i9-13980HX, 24 threads, governor `powersave`, 62 GiB RAM. On AC power.
- Storage: model name not recorded. A direct read of 512 MiB from the first GGUF shard was 3.6 GB/s.
- OS: Omarchy, kernel 7.2.15-1-omarchy.
- Strata: installed control component 0.1.67. Its `COMMIT` file is `d46f5f268f8ed797e83a5c63a69bffbd7a581cf6`. Engine package `0.1.67-linux-x86_64-cuda-cuda13`, CUDA 13.0, `BUILD.json` archs `75;86;89;120`. `BUILD.json` does not name a git commit. Release binary, not a local source build.
- Bandwidth with the server stopped, same machine, same day: VRAM read 563.45 GB/s (12 reps, 561.72 to 563.75), pinned host-to-device 12.25 GB/s, sustained FP16 GEMM 86.49 TFLOPS at 8192. Decimal GB/s (bytes / 1e9 / seconds). These are machine ceilings, not model speeds.
- Background: the laptop was otherwise idle for the runs. Nothing else was pinned.

## Model and configuration

- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS. Served id `qwen3.8-flash-next-iq3_xxs`. Shard filenames follow `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000N-of-00002.gguf`. Repository revision and file SHA-256 were not recorded.
- Vision off. Context served 131072. The checkpoint context is 262144. This card did not hold a 256K cache beside the weights.
- KV int8, `--kv-resident 32768`, expert cache auto, prefill auto, `--pool-workers 15`, `--prompt-cache-spread`, `--conversation-cache-mib 3072`, `--vram-reserve-mib 704`, DFlash2 `--spec 6 --draft-conf 0.7 --draft-fallback none`, `--conversation-cache-min-tokens 1024`.
- No calibration pass and no experimental speed projection were recorded.
- The baseline launch preset was `IQ3_XXS 128K (9)`. The kept long-prefill change is a copy, `IQ3_XXS 128K Hermes NV16`, with one extra variable `STRATA_NV_CHUNK=16`. The parent Hermes preset file was not edited. Public copy: [preset-hermes-nv16.json](preset-hermes-nv16.json). No API key is in that file.

```text
--expert-cache auto --prefill auto --kv int8 --kv-resident 32768
--pool-workers 15 --prompt-cache-spread --max-context 131072
--conversation-cache-mib 3072 --vram-reserve-mib 704
--draft dflash2 --draft-fallback none --spec 6 --draft-conf 0.7
--conversation-cache-min-tokens 1024
```

`STRATA_NV_CHUNK=16` is only on the NV16 rows. The engine then reported a prompt chunk of 16384. Unset, or `STRATA_NV_CHUNK=0`, the chunk was 8192. `STRATA_NV_CHUNK=1` reported 22784 and was not kept.

## Method

Harness and raw ledger: [andyholst/strata-qwen-flash-hillclimb](https://github.com/andyholst/strata-qwen-flash-hillclimb) at `94f3a0d`. The rows for this model are [ledger.jsonl](ledger.jsonl). Ledger stamps are local time on 2026-11-24. The timezone was not written into the file.

Each run discards one warm-up (`Say hello.`, 16 tokens). Short TTFT is one request, cap 8 tokens. Decode repeats use a fixed six-prompt panel, cap 512 tokens, temperature 0.6. The baseline used six decode repeats. The 121530-token rows used one. Concurrency was 1 (`--skip-cN`). Cold prefill asks for about 4096 and 32768 tokens by a 1.6 tokens-per-word estimate, temperature 0, cap 1 token. The server's `prompt_tokens` are the counts in the table, and they came back shorter than the targets (2839 and 21573 on the baseline).

The long document is the same generator. `--long 65536` came back as 43147 tokens. `--long 185000` came back as 121530 tokens. Cap 256, temperature 0. Three codes must match exactly. Each document starts with a unique nonce, so the prompt text is not byte-reproducible. A fresh process was used for each 121530-token row.

Host RSS is the peak of a process whose command line contains `strata --serve`, capped at 58 GiB by the watchdog. It is not a VRAM figure.

## Results

Prompt and decode columns are the client formulas above. One sample where the run count is 1. Reused prompt tokens were not recorded.

| Configuration | Actual prompt tokens | Generated cap | Runs | Prompt tok/s | Decode tok/s | TTFT |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Baseline, chunk 8192, short panel | short | 512 | 6 | not measured | median 79.79, range 75.92 to 90.83 | 501 ms |
| Baseline cold prefill | 2839 | 1 | 1 | 1480.8 | not measured | not measured separately |
| Baseline cold prefill | 21573 | 1 | 1 | 1944.3 | not measured | not measured separately |
| Baseline needle | 43147 | 256 | 1 | 1993.9 | 79.35 | 21.6 s |
| Control A, chunk 8192 | 121530 | 256 | 1 | 2123.0 | 82.99 | 57.2 s |
| Kept, `STRATA_NV_CHUNK=16`, chunk 16384 | 121530 | 256 | 1 | 2801.2 | 83.35 | 43.4 s |
| Control B, `STRATA_NV_CHUNK=0`, chunk 8192 | 121530 | 256 | 1 | 2116.2 | 84.74 | 57.4 s |
| Not kept, `STRATA_NV_CHUNK=1`, chunk 22784 | 121530 | 256 | 1 | 2847.1 | 79.32 | 42.7 s |

Short decode on the 121530-token process starts, one sample each: control A 79.07, NV16 72.40, control B 76.05, chunk 22784 90.40 tok/s. That spread sits inside the baseline's six-run band. The short number was not the reason a row was kept.

Host RSS peak: baseline 46.02 GiB, then 46.95, 45.59, 45.86, and 46.16 GiB on the four 121530-token rows. No failed request on this model. No row was dropped from [ledger.jsonl](ledger.jsonl).

`STRATA_NV_CHUNK=16` is the only change that was kept. Long prefill went from 2123 and 2116 tok/s to 2801 tok/s (about 1.32x) with retrieval still 3/3. The 22784-token chunk at 2847 tok/s did not clear that. Decode at 121530 tokens stayed near 83 tok/s.

## Correctness and limitations

Retrieval was an exact match of three planted codes: 3/3 on every long row, including the rejected chunk. That is recall of those codes, not a quality score. No HumanEval, no tool-call suite, no standard-protocol sanity set.

Not measured: engine timing lines, reused-token counts, file hashes, a prompt the tokenizer counts as the full 131072, concurrency above 1, and any KV format other than int8. CPU governor stayed on `powersave`. The six baseline decode runs spread about ±9% around 79.79, so a one-sample difference inside that band is not a result.
