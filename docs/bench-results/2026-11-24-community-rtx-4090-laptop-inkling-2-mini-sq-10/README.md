# Community benchmark: Inkling-2-Mini SQ-10 on an RTX 4090 Laptop

Measured on 2026-11-24 by [andyholst](https://github.com/andyholst). This folder is only
Inkling-2-Mini SQ-10, served as `inkling-2-mini-sq-10`. SQ-15 and SQ-20 are separate
reports. This is a community speed log from a custom harness, not a `strata bench run`
standard-protocol result, and not an engine change.

Prompt tok/s is `prompt_tokens / time-to-first-token`. Decode tok/s is
`(completion_tokens - 1) / (stream end - first token)`. Both clocks are the client's.
The engine's own timing lines were not kept.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090 Laptop, 16376 MiB, sm 8.9, driver 615.78.08, power cap 175 W, PCIe gen 4 x8. One GPU.
- CPU and RAM: Intel Core i9-13980HX, governor `powersave`, 62 GiB RAM, on AC power.
- Storage model: not recorded.
- OS: Omarchy, kernel 7.2.15-1-omarchy.
- Strata 0.1.67 CUDA 13, engine binary `strata-inkling`, package `0.1.67-linux-x86_64-cuda-cuda13`. Installed control `COMMIT` file: `d46f5f268f8ed797e83a5c63a69bffbd7a581cf6`. Release binary. `BUILD.json` does not name a git commit.
- Machine ceilings, measured the same day with the server stopped: VRAM read 563.45 GB/s, pinned host-to-device 12.25 GB/s. They are not speeds of this model.
- Background: otherwise idle. Nothing else was pinned.

## Model and configuration

- Pack: Inkling-2-Mini SQ-10. Served id `inkling-2-mini-sq-10`. File hashes and the pack revision were not recorded. UD-IQ3_XXS was a partial download and was not run.
- Context 131072. Launch preset `SQ-10 128K`, file not edited. Public copy: [preset-sq-10-128k.json](preset-sq-10-128k.json).
- Engine arguments the process printed: `--max-context 131072 --vram-experts-gb 8.60 --draft dspark --draft-beam 1 --draft-adapt`. The preset file stores `drafting: null`. The engine still started DSpark from its Inkling defaults.
- Log: prefill chunk 512, 5342 MiB free after the experts, KV 82176 VRAM tokens plus 48896 host tokens, 946 MiB free after DSpark (588 MiB).
- The candidate added only `--chunk 2048`. `STRATA_NV_CHUNK` was not set. That variable belongs to the Qwen engine, and this process is `strata-inkling`.

## Method

Harness and the full ledger: [andyholst/strata-inkling-mini-hillclimb](https://github.com/andyholst/strata-inkling-mini-hillclimb) at `a5d21c6`. Rows for this pack are [ledger.jsonl](ledger.jsonl).

One discarded warm-up. One short decode, cap 512, temperature 0.6. Then one long document, cap 256, temperature 0, three exact codes. `--long 185000` is the Qwen document shape. On this tokenizer it is 144469 tokens, and the server returned HTTP 400 because 131072 tokens then have no room to answer. That failure is the first row. `--long 150000` is 117169 tokens and is the measured long prompt. Each document starts with a unique nonce. Concurrency was 1. One fresh process per row. No second control: the candidate lost the first comparison.

Host RSS watches a command line containing `strata --serve` or `strata-inkling`.

## Results

| Configuration | Actual prompt tokens | Generated cap | Runs | Prompt tok/s | Decode tok/s | TTFT | Outcome |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| Too-long document | 144469, rejected | 256 | 1 | not measured | short sample 122.46 before the failure | 86.8 ms short | HTTP 400, `REQ_FAIL` |
| Control, chunk 512 | 117169 | 256 | 1 | 1335.5 | 9.58 | 87.7 s | kept as the control |
| `--chunk 2048` | 117169 | 256 | 1 | 1336.8 | 8.43 | 87.7 s | not kept |

Short decode on the successful rows, one sample: control 122.60 tok/s, chunk 2048 120.43 tok/s. Host RSS peak 3.48 GiB and 3.65 GiB. The failed row recorded 0 because the watchdog had not matched the process yet. Retrieval 3/3 on both successful long rows.

Prefill did not move. Long decode got slower. The preset file was left unchanged.

## Correctness and limitations

The 3/3 check is exact match of three planted codes. It is not a quality score. No standard-protocol sanity set was run.

Not measured: engine timing lines, file hashes, a second control, KV format changes, concurrency above 1. One sample per cell. CPU governor stayed on `powersave`.
