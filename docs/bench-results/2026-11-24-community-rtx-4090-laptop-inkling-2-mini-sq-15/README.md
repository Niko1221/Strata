# Community benchmark: Inkling-2-Mini SQ-15 on an RTX 4090 Laptop

Measured on 2026-11-24 by [andyholst](https://github.com/andyholst). This folder is only
Inkling-2-Mini SQ-15, served as `inkling-2-mini-sq-15`. SQ-10 and SQ-20 are separate
reports. Community speed log from a custom harness. Not a `strata bench run` result,
and not an engine change.

Prompt tok/s is `prompt_tokens / time-to-first-token`. Decode tok/s is
`(completion_tokens - 1) / (stream end - first token)`. Both clocks are the client's.
Engine timing lines were not kept.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090 Laptop, 16376 MiB, sm 8.9, driver 615.78.08, power cap 175 W, PCIe gen 4 x8.
- CPU and RAM: Intel Core i9-13980HX, governor `powersave`, 62 GiB RAM, on AC power.
- Storage model: not recorded.
- OS: Omarchy, kernel 7.2.15-1-omarchy.
- Strata 0.1.67 CUDA 13, binary `strata-inkling`, package `0.1.67-linux-x86_64-cuda-cuda13`. Control `COMMIT` file: `d46f5f268f8ed797e83a5c63a69bffbd7a581cf6`. Release binary.
- Same-day machine ceilings, server stopped: VRAM read 563.45 GB/s, pinned host-to-device 12.25 GB/s.
- Background: otherwise idle.

## Model and configuration

- Pack: Inkling-2-Mini SQ-15. Served id `inkling-2-mini-sq-15`. File hashes were not recorded.
- Context 131072. Launch preset `SQ-15 128K`, file not edited. Public copy: [preset-sq-15-128k.json](preset-sq-15-128k.json).
- Engine line: `--max-context 131072 --vram-experts-gb 11.49 --draft dspark --draft-beam 1 --draft-adapt`.
- Log: prefill chunk 512, 1644 MiB free after the experts, KV 3072 VRAM tokens plus 128000 host tokens, 248 MiB free after DSpark.
- The candidate added only `--chunk 1024`.

## Method

Source ledger: [andyholst/strata-inkling-mini-hillclimb](https://github.com/andyholst/strata-inkling-mini-hillclimb) at `a5d21c6`. This pack's rows are [ledger.jsonl](ledger.jsonl).

One short decode, cap 512, temperature 0.6, then one long document, cap 256, temperature 0. `--long 185000` is 144469 tokens here and the server returned HTTP 400. The measured long prompt is `--long 150000`, 117169 tokens. Unique nonce per document. Concurrency 1. One process per row. No second control.

## Results

| Configuration | Actual prompt tokens | Generated cap | Runs | Prompt tok/s | Decode tok/s | TTFT | Outcome |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| Too-long document | 144469, rejected | 256 | 1 | not measured | short sample 98.60 before the failure | 96.6 ms short | HTTP 400, `REQ_FAIL` |
| Control, chunk 512 | 117169 | 256 | 1 | 837.4 | 3.17 | 139.9 s | control, preset unchanged |
| `--chunk 1024` | 117169 | 256 | 1 | 833.8 | 3.21 | 140.5 s | not kept |

Short decode on the successful rows: control 98.72 tok/s, chunk 1024 95.80 tok/s. Host RSS peak on the control was 5.9 GiB. The candidate row recorded 0: the watchdog on that run still matched only `strata --serve`, and this binary is `strata-inkling`. Retrieval 3/3 on both successful long rows.

Prefill and long decode both sat on the control. The preset file was left unchanged.

## Correctness and limitations

3/3 is an exact code match, not a quality score. No standard-protocol suite. One sample per cell. Engine timings, file hashes, a second control, and other KV formats were not measured. Most of this context's KV is host memory on a link that reads about 12.3 GB/s. A larger prefill chunk does not move that KV onto the card.
