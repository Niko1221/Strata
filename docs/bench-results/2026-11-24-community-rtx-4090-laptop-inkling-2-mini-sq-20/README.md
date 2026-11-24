# Community benchmark: Inkling-2-Mini SQ-20 on an RTX 4090 Laptop

Measured on 2026-11-24 by [andyholst](https://github.com/andyholst). This folder is only
Inkling-2-Mini SQ-20, served as `inkling-2-mini-sq-20`. SQ-10 and SQ-15 are separate
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

- Pack: Inkling-2-Mini SQ-20. Served id `inkling-2-mini-sq-20`. File hashes were not recorded.
- Context 131072. Launch preset `SQ-20 128K`, file not edited. Public copy: [preset-sq-20-128k.json](preset-sq-20-128k.json).
- Engine line: `--max-context 131072 --vram-experts-gb 10.23 --ram-experts-gb 5.45 --draft dspark --draft-beam 1 --draft-adapt`.
- Log: prefill chunk 512, 2042 MiB free after the experts, KV 11520 VRAM tokens plus 119552 host tokens, 356 MiB free after DSpark. Cold experts stay in a pinned host arena and the card reads them itself.
- The candidate added only `--chunk 1024`.

## Method

Source ledger: [andyholst/strata-inkling-mini-hillclimb](https://github.com/andyholst/strata-inkling-mini-hillclimb) at `a5d21c6`. This pack's rows are [ledger.jsonl](ledger.jsonl).

One short decode, cap 512, temperature 0.6, then one long document, cap 256, temperature 0. `--long 185000` is 144469 tokens here and the server returned HTTP 400. The measured long prompt is `--long 150000`, 117169 tokens. Unique nonce per document. Concurrency 1. One process per row. No second control.

## Results

| Configuration | Actual prompt tokens | Generated cap | Runs | Prompt tok/s | Decode tok/s | TTFT | Outcome |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| Too-long document | 144469, rejected | 256 | 1 | not measured | short sample 52.59 before the failure | 632.8 ms short | HTTP 400, `REQ_FAIL` |
| Control, chunk 512 | 117169 | 256 | 1 | 489.2 | 3.37 | 239.5 s | control, preset unchanged |
| `--chunk 1024` | 117169 | 256 | 1 | 465.4 | 3.16 | 251.7 s | not kept |

Short decode on the successful rows: control 51.64 tok/s, chunk 1024 54.24 tok/s. Host RSS was recorded as 0 on both successful rows: the watchdog still matched only `strata --serve`, and this binary is `strata-inkling`. Retrieval 3/3 on both successful long rows.

The larger chunk made prefill slower (489.2 to 465.4 tok/s) and did not move long decode. The preset file was left unchanged.

## Correctness and limitations

3/3 is an exact code match, not a quality score. No standard-protocol suite. One sample per cell. Engine timings, file hashes, a second control, host RSS for this binary, and other KV formats were not measured. Cold experts and most of the KV sit in host memory.
