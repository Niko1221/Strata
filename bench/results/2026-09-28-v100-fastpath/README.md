# V100 fast-path benchmark (2026-09-28, merged main)

Qwen3.8-Flash-Next Q2_0 on the Tesla V100-PCIE-16GB (`sm_70`), Ryzen 5 3600, 48 GB DDR4-3200,
CUDA 12.8, engine 0.1.20 + the prefill-speed PR (FP16 tensor-core projection GEMMs, io_uring
O_DIRECT PLE reads, model on the NVMe). Runtime settings: 262,144-token context, int8 KV,
automatic prefill, five CPU pool workers, 0.28 PCIe fraction, MTP (`--spec 4`, draft floor
0.70).

Each row is a separate, uncached (`reused=0`) OpenAI-compatible chat-completion request with a
unique-prefix repeated-text prompt and 64 generated tokens. Sizes target the exact published
baseline totals (4,156 / 4,390 / 8,801 / 29,512 / 117,833 / 256,073); the row's true token
count comes from the server's `/metrics` endpoint. Prompt and decode timings are the engine's
own (`prompt ... read in T ms`, `G generated in U ms` in the engine log).

Re-run with [`bench/run_v100_bench.py`](../../bench/run_v100_bench.py):

    .venv/bin/python bench/run_v100_bench.py --model-gguf <shard-1.gguf>
    .venv/bin/python bench/run_v100_bench.py --model-gguf <shard-1.gguf> --only '~8K,128K'

## Results

| Prompt size | Exact tokens | Prompt time | Prompt speed | Output speed | Hit rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| ~4K | 4,143 | 4.19 s | 989.1 tok/s | 49.3 tok/s | 84.4% |
| ~4K | 4,376 | 4.34 s | 1007.5 tok/s | 52.1 tok/s | 86.5% |
| ~8K | 8,785 | 9.09 s | 966.5 tok/s | 50.6 tok/s | 88.0% |
| 32K | 29,527 | 25.97 s | 1136.8 tok/s | 50.5 tok/s | 88.7% |
| 128K | 117,837 | 178.72 s | 659.3 tok/s | 44.9 tok/s | 88.4% |
| 256K | 256,076 | 618.39 s | 414.1 tok/s | 38.8 tok/s | 84.3% |

The 8K row was re-measured three times from a cool card: 967.0 / 965.0 / 966.5 tok/s.

## Against the engine 0.1.20 baseline (pre-fast-path)

Same prompt sizes, same settings. The published baseline (docs/DETAILS.md) measured 4K 811-868,
32K 464.6, 128K 324.6, 256K 259.3 tok/s. On today's run 32K is 2.45x, 128K 2.03x and 256K
1.60x faster; output speed is unchanged (the decode path never moved).

## Thermal note

The V100-PCIE-16GB here is passively cooled and idles at 58-60 C on a cool machine but climbs
to ~79 C after long runs; under load it reaches ~83 C and drops from the 1380 MHz boost to
~960-1280 MHz. Short rows measured while the card is hot read up to ~25% lower (the same 8K
prompt spans 616-967 tok/s), and the long rows are the least throttled at the start of a cool
session. The table above is the cool-idle measurement; consecutive long runs on a hot card are
slower.
