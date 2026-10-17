# Community benchmark on 2x NVIDIA RTX PRO 4500 Blackwell (Linux), engine 0.1.41 against 0.1.40.1

> Measured on 2026-10-10 by qni-live. One machine, one model, single stream plus a concurrent-client sweep.
> Main limitation: short prompts for the concurrent runs, no agent workload.

## Hardware and software

- GPUs: 2x NVIDIA RTX PRO 4500 Blackwell (GB203, sm_120), 32 GB GDDR7 each, 200 W limit, PCIe 5.0 x16 both
  (memory clock raised to 14,000 MHz, the rated bin, with an NVML offset of +1270)
- CPU: AMD Ryzen Threadripper 7960X (24 cores / 48 threads, AVX-512); RAM 61 GiB; NVMe storage
- OS: Ubuntu Linux; driver 595.91.07; CUDA 13.x source build for sm_120
- Strata: v0.1.41 (commit fb58e0d) and v0.1.40.1, source build, default Linux release scripts
- Background workloads: none during the runs (service stopped, nothing else on the GPUs)

## Model and configuration

- Model: Swift-Qwen3.8-Flash-Next IQ3_XXS (2 shards), vision encoder `mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf` on GPU
- Context 262,144; KV `k8v4`; `--spec 5`; `--spec-min-p 0.5`; `--pipeline-windows 2`; layer split 25 across both GPUs;
  `--vram-reserve-mib 700`; MTP draft layer with a German draft vocabulary (`--mtp-draft-vocab`, 121,059 ids,
  built with `tools/draft_vocab.py` from German Wikipedia text, coverage 0.99)
- Sampling for the server: temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0.0; the benchmark requests use
  temperature 0 and `max_tokens` 256
- Batch runs: `--batch 8 --trim-stage-weights`, vision off, reserve 700 MiB

## Method

- Single stream: `bench.py` (in this folder), 4k / 32k / 128k / 256k prompt tokens built from the repository's own text,
  a random nonce at the start of every prompt (no prefix reuse), one warm-up and three measured runs per length,
  prompt and decode tok/s and drafts accepted taken from the engine's timing line. Order: 0.1.41, 0.1.40.1, 0.1.41.
- Concurrent clients: `tools/serve_load.py http://127.0.0.1:8080 --clients 1,2,4,8 --rounds 3 --max-tokens 256`.
- Needle test: `tools/needle_bench.py --lengths 32k,128k,262k --depths 10,50,90` (262k skipped by the script because
  the server context is 262,144).
- Memory clock sampled once per second during every arm (`takt-*.csv`): 14,000 MHz in every sample.

## Results

Single stream, medians (range), 3 runs each, cold prompts:

| Engine | Prompt tokens | Prompt tok/s | Decode tok/s | Drafts accepted |
| --- | ---: | --- | --- | ---: |
| 0.1.41 (run 1 / run 2) | 4,233 | 3,075 (3,072-3,075) / 3,065 (3,062-3,070) | 152.0 (146.7-169.4) / 152.4 (150.1-155.6) | 63 % / 62 % |
| 0.1.40.1 | 4,234 | 3,057 (3,051-3,062) | 155.8 (153.3-173.9) | 66 % |
| 0.1.41 | 30,794 | 5,225 / 5,213 | 168.7 (163.6-177.5) / 149.7 (146.0-165.6) | 68 % / 69 % |
| 0.1.40.1 | 30,796 | 5,222 (5,219-5,224) | 175.9 (172.7-177.1) | 69 % |
| 0.1.41 | 120,608 | 6,148 / 6,146 | 131.9 (126.9-133.8) / 132.3 (131.1-140.3) | 55 % / 55 % |
| 0.1.40.1 | 120,608 | 6,142 (6,139-6,152) | 132.1 (127.8-141.1) | 55 % |
| 0.1.41 | 248,490 | 6,031 / 6,025 | 146.4 (141.1-165.3) / 154.8 (141.4-164.3) | 60 % / 61 % |
| 0.1.40.1 | 248,488 | 6,033 (6,031-6,035) | 145.9 (145.6-149.4) | 60 % |

Concurrent clients, `--batch 8`, total tokens/s over the burst (median of 3 rounds); the last column is the same
machine in single-stream mode with its requests queued:

| Clients | 0.1.41, `--batch-groups auto` (2 groups of 4) | 0.1.41, `--batch-groups 1` | 0.1.40.1, `--batch-groups 2` | Single stream (queue) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 115.3 | 108.7 | 116.6 | 128.8 |
| 2 | 152.2 | 125.6 | 76.5 | 132.1 |
| 4 | 208.1 | 159.2 | 147.5 | 133.5 |
| 8 | 256.1 | 175.4 | 245.8 | 132.8 |

Time to first token at 1 / 2 / 4 / 8 clients (p50, s): batch `auto` 0.28 / 0.46 / 0.86 / 1.63; single stream
0.18 / 1.05 / 3.15 / 7.00. No request failed. Eight slot sessions take 12.1 GiB of VRAM on each card and leave 17,537
of 24,576 experts resident (71 %); `--pipeline-windows` is off in batch mode.

Per-run files: `bench-*/`, `load-*.json`, `takt-*.csv`, `timing-*.txt`, `needle-*.json`.

## Correctness and limitations

- Tool calls: 9 of 9 correct (3 single, 6 with two concurrent requests).
- Needle: 6 of 6 found (32k and 128k at depths 10/50/90); 262k skipped by the script.
- `layer_split: auto` chose layer 24 on two identical cards and did not reorder them (issue #1760 not reproduced);
  `"gpu_order": "as_given"` gave the same assignment.
- Not tested: batch 4, long prompts or tool calls in batch mode, vision in batch mode, Windows, cards of different speed.
- Run-to-run spread inside one arm is up to 15 % at 32k decode (149.7 to 175.9 tok/s); the single-stream difference
  between the two versions is inside that spread.
