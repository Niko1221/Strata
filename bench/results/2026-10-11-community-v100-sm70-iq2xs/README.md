# Community benchmark: Tesla V100 32GB, i5-8400, Strata 0.1.42

Measured on 2026-10-11 by [xingyun618](https://github.com/xingyun618), on a Linux
machine called `.34`. This tests Strata 0.1.42 (sm_70, CUDA 12.0) with the
original Flash-Next IQ2_XS, one GPU, and a 65,536-token context limit.

Single-stream decode throughput was **68.4 tok/s** at ~4K prompt tokens
(measured as 3.7s for 256 output tokens, including prefill). This is a
synthetic speed check with temperature 0.7, top_p 0.8, and a 256-token output
cap. It does not establish general answer quality or performance on other
workloads.

## Hardware and software

- NVIDIA Tesla V100-SXM2-32GB; 32 GB VRAM; PCIe gen3.
- Intel Core i5-8400; 6 logical CPUs. No AVX-512.
- 23 GB system RAM; Linux (Ubuntu 24.04).
- Strata 0.1.42, compiled from main for sm_70, CUDA 12.0.
  Source revision: `IQ2_XS@8188` (resident headroom 6 GiB).
- Model: `Qwen/Qwen3.8-Flash-Next`, IQ2_XS quantization.
- Context 65,536; KV cache fp8_e5m2; MTP speculative decoding enabled
  (`--spec 4`).

## Model and configuration

The model was served via the Strata OpenAI-compatible endpoint at
`http://192.168.5.25:8188/v1`. No other GPU services were running during the
measurement (Ollama/vLLM/ComfyUI were stopped to avoid VRAM contention).

Request parameters:
- `temperature: 0.7`
- `top_p: 0.8`
- `max_tokens: 256`
- `stream: true`

The benchmark used a single 256-token prompt with greedy-like sampling. The
3.7s wall time includes both prefill and decode; the 68.4 tok/s figure is
total-output-tokens / total-time.

## Method

Timing was measured with a custom Python script (`strata_compare.py`) that
sends a single chat completion request and records wall-clock time from
request start to the last streamed token. The script reports
`tokens_per_second = output_tokens / elapsed_seconds`.

This is a quick speed check, not a full `strata bench run` protocol execution.
The machine was under light load from system services; GPU clocks were not
fixed.

## Results

| Metric | Value |
| --- | --- |
| Prompt tokens | ~4,096 |
| Output tokens | 256 |
| Total time | 3.7 s |
| **Decode throughput** | **68.4 tok/s** |
| Context limit | 65,536 |
| KV cache dtype | fp8_e5m2 |

Raw output: [strata_compare_out.txt](strata_compare_out.txt)

## Limitations

One run, one prompt length, no repeats. No needle/recall test, no TTFT
measurement, no multi-request concurrency sweep, no thermal run. This is a
single-machine, single-configuration, single-workload data point. Do not
interpret it as a controlled comparison with other GPUs or engines.
