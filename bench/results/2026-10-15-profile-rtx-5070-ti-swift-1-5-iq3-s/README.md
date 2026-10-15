# Where the time goes: NVIDIA GeForce RTX 5070 Ti, swift-1.5-iq3_s, 2026-10-15

Measured by `tools/strata_profile.py` through the HTTP API with `STRATA_PREFILL_TIMING=1` and `STRATA_DECODE_TIMING=1` in the engine's environment; the figures are the engine's own lines.

- GPU: NVIDIA GeForce RTX 5070 Ti, 16 GiB VRAM, driver 595.99.02
- CPU: AMD Ryzen 7 5800X3D 8-Core Processor, 8 cores, 76 GiB RAM
- OS: Linux 7.0.0-38-generic; Strata 0.1.44.1 (0a2f1966), engine 0.1.44.1
- Model: swift-1.5-iq3_s (IQ3_S), context 262,144
- Launch: `strata --pack swift-iq3_s --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768`

## Prompt reads (the GPU timeline by phase)

| Read | Tokens | Prompt tok/s | GPU ms | experts over PCIe | dequant | expert GEMM | attention | GDN | other | Largest phases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2K fresh | 2,023 | 1,118 | 1,614 | 43% | 15% | 18% | 5% | 13% | 6% | wait copy 42%, dequant 15%, gemm gate/up 9% |
| 16K fresh | 16,058 | 2,982 | 5,099 | 6% | 7% | 30% | 18% | 25% | 14% | gemm gate/up 14%, gdn 12%, hc read 10% |
| 32K (long) fresh | 32,049 | 3,041 | 10,257 | 6% | 7% | 27% | 21% | 25% | 15% | gemm gate/up 13%, gdn 12%, hc read 10% |

"experts over PCIe" is the time the card waited for expert copies (wait copy, gather, host grouping); "dequant" their expansion on the card; "expert GEMM" gate/up, down and combine; "attention" the QSA projections, indexer, selection, attention and KV staging; "GDN" the linear-attention layers; "other" the hc read, router, PLE and embeddings.

## Decode windows (ms per window)

| Request | Context | Decode tok/s | Hit rate | PCIe share | ms/window | GPU | CPU experts | draft | adapt wait | other | tokens/window | CPU experts per layer-window | their entries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chat | 44 | 75.7 | 71% | 10% | 33.6 | 10.2 | 19.3 | 1.8 | 1.4 | 0.8 | 2.53 | 4.7 | 7.0 |
| long context | 32,049 | 82.0 | 82% | 6% | 27.4 | 10.0 | 13.1 | 2.3 | 1.3 | 0.8 | 2.25 | 3.6 | 4.7 |
| agent turns | 8,721 | 71.7 | 79% | 6% | 32.2 | 10.1 | 17.9 | 2.0 | 1.3 | 0.9 | 2.38 | 4.7 | 6.6 |

"GPU" is the host's wait for the card to reach the end of the window; "CPU experts" the experts the CPU pool computed because they were not in VRAM (plan, activation quantization, jobs and the compute); "draft" the draft layer; "adapt wait" the adaptive tier's swaps; "other" commit, stage and what the line does not itemize. "Hit rate" is the VRAM share of the experts looked up; "PCIe share" the share of all routed experts the card read over PCIe instead (not in the hit rate). The tokens per second were measured with the timing switches on: for speeds, run tools/strata_bench.py.

KV streaming (this request's block reads): long context 96% of 838,104 reads from VRAM, 119 MiB from RAM

## Agent session, turn by turn

| Turn | Prompt tokens | Reused | Read | Prompt tok/s | Decode tok/s | Hit rate | ms/window | GPU | CPU experts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 496 | 48 | 51% | 39.1 | 9.5 | 26.4 |
| 1 | 3,744 | 623 | 3,121 | 1,452 | 65 | 62% | 37.6 | 10.0 | 23.4 |
| 2 | 5,277 | 3,808 | 1,469 | 769 | 67 | 74% | 35.5 | 10.2 | 20.8 |
| 3 | 8,721 | 5,341 | 3,380 | 1,571 | 80 | 85% | 25.1 | 10.1 | 11.1 |
| 4 | 11,647 | 8,784 | 2,863 | 1,340 | 98 | 86% | 28.8 | 10.3 | 14.5 |
| 5 | 14,596 | 11,697 | 2,899 | 1,428 | 72 | 79% | 34.3 | 10.2 | 19.9 |

## Findings

- The 2K read (2,023 tokens, 1,118 tok/s) waits for the experts streamed over PCIe: 43% of its GPU time waiting for their copies and 15% dequantizing them.
- The 16K read (16,058 tokens, 2,982 tok/s) is compute-bound: expert GEMM 30%, attention 18%, GDN 25% of its GPU time; the experts' copies and dequant 13%.
- The 32K (long) read (32,049 tokens, 3,041 tok/s) is compute-bound: expert GEMM 27%, attention 21%, GDN 25% of its GPU time; the experts' copies and dequant 13%.
- chat (44 tokens of context, 76 tok/s, expert cache hit rate 71%): of a 33.6 ms window the card takes 10.2 ms and the experts computed on the CPU 19.3 ms (4.7 experts, 7.0 entries per layer-window): the misses of the VRAM expert cache take the larger share of the window.
- long context (32,049 tokens of context, 82 tok/s, expert cache hit rate 82%): of a 27.4 ms window the card takes 10.0 ms and the experts computed on the CPU 13.1 ms (3.6 experts, 4.7 entries per layer-window): the misses of the VRAM expert cache take the larger share of the window.
- agent turns (8,721 tokens of context, 72 tok/s, expert cache hit rate 79%): of a 32.2 ms window the card takes 10.1 ms and the experts computed on the CPU 17.9 ms (4.7 experts, 6.6 entries per layer-window): the misses of the VRAM expert cache take the larger share of the window.
- Agent turns: the expert cache hit rate per turn runs from 51% (turn 0) to 86% (turn 4).
- Agent turns: each follow-up turn read 1,469-3,380 tokens after what the conversation cache reused.
- A short read that waits for expert copies is on the path the CPU prefill share works on (STRATA_PREFILL_CPU_SHARE, its range STRATA_PREFILL_CPU_SHARE_MAX; docs/DETAILS.md, "Short prompts: the CPU shares the experts"). Its range stops at 1,024 tokens by default and a wider one is not a sure win (2,048-token chunks measured 2% slower on one PC, docs/DETAILS.md): measure it here before setting it (tools/ab_engine.py A/Bs an environment variable).
- A decode whose larger share is the experts computed on the CPU follows the expert cache's hit rate: the settings on that path are the cache's slots (VRAM: --max-context, --kv-resident, --vram-reserve-mib; docs/DETAILS.md, "The expert cache size is a budget"), the PCIe share and the adaptive tier (setup --calibrate measures both on this PC) and a saved expert profile (expert_profile_save).
