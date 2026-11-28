# AtomicChat Inkling‑2‑Mini AD‑Q4_K on one 48 GiB RTX 4090 (Linux CUDA)

**Community compatibility observation, not a `strata bench run` protocol result.** This tests the custom text GGUF [AtomicChat/Inkling-2-Mini-Uncensored-GGUF](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF/tree/82b34b9b0e5bc2e3e7b6d617ec8cc3e0f9692ee2), not the catalog SQ packs. Engine 0.1.69 loaded the GGUF directly, served chat, emitted a structured tool call with the expected arguments, and used the tool result on the next turn. No conversion, draft model, vision projector, second GPU, or host expert offload was used in those measurements. The model name identifies the checkpoint; this report does not evaluate refusal behavior.

## November 28, 2026 — v0.2.23 release check (Inkling engine 0.1.70)

The official Linux x86_64 CUDA 13 archive and the 551,600,700-byte [DFlash 2 mix drafter](https://huggingface.co/strata-engine/Inkling-2-Mini-DFlash2/tree/04ecd36faef74e696f0a7b6b3c119264442a6234) were independently checked against their published SHA-256 values. Same physical card and same custom GGUF as below. The `v0.2.23` Python frontend used the tokenizer and template exported from this GGUF. This was a **release smoke test**, not a rerun of BFCL or Terminal-Bench.

| Measurement | Draft off | DFlash 2 |
|---|---:|---:|
| Native `--bench`, median prefill / decode, tokens/s (64 in / 128 out, three processes) | 489.7 / 97.8 | Not measured |
| Warm loopback chat, median output tokens / wall-second (one repeated merge-sort prompt, three requests, 128 tokens each) | 99.82 | 124.25 |
| Arithmetic: `17 × 23 = 391`; native `weather(city=Paris)` tool call | Both passed | Both passed |

The DFlash 2 arm was **24.5% faster** on this repeated prompt; its engine log confirmed the drafter loaded. All six greedy chat answers had the same text SHA-256. This is a small, warm, single-request comparison with draft-off measured first: it does not establish the same gain for other prompts, long contexts, or agent workflows. Chat and native `--bench` use different inputs and timers. [Individual release measurements](release-0.1.70.json) · [chat reproducer](check_release.py).

## November 27, 2026 — 0.1.69 source build

Ubuntu 24.04.5 LTS; NVIDIA driver 595.99.02; CUDA 13.0 build targeting SM 89. One GeForce RTX 4090 reporting **49,140 MiB** was visible to inference, not a stock 24 GiB configuration. Host: two Xeon Platinum 8481C CPUs, 125 GiB usable RAM. Power draw and clocks not measured. Strata source: `c234774038d1c9249cb33637d27b895d47b39786`. See [results.json](results.json) for the model SHA-256, individual measurements, and the per-category scores.

| Test | Result |
|---|---|
| Synthetic, one slot, 64 prompt tokens / 128 generated, 4K context, FP16 KV | Median **556.5 prefill** and **96.4 decode tok/s** over three fresh engine processes |
| llama.cpp b12600, same GGUF, prompt token IDs, lengths, one slot | Median **72.2 decode tok/s** across three requests; Strata's median was 33.5% higher on this workload. Engine-reported decode only; timing boundaries and warm-cache behavior can differ. No token-for-token parity claim. |
| BFCL v4: seven non-live categories plus `multi_turn_base` | **1,281/1,590 (80.57%)** micro accuracy; **125/200** multi-turn; **zero inference errors** |
| BFCL request throughput | **67.24 output tok/s** = 631,732 completion tokens / 9,395.11 summed request-seconds, 4,063 requests. Includes prefill and request overhead; not the synthetic decode metric. |
| Terminal-Bench 4 pilot, three preselected tasks | **0/3** (`session-window-debug`, `data-anonymization`, `legacy-utility-triage`); each reached the 20-turn cap and received a verifier score. No infrastructure or timeout errors. |

For the quality runs: one slot, 32,768 context, native tool calling via Strata’s OpenAI-compatible frontend and the GGUF's exported tokenizer/template. BFCL used `bfcl-eval==2026.3.23`, the OpenAI completions FC handler, one worker, temperature 0, seed 0 and `reasoning_effort: none`, up to 4,096 completion tokens/request. Terminal-Bench used Harbor 0.15.0, OptiQ 0.5.12, one attempt/task, temperature 0.6, seed 0, 20 turns/10 minutes per agent. These are not full leaderboard scores. The results establish text/tool compatibility with this GGUF, not general autonomous-agent or multimodal performance.

To reproduce the native microbenchmark with the same weights and a single visible card:

```sh
# ENGINE and MODEL point to strata-inkling and the AD-Q4_K GGUF respectively.
for rep in 1 2 3; do
  CUDA_VISIBLE_DEVICES="$GPU" "$ENGINE" --bench --native "$MODEL" --gpu 0 \
    --parallel 1 --max-context 4096 --prompt-len 64 --gen 128 --draft off
done
```
