# 2026-10-28: RTX 5080, 2x Xeon Platinum 8176, 251 GB RAM — one box, four arms, plus quant quality

Strata `0.2.3-cuda13`, engine `0.1.48`, one RTX 5080 16 GB in a dual-socket server, **measured 2026-10-28**. Four
models were served one at a time, each in its own production launch, and measured with the standard protocol
(`strata bench run`, project `0.2.3`); the arm folders hold the tool-written `README.md`, `results.json`,
`outputs.jsonl`, `config.json` and `provenance/` for each:

| Folder | Model | Quantization | Weights |
|---|---|---|---|
| [iq3s](iq3s/) | qwen3.8-flash-next | IQ3_S | 83.6 GB |
| [nvfp4-qad](nvfp4-qad/) | qwen3.8-flash-next-qad | NVFP4 | 89.3 GB |
| [iq3_xxs](iq3_xxs/) | qwen3.8-flash-next | IQ3_XXS | 75.8 GB |
| [qwen36-35b](qwen36-35b/) | Qwen3.6-35B-A3B (Unsloth UD-Q4_K_XL) | ~4.5-bit | 22.85 GB |

`quality/` holds a second, unrelated measurement: how far each quant's next-token distribution sits from the IQ3_S
reference on a frozen 4,995-position set. It is a fidelity measurement, not a benchmark score, and its method is in
[quality/README.md](quality/README.md).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5080, 15.9 GiB (one card; the box has two sockets but a single GPU)
- CPU: 2x Intel Xeon Platinum 8176 (AVX-512); RAM 250.6 GiB
- OS: Ubuntu 24.04.5 LTS on Unraid 6.18.58; driver 580.196.07; CUDA 13.0
- Strata 0.2.3-cuda13 (release image `ghcr.io/niko1221/strata:0.2.3-cuda13`), engine 0.1.48
- Context 262,144 tokens for the three Qwen3.8 arms; the Qwen3.6 arm runs its own engine (`strata-q36`) at 32,768
- Models and files: named in each arm's `README.md`; no file is re-downloaded between arms
- Background work: the same on every arm (nothing else running); no clock or power capture

## Method

`strata bench run --url http://127.0.0.1:9000 --credential-env STRATA_API_KEY --out <arm folder>`, unmodified protocol
`strata-qwen38-thinking-v1-draft.1`: one warm-up request (excluded), then three repetitions each of short, 4K and 16K
prompts, streaming over loopback, one request at a time. Prompt and decode rates are the engine's own `timings`; TTFT is
the client's monotonic clock. The arms ran back to back inside one 12-minute window, with the previous model stopped
before the next was started, so the box temperature and background load are comparable between arms.

**Read the decode column with the cap in mind.** The protocol caps a request at 512 tokens (reasoning and answer
together). Three of the four arms are thinking models sampling at `reasoning_effort xhigh`, so nearly every performance
request reached the cap: `Capped` is 3/3 in almost every cell, which makes those decode rates *time to the cap*, not
time to an answer, and leaves `First answer s` unmeasured for the arms that were still reasoning at the cap. They are
still directly comparable to each other — every arm was capped in the same way — but they are not "decode until the
model stops".

## Standard protocol results

`Prompt` is prefill; `Decode` is the median of three runs; ranges are in each arm's `README.md`.

| Arm | Prompt 4K tok/s | Prompt 16K tok/s | Decode short | Decode 4K | Decode 16K | TTFT 4K s | TTFT 16K s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IQ3_S (reference) | 990.5 | 1,911.8 | 87.2 | 84.0 | 78.7 | 4.17 | 8.63 |
| NVFP4-QAD | 749.7 | 1,417.8 | 57.4 | 58.8 | 53.3 | 5.50 | 11.63 |
| IQ3_XXS | **1,216.1** | **2,245.3** | 99.3 | 93.0 | **97.6** | **3.40** | **7.36** |
| Qwen3.6-35B-A3B | 710.3 | 1,008.7 | **143.3** | **122.5** | 112.2 | 5.71 | 16.23 |

Relative to IQ3_S on the same box (`+` is better):

| Arm | Decode short | Decode 4K | Decode 16K | Prompt 4K | Prompt 16K | TTFT 4K | TTFT 16K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NVFP4-QAD | −34% | −30% | −32% | −24% | −26% | +32% (worse) | +35% (worse) |
| IQ3_XXS | +14% | +11% | +24% | +23% | +17% | −18% | −15% |
| Qwen3.6-35B-A3B | +64% | +46% | +43% | −28% | −47% | +37% (worse) | +88% (worse) |

Two results are worth naming:

- **IQ3_XXS is faster than IQ3_S on every axis here** — 23% more prefill at 4K, 17% at 16K, 11 to 24% more decode, and
  15 to 18% lower time to first token — for 9.3% fewer bytes (75.8 GB vs 83.6 GB). The quality cost of that is measured
  in `quality/`: 85.23% top-1 agreement with the reference and mean KL 0.1737, against NVFP4's 86.97% and 0.1495.
- **The Qwen3.6-35B arm splits the two regimes.** With only ~3B active parameters and 22.85 GB of weights it decodes
  64% faster than the brain at short prompts and still 43% faster at 16K, but its prefill is 28% lower at 4K and 47%
  lower at 16K, and its time to first token is 88% higher at 16K. On this engine, prompt processing is where the older
  and much smaller model loses.

### The 35B prefill number is a packaging bug, not a model verdict

The q36 engine logs, once per start:

```
strata: K-quant prompt kernels (STRATA_MMQ_KQ=1): libstrata_kq.so not found next to the engine
        - these experts take the FP16 prompt path
```

The library is not shipped next to the `0.2.3-cuda13` engine binary, so prompt processing takes the FP16 path instead of
the K-quant kernels. That is the most plausible cause of the gap above, and it should be fixed or documented before the
prefill column is read as a property of the model.

### Quick suite, for comparison with earlier reports in this directory

The same three arms were also timed with a simple loop (one warm-up call, then three timed runs per case, single request,
engine `timings`), which is closer to the numbers in the older community folders but is **not** the standard protocol:

| Arm | Decode 512 tok | Prefill 3,882 tok |
| --- | --- | --- |
| IQ3_S | 5.375 s = 95.3 tok/s | 1.028 s = 3,778 tok/s |
| IQ3_XXS | 5.00 s = 102.4 tok/s | 0.892 s = 4,350 tok/s |
| Qwen3.6-35B-A3B | 3.632 s = 141.0 tok/s | 3.265 s = 1,189 tok/s |

The 35B improved monotonically across its three timed runs (4.47 s → 3.63 s → 3.08 s) as its expert profile filled,
while the two IQ3 arms stayed flat within ±2%: quote it as 115 to 166 tok/s with a median of 141, and warm it before
measuring.

## Quant quality (fidelity to the reference), separate from the speed runs

`tools/quant_quality.py`, frozen set v1, five segments of 999 teacher-forced positions each, reference = IQ3_S, same
engine and the same fixed conditions for both arms. Files: `quality/cfg-*.json`, `quality/compare*.json`,
`quality/*.seg.json`; the per-position TSVs are about 5.3 MB each and stay out of the repository.

| Arm | top-1 agreement | mean KL | ppl (ref 4.103) | ppl delta |
| --- | --- | --- | --- | --- |
| NVFP4-QAD | 86.97% | 0.1495 | 3.942 | −0.161 |
| IQ3_XXS | 85.23% | 0.1737 | 4.146 | +0.043 |

Per segment (top-1 agreement / KL):

| Segment | NVFP4-QAD | IQ3_XXS |
| --- | --- | --- |
| code | 93.29% / 0.0846 | 91.39% / 0.1146 |
| docs | 78.98% / 0.2108 | **75.48% / 0.2442** |
| english | 86.19% / 0.0970 | 83.78% / 0.1292 |
| multilingual | 91.19% / 0.0611 | 91.99% / 0.0723 |
| agent | 85.19% / 0.2943 | 83.48% / 0.3085 |

The pack *below* the brain is not the pack closest to it: **IQ3_XXS sits further from IQ3_S than the 4-bit NVIDIA format
does**, by 16% in mean KL and 1.7 points of agreement, while being the fastest arm measured. Its damage is uneven —
multilingual and code hold up, while docs falls to one differing next token in four (75.5%) and the agent transcript
reaches KL 0.31.

A note on what this measures: KL and agreement compare a quant with **IQ3_S**, not with an unquantized model. They say
how safe a quant is to swap in, not how good the model is. The softmax is sampled at the engine's own temperature, so
the absolute perplexities are not comparable with a GPU-side `llama-perplexity` run.

## Not measured here

- 32K and 128K prompts: the standard protocol's corpus stops at 16K, and the Qwen3.6 arm runs 32,768-token context
- Concurrency: single-request only, one slot
- Power, clocks and temperature; background load only as "nothing else running"
- Whether the speed holds after hours of serving: each arm was measured minutes after load
- Absolute quality: no task-based evaluation (HumanEval, agentic suites) was run for any arm
