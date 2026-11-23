# Community benchmark: 2x Quadro RTX 6000 (sm_75, NVLink), Inkling-2-Mini SQ-20

Measured on 2026-11-23 by ambu50 with `tools/strata_bench.py` (3 runs per test, medians with
ranges). Speeds are the engine's own timings (`prompt_per_second`, `predicted_per_second`);
first-token times are measured by the client over HTTP.

**This is a `custom` track result, not a release default.** The engine is a local sm_75 source
build of `strata-inkling` 0.1.66 carrying the DSpark speculative-decode parity fix submitted in
PR #6824 (commit `67977b8`), with the draft cost-model default `INK_MOE_RATIO` raised 4.2 → 16.
On stock 0.1.66, DSpark output on Turing (sm_75) is garbled/looping and unusable; this build makes
it lossless and coherent. See "Correctness and limitations" for the before/after and the baseline.

## Hardware and software

| | |
| --- | --- |
| GPU | 2x NVIDIA Quadro RTX 6000, 24 GiB each (Turing, sm_75), driver 595.99.02 |
| GPU-GPU | NVLink present; used via EP2 expert parallelism (`--gpus 2`, direct P2P, experts split 1416 GPU0 + 1656 GPU1) |
| PCIe | gen3 x16; 260 W power limit per card |
| CPU | Intel Xeon Gold 6138 @ 2.00 GHz, 40 cores / 80 threads, 2 sockets |
| RAM | 247.2 GiB |
| Storage | model on `/mnt/d` (SATA SSD); NVMe also present |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-42-generic |
| CUDA | 12.4 (nvcc V12.4.131), source build for `sm_75` only |
| Strata | Terminal 0.1.65; engine `strata-inkling` 0.1.66 + local parity fix (PR #6824, commit `67977b8`); `INK_MOE_RATIO` default 16 |

Build: `engine/src/inkling2/cuda` compiled with nvcc 12.4, `-gencode arch=compute_75,code=sm_75`,
`-DINKLING_ENABLE_CUDA=1 -DINKLING_ENABLE_EP_CUDA=1`, linked against system CUDA 12.4 cublas/cudart.
No prebuilt binary. The three parity changes and the env-tunable draft knobs are exactly PR #6824
(plus the local default-ratio change to 16, which upstream keeps at 4.2).

## Model and configuration

- Model: Thinking Machines Lab **Inkling-2-Mini**, Strata Quant **SQ-20** (bit-exact lossless
  native SF4+MX8; 100% top-1 agreement with the original weights).
- Repository: `strata-engine/Inkling-2-Mini-Strata` (Hugging Face); file
  `Inkling-2-Mini-SQ-20.strata`, 19,346,685,952 bytes, SHA-256
  `0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97` (feed manifest hash).
- Built-in DSpark drafter (in the pack); Vision shard not used (`--vision off`).
- Context 131,072 tokens; KV **k8v4**; parallel slots 1; DSpark on; EP2 across both cards.
- Thinking: Inkling-2-Mini is a non-thinking model; the suite's "thinking off" is its normal mode.
- Sampling: the suite uses `temperature=0` (greedy). Calibration and experimental speed
  projection: not enabled.

```text
strata launch --model sq-20 --family inkling-2-mini --data-dir /mnt/d/strata-inkling \
  --context 128k --kv k8v4 --vision off --parallel 1 --tune skip --access local --port 8080 \
  --gpu 0,1 --split-engine off --drafting auto --no-tui --yes
# engine: strata-inkling 0.1.66 + PR #6824 parity fix, INK_MOE_RATIO default 16 (local build)
# effective engine args: --serve --native Inkling-2-Mini-SQ-20.strata --max-context 131072 \
#   --gpus 2 --kv k8v4 --draft dspark
```

## Method

`python tools/strata_bench.py --url http://127.0.0.1:8080 --runs 3` against the running server's
HTTP API; model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every
fresh prompt carries a unique tag so the conversation cache cannot help the prompt tests.

- short chat: three questions, up to 100 new tokens, 3 runs (9 requests).
- 4K / 16K prompts: generated word lists sized by a calibration request (~1.42 tokens/word),
  24 new tokens, 3 fresh prompts each.
- agent session: system prompt + four tools, 6 turns, 64 new tokens/turn, 3 sessions. **Failed on
  this model** — see limitations.

## Results

| Configuration | Actual prompt tokens | Reused | Generated | Runs | Prompt tok/s (median, range) | Decode tok/s (median, range) | TTFT s (median, range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 44 (43–50) | 0 | 100 | 9 | not measured | **68.6 (64.6–102.8)** | 0.14 (0.14–68.34) |
| 4K prompt, fresh | 4075 | 0 | 24 cap | 3 | **1978.4 (1978.3–1982.5)** | not measured | 2.11 (2.11–2.13) |
| 16K prompt, fresh | 16128 | 0 | 24 cap | 3 | **1833.9 (1833.8–1835.4)** | not measured | 8.86 (8.85–8.86) |
| agent session | — | — | — | 0 ok / 3 failed | — | — | — |

Per-request data: `results.json`. Suite total time: 382 s.

**TTFT outliers:** the 68.34 s and 37.84 s first-token times are the first two requests after
engine load (lazy CUDA module loading / expert-cache and graph warm-up). Steady-state chat TTFT is
0.14 s; the 4K/16K fresh-prompt TTFT (2.11 s / 8.86 s) is dominated by prefill of the fresh prompt.

**Decode note:** chat decode is greedy DSpark (lossless after the fix). Acceptance varies by prompt
(~33–53%); the 64.6–102.8 tok/s spread reflects that. A separate Fibonacci/palindrome HTTP check at
`temperature=0.7` gave 76.0 tok/s sampled (accept 53%) and 80.8 tok/s greedy (accept 52%).

## Correctness and limitations

**Why this build exists (before/after on this machine, sm_75):**
- Stock `strata-inkling` 0.1.66, DSpark on: output garbled and repeating (parity harness 44/128
  tokens exact, first divergence at token 2). Unusable.
- This build (PR #6824): parity harness 128/128 exact for the serving spec paths; coherent code and
  prose; `--selfcheck` PASS; no CUDA errors across repeated 256–512-token generations.
- Plain decode baseline (`--draft off`, same machine/config): ~40 tok/s. DSpark on this build:
  ~68–80 tok/s coherent. The stock engine cannot produce a coherent DSpark baseline on sm_75.

**Speed suite, not a quality suite.** No needle/recall test was run here. The agent-session suite
failed ("no token came back"): it expects content tokens, but Inkling-2-Mini answers the tool
prompts with a tool call (no content), which the suite counts as a failure. This is a tool-format
mismatch in the benchmark suite, not a model defect; Inkling tool calling works via the OpenAI
`/v1/chat/completions` tools API. Agent/cache numbers are therefore not reported.

**Not run / unverified:** other CUDA archs (sm_80/86/89/120), HIP/SYCL/Metal (no hardware); the
standard `strata bench run` protocol (it is Qwen3.8-Flash-Next-specific and still draft, with no
Inkling profile). Single 128K context only; no long-context (32K/128K-prompt) decode sweep.

**Custom track:** modified build + non-default `INK_MOE_RATIO=16`; not comparable to a release
`default` headline. The same-machine baseline above isolates the fix's effect.