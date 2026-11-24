# Community benchmark: 2x Quadro RTX 6000 (sm_75, NVLink), Inkling-2-Mini SQ-20

Measured on 2026-11-24 by ambu50 with `tools/strata_bench.py` (3 runs per test, medians with
ranges). Speeds are the engine's own timings (`prompt_per_second`, `predicted_per_second`);
first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

**Track: `default`** — the official signed **strata-inkling 0.1.67** release, no local build or
patch. Setup choices (all within the release's supported options, listed below): SQ-20 (bit-exact
lossless), 2-GPU EP2, k8v4 KV, 128K context, DSpark on, vision on. On this card (sm_75) 0.1.67
fixes DSpark speculative-decode losslessness upstream (#6671); the parity harness now passes
128/128 exact on sm_75 and output is coherent — this run measures that stock engine.

## Hardware and software

| | |
| --- | --- |
| GPU | 2x NVIDIA Quadro RTX 6000, 24 GiB each (Turing, sm_75), driver 595.99.02 |
| GPU-GPU | NVLink; used via EP2 expert parallelism (`--gpus 2`, direct P2P, experts split 1416 GPU0 + 1656 GPU1) |
| PCIe | gen3 x16; 260 W power limit per card |
| CPU | Intel Xeon Gold 6138 @ 2.00 GHz, 40 cores / 80 threads, 2 sockets |
| RAM | 247.2 GiB |
| Storage | model on a SATA SSD; NVMe also present |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-42-generic |
| CUDA | engine package `0.1.67-linux-x86_64-cuda-cuda13` (CUDA 13 runtime); driver 595.99.02 (supports up to CUDA 13.2) |
| Strata | Terminal 0.1.65; engine **strata-inkling 0.1.67**, official signed release (no source build) |

## Model and configuration

- Model: Thinking Machines Lab **Inkling-2-Mini**, Strata Quant **SQ-20** (bit-exact lossless
  native SF4+MX8; 100% top-1 agreement with the original weights).
- Repository: `strata-engine/Inkling-2-Mini-Strata` (Hugging Face); file
  `Inkling-2-Mini-SQ-20.strata`, 19,346,685,952 bytes, SHA-256
  `0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97` (feed manifest hash).
- Vision: `Inkling-2-Mini-Vision.strata` (608 MB) loaded (`--vision on`, GPU reader).
- Context 131,072 tokens; KV **k8v4**; parallel slots 1; DSpark on; EP2 across both cards.
- Thinking: Inkling-2-Mini is a non-thinking model; the suite's "thinking off" is its normal mode.
- Sampling: the suite uses `temperature=0` (greedy). Calibration and experimental speed
  projection: not enabled.

```text
strata launch --model sq-20 --family inkling-2-mini --data-dir /mnt/d/strata-inkling \
  --context 128k --kv k8v4 --vision on --parallel 1 --tune skip --access local --port 8080 \
  --gpu 0,1 --split-engine off --drafting auto --no-tui --yes
# engine: official strata-inkling 0.1.67 (signed, cuda13)
# effective engine args: --serve --native Inkling-2-Mini-SQ-20.strata --max-context 131072 \
#   --vision Inkling-2-Mini-Vision.strata --gpus 2 --kv k8v4 --draft dspark
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
| short chat | 44 (43–50) | 0 | 100 | 9 | not measured | **71.4 (66.9–104.4)** | 0.14 (0.14–0.15) |
| 4K prompt, fresh | 4075 | 0 | 24 cap | 3 | **2344.2 (2340.2–2348.6)** | not measured | 1.81 (1.80–1.81) |
| 16K prompt, fresh | 16128 | 0 | 24 cap | 3 | **2108.2 (2084.1–2112.2)** | not measured | 7.81 (7.72–7.81) |
| agent session | — | — | — | 0 ok / 3 failed | — | — | — |

Per-request data: `results.json`.

**Notes:** chat decode is greedy DSpark (lossless on this release); the 66.9–104.4 spread is
prompt-dependent draft acceptance. A separate raw-stdio check of the same engine (no HTTP layer)
measured ~92 tok/s greedy / ~91 sampled on short prompts; the HTTP numbers here include the
serving/streaming layer. The 4K/16K TTFT is dominated by prefilling the fresh prompt. A 40K-token
prompt otherwise decoded at ~54 tok/s after prefill.

## Correctness and limitations

**Stock release, not a custom build.** On sm_75, 0.1.67 fixed DSpark speculative-decode
losslessness via #6671 (GEMV K-tile made a multiple of 128, tree split-K count unified, split-K
buffer enlarged). The parity harness passes 128/128 exact on sm_75, and short responses are
coherent. `--selfcheck` PASS.

**Speed suite, not a quality suite.** No needle/recall test was run. The agent-session suite failed
("no token came back"): it expects content tokens, but Inkling-2-Mini answers the tool prompts with
a tool call (no content), which the suite counts as a failure. This is a tool-format mismatch in
the benchmark suite, not a model defect; Inkling tool calling works via the OpenAI
`/v1/chat/completions` tools API. Agent/cache numbers are therefore not reported.

**Not run / unverified:** the standard `strata bench run` protocol (Qwen3.8-Flash-Next-specific,
still draft, no Inkling profile); needle/recall; other CUDA archs; HIP/SYCL/Metal (no hardware);
single 128K context only.