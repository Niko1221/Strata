# Community benchmark: RX 6800 XT + RX 6600, layer split, 128k context (v0.2.11)

Measured on 2026-11-05 by @Kompasby, on release v0.2.11. Two RDNA2 cards in one Windows PC: an RX 6800
XT runs layers 0-37 and an RX 6600 runs layers 38-47 at a 131,072-token context. The main limitation:
this is a hand-built, locally installed engine on a platform where AMD multi-GPU is documented as
Linux-only, so treat it as a report on a configuration rather than a supported path.

## Hardware and software

- GPU 0: AMD Radeon RX 6800 XT, 16 GB, gfx1030. GPU 1: AMD Radeon RX 6600, 8 GB, gfx1032 (x4 slot).
  Layer split 0-37 / 38-47, chosen by the engine's own split search; PCIe link not measured.
  The API names the RX 6600 with a generic name, so results.json lists device 0 as "AMD
  Radeon(TM) Graphics" and device 1 as the RX 6800 XT; that order is not the engine's role
  order, which is CUDA0 for layers 0-37 and CUDA1 for layers 38-47.
- CPU: AMD Ryzen 7 9700X (8C/16T); RAM 63.2 GiB at DDR5-5200; model storage on an NVMe Kingston
  SFYRD20 (1.8 TB). The integrated GPU (gfx1036) is disabled in Windows: with it visible the launcher
  picks it as the helper card (it reports ~43 GiB of shared memory) and the split collapses.
- OS: Windows 11 Pro 26300; GPU driver 32.0.21045.16010; ROCm 10.2.0a20260930 (TheRock wheels).
- **Strata v0.2.11; terminal 0.2.11 (control 0.1.57); engine 0.1.57; source build** for
  `gfx1030;gfx1032;gfx1036`. The engine is built from the release commit because the signed package
  does not carry `gfx1032`; two local patches are described under "Limitations".
- Background work: none — the desktop app that normally drives the model was closed for the run; no
  power limits applied.

## Model and configuration

- Model: `qwen3.8-flash-next-iq2_xs`, IQ2_XS, ISTA-DASLab GSQ-RCO files
  `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`, `...-00002-of-00002.gguf`; no custom pack.
- Vision encoder downloaded, not used here: the reader runs on the CPU (`--vision cpu`).
- Context 131,072; `--kv int8`; `--kv-resident 32768`; expert cache and prefill `auto`; MTP on;
  thinking on (protocol default). The 25.6 GiB expert arena ran on **2 MiB large pages**
  (`expert arena: large pages are resident`).
- Launch command (API key removed):

```text
strata launch --family qwen --model iq2_xs --data-dir E:\Strata\Models --vision cpu \
  --context 128k --access lan --yes --no-tui
```

The split is produced by the engine's search, not by a fixed hand-off:

```text
layer split auto: CUDA0 36 SMs at 2.06 GHz -> 0.98 ms per layer
layer split auto: CUDA1 14 SMs at 2.04 GHz -> 2.54 ms per layer
layer split auto: CUDA1 link 6.9 GB/s -> miss cost x4.10
layer split: CUDA0 runs layers 0-37, expert cache 6700 slots (12.18 GiB)
layer split: CUDA1 runs layers 38-47, expert cache 1531 slots (2.15 GiB)
```

The x4 slot costs the second card bandwidth: its miss cost is weighted x4.10 against CUDA0's x1.00.

## Method

`strata bench run --suite standard-v1` (protocol `strata-qwen38-thinking-v1-draft.1`), one warm-up and
three repetitions per case, 512-token output cap, one request at a time over loopback HTTP, model
hashing before the run. Each case uses a fresh prompt, so `cached_tokens` is 0 everywhere and the
prompt figures are cold prefill. TTFT is measured from request start to the first content delta;
totals include prompt processing.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short (~80 tok) | 77 | 0 | 286-409 | 3 | 36.98 (36.8-40.9), fixed cost dominates | 40.0 (39.4-42.9) | 2.12 (1.94-2.12) |
| prompt-4k | 4,093 | 0 | 512 | 3 | 361.4 (361.3-362.5) | 38.8 (37.5-44.5) | 11.36 (11.34-11.38) |
| prompt-16k | 16,387 | 0 | 512 | 3 | 578.4 (577.0-578.8) | 40.1 (38.5-42.0) | 28.41 (28.38-28.47) |

Capped runs (reasoning plus answer reached the 512-token cap): 0/3 on short, 3/3 on prompt-4k, 3/3 on
prompt-16k. No request failed or timed out; 12 of 12 sanity checks passed.

The short case is 77 prompt tokens: results.json records its prompt throughput at 36.98 tok/s
median (36.8-40.9), which is not comparable with the 4k and 16k figures because fixed cost
dominates at that size.

## Against v0.2.10 on the same machine

Same cards, model and configuration, engine 0.1.56, measured a day earlier:

| Metric | v0.2.10 (engine 0.1.56) | v0.2.11 (engine 0.1.57) | Change |
| --- | ---: | ---: | ---: |
| prompt-4k | 287.2 tok/s | 361.4 tok/s | +26% |
| prompt-16k | 310.9 tok/s | 578.4 tok/s | +86% |
| decode-4k | 40.2 tok/s | 38.8 tok/s | -3% |
| decode-16k | 43.3 tok/s | 40.1 tok/s | -7% |
| TTFT-16k | 52.77 s | 28.41 s | -46% |
| total-16k | 64.5 s | 41.2 s | -36% |

Prefill roughly doubled at 16k, and decode gave back a few percent — a trade that clearly favours the
long-context case on this hardware.

## Correctness and limitations

- 12 of 12 sanity checks passed (instruction 4/4, retrieval 4/4, tool calls 4/4). One model, one
  quantization, one machine; no needle test in this run.
- **Unsupported configuration.** `setup.py` still refuses AMD layer splits on Windows ("Linux-only for
  now") and its `AMD_ARCHS` list lacks `gfx1032`, so the RX 6600 is reported as unsupported. Both
  guards were patched locally; the engine itself splits without complaint. Building `gfx1036` also
  needs it added to `_strata_hip_unvalidated` in `cmake/hip_backend.cmake`.
- The engine is unsigned and installed by hand. `results.json` records `build_kind` as `release`
  although the engine was built from source.
- With a harness in the loop, the same server performed much worse until the harness stopped
  re-injecting a timestamped context block at the start of history: every agent turn then re-read the
  whole prompt (`0 reused`). The standard benchmark uses fresh prompts per case, so it does not show
  that effect — it is reported here because it dominates an interactive session.

## Files

- `results.json` — the machine-readable result, `outputs.jsonl` — every request with raw usage and
  timings, `config.json` — the run configuration, `provenance/` — chat template and corpus manifest.
