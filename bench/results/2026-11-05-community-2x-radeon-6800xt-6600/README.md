# Community benchmark: RX 6800 XT + RX 6600, layer split, 128k context

Measured on 2026-11-05 by @Kompasby. Two RDNA2 cards in one Windows PC: an RX 6800 XT runs layers
0-40 and an RX 6600 runs layers 41-47 at a 131,072-token context. The main limitation: this is a
hand-built, locally installed engine on a platform where AMD multi-GPU is documented as Linux-only,
so treat it as a report on a configuration rather than a supported path.

## Hardware and software

- GPU 0: AMD Radeon RX 6800 XT, 16 GB, gfx1030. GPU 1: AMD Radeon RX 6600, 8 GB, gfx1032 (x4 slot).
  Layer split 0-40 / 41-47, chosen by the engine's own split search; PCIe link not measured.
- CPU: AMD Ryzen 7 9700X (8C/16T); RAM 63.2 GiB at DDR5-5200; model storage on an NVMe Kingston
  SFYRD20 (1.8 TB). The integrated GPU (gfx1036) is disabled in Windows: with it visible the launcher
  picks it as the helper card (it reports ~43 GiB of shared memory) and the split collapses.
- OS: Windows 11 Pro 26300; GPU driver 32.0.21045.16010; ROCm 10.2.0a20260930 (TheRock wheels).
- **Strata commit `ffa80d7` (main, release v0.2.10); engine 0.1.56; source build** for
  `gfx1030;gfx1032`. The benchmark was driven by the 0.2.9 terminal, which is still installed with two
  local patches (see "Limitations").
- Background work: none — the desktop app that normally drives the model was closed for the run; no
  power limits applied.

## Model and configuration

- Model: `qwen3.8-flash-next-iq2_xs`, IQ2_XS, ISTA-DASLab GSQ-RCO files
  `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`, `...-00002-of-00002.gguf`; no custom pack.
- Vision encoder downloaded, not used here: the reader runs on the CPU (`--vision cpu`).
- Context 131,072; `--kv int8`; `--kv-resident 32768`; expert cache and prefill `auto`; MTP on;
  thinking on (protocol default). The 25.6 GiB expert arena ran on **2 MiB large pages**
  (`expert arena: large pages are resident`), which matters: losing them costs 7-11% decode.
- Launch command (API key removed):

```text
strata launch --family qwen --model iq2_xs --data-dir E:\Strata\Models --vision cpu \
  --context 128k --access lan --yes --no-tui
```

The split is produced by the engine's search, not by a fixed hand-off:

```text
layer split auto: CUDA0 36 SMs at 2.06 GHz -> 0.98 ms per layer, 11.44 GiB free
layer split auto: CUDA1 14 SMs at 2.04 GHz -> 2.54 ms per layer, 2.36 GiB free
layer split auto: CUDA1 link 6.9 GB/s -> miss cost x4.10
layer split auto: K=41 - predicted 64.9 ms per decode window; the caches hold
                  9705 of 24576 profiled pairs (~97.2% of the routed mass), best of 48 plans
```

The x4 slot costs the second card bandwidth: its miss cost is weighted x4.10 against CUDA0's x1.00,
which is why the split stops at layer 41 even though that card has VRAM to spare.

## Method

`strata bench run --suite standard-v1` (protocol `strata-qwen38-thinking-v1-draft.1`), one warm-up and
three repetitions per case, 512-token output cap, one request at a time over loopback HTTP, model
hashing before the run. Each case uses a fresh prompt, so `cached_tokens` is 0 everywhere and the
prompt figures are cold prefill. TTFT is measured from request start to the first content delta;
totals include prompt processing.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short (~80 tok) | 77 | 0 | 250-398 | 3 | not measured (fixed cost dominates) | 46.4 (41.0-46.9) | 1.91 (1.91-20.84) |
| prompt-4k | 4,093 | 0 | 512 | 3 | 287.2 (286.1-292.3) | 40.2 (40.1-43.0) | 14.38 (14.28-27.88) |
| prompt-16k | 16,387 | 0 | 512 | 3 | 310.9 (301.6-326.1) | 43.3 (40.8-48.1) | 52.77 (50.31-54.39) |

Capped runs (reasoning plus answer reached the 512-token cap): 0/3 on short, 3/3 on prompt-4k, 3/3 on
prompt-16k. No request failed or timed out; 12 of 12 sanity checks passed.

The wide TTFT ranges on the short and 4k cases come from the first repetition paying for a cold cache;
the medians are the steady state.

## What the CPU does and does not change

The same machine was measured on a Ryzen 5 7600 (6C/12T) two days earlier with the same engine. The
9700X raises the expert-pool workers from 5 to 7 (one per physical core, host on core 0) and moves
decode by single digits either way, while the 16k prompt drops from 73.5 s to 64.5 s end to end. With
two DDR5-5200 channels the CPU-side expert path looks bandwidth-bound rather than core-bound, so more
cores are not the lever here; the expert cache and memory bandwidth are.

## Correctness and limitations

- 12 of 12 sanity checks passed (instruction 4/4, retrieval 4/4, tool calls 4/4). One model, one
  quantization, one machine; no needle test in this run.
- **Unsupported configuration.** `setup.py` refuses AMD layer splits on Windows ("Linux-only for
  now") and its `AMD_ARCHS` list lacks `gfx1032`, so the RX 6600 is reported as unsupported. Both
  guards were patched locally; the engine itself splits without complaint.
- The engine is unsigned and installed by hand. Because the *terminal* is still 0.2.9 while the
  engine is built from 0.2.10's commit, `results.json` records `strata_commit` and `build_kind` from
  the terminal's manifest (`c62d5e3`, `release`); `engine_version` is correct at `0.1.56`. The commit
  the engine was actually built from is `ffa80d7`.
- With a harness in the loop, the same server performed much worse until the harness stopped
  re-injecting a timestamped context block at the start of history: every agent turn then re-read the
  whole prompt (`0 reused`). The standard benchmark uses fresh prompts per case, so it does not show
  that effect — it is reported here because it dominates an interactive session.

## Files

- `results.json` — the machine-readable result, `outputs.jsonl` — every request with raw usage and
  timings, `config.json` — the run configuration, `provenance/` — chat template and corpus manifest.
