# Community benchmark: RX 6800 XT + RX 6600, layer split, 128k context

Measured on 2026-11-01 by @Kompasby. Two RDNA2 cards in one Windows PC: an RX 6800 XT runs layers
0-40 and an RX 6600 runs layers 41-47 at a 131,072-token context. The main limitation: this is a
hand-built, locally installed engine on a platform where AMD multi-GPU is documented as Linux-only,
so treat it as a report on a configuration rather than a supported path.

## Hardware and software

- GPU 0: AMD Radeon RX 6800 XT, 16 GB, gfx1030. GPU 1: AMD Radeon RX 6600, 8 GB, gfx1032.
  Layer split 0-40 / 41-47; PCIe link width not measured.
- CPU: AMD Ryzen 5 7600 (6C/12T); RAM 63.2 GiB; model storage on an NVMe Kingston SFYRD20 (1.8 TB).
- OS: Windows 11 Pro 26300; GPU driver 32.0.21045.16010; ROCm 10.2.0a20260930 (TheRock wheels).
- Strata commit `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`; engine 0.1.55; **source build** for
  `gfx1030;gfx1032` (see [Building](../README.md) note below).
- Background work: the desktop app that normally drives the model was closed for the run; no power
  limits applied.

## Model and configuration

- Model: `qwen3.8-flash-next-iq2_xs`, IQ2_XS, ISTA-DASLab GSQ-RCO files
  `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`, `...-00002-of-00002.gguf`; no custom pack.
- Vision encoder downloaded, not used here: the reader runs on the CPU (`--vision cpu`).
- Context 131,072; `--kv int8`; `--kv-resident 32768`; expert cache and prefill `auto`; MTP on;
  thinking on (protocol default).
- Launch command (API key removed):

```text
strata launch --family qwen --model iq2_xs --data-dir E:\Strata\Models --vision cpu \
  --context 128k --access lan --yes --no-tui
```

The two cards are configured with `"gpu_roles": {"0": "main", "1": "off", "2": "helper"}` in the run
config, which turns into `layers 0-40 (CUDA0), 41-47 (CUDA1), one hand-off per window`.

## Method

`strata bench run --suite standard-v1` (protocol `strata-qwen38-thinking-v1-draft.1`), one warm-up and
three repetitions per case, 512-token output cap, one request at a time over loopback HTTP, model
hashing before the run. Each case uses a fresh prompt, so `cached_tokens` is 0 everywhere and the
prompt figures are cold prefill. TTFT is measured from request start to the first content delta;
totals include prompt processing.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short (~80 tok) | 77 | 0 | 326-512 | 3 | not measured (fixed cost dominates) | 41.4 (38.2-42.8) | 2.02 (1.78-2.03) |
| prompt-4k | 4,093 | 0 | 512 | 3 | 246.4 (242.2-247.1) | 40.9 (37.4-41.3) | 16.66 (16.59-16.94) |
| prompt-16k | 16,387 | 0 | 512 | 3 | 255.9 (249.0-262.1) | 36.5 (35.9-40.8) | 64.09 (62.58-65.84) |

Capped runs (reasoning plus answer reached the 512-token cap): 1/3 on short, 3/3 on prompt-4k, 3/3 on
prompt-16k. No request failed or timed out.

On this machine prompt processing dominates wall time: the 16k case spends 64 s before the first
token and about 14 s generating.

## Correctness and limitations

- 12 of 12 sanity checks passed (instruction 4/4, retrieval 4/4, tool calls 4/4).
- One model, one quantization, one machine. No needle test in this run.
- **Unsupported configuration.** `setup.py` refuses AMD layer splits on Windows ("Linux-only for
  now") and its `AMD_ARCHS` list lacks `gfx1032`, so the RX 6600 is reported as unsupported. Both
  guards were patched locally; the engine itself splits without complaint.
- The engine is unsigned and was installed by hand, so `strata components verify` will not treat it
  as a release component.
- With a harness in the loop, the same server performed much worse until the harness stopped
  re-injecting a timestamped context block at the start of history: every agent turn then re-read the
  whole prompt (`0 reused`). The standard benchmark uses fresh prompts per case, so it does not show
  that effect — it is reported here because it dominates an interactive session.

## Files

- `results.json` — the machine-readable result, `outputs.jsonl` — every request with raw usage and
  timings, `config.json` — the run configuration, `provenance/` — chat template and corpus manifest.
