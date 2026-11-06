# Community benchmark: RX 6800 XT + RX 6600 as an expert-cache helper, 128k context (v0.2.11)

Measured on 2026-11-06 by @Kompasby, on release v0.2.11 (engine 0.1.57). This is a follow-up to the
2026-11-05 report on the same machine, where the RX 6600 was a layer-split stage (layers 38-47). Here
the 6800 XT runs the whole model and the 6600 holds the part of the expert cache that does not fit on
the primary. Every number below is set against a control run of the layer split on the same day,
same engine, minutes apart, so the two tables differ only in how the second card is used.

## Hardware and software

- GPU 0: AMD Radeon RX 6800 XT, 16 GB, gfx1030. GPU 1: AMD Radeon RX 6600, 8 GB, gfx1032, in an x4
  slot. The engine's own host-to-device probe reads 28.2 GB/s. Windows budgets 7,378 of the card's
  8,176 MiB to the process.
- CPU: AMD Ryzen 7 9700X (8C/16T); RAM 61.7 GiB as the runner reports it, at DDR5-5200; model files on
  an NVMe Kingston SFYRD20. The integrated GPU (gfx1036) is disabled in Windows: visible, the launcher
  gives it the helper role and the run collapses.
- OS: Windows 11, build 26300; GPU driver 32.0.21045.16010; ROCm 10.2.0a20260930 (TheRock wheels).
- **Strata v0.2.11; terminal 0.2.11 (control 0.1.57); engine 0.1.57; source build** for
  `gfx1030;gfx1032;gfx1036` — the signed package still carries no `gfx1032`. The same local patches as
  in the 2026-11-05 report apply.
- Background work: none.

## Model and configuration

- Model: `qwen3.8-flash-next-iq3_xxs`, IQ3_XXS, ISTA-DASLab GSQ-RCO files, 43.81 + 26.82 GiB.
- Context 131,072; `--kv int8`; `--kv-resident 32768`; expert cache and prefill `auto`; the DFlash2
  drafter with `--spec 6`; vision reader on the CPU.
- The shape is the point. A config with one entry in `"gpu"` makes the server write
  `HIP_VISIBLE_DEVICES=0`, which hides the 6600, so the second card is kept visible through the config's
  `env` block and the helper is requested by hand:

```json
{
  "gpu": 0,
  "env": { "HIP_VISIBLE_DEVICES": "0,1", "STRATA_HIP_PROMPT_F16": "1" },
  "user_args": ["--expert-cache-device1", "auto"]
}
```

- `STRATA_HIP_PROMPT_F16=1` is the same setting the launcher uses on RDNA2 by default; it is repeated
  here because the run bypasses the launcher's choice list.
- Effective command (paths shortened, API key absent):

```text
strata.exe --serve --pack ...\iq3_xxs --native ...IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf ...IQ3_XXS-00002-of-00002.gguf --expert-profile ...\expert-profile.bin \
  --expert-cache auto --prefill auto --max-context 131072 --kv int8 --kv-resident 32768 \
  --vision --expert-cache-device1 auto --draft dflash2 --draft-fallback none --spec 6 \
  --draft-conf 0.35
```

No `--layer-split` and no `--remote-expert-opt` (the latter switches `--pipeline-windows` off and would
confound the comparison).

## Method

`strata bench run --suite standard-v1` (protocol `strata-qwen38-thinking-v1-draft.1`), one warm-up and
three repetitions per case, 512-token output cap, one request at a time over loopback HTTP. Same corpus,
same client, same sampling parameters as the control run; the server had answered one request before the
suite started. TTFT is the client's monotonic clock from dispatch to the first nonempty delta.

The control run is `2026-11-06-community-6800xt-6600-layer-split-control/`: the same machine, stopped
(`strata launch --stop`), restarted with the stored split preset (`--layer-split auto`, found by the
engine's own search as 0-37 / 38-47), and benchmarked at 19:26Z, a minute after this run finished
(run `20261106T192002Z-db7dbaa7`). Both runs loaded from a warm page cache (5 s and 35 s to ready), so
the load time is not part of the comparison.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 52.6 (44.4-52.6) n=3 | 1.48 (1.36-1.50) n=3 | 3.14 (2.84-3.38) n=3 | 9.4 (7.3-9.6) n=3 | 0/3 |
| prompt-4k | 631.2 (629.0-651.8) n=3 | 43.5 (43.4-44.2) n=3 | 6.52 (6.33-6.58) n=3 | 15.80 (14.56-17.05) n=2 | 18.1 (18.1-18.3) n=3 | 3/3 |
| prompt-16k | 772.4 (770.1-786.9) n=3 | 38.7 (38.5-48.0) n=3 | 21.28 (20.88-21.34) n=3 | 25.50 (n=1) | 34.1 (31.9-34.5) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens, so its prompt rate is fixed
cost and is not shown. Capped requests reached the 512-token cap (reasoning plus answer) and therefore
have no first-answer time.

### With `--remote-expert-opt`

Adding the flag to this shape does not pay on this machine: the prompt is unchanged (629.3 and 762.2
tok/s against 631.2 and 772.4), and decode drops to 38.4 / 35.8 / 34.7 tok/s (short / 4K / 16K) from
52.6 / 43.5 / 38.7, because the flag switches `--pipeline-windows` off and the 6600 has to sum expert
rows over a x4 link. The primary cache also shrinks slightly (3,940 slots against 3,983).
([round 2 report](../2026-11-06-community-6800xt-6600-helper-cache-remote-opt/README.md))

## Against the layer split on the same cards, same day

| Metric | Layer split (control) | 6600 as helper cache | Change |
| --- | ---: | ---: | ---: |
| prompt-4k, tok/s | 375.5 | 631.2 | +68% |
| prompt-16k, tok/s | 640.4 | 772.4 | +21% |
| decode short, tok/s | 42.8 | 52.6 | +23% |
| decode-4k, tok/s | 37.4 | 43.5 | +16% |
| decode-16k, tok/s | 38.6 | 38.7 | ±0 |
| TTFT-4k, s | 10.95 | 6.52 | -40% |
| TTFT-16k, s | 25.64 | 21.28 | -17% |
| total-4k, s | 24.6 | 18.1 | -26% |
| total-16k, s | 38.7 | 34.1 | -12% |

The 05.11 run of the split on this machine (361.4 / 578.4 tok/s, 40.0 tok/s short decode) lands within
4-11% of the control, so the gap is the configuration and not run-to-run noise.

## Why it is faster (engine evidence)

Both runs start from the same expert profile, so the difference is what the engine can keep and how big
a prompt chunk it can afford:

```text
strata generate: GPU 0: AMD Radeon RX 6800 XT (gfx1030)
strata generate: expert cache 3983 slots, 6.49 GiB of VRAM; policy is PROFILE, ranked by routing frequency
strata generate: CUDA1 context ready, 7.07 GiB free before expert arena registration
strata generate: CUDA1: 4097 additional experts, 6.57 GiB; results return through pinned host rows
strata serve: prompt chunk auto: 6144 tokens, a 96-slot ring
strata serve: the prompt path borrows 1703 CUDA0 cache slots (2.75 GiB)
```

- The split keeps 5,227 slots (8.21 GiB) on CUDA0, spends 5.66 GiB of the 6600 on layers 38-47 and the
  head, and has 2.23 GiB left for a 1,229-slot cache there; the prompt chunk falls to 3,840 tokens by
  the second card's geometry, and the engine says why:

  ```text
  WARNING: prompt chunk 3840 tokens, not 6144: CUDA1 (AMD Radeon RX 6600) has 1229 expert-cache slots,
  and a 6144-token chunk borrows 1229 of them (it must keep 128, and lend at most 85%) - prompts read
  slower than on CUDA0 alone (#448)
  ```

  The helper shape has no second stage to satisfy: 3,983 slots on the 6800 XT, 4,097 experts (6.57 GiB)
  on the 6600, and a 6,144-token chunk.
- Decode hit rate on the primary rises as the run warms: 51.4% on the first request, then 71.6-84.7%
  with 8-32% of routed experts fetched over PCIe or from CUDA1. Per request the 6600 returns between
  1,886 and 17,162 expert entries, up to 212 MiB, roughly 9% of routed work in steady state.
- The x4 link never becomes the limit: the helper returns rows through pinned host memory at 28.2 GB/s
  for a probe, and the engine waits 5-36 ms per request to stage it.

## Correctness and limitations

- 12 of 12 sanity checks passed (instruction 4/4, retrieval 4/4, tool calls 4/4), as in the control run.
  No needle test in this run.
- The short case differs in cap: 0/3 capped here, 1/3 capped on the split, so its decode numbers compare
  responses of different lengths. Treat +23% as indicative, not exact.
- Decode at 16K is flat: the 6600 helps the prompt path and short-context decode, and stops mattering
  once the KV stream dominates. The win is a prefill and TTFT win.
- **Large pages were refused in all three 06.11 runs** (`virtual alloc error 1450`, 4 KB pages), while the
  2026-11-05 report ran its arena on 2 MiB pages. That is why the control run is from the same day: the
  05.11 numbers are not a clean baseline for either shape on this machine right now.
- **Unsupported configuration**, unchanged from the 2026-11-05 report: `setup.py` refuses AMD layer
  splits on Windows and does not list `gfx1032`; the engine is unsigned and installed by hand, which
  `results.json` records as `build_kind: release`.
- The helper shape is not what `strata launch` selects: the launcher rebuilds its config from stored
  choices at every boot and always picks the split when two AMD cards are visible. This run was started
  from a saved runtime profile carrying the config above.

## Files

- `results.json` — the machine-readable result; `outputs.jsonl` — every request with raw usage, timings
  and client event offsets; `config.json` — the run configuration; `provenance/` — chat template and
  corpus manifest.
- `../2026-11-06-community-6800xt-6600-layer-split-control/` — the control run.
