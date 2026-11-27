# Community benchmark on RTX 5080 16 GB (Windows, CUDA 13) — refusal-direction control vector on IQ3_XXS

Measured on 2026-11-27 by Martin Gmuca (GitHub: [@MartinGmc](https://github.com/MartinGmc)).
One thing was measured, and it is deliberately narrow: **the same weights with and
without one control vector**. Nothing else differs between the two runs — same pack,
same KV, same context, same drafting, same engine build. The main limitation is that
this is a single machine and three repetitions.

## What the vector is

The package ships `Qwen3.8-Flash-Next-experimental-speed-projection.gguf` (480 KB) and
its own `README.md`. Quoting that document, because the name is misleading:

> The package's own documentation describes the vector as a **refusal-direction
> projection**: the model declines far fewer requests (it reports 1 of 50 vs 50 of 50
> on its test set), and removing refusals removes a safety behaviour — you are
> responsible for what the model writes with it on. It also shifts the model's
> predictions slightly on ordinary text (…) It is not an optimization in the engine:
> on the same text it costs 0.2–0.4% per token.

The configuration it was selected with (`…speed-projection.json`) reports
`refusal_harmless_thinkoff: 4.0`, `refusal_harmful_thinkoff: 2.0`, `kl_mean: 0.186`,
`kl_p99: 3.14`, `layer_range: [4, 44]`, `mode: project`, `dir: per-layer`.

**This report does not measure refusal rates.** Its tasks are ordinary programming
tasks that the aligned model answers as well, so the point is the opposite one:
**does removing the direction cost capability?**

## Hardware and software

- **GPU**: NVIDIA GeForce RTX 5080, 16 303 MiB VRAM (17 094 934 528 bytes), single
  card, PCIe 5.0 ×16 (the link reports Gen1 while idle); power limit 360 W
- **CPU**: AMD Ryzen 7 9800X3D (8 cores); **RAM**: 61.4 GiB (65 940 058 112 bytes);
  **storage**: Samsung SSD 9100 PRO 2 TB, NVMe
- **OS**: Windows 10.0.26300; **driver**: 621.14; **CUDA**: 13.5
- **Strata**: `0.1.62-windows-x86_64-cuda-cuda13`, prebuilt release binary,
  commit `64524fb9fc82222ddce554c0643b64f056ec8366` (control 0.2.16)
- **Background load**: nothing else on the GPU; power limit unchanged (360 W stock)

## Model and configuration

- **Qwen3.8-Flash-Next IQ3_XXS** in both runs, 262 144-token context, vision reader on
  the CPU. The two runs use the **same pack and the same two GGUF shards**;
  `results.json` carries their SHA-256.
- Only the four control-vector flags differ:

```text
--control-vector-scaled <...>\Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 ^
  --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

Everything else is identical in both command lines: `--expert-cache auto`,
`--prefill auto`, `--max-context 262144`, `--kv q4_0`, `--kv-resident 32768`,
`--vision`, `--conversation-cache-mib 4096`, `--vram-reserve-mib 1536`,
`--draft dflash2`, `--draft-fallback none`, `--spec 6`, `--draft-conf 0.7`.

The paired design was verified from the running command lines, not from the preset
files: the launcher rewrites presets, so the preset file is not evidence. Both
configurations were started from saved profiles (`launch-qwen-256k` and
`launch-qwen-256k-esp`) with `--no-start`, so `results.json` carries the settings and
neither run started or restarted the engine.

## Method

**Standard suite.** One `complete` run per configuration, one run per engine start,
`score >= 2.20` in the suite's own units. Before each run I sent **four 20k-token
requests of unrelated text**: on this card the expert cache of a 76 GB model is
RAM-tiered and a cold engine measures 16–21 tok/s instead of ~100. The warm-up text
never touches the suite's prompts.

**The .NET eval** is behavioural: the model receives a task statement plus the starting
files, returns complete files, and the harness builds them with `dotnet build` and runs
hidden tests. Four tasks, **3 repetitions each**, **up to 10 repair rounds** per
repetition with the compiler or test output as feedback. `rounds_to_green` is the
headline: 0 means right on the first try. Time and tokens cover the whole task —
generation, repairs, builds and test runs.

## Results

### Standard suite

| Case | Base | With the vector | Difference |
| --- | --- | --- | --- |
| prompt-4k prefill tok/s | 2806.7 (2805.0–2821.8) | 2757.9 (2755.3–2768.5) | **−1.7%**, ranges do not overlap |
| prompt-16k prefill tok/s | 3791.1 (3771.7–3801.8) | 3703.9 (3681.6–3708.0) | **−2.3%**, ranges do not overlap |
| prompt-4k decode tok/s | 100.6 (100.0–116.4) | 104.3 (97.4–115.5) | +3.7%, within spread |
| prompt-16k decode tok/s | 103.9 (97.1–119.7) | 102.2 (98.5–106.0) | −1.6%, within spread |
| prompt-4k TTFT s | 1.50 (1.48–1.52) | 1.51 (1.49–1.52) | +0.7% |
| prompt-16k TTFT s | 4.36 (4.36–4.38) | 4.47 (4.47–4.49) | **+2.5%**, ranges do not overlap |
| prompt-4k total s | 6.6 (5.9–6.6) | 6.4 (5.9–6.8) | −3.0%, within spread |
| prompt-16k total s | 9.3 (8.6–9.6) | 9.5 (9.3–9.7) | +2.2%, within spread |
| sanity checks | 11/12 (tool call) | **12/12** | — |
| run duration | 99.8 s | **86.8 s** | −13.0% |

Prefill is consistently about **2 % slower** with the vector, and the ranges of the
two runs do not overlap — this is the cost the package's README describes as
"0.2–0.4 % per token". Decode and total time are inside the run-to-run spread.
The shorter total run duration is not a speed-up of the vector: the suite's wall time
depends on how long the model's own answers run.

### .NET eval

| Task | Base | With the vector |
| --- | --- | --- |
| T1 async | 4/5 ×3 — 20.8 s, 821 tok | 4/5 ×3 — 19.3 s, 1134 tok |
| T2 nullable | 9/9 ×3 — 13.9 s, 516 tok | 9/9 ×3 — 12.3 s, 627 tok |
| T3 minimal API | 9/9 ×3 — 18.2 s, 1193 tok | 9/9 ×3 — 15.8 s, 1218 tok |
| T4 app | 10/10 ×3 — 118.6 s, 15 520 tok | 10/10 ×3 — **142.5 s**, 16 735 tok |
| T4 `rounds_to_green` | 2, 1, 2 | 1, 1, 4 |
| green / fully green tasks | 9/12 · 3/4 | 9/12 · 3/4 |
| eval wall time, all tasks | 171.5 s | 189.9 s |

Quality is unchanged on the four tasks (9/12 both). The app task took **20 % longer
and 8 % more tokens** with the vector, driven by one repetition that needed four
corrections instead of one or two. With three repetitions this is a hint, not a
measurement of a degradation.

## Correctness and limitations

- **Single variable, but not a controlled experiment in the laboratory sense**: the two
  runs happen on different engine sessions, so run-to-run variance applies. The prefill
  and TTFT differences are outside the spread; the decode and total-time differences
  are inside it.
- **Three repetitions** for the eval. `rounds_to_green` of 1, 1, 4 against 2, 1, 2 is
  suggestive; it does not establish that the vector hurts coding.
- **Only one scale** (1.0) and one layer range (4–44), as shipped by the package.
  Intermediate scales are untested.
- **Refusal behaviour is not measured here.** The package's own numbers (1 of 50 vs
  50 of 50) are the reference for that; this report deliberately stays on capability.
- **Perplexity not measured**: this engine exposes no token logprobs.
- **Local paths in the artifacts are placeholders** (`<home>`, `<tmp>`, `<strata-home>`).
- `harness/` contains the runner, the four eval tasks and the reference solutions;
  `-SelfTest` proves that a reference solution passes while the starting template fails.
