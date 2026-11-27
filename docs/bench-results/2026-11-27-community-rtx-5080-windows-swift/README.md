# Community benchmark on RTX 5080 16 GB (Windows, CUDA 13) — Swift 1.5 IQ3_XXS

Measured on 2026-11-27 by Martin Gmuca (GitHub: [@MartinGmc](https://github.com/MartinGmc)).
Two things were measured: the standard suite on **Swift 1.5 IQ3_XXS** against the base
**Qwen3.8-Flash-Next IQ3_XXS** on the same machine, and a **three-model .NET
code-generation comparison**. The main limitation is scale: one machine, one engine
build, three repetitions.

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

- **Swift 1.5** `swift-1.5-iq3_xxs`, **IQ3_XXS**, 262 144-token context, vision reader
  on the CPU. Two GGUF shards; `results.json` carries their SHA-256.
- Same settings as the base model's report: `--kv q4_0`, `--kv-resident 32768`,
  `--conversation-cache-mib 4096`, `--vram-reserve-mib 1536`, prefill auto,
  expert cache auto, DFlash2 drafting (`--spec 6 --draft-conf 0.7`), MTP fallback off.

```text
strata.exe --serve --pack <PACK> --native <SWIFT-GGUF1> --ple-gguf <SWIFT-GGUF2> ^
  --expert-profile ... --expert-cache auto --prefill auto --max-context 262144 ^
  --kv q4_0 --kv-resident 32768 --vision --conversation-cache-mib 4096 ^
  --vram-reserve-mib 1536 --draft dflash2 --draft-fallback none ^
  --mtp-draft-vocab <...>\draft_vocab.bin --spec 6 --draft-conf 0.7
```

## Method

**Standard suite.** `strata bench run --profile launch-swift-1-5-256k --no-start`
against the already-running server, one run on one engine start, status `complete`.
Because it is started from the profile, `results.json` carries the full engine
configuration. Before the run I sent **four 20k-token requests of unrelated text**:
on this card the expert cache of a 76 GB model is RAM-tiered, and a run started right
after the engine loads measures 16–21 tok/s instead of ~100. The warm-up text never
touches the suite's prompts, so the run stays `complete`.

**The .NET eval** takes the model out of the loop: it receives a task statement plus
the starting files and must return complete files; the harness builds them with
`dotnet build` and runs **hidden tests**. No NuGet, so it runs offline. Four tasks:

| Task | Tests | What it covers |
|---|---:|---|
| T1 | 5 | async/await, timeout, cancellation, exception propagation |
| T2 | 9 | nullable reference types under `-warnaserror`, input validation |
| T3 | 9 | ASP.NET Core minimal API, HTTP status codes, JSON binding |
| T4 | 10 | a runnable CLI app: Open-Meteo air quality, `--file`/`--hours`/`--json`, exit codes |

Each task runs **3 times**. T4 allows **up to 10 repair rounds**: after a failing
build or a failing test the model gets that output and may resubmit. The headline
metric is `rounds_to_green` — 0 means it was right on the first try, a number means
how many corrections it needed. Time and tokens below are measured for the whole
task: generation, every repair, the builds and the test runs.

## Results

### Standard suite: Swift 1.5 vs base (same machine, same settings)

| Case | Base IQ3_XXS | Swift 1.5 IQ3_XXS | Difference |
| --- | --- | --- | --- |
| prompt-4k prefill tok/s | 2806.7 (2805.0–2821.8) | **3537.0 (3486.7–3567.5)** | **+26.0%** |
| prompt-16k prefill tok/s | 3791.1 (3771.7–3801.8) | **4137.0 (4118.4–4144.5)** | **+9.1%** |
| prompt-4k decode tok/s | 100.6 (100.0–116.4) | **131.6 (112.7–144.1)** | **+30.8%** |
| prompt-16k decode tok/s | 103.9 (97.1–119.7) | **125.4 (124.2–132.1)** | **+20.7%** |
| prompt-4k TTFT s | 1.50 (1.48–1.52) | **1.19 (1.19–1.20)** | **−20.7%** |
| prompt-16k TTFT s | 4.36 (4.36–4.38) | **4.00 (3.98–4.01)** | **−8.3%** |
| prompt-4k total s | 6.6 (5.9–6.6) | **5.1 (4.7–5.7)** | −22.7% |
| prompt-16k total s | 9.3 (8.6–9.6) | **8.1 (7.9–8.1)** | −12.9% |
| sanity checks | 11/12 (tool call) | **12/12** | — |
| run duration | 99.8 s | **75.2 s** | −24.6% |

Prompt and decode medians are the suite's own engine timings; TTFT is the client's
first-token time. The prefill and TTFT ranges of the two runs do not overlap, so the
difference is larger than the run-to-run spread; the decode ranges overlap at the
edges and should be read as directional.

### Three-model .NET eval (3 repetitions, up to 10 repair rounds)

| Task | Base IQ3_XXS | IQ3_S | Swift 1.5 |
| --- | --- | --- | --- |
| T1 async | 4/5, 4/5, 4/5 — 20.8 s, 821 tok | 4/5, **5/5**, 4/5 — 24.2 s, 1198 tok | 4/5 ×3 — 19.4 s, 1177 tok |
| T2 nullable | 9/9 ×3 — 13.9 s, 516 tok | 9/9 ×3 — 13.8 s, 438 tok | 9/9 ×3 — **11.8 s**, 525 tok |
| T3 minimal API | 9/9 ×3 — 18.2 s, 1193 tok | **2/9 ×3** — 19.4 s, 1229 tok | 9/9 ×3 — **16.9 s**, 1455 tok |
| T4 app | 10/10 ×3 — 118.6 s, 15 520 tok | 10/10 ×3 — **139.1 s**, 14 676 tok | 10/10 ×3 — **100.2 s**, 14 602 tok |
| T4 `rounds_to_green` | 2, 1, 2 | 3, 0, 1 | **0, 1, 2** |
| green / fully green tasks | 9/12 · 3/4 | **7/12 · 2/4** | 9/12 · 3/4 |
| eval wall time, all tasks | 171.5 s | 196.5 s | **148.3 s** |

1. **Swift finishes a task fastest end-to-end**: 100.2 s on the app task against
   118.6 s (base) and 139.1 s (IQ3_S), at essentially the same token count
   (14.6 k vs 15.5 k). The gain is engine speed, not less work.
2. **IQ3_S is the only configuration with a regression**: T3 scores 2/9 in all three
   repetitions while both other models score 9/9. The code compiles, but `POST /todos`
   returns 500 and every endpoint that depends on a stored todo fails; the two tests
   that pass are `GET /todos` on an empty list and `GET /todos/999`. More bits did not
   help this task.
3. **T1 is a blind spot for all three models.** Every repetition fails the same check,
   `cancelled token -> OperationCanceledException`, and no model fixed it even with
   10 rounds and the exact test output in hand.

Both generated apps are in `results/dotnet-eval-apps/` — the passing one and the
failing one, as they came out of the harness.

## Correctness and limitations

- **This is model behaviour, not an engine comparison** for the eval part. The engine
  is identical in all three runs; only the weights differ.
- **One machine, one engine build, three repetitions.** Nine runs per task per model.
  A difference of one test is not a finding; T3's 2/9 vs 9/9 and T4's round counts are.
- **T1 was not given repair rounds in the first report** (#7322, measured with a single
  repair). This report measures all three models with up to 10 rounds and reports
  `rounds_to_green`, so its numbers are not directly comparable with that earlier run.
- **Perplexity not measured**: this engine exposes no token logprobs.
- **Local paths in the artifacts are placeholders** (`<home>`, `<tmp>`, `<strata-home>`),
  the same way the standard runner sanitizes its own output.
- The `.NET eval` and its four tasks are model-agnostic; `harness/` has the runner, the
  tasks and the reference solutions, and `-SelfTest` proves that a reference solution
  passes all tests while the starting template fails them.
