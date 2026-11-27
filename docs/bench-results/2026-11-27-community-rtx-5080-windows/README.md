# Community benchmark on RTX 5080 16 GB (Windows, CUDA 13 build)

Measured on 2026-11-27 by Martin Gmuca (GitHub: [@MartinGmc](https://github.com/MartinGmc)). Two things were measured:
a **quantization A/B** (IQ3_S vs IQ3_XXS, quality and throughput at 5k–248k prompt
tokens) and a **.NET code-generation eval**. The main limitation is scale: one
machine, one engine build, three rounds per model.

## Hardware and software

- **GPU**: NVIDIA GeForce RTX 5080, 16 303 MiB VRAM (17 094 934 528 bytes), single
  card, **PCIe 5.0 ×16** (the link reports Gen1 while idle); power limit 360 W
- **CPU**: AMD Ryzen 7 9800X3D (8 cores); **RAM**: 61.4 GiB (65 940 058 112 bytes);
  **storage**: Samsung SSD 9100 PRO 2 TB, NVMe; 340 GiB free at measurement time
- **OS**: Windows 10.0.26300; **driver**: 621.14; **CUDA**: 13.5
- **Strata**: `0.1.62-windows-x86_64-cuda-cuda13`, prebuilt release binary,
  commit `64524fb9fc82222ddce554c0643b64f056ec8366` (control 0.2.16)
- **Background load**: nothing else on the GPU during the runs; the power limit was
  not changed (360 W stock). SM clock was 2910 MHz under load; no clock log is kept.

## Model and configuration

- Models: `qwen3.8-flash-next` in **IQ3_S** (77.88 GiB, 2 GGUF shards) and
  **IQ3_XXS** (70.63 GiB, 2 shards); expert packs `packs\iq3_s` / `packs\iq3_xxs`
  (1476 MiB each). The second shard is byte-identical between the two quants
  (26.82 GiB), so the real difference is ~12% of the bits.
- Vision reader on the CPU; no custom profile; no calibration.
- Context **262 144** (both), `--kv q4_0`, `--kv-resident 32768`,
  `--conversation-cache-mib 4096`, `--vram-reserve-mib 1536`, prefill auto,
  expert cache auto, drafting DFlash2 (`--spec 6 --draft-conf 0.7`), MTP fallback off.

```text
strata.exe --serve --pack <PACK> --native <GGUF1> --ple-gguf <GGUF2> ^
  --expert-profile ... --expert-cache auto --prefill auto --max-context 262144 ^
  --kv q4_0 --kv-resident 32768 --vision --conversation-cache-mib 4096 ^
  --vram-reserve-mib 1536 --draft dflash2 ^
  --draft-model <data-dir>\dflash2\qwen3.8-flash-next-dflash2.q4_0-selq4.pack ^
  --draft-fallback none --mtp-draft-vocab <data-dir>\mtp\rt\draft_vocab.bin ^
  --spec 6 --draft-conf 0.7
```

Identical command line for both packs; only `<PACK>` changes. The config hash used
for pairing is computed over the **sorted** flag set, because the launcher emits
`--vision` and `--kv-resident` in a different order per model.

## Method

- **Paired A/B/A**: IQ3_S (round A1) → IQ3_XXS (B1) → IQ3_S again (A2), each round
  from a fresh engine start. A1↔A2 gives the noise floor; any difference smaller
  than that is noise.
- **Prompt reuse is forced to zero.** Every document gets a per-paragraph salt
  (`[C32k-17]`) and each round asserts `reused == 0` in the engine log. Without the
  salt the conversation cache returned 5108 of 5115 tokens and "prefill" measured
  7 tokens at 23.7 tok/s — a physically impossible number that is easy to publish
  by accident.
- **Quality** (pre-registered primary metric): arithmetic items, 40 items × 10
  repetitions with a different draw each repetition (400 items per model),
  `temperature 0`. Aggregation is task-balanced (each task equal weight) with a
  bootstrap CI, because three ceiling tasks otherwise drown the signal.
- **Speed**: one request per prompt length; documents are generated deterministically
  from a seed; `max_tokens` for the C layer is 4, so **decode figures below are
  indicative only** — they are not a decode benchmark.
- **.NET eval**: model receives a task statement plus the starting files and must
  return complete files; the harness builds them with `dotnet build` and runs hidden
  tests. 3 repetitions per task; the app task allows one repair round after a
  compiler/test error.

Scripts and fixtures: `harness/` and `results/` in this folder.

## Results

### Throughput (fresh prompts; decode indicative, see Method)

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| IQ3_S (A1) | 5 356 | 0 | 4 | 1 | 2118 (—) | 47.8 (—) |
| IQ3_S (A1) | 32 785 | 0 | 4 | 1 | 3015 (—) | 51.3 (—) |
| IQ3_S (A1) | 131 692 | 0 | 4 | 1 | 3232 (—) | 54.2 (—) |
| IQ3_S (A1) | 248 482 | 0 | 4 | 1 | 3157 (—) | 69.8 (—) |
| IQ3_XXS (B1) | 5 356 | 0 | 4 | 1 | 3196 (—) | 70.7 (—) |
| IQ3_XXS (B1) | 32 785 | 0 | 4 | 1 | 3615 (—) | 81.9 (—) |
| IQ3_XXS (B1) | 131 692 | 0 | 4 | 1 | 3630 (—) | 86.6 (—) |
| IQ3_XXS (B1) | 248 482 | 0 | 4 | 1 | 3455 (—) | 79.9 (—) |

Per-prompt wall time at 248k tokens: **72.1 s** (IQ3_XXS) vs **78.9 s** (IQ3_S).

### Standard suite (`strata bench run --suite standard-v1`)

Started from the saved profile (`--profile launch-qwen-256k --no-start`), status
**`complete`** (21/21 required requests, 0 failed, 0 skipped, 99.8 s after the model
loaded). Because it is started from the profile, `results.json` carries the full
engine configuration (KV type and residency, expert cache, prefill, drafting,
sampling, chat template hash). Artifacts in `standard-v1/`.

| Case | Prompt tok/s median (min–max) | Decode tok/s median (min–max) | TTFT s median (min–max) | Total s | Capped |
| --- | --- | --- | --- | ---: | --- |
| short (~80 tokens) | not shown by the suite | 120.0 (105.7–129.5) n=3 | 0.39 (0.33–0.41) | 4.6 (3.1–5.2) | 1/3 |
| prompt-4k | 2806.7 (2805.0–2821.8) n=3 | 100.6 (100.0–116.4) n=3 | 1.50 (1.48–1.52) n=3 | 6.6 (5.9–6.6) | 3/3 |
| prompt-16k | 3791.1 (3771.7–3801.8) n=3 | 103.9 (97.1–119.7) n=3 | 4.36 (4.36–4.38) n=3 | 9.3 (8.6–9.6) | 3/3 |

**This machine needs a warm-up before the suite.** A run started right after the
engine loaded measured **16–21 tok/s decode** instead of ~104. Four 20k-token
requests with unrelated text restore it (71.5 → 106.6 → 103.7 → 112.8 tok/s), and
the suite then reproduces the numbers above. The expert cache of a 76 GB model on a
16 GB card is RAM-tiered; after a restart it is empty, and the suite's own single
warm-up request (≤128 tokens) is not enough to fill it. The warm-up text is
unrelated to the suite's prompts, so the run stays `complete`.

**Reconciling the two decode sets.** The C-layer decode figures above (70–87 tok/s)
come from requests that generated **4 tokens**, so they include the fixed cost of a
request and are not a decode measurement. The suite's numbers (100.6–120.0 tok/s)
come from 512-token caps and are the authoritative decode figures for this machine.
The C-layer table stays in this report because its *prompt* rates are measured on
long prompts (5k–248k) that the standard suite does not cover.

Sanity checks: **11 of 12** (instruction 4/4, retrieval 4/4, **tool calls 3/4**;
the miss was `T2: expected one structured tool call, got 0`). Across three runs of
the suite on this machine the tool-call check failed **twice, on a different test
each time** (T4, then T2), and passed 12/12 once — so it is intermittent, not a
stable property of the model.

### Quality A/B (arithmetic, 400 items per model)

| | IQ3_S | IQ3_XXS | Difference |
| --- | ---: | ---: | --- |
| accuracy | 86.25% (690/800) | 83.75% (335/400) | **+2.5 pp, 95% CI [−1.8 .. +6.9]** |
| per-round | 86.3 / 86.8 (A1 / A2) | 83.8 (B1) | noise floor A1↔A2 = **1.0 pp** |
| recall with decoys | 0 errors / 10 | 0 errors / 10 | ceiling |
| table → JSON | 6000/6000 fields | 3000/3000 | ceiling |
| bug hunt in code | 5/5 ×10 | 5/5 ×10 | ceiling |

**Conclusion: a quality advantage for IQ3_S is not demonstrated.** The CI includes 0
and the point estimate is below the pre-registered practical threshold (+5 pp).
The cost is clear: −30.6% decode.

### .NET code generation (4 tasks, 3 repetitions, hidden tests)

| Model | green runs | fully green tasks | app task (10 tests) |
| --- | ---: | ---: | --- |
| qwen3.8-flash-next IQ3_XXS | 6/12 | 2/4 | **2/10, 0/0, 2/10** |
| Swift 1.5 IQ3_XXS (separate run, same tasks) | 9/12 | 3/4 | **10/10 ×3, first try** |

The IQ3_XXS failures were reproducible: twice it did not consume the value of
`--file` and exited 2 ("unknown argument '<path>'"), once it did not compile
(`CS0165`). The repair round did not fix either. Both generated apps are in
`results/dotnet-eval-apps/`.

## Correctness and limitations

- **Perplexity not measured**: the engine exposes no token logprobs
  (`/openapi.json`, `/docs`, `/api/tags` → 404), so the logprob-based check could
  not run. Everything here is behavioural.
- **Fields the operator must supply** are still `missing` in `results.json`:
  `model/revision`, `source_revision`, `tokenizer_revision`, `custom_pack`,
  `build_options` and `background_workloads`. None can be read from a running
  server; the first four need the quantization repository's immutable revision.
- **The structured tool-call check is intermittent**: it failed in 2 of 3 suite runs
  on this machine, on a **different test each time** (`T4`, then `T2`), and one run
  passed 12/12. Treat it as flaky, not as a model defect.
- **The profile-started repeat run is `nonconforming` on purpose, and that is
  informative**: it re-sent the same prompts to an engine that had already answered
  them, and the runner flagged *"unexpected reuse of the user payload"* on all six
  prompts while prefill collapsed from ~2 843 to ~22 tok/s. The same trap that
  falsified the prefill numbers in the Method section is detected automatically by
  the protocol. That run is kept in `standard-v1-nonconforming-reuse/` as evidence,
  not as a performance result.
- **Paths in the artifacts are placeholders.** Local paths in `results/*.json` are
  replaced (`<home>`, `<tmp>`, `<strata-home>`, `<data-dir>`), the same way the
  standard runner sanitizes its own output. The reproducible command line is in
  *Model and configuration* above.
- **The three ceiling tasks do not discriminate** (B2/B3/B4 at 100% for both
  models). Only the arithmetic task and the .NET app task separate the models.
- **Decode numbers are indicative** (4 generated tokens per request).
- **Scale**: one machine, one engine build, three rounds; the quality CI is wide
  enough that "no difference demonstrated" is the honest wording, not "equal".
- **Not tested**: multi-GPU, other quants, other context sizes, concurrency > 1,
  power-limited operation.
- The `.NET eval` and the quality harness are model-agnostic and can be re-run
  against any pack; see `harness/README.md`.
