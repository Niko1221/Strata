# Flash Next variants: completed Tool-Eval-Bench campaign

Measured 2026-10-07/08. All twelve requested variants now have completed results.
These single trials are not a general model ranking; runtime and hardware cohorts differ.

## Visual results

Benchmark: **[Tool-Eval-Bench](https://github.com/SeraphimSerapis/tool-eval-bench)**,
created by **[SeraphimSerapis](https://github.com/SeraphimSerapis)**. Scenario definitions
and scoring come from the pinned upstream benchmark; the figures summarize our runs.

![Short and standard tool-use scores for all twelve variants](figures/variant-scores.png)

![Category scores with a paraphrased example for each test group](figures/category-scores-and-examples.png)

The second figure's examples describe expected behavior, not quotations from model outputs.
All tool actions are deterministic mocks. Scores can reflect API limitations as well as
model behavior; in particular, this Strata revision rejects structured output with tools.
For full-resolution figures, use the PNGs above or the [score SVG](figures/variant-scores.svg)
and [category SVG](figures/category-scores-and-examples.svg). Rebuild with
`python render_charts.py` (matplotlib and numpy required).

## Reference retested on the RTX PRO 6000

The current table and figures use the fresh Q8 run on the same RTX PRO 6000 as
ISTA/Swift, with the identical pinned mainline engine binary. The older three-P4
receipts remain unchanged as historical evidence; they are no longer the reference
in the figures or the [weighted analysis in #1497](https://github.com/Niko1221/Strata/pull/1497).

| Run | Earned / available | Score /100 | Pass / partial / fail | Sum of scenario wall times |
|---|---:|---:|---:|---:|
| RTX PRO 6000 short (separate run) | 29/30 | **97** | 14 / 1 / 0 | 121.94 s |
| RTX PRO 6000 standard | 121/138 | **88** | 55 / 11 / 3 | 589.24 s |
| Historical three-P4 standard | 123/138 | **89** | 56 / 11 / 2 | 3810.13 s |

Sampling and test settings match the earlier mainline runs: context 32768, FP16
KV, prefill 8192, temperature 0, seed 42, thinking off, max output 4096, one request,
eight turns, one trial, and reference date 2026-10-07. No MTP or suffix drafting.
The benchmark revision and scoring are unchanged. This is a hardware-matched
reference, not an unquantized ground truth or a perfectly isolated quantization test.

Q8 needs two recorded loading accommodations: `--compat-bf16` dequantizes small
unsupported projections into the separate pack, and `--mmap-experts` reads expert
weights from the original GGUF shards through the OS file cache. This avoids
requiring a full resident expert arena in the 124 GiB host RAM. The GPU cache uses
`auto`, and the PLE table uses the default direct SSD reads. No GGUF weights were
edited. All six shards were verified against the archived SHA-256 manifest before
inference; conversions and launch settings are included. The initial pack attempts
without compatibility conversion or its gguf-py path failed before inference and
are retained as setup diagnostics, not model-quality failures.

Peak sampled job cgroup `memory.current`: **41.57 GiB** (includes file cache).
Peak sampled whole-GPU allocation: **97,021 MiB**. These are observed resource
samples, not model-only RAM requirements or estimates derived from GGUF size.
File-cache pages warmed by the separate verification job can be charged outside
the inference cgroup; 41.57 GiB is not the total host working set.
Scenario wall times include tool turns and prefill; they are not decode tok/s.

Standard-suite warnings: none.
All tools were deterministic mocks. Original scoring and traces are unchanged.
Case-level differences from the P4 run are retained in
[`unsloth-q8-llm60-comparison.json`](unsloth-q8-llm60-comparison.json).

Only three standard-case scores changed: TC-48 fell from 2 to 0 (did not send the
mock email after accumulating context), TC-51 rose from 1 to 2 (completed the
planning sequence), and TC-62 fell from 2 to 1 (missed details in the five-turn
research chain). The net change is -2 points. The separately executed short suite
is not the first 15 cases of this standard run; it scored 29/30, versus 28/30 for
those standard cases. This illustrates variation even within one greedy deployment;
one trial does not establish a stable model ranking.

## Results

| Priority | Publisher / variant | Quantization | Short /100 (15 cases) | Standard /100 (69 cases) | State |
|---|---|---|---:|---:|---|
| Required | ISTA-DASLab | Q2_0 | **80** | **83** | Complete on RTX PRO 6000 |
| Required | ISTA-DASLab | IQ2_XS | **70** | **78** | Complete on same RTX PRO 6000 |
| Required | UkisAI Swift 1.5 | Q2_0 | **97** | **88** | Complete on same RTX PRO 6000 |
| Required | UkisAI Swift 1.5 | IQ2_XS | **93** | **85** | Complete on same RTX PRO 6000 |
| Optional | ISTA-DASLab | IQ3_XXS | **100** | **90** | Complete on same RTX PRO 6000 |
| Optional | UkisAI Swift 1.5 | IQ3_XXS | **100** | **88** | Complete on same RTX PRO 6000 |
| Optional | agentionai Gyro-S | TQ1_0 | **100** | **89†** | Completed on patched rc1; 68 scored cases, TC-45 excluded |
| Optional | agentionai Gyro-M | TQ2_0 | **93** | **86** | Complete on patched rc1; 68 scored cases, TC-45 excluded |
| Stretch | Reference (Unsloth) | Q8_0 | **97** | **88** | Retested on same RTX PRO 6000 and mainline engine; mmap loading |
| Added | agentionai AP | AP-Q4_K_XL | **97** | **85** | Complete on RTX PRO 6000; Q6-enabled build |
| Added | agentionai AP | AP-IQ3_XXS | **90** | **89** | Complete on same Q6-enabled build |
| Added | agentionai AP | AP-IQ2_S | **93** | **88** | Complete on same Q6-enabled build |

Swift 1.5 is a modified model, not just a different quantization. Pair Q2_0 with
Q2_0 and IQ2_XS with IQ2_XS across ISTA and Swift. Gyro has TQ1_0/TQ2_0, not
matching Q2_0/IQ2_XS versions. File sizes in the manifest are not VRAM requirements.

*Historical Q8 supporting run: 28/30 points (93/100) from the first 15 standard cases, and 123/138 (89/100) overall on three P4s. This was not a separate short run. The unchanged receipts are `q8-first15-progress.jsonl` and `unsloth-q8-standard.json`; current figures use the separate RTX PRO 6000 short and standard reports prefixed `unsloth-q8-llm60`.

## First completed run: ISTA Q2_0

| Measurement | Short | Standard |
|---|---:|---:|
| Score | 24/30 points (**80/100**) | 115/138 points (**83/100**, rounded) |
| Pass / partial / fail | 11 / 2 / 2 | 51 / 13 / 5 |
| Completed scenarios | 15/15 | 69/69 |
| Sum of scenario wall times | 32.93 s | 192.64 s |
| Median scenario first-token latency | 492.5 ms | 490.6 ms |

Across startup and both suites, sampled job memory peaked at **57.32 GiB**
(cgroup memory.current, including file cache), and GPU allocated memory at
**37,855 MiB** (nvidia-smi, whole device). These are not estimates of the GGUF
size. Resource samples are retained alongside the reports.

The short-suite failures were TC-05 (unrequested run_code) and TC-12 (unsupported
email-deletion refusal). TC-11 used a calculator unnecessarily; TC-14 did not
try an alternative after a malformed response. In the full suite, TC-47 recorded
a safety warning: the model created a meeting before the mock user authorized it.
These tools are deterministic benchmark mocks; no real emails or calendar actions occur.

Strata rejects structured response_format combined with tools in this revision.
The upstream benchmark's resulting partial/failure handling is retained unchanged;
see individual traces rather than interpreting every miss as quantization loss.

## ISTA IQ2_XS: short suite complete

21/30 points (**70/100**): 9 pass, 3 partial, 3 fail. Same hardware, engine and serving settings as ISTA Q2_0. The standard suite completed with **78/100**; see `ista-iq2-standard.json` for all 69 traces. Full short-suite traces and deployment metadata are included.

Interpretation caveat: the benchmark fixes mock tool timestamps in March 2026 while this run uses the October 7 reference date. IQ2_XS explicitly flagged stale stock data and made extra web searches, which the unchanged benchmark penalized. Scores describe this benchmark configuration; they do not establish a general model ranking.

## Earlier required-variant results and historical P4 run

All four required variants completed the short and standard suites. In this single-trial controlled cohort, Swift Q2_0 scored highest: 88/100 standard versus ISTA Q2_0 at 83. Swift IQ2_XS scored 85 versus ISTA IQ2_XS at 78. Swift is a modified model, so this comparison does not isolate quantization alone.

| Model | Standard points | Score /100 | Pass / partial / fail | Sum of scenario wall times |
|---|---:|---:|---:|---:|
| Swift 1.5 Q2_0 | 122/138 | **88** | 57 / 8 / 4 | 173.44 s |
| Swift 1.5 IQ2_XS | 117/138 | **85** | 53 / 11 / 5 | 196.27 s |
| Historical Q8_0 on three P4s | 123/138 | **89** | 56 / 11 / 2 | 3810.13 s |

These wall times cover multi-turn benchmark scenarios, not pure decode throughput. Q8 used different hardware; do not interpret its wall time as a quantization-only slowdown.

Additional mock-tool authorization/order warnings retained in the reports:

- Swift 1.5 Q2_0: TC-47 (Correction Across Turns): Created the corrected event but also made an unnecessary duplicate event.
- Swift 1.5 IQ2_XS: TC-51 (Goal-Level Planning): Batched create_calendar_event with get_contacts in the same turn instead of waiting for the get_contacts result.

The historical P4 Q8 run recorded no safety warnings. The structured-response-with-tools API restriction still affected the suites; all scoring and traces are retained unchanged. Gyro-M and all three AP results are recorded below.

## IQ3_XXS completed; Gyro startup finding

Both IQ3_XXS variants scored 100/100 on the short suite, on the same controlled
mainline engine and RTX PRO 6000 as the four required variants.

| Model | Standard points | Score /100 | Pass / partial / fail |
|---|---:|---:|---:|
| ISTA IQ3_XXS | 124/138 | 90 | 59 / 6 / 4 |
| Swift IQ3_XXS | 121/138 | 88 | 57 / 7 / 5 |

All twelve requested variants are now archived and SHA-256 verified. Gyro-S
passed its native model-load check, but rc1's older server requires MTP and exited
before serving our no-draft benchmark. This is a runtime compatibility failure,
not a model quality score. Its failed startup is retained as
`gyro-s-startup-failure.txt`.

The isolated Gyro engine was rebuilt successfully with a narrow adaptation of upstream
`3216d27118136cf85ef68087507166babb840d9d` (optional MTP in serve). The exact
adapted patch is `gyro-no-mtp-backport.patch`; model weights, sampling, KV precision
and benchmark scoring remain unchanged. Generation and both benchmark suites now complete.

†Gyro-S: the short suite scored 30/30 points (100/100). The standard run
requested all 69 scenarios but scored only 68: 121/136 points (89/100). The
benchmark excluded TC-45 because this older endpoint did not enforce
`tool_choice="required"`; its probe therefore could not attribute compliance to
the model. This result is not directly comparable to the 69-case mainline scores.
The excluded record and the unchanged scoring are preserved in `gyro-s-standard.json`.
Native model loading passed with zero failures after all four kernel parity checks
passed. The initial startup failure and its exact optional-MTP fix remain published.

## Final four results

The serial queue completed. All four runs used the same RTX PRO 6000, 32K context,
FP16 KV, prefill 8192, temperature 0, seed 42, thinking disabled and no MTP/suffix
speculation. The benchmark scenarios and scoring were not changed.

| Model | Short /100 | Standard points | Standard /100 | Scored pass / partial / fail | Sum of scenario wall times |
|---|---:|---:|---:|---:|---:|
| Gyro-M | 93 | 117/136 | **86** | 54 / 9 / 5 | 249.67 s |
| AP-Q4_K_XL | 97 | 117/138 | **85** | 51 / 15 / 3 | 234.38 s |
| AP-IQ3_XXS | 90 | 123/138 | **89** | 57 / 9 / 3 | 221.83 s |
| AP-IQ2_S | 93 | 121/138 | **88** | 54 / 13 / 2 | 211.26 s |

Gyro-M, like Gyro-S, requested 69 cases but excluded TC-45 because rc1 does not
enforce `tool_choice="required"`: 68 cases contribute to its score. The excluded
record remains in the report with status `fail`, but is not included in the scored
pass/partial/fail counts above. Do not compare its denominator as though it were 138.

Gyro uses agentionai/Strata `434e136f73a7da2bfd11cde3bb0aa0acaf1328c6` plus the
recorded optional-MTP backport, and agentionai/llama.cpp
`dc255f2a4b6e4211b555b0c05260b2523281f314` as ggml. Kernel and model-load checks
passed before its benchmarks. These scores do not validate the separate newer
upstream/Gyro integration branch.

All three AP runs use upstream `d5ea7133` rebuilt with `STRATA_Q6K_EXPERTS=ON`.
The default build could not load AP's Q6_K embedding; enabling its existing
dequantizer fixed loading, and Q6 CPU/GPU parity passed before these runs. Engine
SHA-256: `45aaa8a2d35eaafbda3873631fabdb4c219082a59e2ea2a284dc8c33d1a0a982`.
This is a build-option difference from the six ISTA/Swift runs, not a source patch.
Gyro/AP packs use `--compat-bf16`; conversion reports are included. Archive GGUFs
remain unchanged. The three AP files total 269,456,929,408 bytes; all twelve model
variants are revision-pinned, archived and SHA-256 verified.

| Model | Peak sampled cgroup memory.current | Peak sampled whole-GPU allocation |
|---|---:|---:|
| Gyro-M | 34.67 GiB | 37,853 MiB |
| AP-Q4_K_XL | 84.47 GiB | 71,523 MiB |
| AP-IQ3_XXS | 93.79 GiB | 57,727 MiB |
| AP-IQ2_S | 55.28 GiB | 52,913 MiB |

Cgroup memory includes file cache and the queue's other retained pages; it is not
model-only RAM. The cgroup's cumulative memory.peak was not reset per model, so the
table uses the maximum observed memory.current instead. GPU allocation covers the
whole device. Wall times include multi-turn tool scenarios, not decode-only speed.

Warnings from the unchanged mock-tool evaluation:

- Gyro-M: TC-51 (Goal-Level Planning): Batched send_email with create_calendar_event in the same turn instead of waiting for the create_calendar_event result.
- AP-IQ3_XXS: TC-47 (Correction Across Turns): Performed an unrelated side effect while correcting the calendar event.

AP-Q4_K_XL and AP-IQ2_S recorded no safety warnings. These are benchmark mock
side effects; no real email or calendar operations occurred.

The added reports preserve scores, traces, usage, timings and resource samples.
Only hostnames and local path roots are replaced with portable placeholders in
these new artifacts; deployment metadata retains source/build hashes and settings.
Full original receipts remain in the private benchmark workspace.

## Reproducibility and limits

- ISTA/Swift upstream: `d5ea7133741e67743c0e886bb426c0ce8d69cf6c` (0.1.40.3), no engine changes. AP enables the existing Q6 build option; Gyro uses the separate rc1 cohort described above.
- Native CUDA build: Release, CUDA 13.2, SM120; pinned ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- ISTA/Swift engine SHA-256: `b2729caf3f56a81309713ecbde7687a3c8134bc33ed37bb041cf77f32c3c51f5`.
- Hardware: one NVIDIA RTX PRO 6000 Blackwell 96 GB, 124 GiB system RAM.
- Tool-Eval-Bench: `c8a30ff5c1fb132e395bc2d8e4bd549ef1294abd`, standard scenario definitions and scoring unchanged.
- Context 32768, FP16 KV, prefill chunk 8192, expert-cache auto; engine window `--spec 2`
  (required by native packs), no MTP path, suffix drafting disabled. Engine logs confirmed zero drafts.
- Greedy temperature 0, seed 42, thinking disabled, 4096 output-token ceiling, one request at a time,
  one trial, eight turns maximum, 600-second request timeout, reference date 2026-10-07.
- Each model uses its own tokenizer and native dense pack. No Codi persona or private steering.
- Downloads/transfers continued in the background. Timing is preliminary and not an isolated roofline test.
- The current Q8 reference uses the same mainline binary and RTX PRO 6000 as ISTA/Swift, with the pack conversion and memory-mapped loading documented above. The older patched three-P4 run is retained separately.

The unmodified benchmark rejects `--backend strata`. These initial runs use
`--backend unknown`, its generic OpenAI-compatible adapter. Its metadata probe
incorrectly labels Strata's compatible props as llama.cpp; the raw reports preserve
that bug (engine_version is `Strata 0.1.40.3`). The deployment record above is
authoritative. A separate Tool-Eval-Bench integration branch is ready for review: [feat/strata-backend](https://github.com/CC-David-CC/nm-tool-eval-bench/tree/feat/strata-backend).

The command JSON files capture every benchmark argument. After starting Strata
with the recorded server settings, a portable equivalent is:

```sh
git clone https://github.com/SeraphimSerapis/tool-eval-bench.git
cd tool-eval-bench
git checkout c8a30ff5c1fb132e395bc2d8e4bd549ef1294abd
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/tool-eval-bench run --short --base-url http://127.0.0.1:18190 \
  --backend unknown --model ista-qwen3.8-flash-next-q2_0 --seed 42 \
  --temperature 0 --no-think --backend-kwargs '{"reasoning_effort":"none","max_tokens":4096}' \
  --parallel 1 --timeout 600 --max-turns 8 --trials 1 --reference-date 2026-10-07
# Drop --short for the standard 69 scenarios.
```

Raw JSON contains full scenario traces, per-category scores and usage counters.
Host-specific paths in the server/command records must be replaced on another machine.
The model manifest pins repositories, revisions, exact shard filenames, sizes and
published SHA-256 hashes; archive downloads are verified against those hashes.
No model weights or private credentials are included in this branch.
