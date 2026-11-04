# RTX 4070 Ti SUPER: a 14 GiB RAM budget improves prefill at 128K capacity

Measured on 2026-11-03 by fm5454, with AI assistance from **Sol 6.1 Sol High**
(assistant label supplied by the contributor). This identifies the assistant
used to orchestrate the tests; measured inference used the Strata model below.
This is a custom prefill
comparison on **one NVIDIA GeForce RTX 4070 Ti SUPER 16 GB**, not a standard
benchmark, quality evaluation, or recommendation to use 14 GiB on other PCs.
No engine code or defaults change in this report.

Adding `--resident-budget-gib 14` to the working `--mmap-experts` configuration
raised three-run median prefill throughput by **458.5% at approximately 4K**
(188.2 to 1,051.1 tok/s; 5.59x) and **389.1% at approximately 16K**
(315.7 to 1,544.0 tok/s; 4.89x). Configured context stayed at **131,072 tokens**
and KV at **INT8** throughout. These percentages compare this PC with itself,
not another GPU or a screenshot's results.

## Hardware and software

- Single RTX 4070 Ti SUPER, 16 GB graphics memory; CUDA compute capability 8.9.
- AMD Ryzen 7 7800X3D, 8 cores / 16 logical processors; 31.1 GiB usable RAM.
- Model files on the C: NVMe volume, Samsung SSD 990 PRO 1 TB.
- Windows; exact OS build and NVIDIA driver version were not captured.
- Desktop app 0.2.10; prebuilt engine 0.1.56, `windows-x86_64-cuda-cuda13`.
  The exact engine source revision, binary digest, power limits and clock
  telemetry were not captured. The current PR revision was not the measured build.
- Desktop applications remained open. Background CPU/GPU load was not rigorously
  controlled. Inference conditions ran sequentially, with the normal app model
  unloaded before the isolated workers; no simultaneous benchmark workers.

## Model and configuration

Swift 1.5 Qwen3.8-Flash-Next GSQ-RCO IQ2_XS, native pack `swift-iq2_xs`.
The installed GGUF argument was
`Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`.
Immutable model/tokenizer revisions, weight hashes and expert-profile hash were
not captured. The profile was the installed `data/expert-profile.bin`.

Both conditions used `--expert-cache auto --prefill auto --spec 4
--spec-min-p 0.5 --max-context 131072 --kv int8 --mmap-experts --vision
--vram-reserve-mib 1536`, with the same pack, native/PLE GGUF and MTP directory.
Vision was enabled with a CPU encoder (8 threads); requests contained text only.
No KV streaming flag or control vector was added. One request ran at a time.
The engine selected 7 CPU pool workers and 8,192-token prompt chunks.

The sole intentional inference-setting change was adding
`--resident-budget-gib 14`. Sanitized effective argument/vision excerpts are in
[config-baseline.json](config-baseline.json) and
[config-resident14.json](config-resident14.json). They are descriptive excerpts,
not complete runnable configs. `<DATA>`, `<CONTROL>` and `<ENGINE>` replace local
paths. The normal app load was subsequently verified with the flag persisted in
`user_args` as well as `args`.

Automatic choices were not frozen: resident mode, file I/O strategy, available
RAM, expert adaptation, conversation-cache sizing and GPU cache slots can differ.
For example, the attached resident run had 4,450 GPU slots; its fresh stress
restart had 4,494. The logs record the actual choices.

## Method and results

[prefill_probe.py](prefill_probe.py) is the script used for the measurements.
It sends non-streaming local Chat Completions requests with `temperature: 0`,
`max_tokens: 8` and `strata_checkpoint: false`. Thinking was not explicitly
disabled; some replies exhausted eight tokens while thinking. These completions
are sufficient to time prefill, but do not establish answer correctness or useful
answer latency. Decode rates from such short completions are not headline results.

Prompts are synthetic, repetitive water-flow records with deterministic random
numbers and a fresh leading nonce. Character/token calibration changes later
prompt lengths; nonce tokenization also differs slightly between conditions.
The actual lengths, not the nominal target, are in every raw record. These are
the same generator and workload family, not byte-identical fixed prompts or
representative documents. The original nonce values were not retained.

Six requests per main condition: three approximately 4K prompts, then three
approximately 16K prompts. The baseline used the already-loaded app engine after
earlier tuning; the resident condition used a freshly loaded isolated worker.
There was no separate warm-up request for this probe. Expert adaptation and OS
cache state were retained within each engine; the OS file cache was not flushed.
Loading is excluded. Every measured request reported **zero cached prompt tokens**.

| Nominal size | Baseline actual tokens | Baseline tok/s (median; range) | Resident14 actual tokens | Resident14 tok/s (median; range) | Gain |
| --- | --- | --- | --- | --- | --- |
| 4K | 4,163–4,290 | 188.2; 172.9–212.4 | 4,159–4,287 | 1,051.1; 1,036.1–1,090.2 | +458.5% |
| 16K | 16,318–16,517 | 315.7; 293.6–365.8 | 16,321–16,523 | 1,544.0; 1,482.9–1,766.1 | +389.1% |

Gain is `(resident median / baseline median - 1) * 100`, rounded to one decimal.
[results.json](results.json) preserves all 18 requests, including engine timing,
actual input/output counts, cached counts and client wall time.
`prompt_ms` is engine prefill time, not measured streaming TTFT; `wall_s` is total
non-streaming request latency, including the eight generated tokens.

The sequence after the two main conditions was:

1. A fresh resident worker: 4,289-token calibration request (1,002.5 tok/s), then
   **124,529 fresh input tokens**, 8 generated, at **2,095.1 tok/s** prefill
   (59.4373 s engine prompt time; 59.688 s client wall time).
   Available system RAM, sampled approximately every 0.5 s with Strata's
   `tune_machine.memory()` / Windows `GlobalMemoryStatusEx`, had a minimum of
   **4,443 MiB** during these requests. This is sampled free memory, not a process
   memory peak. Raw memory samples were not retained.
2. A fresh return to the original setting: one request per size, **194.8 tok/s**
   at 4,290 tokens and **340.2 tok/s** at 16,317 tokens. The slowdown returned
   after the fast resident runs; warm-up alone does not explain the difference.
3. After saving the override to the app's canonical model config and loading
   normally: one request per size, **1,003.2 tok/s** at 4,289 tokens and
   **1,825.5 tok/s** at 16,317 tokens. This checks normal-load persistence,
   not an additional three-run median.

Recorded probe commands, with local paths replaced by placeholders:

```text
python prefill_probe.py <active-instance-config.json> baseline
python prefill_probe.py <active-instance-config.json> verified_app --repeats 1
```

The isolated worker called `benchmark(cfg, "resident14")`; the return-to-baseline
worker called `benchmark(cfg, "baseline_recheck", repeats=1)`. The stress worker
called `benchmark(cfg, "resident14_stress", sizes=(4096, 126000), repeats=1)`.
Worker launch configs carried ephemeral loopback API credentials and are not
published. To repeat the probe, use a private running-instance config containing
the actual `port` and `api_key`; the script never prints the key. Start only one
engine at a time, compare the same settings above, and keep the engine's logs.

## What the logs show, and what remains unproven

[baseline_recheck-engine.log](baseline_recheck-engine.log) and
[resident14-engine.log](resident14-engine.log) are supporting logs from the actual
runs, with local paths sanitized. The resident log includes both the three-repeat
run and its fresh stress restart. The baseline log supports the later baseline
recheck; a separate original six-request baseline log is not included.

The original mmap condition had no resident expert RAM copy. The resident log
confirms 14 GiB of page-locked experts and **unbuffered file-tier reads** for the
remaining experts. Thus the gain measures a combined change to RAM residency and
I/O behavior; it does not isolate the individual contribution of either one.
Roughly 13 GiB of experts still remained in the file tier; the whole model did
not fit in RAM plus VRAM.

There are memory tradeoffs in the logs: the resident stress restart disabled the
parked-conversation RAM cache, warned about remaining Windows commit headroom,
and later disabled CPU prefill share when its rows buffer could not be allocated.
Inference still completed. These warnings are preserved; this is not evidence of
an optimal or universally safe budget, nor of faster cached multi-chat workflows.

No measured probe request failed or was cancelled. The initial harness attempt
was blocked by local socket permissions before sending an inference request;
the subsequent authorized run produced the recorded baseline. An initial
persistence attempt wrote the tuner's working copy, which the app regenerated;
the corrected canonical-config load is the `verified_app` series.

Not tested: full-capacity 131,072-token inputs, recall/quality, natural-document
performance, long output generation, concurrency, other GPUs or systems, an
optimal RAM-budget sweep, or a source build of this PR. The 124,529-token result
is an execution/memory stress check, not a long-context accuracy result.
## Recommendation

This report recommends investigating a **bounded resident-RAM candidate in the
opt-in tuner** when a Windows mmap configuration repeatedly reads experts that
do not fit the available file cache. Size any candidate against available RAM,
Windows commit headroom, configured context and other host allocations. Compare
repeated uncached prefill measurements against the unchanged configuration, and
retain it only when throughput improves with acceptable memory headroom and
parked-conversation behavior.

The tested **14 GiB** value is a manual setting for this specific PC/model pair.
The report does not establish a universal budget, an optimal sizing formula,
or a benefit for other models. The recommendation is a follow-up investigation;
this benchmark PR implements no tuner logic or default change.

Related discussions: [#5059](https://github.com/Niko1221/Strata/issues/5059)
(app RAM-budget handling; different model/RAM and startup symptom) and
[#3072](https://github.com/Niko1221/Strata/pull/3072)
(Windows mapped-expert prompt I/O; different hardware and proposed engine change).
Neither is claimed as reproduced here. This report is also distinct from
[#4716](https://github.com/Niko1221/Strata/pull/4716), which uses a different CPU,
64 GB RAM, model variant and standard workload on the same GPU model.
