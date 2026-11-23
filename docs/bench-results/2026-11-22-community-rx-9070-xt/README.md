# Community benchmark on Radeon RX 9070 XT 16 GB (gfx1201), Linux, Strata 0.2.20

Measured on 2026-11-22 by Vyeche. Four models on one 16 GB RDNA4 card, one at a time: Strata's standard benchmark
(`strata bench run`, draft protocol `strata-qwen38-thinking-v1-draft.1`) and a small set of 22 checked tasks. Main
limitations: one run per model, every standard run is `nonconforming` (reasons below), and the Hy4-Flash and
Inkling-2-Mini runs use a protocol written for Qwen3.8-Flash-Next.

## Hardware and software

- GPU and VRAM; CPU; installed RAM; storage; PCIe link if known: AMD Radeon RX 9070 XT, 16 GB (15.9 GiB reported), gfx1201;
  AMD Ryzen 9 9950X3D (16 cores); 92 GB (91.9 GiB reported); model files on NVMe (Hy4-Flash, Inkling-2-Mini) and on a SATA
  SSD (Qwen3.8-Flash-Next); PCIe link not measured
- OS; driver; CUDA or ROCm: Pop!_OS 24.04 LTS, Linux 7.2.15, in-kernel amdgpu. Qwen3.8-Flash-Next: engine built against
  ROCm 7.2.4 and run with the ROCm 10.0.0 runtime libraries (`lib_dirs` in the run config). Hy4-Flash and
  Inkling-2-Mini: the libraries inside their `hip-rx9070` packages
- Strata commit; engine version; release binary or source build: `923e2df4d` (one commit after `v0.2.20`); engines 0.1.66.
  `strata-engine` is a source build (`setup.sh --update`), `strata-inkling` and `strata-hy4` are the release
  `linux-x86_64-hip-rx9070` packages. The generated reports say "runtime rocm 7.2.4" and "release build" for all four;
  for Qwen3.8-Flash-Next read the line above instead
- Background workloads and any power limits: the desktop compositor runs on the same card (about 0.4 GiB of VRAM in
  use before a model loads); no other inference; no VM; `vm.compaction_proactiveness` at its default 20; system76-power
  profile "performance"; no GPU power limit set

## Model and configuration

- Model repository and revision; quantization; GGUF filenames: see each run's `README.md` (file names and hashes are
  recorded there); repository revisions not measured
- Vision encoder; custom packs or profiles: vision on for all four (CPU reader for Inkling-2-Mini); Qwen3.8-Flash-Next
  with the native IQ3_S pack and the shipped expert profile
- Context; KV type and streaming; cache; prefill; low-RAM mode: Qwen3.8-Flash-Next 262,144 context, `--kv int8
  --kv-resident 32768`, `--expert-cache auto`, `--prefill auto`; the other three 32,768 context with the placement
  `strata launch` wrote
- MTP; reasoning; sampling; calibration; experimental speed projection: DFlash2 (Qwen3.8-Flash-Next, `--draft-conf 0.5`),
  DSpark (Inkling-2-Mini), DFly2 (Hy4-Flash); reasoning and sampling as each benchmark sets them; no calibration
  (`--tune skip`); route tail skip at the 0.2.20 default (off)

```text
# Qwen3.8-Flash-Next IQ3_S (serve/server.py --engine strata --config <run config>)
--pack <models>/iq3_s --native <models>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
--ple-gguf <models>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile <models>/expert-profile.bin
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <models>/rt --max-context 262144 --kv int8
--kv-resident 32768 --vision --vram-reserve-mib 2298 --draft-conf 0.5
env: STRATA_HIP_WMMA=1 STRATA_PF_FUSED=1 STRATA_HIPBLASLT_TUNING=<tools/hip>/gfx1201-hipblaslt-100202.txt
     (the table is for hipBLASLt 1.2.2; with the ROCm 10.0.0 runtime the engine reports a version mismatch and uses hipBLAS)

# Inkling-2-Mini SQ-10 / SQ-15 (strata launch --family inkling-2-mini --model sq-10|sq-15 --context 32k --vision cpu --tune skip)
--native <models>/Inkling-2-Mini-SQ-10.strata --max-context 32768 --vision <models>/Inkling-2-Mini-Vision.strata --vram-experts-gb 8.60 --ram-experts-gb 0.00
--native <models>/Inkling-2-Mini-SQ-15.strata --max-context 32768 --vision <models>/Inkling-2-Mini-Vision.strata --vram-experts-gb 12.28 --ram-experts-gb 0.00

# Hy4-Flash SQ-64 (strata launch --family hy4-flash --model sq-64 --context 32k --vision on --tune skip)
--native <models>/Hy4-Flash-SQ-64.strata --engram <models>/Hy4-Flash-Engram-Q8.strata --max-context 32768
--vision <models>/Hy4-Flash-Vision.strata --vram-experts-gb 5.55 --ram-experts-gb 50.49
```

The configs were written by `strata launch` on 0.2.20 and then served with `serve/server.py` on port 8080.

## Method

- **Standard benchmark:** `strata bench run --url http://127.0.0.1:8080 --track custom`, one run per model on an engine
  started for that run, nothing else loaded. Folders: [`standard/`](standard/), each with the generated `README.md`,
  `results.json`, `outputs.jsonl`, `config.json` and `provenance/`. The local output path in `results.json` was replaced
  by `<out>` after the run; nothing else was edited.
- **22 checked tasks:** [`tasks/bench_models.py`](tasks/bench_models.py) against the same servers, a separate engine
  start per model. 6 coding tasks (the returned function is run against asserts), 6 reasoning questions with one exact
  answer, 4 tool-call requests (right tool and argument), 3 questions on a 12K-token document with planted facts, 3
  format tasks. Streamed, `reasoning_effort: "high"`, `temperature: 0`, `max_tokens: 6000`, one warm-up request, each
  task once. "First" is the time to the first visible answer text or tool call, thinking not counted. Per-task data:
  [`tasks/*.json`](tasks/).

## Results

### Standard benchmark (median and range of 3)

| Model | Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next IQ3_S | short | n/a | 70.8 (70.1-76.5) n=3 | 0.44 (0.38-0.51) n=3 | 1.72 (1.25-1.78) n=3 | 4.9 (4.0-6.2) n=3 | 0/3 |
| Qwen3.8-Flash-Next IQ3_S | prompt-4k | 2529.0 (2515.4-2545.1) n=3 | 64.2 (57.8-77.7) n=3 | 1.65 (1.64-1.65) n=3 | not measured | 9.6 (8.2-10.5) n=3 | 3/3 |
| Qwen3.8-Flash-Next IQ3_S | prompt-16k | 3513.5 (3508.8-3554.2) n=3 | 68.6 (68.2-73.6) n=3 | 4.70 (4.65-4.71) n=3 | 10.50 (9.03-10.72) n=3 | 12.1 (11.6-12.2) n=3 | 3/3 |
| Inkling-2-Mini SQ-10 | short | n/a | 62.0 (59.7-62.7) n=3 | 0.09 (0.09-0.09) n=3 | 0.09 (0.09-0.09) n=3 | 3.3 (2.4-3.4) n=3 | 0/3 |
| Inkling-2-Mini SQ-10 | prompt-4k | 2376.4 (2367.1-2387.7) n=3 | 56.7 (54.1-59.4) n=3 | 1.45 (1.44-1.45) n=3 | 1.45 (1.44-1.45) n=3 | 4.8 (4.8-5.3) n=3 | 0/3 |
| Inkling-2-Mini SQ-10 | prompt-16k | 2274.5 (2271.2-2280.4) n=3 | 44.1 (41.9-45.5) n=3 | 6.07 (6.06-6.08) n=3 | 6.07 (6.06-6.08) n=3 | 9.6 (8.2-9.8) n=3 | 0/3 |
| Inkling-2-Mini SQ-15 | short | n/a | 60.7 (60.2-61.5) n=3 | 0.09 (0.09-0.09) n=3 | 0.09 (0.09-0.09) n=3 | 3.3 (2.8-3.4) n=3 | 0/3 |
| Inkling-2-Mini SQ-15 | prompt-4k | 1738.4 (1702.6-1742.2) n=3 | 47.9 (46.9-57.2) n=3 | 1.98 (1.98-2.01) n=3 | 1.98 (1.98-2.01) n=3 | 5.1 (4.5-5.4) n=3 | 0/3 |
| Inkling-2-Mini SQ-15 | prompt-16k | 1656.8 (1535.1-1663.7) n=3 | 37.1 (36.5-42.8) n=3 | 8.33 (8.29-8.99) n=3 | 8.33 (8.29-8.99) n=3 | 13.7 (11.5-14.2) n=3 | 0/3 |
| Hy4-Flash SQ-64 | short | n/a | 22.0 (11.9-28.3) n=3 | 0.53 (0.51-1.16) n=3 | 24.00 (13.15-34.85) n=2 | 23.8 (18.6-44.1) n=3 | 3/3 |
| Hy4-Flash SQ-64 | prompt-4k | 411.7 (392.3-419.4) n=3 | 20.4 (15.1-26.1) n=3 | 6.79 (6.67-7.13) n=3 | 25.50 (19.97-27.95) n=3 | 31.9 (26.3-40.8) n=3 | 2/3 |
| Hy4-Flash SQ-64 | prompt-16k | 502.0 (431.8-503.3) n=3 | 21.0 (11.0-22.2) n=3 | 22.34 (22.28-25.96) n=3 | 43.69 (43.69-43.69) n=1 | 49.0 (46.6-69.1) n=3 | 3/3 |

Sanity checks: 12 of 12 passed on all four models (instruction 4/4, retrieval 4/4, tool calls 4/4). 21 of 21 required
requests finished in every run, 0 failed, 0 skipped. Time after loading: Qwen3.8-Flash-Next 1 min 49 s, Inkling-2-Mini
SQ-10 1 min 14 s, SQ-15 1 min 32 s, Hy4-Flash 7 min 45 s.

Why each run is `nonconforming`:

- **Qwen3.8-Flash-Next IQ3_S:** only "the Strata commit is unknown (not a git checkout and no COMMIT file)". The checkout
  is a git clone at `923e2df4d`; the runner did not find it from `terminal/` or from the repository root.
- **Inkling-2-Mini and Hy4-Flash:** the same commit line, and the prompts tokenize to fewer tokens than the targets
  (3,423 and 13,775 for Inkling-2-Mini, 2,795 and 11,206 for Hy4-Flash, against 4K and 16K), so "prompt-4k" and
  "prompt-16k" are shorter prompts for these models. The full lists are in each `README.md`.

### 22 checked tasks

| Model | Passed | Coding | Reasoning | Tool calls | Reading | Format | Coding first / worst total s | Reasoning | Tool calls | Reading | Suite minutes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: |
| Qwen3.8-Flash-Next IQ3_S | 22/22 | 6/6 | 6/6 | 4/4 | 3/3 | 3/3 | 8.0 / 46.3 | 1.9 / 3.1 | 1.3 / 1.9 | 4.7 / 5.0 | 2.2 |
| Inkling-2-Mini SQ-10 | 20/22 | 5/6 | 6/6 | 3/4 | 3/3 | 3/3 | 0.1 / 2.3 | 0.1 / 3.0 | 0.8 / 0.8 | 5.5 / 5.7 | 0.7 |
| Inkling-2-Mini SQ-15 | 21/22 | 6/6 | 6/6 | 3/4 | 3/3 | 3/3 | 0.1 / 3.1 | 0.1 / 3.6 | 0.9 / 1.0 | 7.7 / 8.1 | 0.8 |
| Hy4-Flash SQ-64 | 21/22 | 6/6 | 6/6 | 4/4 | 2/3 | 3/3 | 28.7 / 132.7 | 5.9 / 9.2 | 8.1 / 10.2 | 29.1 / 30.1 | 7.0 |

Failures: Inkling-2-Mini SQ-10 failed the small cache task (`lru`); both Inkling-2-Mini packs answered "Create a file
notes/todo.txt containing exactly: buy milk" with a `run_command` call (`pwd && ls -la`) instead of `write_file`, which
the check counts as a miss; Hy4-Flash missed one of the three planted facts.

### Memory, from earlier runs the same day (4 requests per model, not part of the runs above)

| Model | VRAM in use after loading | VRAM after a 7.8K-9.8K-token prompt | RAM in use |
| --- | ---: | ---: | ---: |
| Inkling-2-Mini SQ-10 | 13,223 MiB | 14,499 MiB | 15 GB |
| Inkling-2-Mini SQ-15 | 16,061 MiB | 16,127 MiB | 16 GB |
| Hy4-Flash SQ-64 | 13,507 MiB | 13,511 MiB | 68 GB |

Card total 16,304 MiB (`mem_info_vram_used` / `mem_info_vram_total`). Qwen3.8-Flash-Next not measured in that run.

## Correctness and limitations

- One run per model and one machine. No repeat on another day, no comparison against another build.
- The 22 tasks are small and were written for this report; they show that a model answers correctly and how long the
  first text takes, not how it does on long agent work. Only Qwen3.8-Flash-Next was run with a context above 32K.
- Inkling-2-Mini SQ-15 was not tested beyond a 13.8K-token prompt; it leaves under 0.2 GiB of VRAM free here.
- Hy4-Flash decode is 20-22 tok/s in the standard run on this 16 GB card (11-28 across requests); a 4K context was
  not tested (`strata launch` offers 32K as the smallest).
- Not measured: PCIe link, power draw, concurrency (`--optional-concurrency` was not run), model repository revisions.

This report was prepared with the help of an AI assistant; the measurements were run on the hardware above.
