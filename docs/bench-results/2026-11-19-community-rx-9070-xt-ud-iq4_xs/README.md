# Community benchmark on AMD Radeon RX 9070 XT (gfx1201)

Measured on 2026-11-19 by Thxeverybody. Single RX 9070 XT 16 GB + Ryzen 7 9700X + 123 GiB RAM,
Strata 0.2.16 with the **official prebuilt engine 0.1.62 (linux-x86_64-hip-rx9070 package)**,
Unsloth UD-IQ4_XS v3.1, 262,144-token context, MTP drafter, calibrated settings.
Speed suite (`tools/strata_bench.py --long`, thinking off) plus a 6-needle recall check.
Main limitations: one GPU, speed-only suite (no coding/tool correctness beyond recall),
GPU power limit not captured, and the engine is the release package — not a source build
against the system ROCm (a local source build segfaults at startup on ROCm 7.16, see #6322).

## Hardware and software

- GPU and VRAM: AMD Radeon RX 9070 XT, 16 GiB (gfx1201). The desktop is composited on the
  iGPU (Radeon 610M); the dGPU is dedicated to the engine. PCIe 5.0 x16 (32.0 GT/s, `rocm-smi`
  pcie clock level 2). GPU power limit/draw: not captured (`rocm-smi` power unreadable in this environment).
- CPU: AMD Ryzen 7 9700X (8 cores / 8 threads, Zen 5); installed RAM: 123.4 GiB DDR5-6000;
  storage: NVMe Lexar SSD NM790 4 TB (model + pack), SATA HDDs/SSD present (not used for the model).
- OS: CachyOS (Arch-based), kernel 7.3.5-1-cachyos; amdgpu driver 7.3.5-1-cachyos;
  ROCm userspace 7.16.0 (hipcc 7.16.0-4f6d503, AMD clang 24.0.0git), system hipBLASLt 1.4.1.
- Strata: project **0.2.16** (tag `v0.2.16`, terminal/server commit `64524fb9f`); engine **0.1.62**,
  the **official prebuilt `strata-engine-0.1.62-linux-x86_64-hip-rx9070` package** (sha256-verified;
  `BUILD.json` source commit `18526b879`, variant rx9070, archs gfx1200/gfx1201, bundled ROCm libs,
  `hipblaslt_version` 100300). **Release binary, not a source build.** The package bundles its own
  ROCm libraries (`RUNPATH $ORIGIN`); `lib_dirs` points only at the package dir so the system ROCm 7.16
  does not override them. A local source build of the same tag against system ROCm 7.16 segfaults in
  `prime_hw_queues` before any output (#6322); the package is unaffected.
- Background workloads: desktop on the iGPU, no game/video during the suite; the assistant backend
  was remote (no local GPU use). Exact background load not recorded.

## Model and configuration

- Model: Qwen3.8-Flash-Next, **Unsloth UD-IQ4_XS dynamic v3.1** (`unsloth/Qwen3.8-Flash-Next-GGUF`,
  `UD-IQ4_XS`). GGUF: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf` … `-00004-of-00004.gguf`.
  Native expert pack built with `tools/iq_pack.py` (pack format v3); expert profile trained on this
  rig (24,576 ranked pairs). Routed-expert layout: 33/48 layers Q5_0 down, 13 IQ4_NL, 2 Q2_0;
  gate/up IQ3_XXS / IQ4_XS / IQ2_S. The engine reads the prompt on **48 of 48 layers fused**
  (`STRATA_PF_FUSED_Q50` default on for gfx12/Linux since 0.2.15).
- Vision: enabled, CPU reader (`strata-vision`), BF16 mmproj, `max_tokens` 300 — **not exercised**
  by this text-only suite.
- Context 262,144; KV `int8` with `--kv-resident 32768` (KV streaming); expert cache `auto`
  → **3812 slots / 8.74 GiB VRAM**; prefill `auto` → chunk 21,760 (96-slot ring);
  `--ple-io ram` (n-gram table mlocked, ~28.8 GiB); `--vram-reserve-mib 512`.
- Drafter: **MTP** (`--spec 4 --mtp rt`; runtime verify window 6, suffix-draft 3, lookup 3).
  `--pool-workers 2`, `--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`,
  `--pcie-frac 0.75`, `--spec-min-p 0.50`. Env: `STRATA_GDN_CHUNKED=1`; `STRATA_SH_STREAM` off (default).
- **Calibration: yes** — `setup.sh --calibrate` on this PC (2026-11-19) chose pcie-frac 0.75,
  spec-min-p 0.50, pool-workers 2, adapt 1/80/0.97, SH_STREAM off (decode-only sweep). No experimental
  speed projection for this model (the control-vector projection is IQ3_S-only on this rig).

```text
strata --serve --pack <data>/packs/unsloth-ud-iq4_xs-v31 \
  --native <models>/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf \
  --expert-profile <strata>/data/expert-profile-ud-iq4_xs-v31.bin \
  --expert-cache auto --prefill auto --spec 4 --mtp <data>/mtp/rt \
  --max-context 262144 --kv int8 --ple-io ram --kv-resident 32768 \
  --vram-reserve-mib 512 --vision --pcie-frac 0.75 --spec-min-p 0.50 \
  --pool-workers 2 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
# server: python serve/server.py --engine strata --config <config>.json --port 8080
# engine: official 0.1.62 rx9070 package; lib_dirs = [package dir] only
```

## Method

Run by `python terminal/tools/strata_bench.py --url http://127.0.0.1:8080 --long` against the HTTP API
of an already-running server; model loading is not timed. One warm-up chat and a 1K prompt are discarded
first. Every prompt carries a unique tag, so the conversation cache cannot help the fresh-prompt tests.
Thinking is off for the suite; temperature 0. Speeds are the engine's own timings
(`prompt_per_second`, `predicted_per_second`); first-token times are measured by the client over HTTP.

- short chat: three questions, up to 100 new tokens, 3 runs (9 requests)
- 4K / 16K / 32K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt + four tools, then 5 turns each adding ~1500 words of a project file and
  continuing from the model's own earlier answers, 64 new tokens/turn, 3 sessions; cache numbers from
  the answer's `usage.cached_tokens` and `timings.prompt_n`
- recall: `python terminal/tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k --depths 10,50,90`

Expert cache warmed from the trained profile at startup; conversation cache 3840 MiB RAM-tier with SSD spill.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43–51) | 0 | 100 | 9 | not measured | 67.4 (57.8–78.1) | 0.49 (0.41–0.64) |
| 4K prompt, fresh | 4019 | 0 | 24 cap | 3 | 2376.3 (2366.3–2385.9) | not measured | 1.72 (1.71–1.73) |
| 16K prompt, fresh | 16058 | 0 | 24 cap | 3 | 3492.7 (3458.5–3506.3) | not measured | 4.63 (4.61–4.67) |
| 32K prompt, fresh | 32049 | 0 | 24 cap | 3 | 3406.5 (3399.9–3455.2) | not measured | 9.45 (9.31–9.46) |
| agent session, follow-up turns | 9021 (9016–9057) last turn | see below | up to 64 | 3 | see below | 68.1 (63.0–81.7) | 1.11 (0.92–1.26) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.
Prefill tok/s rises with prompt size (4K → 16K) then plateaus (16K ≈ 32K); compare only within a size.

### Agent session (conversation cache)

Cache hit over the follow-up turns: **71.0 %**; earlier-turn tokens read again per session: 5
(the ~8207 new file tokens are read once); first turn (nothing cached) TTFT 1.32 s (1.26–1.37).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | TTFT (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 494.6 | 1.264 | 43.3 |
| 1 | 2263 | 621 | 1642 | 1724.4 | 0.976 | 66.2 |
| 2 | 3969 | 2326 | 1643 | 1572.2 | 1.070 | 73.8 |
| 3 | 5675 | 4032 | 1643 | 1377.9 | 1.217 | 64.3 |
| 4 | 7378 | 5736 | 1642 | 1341.8 | 1.252 | 63.0 |
| 5 | 9057 | 7415 | 1642 | 1333.4 | 1.256 | 66.3 |

### Long-context recall (needle)

`needle_bench.py`, code word at 10/50/90 % depth. **6 of 6 found.**

| Label | Actual prompt tokens | Depths | Found |
| --- | ---: | --- | --- |
| 32k | 27,870–27,871 | 10/50/90 % | 3/3 |
| 128k | 106,321–106,323 | 10/50/90 % | 3/3 |

Per-request data: `results.json` (speed suite) and `needles.json` (recall).
Engine startup lines: `engine-log-excerpt.txt` (paths scrubbed).

### Memory observations

- VRAM: 640 MiB free after load; verify-window capture down to ~564 MiB; expert cache 3812 slots / 8.74 GiB.
  During long prompts the prompt path borrows cache slots; 3 request-start warnings at 496–502 MiB
  (just under the engine's 512 MiB safety line), **no paging, no stall, no graph eviction** across the suite.
- RAM: expert arena ~56.2 GiB pinned (`cudaHostRegister PORTABLE ok`, transparent huge pages;
  `MAP_HUGETLB` unavailable, `vm.nr_hugepages=0`); PLE table ~28.8 GiB mlocked (`--ple-io ram`);
  MemAvailable ~11 GiB with everything loaded; conversation cache 3840 MiB RAM-tier + SSD spill.
- The conversation-cache SSD tier occasionally spills a large parked conversation in the background
  (writes up to ~2.8 s), which can dip a single request; medians are robust to it.

## Data format

Per-run data is the benchmark tool's own JSON (`schema: 1`). All speeds are **tokens per second
(tok/s)**; `*_s` fields are **seconds**, `*_ms` **milliseconds**, `*_tokens` **token counts**,
`*_mib` **MiB**, `ram_gib` **GiB**, needle `depth` **percent**.

`results.json` top level: `schema`, `tool`, `date`, `params` (`runs`, `think`, `clients`, `long`,
`started_server`, `command`), `elapsed_s` (suite wall time), `machine` (GPU/VRAM/CPU/RAM/OS/engine_build),
and `results` with sub-objects `chat`, `prompts` (`4K`/`16K`/`32K`), `agent`, `clients`, `skipped`, `failures`.
Each test object holds aggregate medians/ranges plus a `requests` (or `sessions`) array. Per-request fields:

| Field | Unit | Meaning |
| --- | --- | --- |
| `ok` / `err` | bool / string\|null | request succeeded / error text |
| `ttft_s` | s | time to first token, **client-measured** over HTTP |
| `total_s` | s | total request latency, **client-measured** |
| `prompt_tokens` | tokens | total prompt tokens |
| `reused_tokens` / `read_tokens` | tokens | conversation-cache reused / freshly read |
| `generated_tokens` | tokens | tokens generated (answer; thinking off) |
| `prompt_tps` / `decode_tps` | tok/s | engine's own `prompt_per_second` / `predicted_per_second` |
| `prompt_tps_source` / `decode_tps_source` | string | `engine` (from the answer's `timings`) |
| `prompt_ms` / `decode_ms` | ms | engine prompt-processing / decode time |
| `drafts_offered` / `drafts_accepted` | count | MTP draft proposals / accepted |
| `run` / `prompt` | index | run number / chat-question index |

`needles.json` is an array of `{length` (label), `depth` (percent), `prompt_tokens` (actual),
`seconds` (wall), `found` (bool), `word` (the needle code word), `answer` (model reply)`}`.

**Reproducibility:** prompts are generated by the repository tools — word lists sized by a calibration
request (`tools/strata_bench.py`), the three chat questions from `tools/ab_engine.py`, and random needle
code words (`terminal/tools/needle_bench.py`). Every prompt carries a unique tag, so no external fixture
files are needed and the fresh-prompt tests never reuse cache. Re-run with the exact commands in Method.
The native expert pack and trained profile are derived from the public Unsloth GGUF via `tools/iq_pack.py`
and the engine's profile training; they are not included here (large generated files stay out of the PR).

## Correctness and limitations

- Recall: 6/6 needles at 27.9k and 106.3k actual tokens (depths 10/50/90 %). This measures recall on
  those inputs, not overall model quality. No coding-task or tool-call correctness check was run.
- Speed suite with thinking off and short output caps; production use with reasoning enabled will differ.
- Single GPU, official release package (bundled ROCm libs), not a source build against system ROCm 7.16.
- GPU power limit/draw not captured; exact background load not recorded (desktop on iGPU, no game/video).
- Total time of the speed suite: 103 s; needle suite ~2.5 min. Model loading not included in any timing.
