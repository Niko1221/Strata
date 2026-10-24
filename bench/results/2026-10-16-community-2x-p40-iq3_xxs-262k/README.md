# Community benchmark: 2x Tesla P40 (sm_61), EPYC 7302, Flash-Next IQ3_XXS at 262,144 context, engines 0.1.40.3, 0.1.40.4 and 0.1.44.1

Measured 2026-10-14 to 2026-10-16 by [dev-brewery](https://github.com/dev-brewery) on a Linux server, Strata in Docker.
Three engine versions on the same machine, model, config and prompts. It confirms the Pascal decode regression in
#1469 and its fix on a second P40 pair, adds a container vs bare-metal pair, a per-request `strata_tune` sweep of
`--spec-min-p` and `--pcie-frac` (the knobs from #1799), and two direct clients with `--batch 2 --batch-groups 2`.
The report was prepared with an AI assistant (Claude Code, Claude Opus 5.5) from the raw run files in `raw/`;
every number below was checked against them.

Single-stream decode with 1,024-token answers at a 4K prompt, three runs each, same config apart from the noted
environment variables:

| engine | decode tok/s | draft acceptance | run |
| --- | --- | --- | --- |
| 0.1.40.3 (`d5ea713`) | 17.5, 18.0, 18.1 | 0.559, 0.604, 0.597 | `raw/proxy/20261016T142027Z` |
| 0.1.40.4 (`6674a00`) | **35.4, 32.6, 33.2** | 0.671, 0.623, 0.588 | `raw/proxy/20261016T142855Z` |
| 0.1.44.1 (`296b59d2`), new defaults held off | **36.0, 35.7, 40.3** | 0.582, 0.573, 0.707 | `raw/proxy/20261016T154228Z` |

These are not standard-protocol results (they predate `strata bench run` on this machine). The Strata single-stream
answers are sampled with the model card's thinking-mode values, not greedy, and those requests went through a local
OpenAI-compatible proxy on the same host. See Limits.

## Hardware and software

- **GPUs:** 2x NVIDIA Tesla P40 24 GB (GP102, compute capability 6.1, 23,040 MiB usable each), VBIOS 86.02.23.00.01,
  power limit 250 W (card default), PCIe 3.0 x16 on both (current and max), no NVLink, both on NUMA node 0 (`machine-facts`
  excerpt below). The engine's PCIe probe: 13.1 GB/s host to device on each card, `pcie_frac` auto 0.36.
- **CPU and RAM:** AMD EPYC 7302 (16 cores / 32 threads, AVX2, no AVX-512, NPS1, one NUMA node), 128 GB DDR4-2666 ECC
  (125 GiB MemTotal). Engine: 15 expert-pool workers plus the host thread.
- **OS and runtime:** Ubuntu 22.04.5, kernel 5.15.0-198-generic, NVIDIA driver 580.178.04, Docker 29.8.1,
  nvidia-container-toolkit 1.20.1.
- **Other load:** the server also runs the host's other services (the proxy and small CPU-side models).
  Each run records the 1-minute load of other processes before and after (`ext load` in `raw/*/summary.md`); it was
  0.0 or 1.0 throughout. The GPUs ran only Strata.

## Build

`Dockerfile.p40` in this folder is Strata's own `Dockerfile` at each tag with these changes (its header lists them):
base `nvidia/cuda:12.9.1-devel-ubuntu24.04` (CUDA 13 dropped sm_6x), `CUDA_ARCHITECTURES=61`, `BUILD_VISION=0`, the
engine built into `engine-cuda12/` with `-DSTRATA_EXPERIMENTAL_SM60=ON` and `"toolkit": 12` in `BUILD.json` (what
`setup.build_engine()` writes for a CUDA 12 engine), `ENV STRATA_CUDA=12 STRATA_EXPERIMENTAL_SM60=1`, and a 900 s
health-check start period. CUDA compiler 12.9.86, host compiler GCC 13.3.0, 0 errors. Strata sources are unmodified
clones at the tags; llama.cpp at Strata's pin `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

| engine | tag commit | image ID | `BUILD.json` src |
| --- | --- | --- | --- |
| 0.1.40.3 | `d5ea7133741e67743c0e886bb426c0ce8d69cf6c` | `sha256:69dca0eca2b5…` | `3950c1f221aa6e1b` |
| 0.1.40.4 | `6674a00` | `sha256:3753c531cb42…` | `2d98525b8162b1fa` |
| 0.1.44.1 | `296b59d2` (the tag resolves to this merge commit) | `sha256:54dd82ad9286…` | `7dfd34378bd2d32a` |

The bare-metal pair (E2 below) used a host build of the same 0.1.40.3 source: nvcc 12.9.2, GCC 12.3, same run config.

## Model

[`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) at
revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (the revision `setup.py` pins), IQ3_XXS, used in place through
`GGUF_DIR`. SHA-256 verified locally on 2026-10-08 against the published values:

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` 47,039,860,096 bytes `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` 28,800,138,432 bytes `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`

Native IQ pack and MTP draft layer built by setup on first start; expert profile `data/expert-profile.bin` as bundled.

## Configuration

`config/` holds the run configs (container paths). All runs: `--expert-cache auto --prefill auto --spec 4 --mtp
--max-context 262144 --kv int8 --kv-resident 32768 --remote-expert-opt --conversation-cache-mib 16384`,
`layer_split "26"` (layers 0-25 on CUDA0, 26-47 and the head on CUDA1), `STRATA_STAGE_TRIM=1`, sampling
temperature 1.0 / top_p 0.95 / top_k 20 / min_p 0 (the model card's thinking-mode values), `fit_max_tokens` on.

| config | difference | used for |
| --- | --- | --- |
| `strata-iq3_xxs.json` | `--spec-min-p 0.5` | 0.1.40.3 and 0.1.40.4 |
| `strata-iq3_xxs.ctx256k-fit.json` | identical args to the above | the E2 container/bare-metal pair |
| `strata-iq3_xxs.v0.1.44.1-held.json` | + `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_PREFILL_CPU_SHARE=0` | 0.1.44.1 |
| `strata-iq3_xxs.v0.1.44.1-held.minp0.7.json` | held + `--spec-min-p 0.7` | the gate deployed as the running value |
| `strata-iq3_xxs.v0.1.44.1-held.batch2.json` | held + `--batch 2 --batch-groups 2` | two direct clients |

The two held variables follow `docs/OLDER_GPUS.md` at v0.1.44.1 (pin both off when comparing builds). Route tail skip
would have been auto-off here anyway: 93% of the experts are cached.

Placement from the engine log at each start (`engine-log/startup-blocks.txt`): 22,961 to 22,964 resident experts (93%),
543 MiB VRAM free, 99.9% expert cache hit rate at decode, expert arena 39.97 GiB in pinned RAM. With `--batch 2`:
21,668 resident (88%), 99.5 to 99.8% hit rate.

## Method

`raw/bench.py` sends fresh prompts (a unique nonce per request, `cache_prompt: false`, `ignore_eos: true`) and reads
the engine's own `timings` (`prompt_per_second`, `predicted_per_second`, `draft_n`, `draft_n_accepted`). A "standard"
pass is 256-token answers at about 256, 1K, 4K, 16K and 32K prompt tokens; a "long" pass is three 1,024-token
answers at a 4K prompt. TTFT is a separate streaming request with 64 tokens. The Strata proxy and tune runs send no
temperature, so the server's sampling defaults apply (temperature 1.0); the two-client runs, the llama.cpp run and the
dense baseline send temperature 0. The first line of each `summary.md` records which. `raw/<kind>/<run>/runs.jsonl` holds every request, including the
1-minute load before and after and `nvidia-smi` after each one; `summary.md` is the table built from it.

Correctness: `raw/correctness_test.py`, 20 fixed prompts (`correctness/prompts.json`), greedy (temperature 0),
compared build to build as full text and as final answer.

## Results

### 1. The regression and the fix (#1469)

Same session, same config, one tag change. Long pass, 1,024-token answers at a 4K prompt:

| | 0.1.40.3 | 0.1.40.4 |
| --- | --- | --- |
| decode tok/s | 17.5, 18.0, 18.1 | 35.4, 32.6, 33.2 |
| prefill tok/s at 4K | 314.1, 313.3, 317.4 | 335.6, 337.0, 334.3 |
| draft acceptance | 0.559, 0.604, 0.597 | 0.671, 0.623, 0.588 |
| resident experts / VRAM free | 22,964 / 543 MiB | 22,964 / 543 MiB |
| greedy, final answer identical to 0.1.40.3 | | 16/20 (full text 6/20) |

The 17 to 18 tok/s matched the regressed 17.7 in #1469 on the same quantization. Every differing answer in the
correctness compare was reviewed against both builds; none was wrong, the differences were wording ("3 hours 25 minutes" vs
"3 hours and 25 minutes"). The same two prompts reached the 400-token cap on every build.

### 2. 0.1.44.1 with the new defaults held off

| | 0.1.40.4 | 0.1.44.1 held |
| --- | --- | --- |
| decode tok/s, long pass | 35.4, 32.6, 33.2 | 36.0, 35.7, 40.3 |
| greedy, final answer identical to 0.1.40.4 | | 14/20 (full text 6/20) |

Standard pass, 256-token answers (`raw/proxy/20261015T210525Z` for 0.1.40.3, `20261016T142535Z` for 0.1.40.4,
`20261016T153800Z` for 0.1.44.1):

| prompt tokens | decode 0.1.40.3 | decode 0.1.40.4 | decode 0.1.44.1 | prefill 0.1.40.3 | prefill 0.1.40.4 | prefill 0.1.44.1 |
| ---: | --- | --- | --- | --- | --- | --- |
| 256 | 17.7, 26.7 | 33.3, 33.2 | 36.9, 38.4 | 156.0, 150.8 | 152.5, 154.3 | 172.9, 167.4 |
| 1,024 | 18.5, 16.8 | 35.1, 32.5 | 37.4, 37.2 | 259.7, 261.3 | 262.3, 260.2 | 286.4, 316.1 |
| 4,096 | 17.8, 17.2 | 36.0, 44.8 | 37.4, 36.9 | 321.2, 320.7 | 332.1, 332.3 | 356.5, 356.4 |
| 16,384 | 18.9 | 29.8 | 36.5 | 505.0 | 531.4 | 560.6 |
| 32,768 | 17.0 | 33.3 | 35.6 | 593.0 | 622.7 | 652.8 |

Single runs at 256 tokens swing with draft acceptance: the 26.7 on 0.1.40.3 had acceptance 0.910, the 44.8 on
0.1.40.4 had 0.877. The long pass is the steadier number. TTFT at 1K, streaming 64 tokens: 4.27 to 4.78 s on 0.1.40.4,
3.44 to 4.81 s on 0.1.44.1 (one cold 38.6 s TTFT on the first request after the 0.1.44.1 deploy is in the raw file).

### 3. Container vs bare metal (E2, 0.1.40.3)

Same engine source, same config, same prompts, through the same proxy, on 2026-10-15:

| | container | bare metal |
| --- | --- | --- |
| long pass decode tok/s | 17.8, 17.8, 18.0 (`20261015T210945Z`) | 16.6, 17.0, 16.8 (`20261015T213228Z`) |
| standard pass decode at 4K | 17.8, 17.2 (`20261015T210525Z`) | 17.2, 17.4 (`20261015T212756Z`) |
| standard pass prefill at 32K | 593.0 | 588.5 |

Identical placement. The container costs nothing measurable on this machine.

### 4. `strata_tune` sweep: `--spec-min-p` and `--pcie-frac` (0.1.44.1 held)

`raw/spec_minp_sweep.py` sends `"strata_tune": {...}` per request on the live server, no restart, 1,024-token answers at a
4K prompt, arms rotated each round (`raw/tune/20261016T183446Z`). An earlier run, `20261016T181038Z`, is included
but stopped after one round: the first request of round 2 got HTTP 502 when another client asked the proxy for a
different model and the proxy swapped Strata out. Every round 1 request completed normally.

| arm | decode tok/s per round | median | acceptance (median) | drafts issued |
| --- | --- | --- | --- | --- |
| control (gate 0.5, `pcie_frac` auto 0.36) | 36.2 / 36.5 / 36.9 | 36.5 | 0.607 | 744 / 806 / 769 |
| `spec_min_p` 0.7 | 40.2 / 42.8 / 34.4 | 40.2 | 0.787 | 634 / 629 / 483 |
| `spec_min_p` 0.3 | 34.2 / 34.8 / 34.5 | 34.5 | 0.434 | 1,213 / 1,176 / 1,141 |
| `pcie_frac` 0.34 + `spec_min_p` 0.7 (the #1799 pair) | 35.7 / 36.5 / 37.0 | 36.5 | 0.747 | 531 / 549 / 577 |

The gate at 0.7 was then deployed as the running value and measured again (`raw/proxy/20261016T185455Z`, long pass):
**36.5, 41.3, 35.3** tok/s, acceptance 0.754 / 0.850 / 0.738, against 36.0 / 35.7 / 40.3 at 0.5. Standard pass
`20261016T185153Z`: 44.8 / 40.9 at 256 tokens, 36.4 / 36.6 at 1K, 35.4 / 38.1 at 4K, 36.4 at 16K, 37.1 at 32K.
Greedy final answers identical to the 0.5 build: 13/20, none wrong. Acceptance rose as the sweep predicted; tokens
per second did not follow outside the short prompts. A higher gate makes each verify window surer and shorter, and on
this machine the two cancel from 1K to 16K. Reverted to 0.5.

A fixed `pcie_frac` cost about 4 tok/s against auto in every round. With 99.9% of expert lookups already hitting VRAM,
there is little for the share to govern here. Left on auto. On this machine, the #1799 pair raises acceptance but is
not shown to raise throughput.

### 5. Two direct clients, `--batch 2 --batch-groups 2` (0.1.44.1 held)

Two simultaneous requests straight to the engine's port, 256-token answers (`raw/direct-concurrency/20261016T194933Z`),
against the single-sequence FIFO on 0.1.40.3 (`20261014T125520Z`, same protocol):

| | FIFO, 0.1.40.3 | `--batch 2`, 0.1.44.1 |
| --- | --- | --- |
| 256-token prompt: per stream / aggregate tok/s | 17.0 + 18.6 / 15.3 | 27.0 + 22.0 / 37.0; 22.3 + 27.3 / 37.8 |
| 4K prompt: per stream / aggregate tok/s | 18.7 + 18.1 / 9.6 | 11.9 + 26.8 / 15.3; 12.0 + 26.9 / 15.3 |
| 4K prompt: pair wall time | 53.4 s | 33.4 s, 33.4 s |

The 4K asymmetry matches `docs/BATCHING.md`: admissions are one at a time, so the second stream decodes while the first
prompt is read. The slots run without MTP drafts (`--batch-mtp` does not combine with `--remote-expert-opt` or
`--batch-groups` above 1). A single request on the batch config goes back to the solo path with drafts: 37.0, 35.5, 35.8
tok/s (`raw/proxy/20261016T195116Z`). Cost: 22,961 to 21,668 resident experts.

### 6. Long live turn

A 13,083-token prompt from a chat client on the final config: 13,327 tokens generated at **38.9 tok/s** sustained over
342 s, prompt read at 507.5 tok/s, drafts accepted 6,694 of 10,075 (engine log line in `engine-log/live-turn.txt`).

## Context: llama.cpp on the same machine

The same IQ3_XXS files on llama.cpp b11330 (`--fit` placement, q8_0 KV, FA on, one slot, no MTP),
same proxy and harness but greedy (`raw/llamacpp/20261010T150344Z`): decode 21.85 / 23.24 at 256 tokens, 21.57 / 22.63 at 4K, 20.77 at 16K;
prefill 144 / 231 / 217 tok/s. Row split, the fastest mode for dense models on these cards, is gone from llama.cpp
since July 2026, and `-sm tensor` is not implemented for this architecture. The dense 27B this replaced (Qwen3.8 27B
Q6_K on llama.cpp, MTP, 4 slots) ran 23.25 / 14.43 tok/s at 256 tokens and 14.68 at 4K on the same harness, also
greedy (`raw/baseline/20261008T010100Z`). Greedy and sampled decode rates are not strictly comparable; treat this
section as context, not as an A/B.

## Limits

- Not the standard protocol: own harness, Strata single-stream runs sampled at temperature 1.0 with thinking on,
  through a local proxy on the same host. Absolute numbers are not comparable with `strata bench run` results; the
  comparisons between Strata versions are within one harness and one sampling setup.
- One failed request (HTTP 502, proxy model swap) in `raw/tune/20261016T181038Z`, recorded in that run's `summary.md`;
  the harness wrote no `runs.jsonl` line for it. Every other request finished with `finish_reason: length` and engine timings.
- Three runs per long-pass cell, one or two per standard-pass cell. Draft acceptance moves single runs by up to
  ~25%; see the 26.7 and 44.8 outliers above.
- Engine versions are 0.1.40.3 to 0.1.44.1; nothing here was measured on 0.2.x.
- No needle or quality suites beyond the 20-prompt greedy compare. No vision (`BUILD_VISION=0`). No power or clock capture.
- The engine log excerpt is bounded by engine starts, because only start lines carry timestamps; runtime lines whose
  date cannot be established were left out rather than dated by guess.
- Not an isolated OS: other services ran on the CPU side during the runs (see `ext load`).

## Files

- `raw/proxy/`, `raw/tune/`, `raw/direct-concurrency/`, `raw/llamacpp/`, `raw/baseline/`: one folder per run with
  `summary.md`, `runs.jsonl` (every request) and `config.json` (the harness settings).
- `raw/bench.py`, `raw/spec_minp_sweep.py`, `raw/correctness_test.py`: the scripts that produced them.
- `correctness/`: the three greedy compares (Markdown and JSON), the per-build outputs, the prompt set.
- `config/`: the run configs. `Dockerfile.p40`: the image recipe.
- `engine-log/startup-blocks.txt`: the engine's start-up lines for every start in the window (placement, residency,
  VRAM free, PCIe probe). `engine-log/live-turn.txt`: the one log line for section 6.
- `machine-facts.txt`: `nvidia-smi -q`, `nvidia-smi topo -m`, `lscpu`, `free`, `uname`, OS release, Docker version.

Host names, user names, home folders, LAN addresses and storage roots are replaced by placeholders (`<HOME>`,
`<HOST>`, `<STORAGE>` and so on). Timings, token counts, commit hashes, image IDs and file hashes are unchanged.
