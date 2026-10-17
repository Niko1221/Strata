# Community benchmark on NVIDIA GeForce RTX 5070 Ti (16 GiB) with 31.6 GiB of RAM

Measured on 2026-10-17 by [Lithivm](https://github.com/Lithivm) with `tools/strata_bench.py`
(suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings
(`prompt_per_second`, `predicted_per_second`); first-token times are measured by the client over
HTTP. Speed only: no correctness check was run.

**What this report is:** the same GPU as `2026-10-04-community-rtx-5070ti-windows` and
`2026-10-06-community-rtx5070ti-5900x`, but with **31.6 GiB of system RAM instead of 64 GiB**. Both
earlier reports run the experts out of RAM; at 32 GiB this install cannot, so it runs Strata's
low-RAM mode (`--mmap-experts`) with the experts served through the OS file cache. Decode is roughly
half of the 64 GiB numbers and prompt processing is 4-8x slower — the tables below measure that
boundary, which is the main limitation of the report and, I think, its point.

## Hardware and software

- **GPU and VRAM:** NVIDIA GeForce RTX 5070 Ti, 16,303 MiB reported VRAM (16 GiB), compute
  capability 12.0. Power limit **300 W**, maximum also 300 W (not reduced).
- **PCIe link:** **Gen5 x16** (current and maximum), no downshift observed.
- **CPU:** AMD Ryzen 7 9700X, 8 cores / 16 threads, AVX-512.
- **Installed RAM:** **31.6 GiB** (33.9 GB) — reported as GiB throughout this report.
- **Storage:** WD_BLACK SN770 1 TB, NVMe SSD (PCIe 4.0). The model files, the prepared pack and the
  draft layer are all on `D:`; the OS is on the same physical drive. No separate model disk.
- **OS:** **Windows 11 Pro, 10.0.26200** (the benchmark's own line says "Windows 10 (10.0.26100)",
  which is wrong: Python reports Windows 11 as "Windows 10" and the build number is one behind).
- **Driver:** 616.64. **CUDA:** 13.0 (from `engine/BUILD.json`).
- **Strata:** project **0.1.44.1**, engine **0.1.44.1**, both from the **release binary**; this
  install is a **zip install, not a git checkout** (`git_commit` is therefore `null` in
  `results.json`). No source build, no changed build options.
- **Background workloads:** *not an isolated machine.* Clash Verge Rev (`verge-mihomo`, the local
  proxy), Steam, Motrix, a ComfyUI MCP bridge, Unsloth Studio, Paseo, Edge/EdgeWebView2 and a
  third-party antivirus daemon were running; the desktop was in use. The GPU also drives the
  display, so the desktop compositor holds some VRAM. **Measured idle VRAM headroom before the run
  was about 1.0-1.2 GiB, which is the threshold Strata's engine itself warns about** (see
  Limitations).

## Model and configuration

- **Model:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at revision
  `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (2026-09-29), quantization **IQ2_XS** (2-bit i-quant).
- **GGUF files** (verified against the revision's published SHA-256 before they were installed):
  - `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` — 39,225,954,592 bytes,
    sha256 `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7`
  - `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` — 28,800,138,432 bytes,
    sha256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`
- **Vision encoder:** not installed (text only). **No** custom pack, custom expert profile (the
  shipped `data/expert-profile.bin` is used) or custom draft vocabulary.
- **Context:** 131,072 tokens (half of the model's trained 262,144). **KV:** `q4_0`
  (Hadamard-rotated 4-bit), **KV streaming off** (setup left it off: it would need about 1.0 GB of
  RAM beside the ~48 GB the model wants, and this PC has 31.6 GiB). The KV cache therefore lives
  entirely in VRAM and the context is paid for out of the expert cache.
- **Low-RAM mode:** **on**, `--mmap-experts`: the 36 GB arena is read through the OS file cache
  instead of being copied into RAM. The GPU holds roughly 29-31% of the experts.
- **Expert cache:** `auto`; it settled at **4,966-5,089 slots, about 6.7-6.8 GiB of VRAM**
  (`PROFILE` policy, no eviction, pre-filled from the shipped profile).
- **Prefill:** `auto` (the default 8 K chunk; `auto:32768` is *not* used — it is a known pessimization
  on a 32 GiB box, see Limitations).
- **MTP:** `--spec 4 --spec-min-p 0.5`, draft layer from `Qwen/Qwen3.8-Flash-Next` as installed by
  setup, default draft vocabulary (`cjk`).
- **Reasoning:** off for the suite (the benchmark's default). **Sampling:** the suite's own requests
  set `temperature 0`; the install's `sampling` block
  (`temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0`)
  is therefore overridden and **does not apply to these numbers**.
- **Calibration:** none. **Experimental speed projection:** off.

```text
strata.exe --serve --pack D:\Strata-data\packs\iq2_xs
  --native   ...\IQ2_XS\Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
  --ple-gguf ...\IQ2_XS\Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
  --expert-profile D:\Strata\data\expert-profile.bin
  --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp D:\Strata-data\mtp\rt
  --max-context 131072 --kv q4_0 --mmap-experts --vram-reserve-mib 1600
```

One environment variable was set for the server the benchmark started:
`STRATA_RESIDENT_HEADROOM_GIB=6`. It limits the page-locked RAM copy the prompt path takes; the
engine's default (4) leaves about 1.4 GiB of `MemAvailable` on a 32 GiB machine, which is close
enough to paging to be a risk (the same finding as `2026-10-04-community-4090-32gb-ram-512k`).

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server started by the
benchmark itself from the install configuration. Model loading is not timed (it was 1.3 s of weights
plus the profile pre-fill; ready in about 20 s). One warm-up chat and a 1K prompt are discarded
first. Every prompt starts with a unique tag, so the conversation cache cannot help the
fresh-prompt tests. Output caps: 100 new tokens for chat, 24 for the prompt tests, 64 per agent
turn. Total suite time: **420 s**.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh
  prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a
  project file and continue from the model's own earlier answers, 64 new tokens per turn, 3
  sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 57.2 (36.8-87.0) | 0.54 (0.42-1.06) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 270.0 (226.5-270.2) | not measured | 14.94 (14.94-17.80) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 447.8 (446.7-518.1) | not measured | 35.94 (31.07-36.03) |
| agent session, follow-up turns | 14572 (14561-14582) at the last turn | see below | up to 64 | 3 | see below | 45.2 (32.3-51.4) | 12.79 (10.82-17.17) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

**Note the ordering inside the chat row:** the nine short-chat requests are three runs of three
questions back to back, and decode climbs across them — 42.5, 41.9, 36.8 / 38.8, 57.2, 69.7 / 87.0,
68.0, 86.0 tok/s. That is the low-RAM mode warming up: the first requests still miss experts on the
SSD, and by the end of the suite the steady state is about 85-87 tok/s. A single cold request
measures the low end of the range, not the machine.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session:
73 (2-73) (earlier turns that the cache did not hold; the 13729 (13728-13729) new tokens of the
files are read once); first turn (nothing cached) first token: 6.27 s (3.82-9.61).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 64.4 | 9.605 | 50.2 |
| 1 | 3760 | 639 | 3121 | 293.2 | 10.824 | 41.4 |
| 2 | 5293 | 3753 | 1540 | 130.8 | 11.862 | 47.3 |
| 3 | 8737 | 5357 | 3380 | 198.0 | 17.173 | 41.0 |
| 4 | 11643 | 8781 | 2862 | 193.4 | 14.893 | 51.4 |
| 5 | 14582 | 11683 | 2899 | 216.6 | 13.453 | 45.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt`.

### How this compares with the 64 GiB RTX 5070 Ti reports

| | this run (31.6 GiB) | `...rtx5070ti-5900x` (64 GiB) | `...rtx-5070ti-windows` (64 GiB) |
| --- | --- | --- | --- |
| decode, 4K prompt | not measured | 102.0 tok/s | 75.8 tok/s |
| decode, 32K prompt | not measured | 101.3 tok/s | 82.8 tok/s |
| prompt, 4K | **270.0 tok/s** | 1,992 tok/s | 3,012 tok/s (5.2K) |
| prompt, 16K | **447.8 tok/s** | ~2,482 tok/s (32K) | ~3,523 tok/s (131K) |
| decode, agent turns | 45.2 tok/s | — | — |
| decode, short chat, steady state | ~85-87 tok/s | — | — |

The prompt numbers are not directly comparable (different prompt sizes and engines), but the gap is
4-8x and consistent with the cause: at 32 GiB the prompt path re-reads lent experts from the SSD,
while at 64 GiB they stay in RAM.

## Correctness and limitations

- **Not measured:** this is a speed suite. No recall, needle or coding check was run, and the suite's
  own answers are not inspected. The conversation uses the model's own replies, so token counts
  differ a little between runs.
- **Isolation:** the machine was **not** isolated (see Hardware). The desktop holds some VRAM, so the
  expert cache is smaller than it would be on a dedicated machine, and the SSD is shared with the
  OS.
- **Warm-up:** the low-RAM expert cache warms over the whole suite, so the chat row spans 36.8-87.0
  tok/s. Numbers from a cold process are not comparable to these.
- **The engine's VRAM warning:** at a smaller `--vram-reserve-mib` the engine logged
  `WARNING: only 853 MiB of VRAM free at the start of this request (below 1024 MiB)` and named
  issue #2065 (the driver can page GPU memory to system RAM and decode falls to a few tok/s). This
  run used `--vram-reserve-mib 1600`, which cleared the warning at the cost of about 15 % of the
  expert cache. Machines that also drive a display should expect to pay this.
- **Prefill:** `--prefill auto` (8 K chunks) is used deliberately. On a 32 GiB box
  `auto:32768` was measured 2.7-3.4x **slower** in `2026-10-04-community-4090-32gb-ram-512k`
  because the lent expert slots do not fit in RAM and are re-read per chunk. The low prefill numbers
  above are therefore the 32 GiB behaviour and not a tuning accident.
- **Skipped cases:** `--clients` and `--long` were not run; the 384K/512K contexts were not tested.
  Vision and image generation are not installed.
- **Tool bug seen while preparing this report:** `tools/strata_bench.py:564` calls
  `(ROOT / "CMakeLists.txt").read_text()` without an encoding, so on a Chinese/Japanese Windows
  (GBK code page) it dies with
  `UnicodeDecodeError: 'gbk' codec can't decode byte 0xbf in position 2` on the file's UTF-8 BOM,
  before any measurement. `setup.py:4649` reads the same file with `encoding="utf-8"`. Setting
  `PYTHONUTF8=1` works around it. Report filed separately.

Machine details come from the server, `nvidia-smi` and the OS; background load is described above
and otherwise not controlled. Total time of the suite: 420 s.
