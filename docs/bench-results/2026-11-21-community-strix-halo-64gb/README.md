# Community benchmark on Radeon 8060S (Strix Halo, **64 GB unified memory**) — the 64 GB data point

Measured on 2026-11-20 and 2026-11-21 by **ryougishiki00**. Strata 0.2.15 (engine 0.1.60 `linux-x86_64-hip-halo`) with the
`docs/STRIX_HALO.md` §5 fast config on a **64 GB** Ryzen AI Max+ 395. As far as I can tell this is the first
public 64 GB unified-memory result: the existing community entry and the maintainer numbers are both 128 GB machines that can
hold all 24,576 experts in the GPU-visible pool. Here only ~2.6k–4.7k expert-cache slots fit, and two things
turn out not to matter as much as expected:

- **the SSD file tier is never used.** The experts stay resident in host RAM on this machine, so a small GPU
  cache buys almost nothing on decode at high hit rates (81% vs 88% measures the same, ±0.5 tok/s; only
  going from 61% to 69% moves decode by ~4%);
- **prompt throughput reaches the 128 GB numbers** (1,122–1,173 tok/s at 110K tokens), and the whole 256K
  window is usable (260,513-token prompt at 1,058 tok/s).

Decode depends much more on the content class and on which drafter runs than on the cache size — see Results.

## Hardware and software

- GPU: Radeon 8060S (gfx1151), unified memory
- Graphics carve-out (BIOS): **512 MiB** (`mem_info_vram_total` = 536,870,912), the smallest value used here, so
  the OS sees the most system RAM: `MemTotal` = 63,428 MiB (61.94 GiB); the rest of the 64 GB is firmware/ACPI reserved
- GTT: 48 GiB (`ttm.pages_limit=12582912 ttm.page_pool_size=12582912`)
- CPU: Ryzen AI Max+ 395 (16C/32T); RAM: 64 GB LPDDR5X-8000 (256 GB/s); storage: KIOXIA KBG60ZNS1T02 NVMe
- Machine: ASUS ROG Flow Z13 GZ302EA, BIOS `GZ302EA.314`, platform profile `performance`
  (`power-profiles-daemon` disabled, a small unit re-applies `performance` + EPP after boot); package power
  ~86 W during prefill, ~4 W idle (RAPL package and `amdgpu/power1_average` agree to <0.5 W)
- OS: Ubuntu 26.04.1, kernel 7.0.0-43-generic, `multi-user.target` (headless), `vm.swappiness=10`
- Kernel args: `ttm.pages_limit=12582912 ttm.page_pool_size=12582912 crashkernel=no amd_iommu=off
  amdgpu.lockup_timeout=10000,60000,10000,10000`
- Strata: terminal 0.2.15 / build 0.1.61, engine 0.1.60 installed with
  `strata components install-file --backend hip` (binary distribution, **not** a source build)
- Background: no desktop session, no other GPU work; the engine runs under systemd

## Model and configuration

- Model: [SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF),
  IQ3_XXS, revision `90a3350cd800ec1fe0d1660a38414ad9e985782f` — shard 1 = 47,342,144,896 B,
  sha256 `03b11926b41a3d7f7a008882b1bc7a6330a3c9c43d3cad4598d31dc73042fdc5`; shard 2 (the shared n-gram table) =
  28,800,138,432 B, byte-identical to the ISTA-DASLab pack, hard-linked instead of stored twice
- Pack: rebuilt locally from this model's shards with `tools/iq_pack.py --gguf <shard1> --out <dir>`
  (`--base` needs a canonical Q2_0 `manifest.json`, which this pack does not have; run it with the engine's
  bundled `python/bin/python3`, which needs `numpy`). Without the rebuild the engine refuses to start:
  *"the pack's native_experts.txt does not match the model: blk.0.ffn_gate_exps.weight … starts at byte
  1228652416, the pack says 1227833216"*
- Vision: `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,543,008 B), GPU reader (`--vision gpu`, 1444 image tokens)
- Context: `--max-context 262144`, `--kv int8`, `--kv-resident 32768` (default), `--parallel 1`
- Experts: `--expert-cache 2000` (2,649 slots, 4.34 GiB of VRAM), `--vram-reserve-mib 1536`,
  `--prefill 8192` (borrows 1,760–2,331 cache slots, 2.88–3.81 GiB), `--conversation-cache-mib 1024`
- Drafting: the MTP layer, packed locally from the 31 `mtp.*` tensors of the BF16 checkpoint with
  `tools/mtp_pack.py --experts q2_0` + `tools/mtp_rt.py` (889 MB GGUF → `rt/{experts.bin,dense.bin}`, 801 MiB of
  VRAM), then `--spec 4 --mtp-window 8192 --spec-min-p 0.5`. The MTP folder must be named on the command line
  (`--mtp`/`--set mtp=`): the launcher does not add it, and without it the engine drafts nothing at all.
- `STRATA_RESIDENT_HEADROOM_GIB=6`

```text
STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_HC_UPMIX=1 STRATA_PA_FAST=1 STRATA_HIP_WMMA=1 STRATA_SELECT_WMMA=1 \
STRATA_PF_SWITCH_MIN_T=4096 STRATA_PREFILL_STREAM_MIN=128 STRATA_REFILL_PARTS=1 \
STRATA_HIPBLASLT_TUNING=<engine>/tuning/gfx1151-hipblaslt-100401.txt   # shipped with the engine

strata launch --restart --family qwen --model iq3_xxs --model-path <model dir> \
  --context 256k --kv int8 --vision gpu --mtp on --drafting mtp --backend amd --parallel 1 \
  --set expert-cache=2000 --set vram-reserve-mib=1536 --set prefill=8192 \
  --set mtp=<data>/mtp/rt --set mtp-window=8192 --set spec-min-p=0.5 \
  --set conversation-cache-mib=1024 --access lan --port 8095 --no-tui --yes
```

## Method

- Prompt: Chinese prose (novel-style text), 1.478 tokens/char; the full-context runs at the bottom of Results
  use random CJK characters (1.55 tokens/char) so that nothing can be cached and the prompt can be pushed to the
  context limit.
- Timings are the engine's own `timings.prompt_per_second` / `timings.predicted_per_second` as reported by the
  OpenAI-compatible server (`/v1/chat/completions`), `temperature 0`,
  `chat_template_kwargs.enable_thinking = false`, `max_tokens` 8 for prompt runs and 200 for decode runs. Every
  measurement uses a fresh salt so the conversation cache cannot reuse a prefix.
- Decode is reported per content class, because draft acceptance depends on it: **prose** and **code** (a
  question whose answer is source code). Medians of 3–5 runs.
- Ablation rows differ from their baseline only in the listed change.

## Results

| Configuration | Prompt tokens | Runs | Prompt tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| fast config, abliterated pack, prose (drafter: MTP; the 3rd run also adds `amd_iommu=off`) | 110,454 | 3 | **1,122.8 / 1,131.0 / 1,173.4** | 35.7 (25.7 with no drafter) |
| same, prose | 177,198 | 1 | 1,081.9 | — |
| same, code (200-token decode, MTP) | 36 | 5 | — | **50.8** |
| same, prose (200-token decode, MTP) | 36 | 5 | — | **35.7** |
| full context, random CJK, 88.9% of the window | 232,930 | 1 | 1,078.2 | — |
| full context, random CJK, 94.5% of the window | 247,867 | 1 | 1,066.4 | — |
| full context, random CJK, 99.4% of the window | **260,513** | 1 | **1,058.1** | — |
| official ISTA-DASLab IQ3_XXS pack, prose (drafter: DFlash2) | 134,061 | 1 | **1,141.1** | 29.6 |
| official pack, code (200-token decode, DFlash2) | 36 | 1 | — | **54.4** |
| official pack, prose (200-token decode, DFlash2) | 36 | 1 | — | 29.6 |

Draft acceptance is the decode lever: DFlash2 accepted 52% (prose) / 69% (code) on the abliterated pack and
46% / 80% on the official pack; the converted MTP layer accepts 60% / 85% and is the faster drafter here. The
full-context rows are the 256K window being used almost completely; a prompt beyond `--max-context` is rejected
with `HTTP 400` in under a second and does not disturb the engine.

For reference, the 128 GB entries (same model family, fast config):

| | 128 GB (HP ZBook Ultra G1a, 70 W) | this 64 GB machine |
| --- | ---: | ---: |
| prompt 8K–128K tok/s | 969–1,031 (greedy, engine-level) | **1,058–1,173** (110K–260K tokens, greedy; 1,180.5 thinking-on at 16K in the standard run) |
| decode tok/s | 59.2–42.0 (C++ prompts, **greedy**) | **50.8 (code) / 35.7 (prose)** (MTP, n=5) at the default 2,649 slots, greedy; 51.9 / 39.2 at 5,885 slots |

The prose/code split is content, not configuration: with identical settings DFlash2 accepts 69% of drafts on
code and 52% on prose, which alone moves decode by ~1.6×.

## Standard benchmark (`strata bench run`)

Protocol `strata-qwen38-thinking-v1-draft.1`, run `20261121T005308Z-7d853232`: **status complete, 21 of 21 required
requests finished, 0 failed, 0 skipped, 12 of 12 sanity checks passed** (instruction 4/4, retrieval 4/4, tool calls
4/4), 4 min 45 s after the model loaded. The runner started the engine from the saved profile itself, so the run is
cold ("the server had answered no request before this run") and the effective engine arguments are recorded in its
`config.json`. Run against the **official ISTA-DASLab IQ3_XXS** weights with this machine's MTP drafter, `track: custom`,
thinking on at the model card's sampling. Every request, its raw usage and timings:
[2026-11-21-standard-8060s-7d853232](../2026-11-21-standard-8060s-7d853232/README.md).

| Case | Prompt tok/s | Decode tok/s | TTFT s | Total s |
| --- | ---: | ---: | ---: | ---: |
| short (~80-token prompt) | n/a (fixed cost dominates) | 42.4 (39.8–46.6) n=3 | 0.81 (0.70–0.81) n=3 | 11.7 (9.5–12.9) n=3 |
| prompt-4k | 939.9 (840.5–962.8) n=3 | 35.2 (35.1–40.7) n=3 | 4.39 (4.29–4.91) n=3 | 18.8 (17.5–18.9) n=3 |
| prompt-16k | 1,180.5 (1,155.8–1,217.4) n=3 | 34.0 (34.0–39.3) n=3 | 13.92 (13.51–14.22) n=3 | 28.5 (26.9–29.2) n=3 |

Cells are median (min–max) n=samples. prompt-4k and prompt-16k reached the 512-token cap with thinking on (short: 2 of 3), so
decode is the rate while reasoning; the sweep rows above run with thinking off and are therefore not directly
comparable — the standard numbers are the ones to compare with other standard runs. The twelve sanity checks above
are also the only quality evidence in this report: they passed 12/12 at `--kv int8`, expert cache 2000 and the
MTP drafter (packed locally from the BF16 checkpoint), so none of those settings was observed to break instruction following,
retrieval or tool calls in these cases.

## Ablations

| Change | Effect |
| --- | --- |
| `/etc/strata-fast.env` off → on (the §5 switches + hipBLASLt table) | prompt 110,454: **691.6 → 1,122.8–1,131.0 tok/s (+62–64%)** (decode not compared cleanly: the off baseline ran without a drafter) |
| `amd_iommu=off` added to the kernel args | prompt 1,131.0 → **1,173.4 (+3.7%)**, n=1, applied together with `amdgpu.lockup_timeout` and a reboot |
| expert cache 3000 → 3500 → 4000 slots | hit rate 81% → 88%, decode unchanged (±0.5 tok/s): on UMA, cache slots do not buy throughput |
| expert cache 2000 (2,649 slots) vs 1000 (1,318 slots) | decode unchanged; prompt 110K **676 → 505 tok/s** — the prompt path borrows cache slots, so < ~1,600 slots forces smaller chunks |
| `--prefill 4096 → 8192` (before the fast config) | prompt 110K 668 → 707 tok/s (+7.5%); with the fast config the chunk size stops mattering |
| `--kv int8 → k8v4` | KV 3.09 → 2.39 GiB pinned, decode −2.8% (`q4_0` would be 1.55 GiB) |
| context 256K → 128K, expert cache 2000 → 8000 (engine clamps to 5,885 slots, 9.53 GiB) | hit rate 61% → 69%, decode prose 37.9 → 39.2 and code 50.0 → 51.9 (+4–4.6%) — not worth half the window; 24,576 slots (100% hit) would need another ~40 GiB of cache beside the 40 GiB arena, which did not fit in the runs above |
| `--draft-conf` 0.35 / 0.5 / 0.6 / 0.7 / 0.8 | prose 30.8/30.5/**30.1**/29.7/29.0, code 51.8/52.9/**54.7**/54.0/51.9 → 0.6 |
| MTP layer, converted (`mtp_pack.py`/`mtp_rt.py`, `--mtp rt`, `--spec 4 --mtp-window 8192 --spec-min-p 0.5`) vs DFlash2 | prose **35.7** (61/102 accepted) vs 30.5 (35/67); code **50.8** (140/165) vs 49.8 (141/204) → MTP wins prose by 17%, ties code. Without `--mtp` the engine emits no MTP drafts at all (`0 of 0`), which is easy to mistake for 'MTP is slow' |
| tuned DFlash2 pack (`strata1.q4_0-selq4`) vs stock | prose 30.3 vs 30.4, code 54.8 vs 55.7 → no gain |
| vision reader CPU (300 tokens) → GPU (1444 tokens) | prompt tokens 315 → 1,479; warm image 2.7 s; ~1 GB of GTT |

## Memory

- `MemTotal` 63,428 MiB; engine `VmRSS` 46.1 GiB (39.97 GiB expert arena + KV + graph buffers), `VmSwap` 0 while
  decoding; `MemAvailable` 3.1 GiB idle with the MTP layer loaded (4.5 GiB when DFlash2 was the drafter)
- Startup log: `memory guard: limit 61.9 GiB … available 60.4 GiB, floor 2560 MiB`,
  `expert arena read through the file cache (buffered (not Windows))`, `loaded 39.97 GiB at 3.49 GiB/s`,
  `expert cache 2649 slots, 4.34 GiB of VRAM … pre-filled 2649 of 2649`
- The SSD file tier is **never used** here: `file tier` / `unbuffered` / `blob reads` appear 0 times in the log.
  (On Windows with a 30 GiB RAM budget the same model logged `file tier reads unbuffered … 10.0 GiB`.)
- Long prompts: 232,930 / 247,867 / **260,513** tokens all completed (99.4% of the window, 247 s), with the
  lowest `MemAvailable` seen = **2,836 MB** against the 2,560 MiB guard floor; no `no progress for 60 s` event.
  125K prompts used to stop the engine before `STRATA_RESIDENT_HEADROOM_GIB=6` and
  `--conversation-cache-mib 1024` were set.

## Limitations

- Server-level timings (`/v1/chat/completions`), not engine `--stats`: first-token path and template rendering are
  included in the prompt time, so these numbers are slightly conservative versus engine-level reports.
- 1–5 runs per cell (3–5 for decode, 1 for the full-context rows); run-to-run spread on decode is ±2 tok/s.
- Two packs are reported (official ISTA-DASLab IQ3_XXS and an abliterated build), not the UD-IQ4_XS pack the 128 GB
  entries used, so absolute values compare loosely with those reports. The MTP drafter works on both packs (it loads
  and drafts against the official weights in the standard run, where the draft layer is not part of the pack); the
  official-pack decode rows in the sweep below were taken with DFlash2 before the MTP layer was packed.
- Vision was enabled for all runs (`--vision gpu`): 801 MiB drafter + ~1 GB reader.
- Two kinds of number are reported: the **standard protocol** run (`strata bench run`, thinking on, model-card
  sampling, official weights) and a **hand-run sweep** (fresh salt per request, thinking off, greedy) used for the
  ablations. Compare the standard rows only with other standard runs.
- **Where the ~10% decode gap to the 128 GB entry comes from is not established.** The most obvious candidate is
  expert-cache coverage — 61% of routed experts are served from VRAM here against 100% there, because 24,576 slots
  would need roughly another 40 GiB of cache on top of the 40 GiB arena and that does not fit in 64 GB — but the
  evidence is weak: raising the cache to 5,885 slots (69% hit) moved decode by only ~4%, and the two machines differ
  in engine version (0.1.60 vs 0.1.41), quantization (IQ3_XXS vs UD-IQ4_XS), prompt content, package power limit
  (86 W vs 70 W) and RAM (64 vs 128 GB). There is also no prose decode number on the 128 GB side to compare with.
  Read the gap as "roughly 10% on code-like content, cause open", not as a diagnosed cache effect.

## Reproduction notes

- The BIOS graphics carve-out is the single most important setting on a 64 GB machine: 512 MiB leaves 61.94 GiB of
  `MemTotal`; the 8 GiB option costs 7 GiB of system RAM for no gain (the engine never puts the experts in VRAM),
  and 32/48 GiB starve the OS (`MemTotal` 31.6 / 15.6 GiB) and push the experts back into the file tier.
- Second most important: `docs/STRIX_HALO.md` §5. Without the eight prompt switches the same machine measures
  691.6 tok/s at 110K.
- An engine stop on very long prompts is a soft failure — the serve layer does not restart it, so a supervisor
  (systemd with `KillMode=none`, or a watchdog) is worth having.




