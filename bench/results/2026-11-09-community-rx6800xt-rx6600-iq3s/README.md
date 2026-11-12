# Community benchmark on 2× AMD Radeon (RX 6800 XT 16 GiB + RX 6600 8 GiB)

Measured on 2026-11-09. Qwen3.8-Flash-Next **IQ3_S**, 131,072-token context, KV int8,
layer split over two different RDNA2 cards with the expert arena in system RAM. A second run of the
same suite on the same machine, immediately before a reboot, is included below as a page-size
comparison (4 KB vs 2 MB pages). The main limitation: the run is **nonconforming** — the model ships
as a custom pack plus GGUF, so no model file could be hashed and the quantization revision is unknown.

## Hardware and software

- GPU 0 (main): AMD Radeon RX 6800 XT 16 GiB, **gfx1030**, PCIe 3.0 x16 (host→device probe 28.3 GB/s)
- GPU 1 (second stage): AMD Radeon RX 6600 8 GiB, **gfx1032**, PCIe 3.0 x16 (host→device probe 6.9 GB/s)
- CPU: AMD Ryzen 7 9700X, 8 cores / 16 threads (AVX-512); AMD Radeon integrated graphics **disabled in
  the BIOS** (with it enabled the engine picked gfx1036 as the helper card and ran 36 layers on one SM)
- RAM: 63.7 GiB installed (2×32 GiB DDR5-5200, dual channel); 53.7 GiB free with the model stopped
- Storage: NVMe SSD, model files read unbuffered (`STRATA_UNBUFFERED_LOAD` behaviour, 51.1 GiB of files)
- OS: Windows 10.0.26300; GPU driver 32.0.21045.16010; HIP runtime `amdhip64_7.dll` from the engine package
- Strata commit `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d`; engine **0.1.57**, hand-built for
  `STRATA_HIP_ARCHS=gfx1030;gfx1032;gfx1036` (the prebuilt rdna2 package does not contain gfx1032)
- Background: no other GPU or CPU work; the browser was closed for the run. The model had just been
  started after a reboot, one engine session, no other requests before the suite.

## Model and configuration

- Model: Qwen3.8-Flash-Next **IQ3_S** — 46.84 GiB of routed experts, served from a native pack
  (`packs/iq3_s`) with the token embedding IQ4_XS in mapped host memory (322 MiB)
- Vision: on, encoder on the CPU (8 threads)
- Context 131,072; `--kv int8`; KV streaming: 16,384 of 131,072 cells per QSA layer in VRAM, the K/V in
  1.16 GiB of pinned RAM
- Expert cache: `auto` → **7,362 experts** in VRAM (5,527 slots / 10.26 GiB on the 6800 XT, 1,835 slots /
  3.86 GiB on the 6600), pre-filled from the routed-expert profile
- Expert arena: **40,290 MiB locked** in system RAM, 2 MB large pages, 8 slices (7 GiB) pinned via
  `cudaHostRegister`; under WDDM the engine caps that at 8 GiB (`STRATA_ARENA_PIN_GIB`)
- Prompt path: `--prefill auto` → chunk 6,144 tokens, 96-slot ring; borrows 1,502 CUDA0 slots and 1,322
  CUDA1 slots (2.78 GiB each)
- Drafting: DFlash2 drafter (Q4_0 weights, 625 MiB of VRAM), `--spec 6 --draft-conf 0.35`
- VRAM free with everything loaded: 842 MiB (6800 XT) and 810 MiB (6600) — close to the edge; the engine
  warns below 1,024 MiB (#2065)

```text
strata.exe --serve --pack <HOME>\Strata\Models\packs\iq3_s
  --native <HOME>\Strata\Models\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
  --ple-gguf <HOME>\Strata\Models\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
  --expert-profile <HOME>\Strata\components\control\0.1.57\data\expert-profile.bin
  --expert-cache auto --prefill auto --max-context 131072 --kv int8 --kv-resident 16384
  --layer-split auto --vram-reserve-mib 1024 --vram-reserve-later-mib 864
  --draft dflash2 --draft-model <HOME>\Strata\Models\dflash2\qwen3.8-flash-next-dflash2.q4_0-selq4.pack
  --draft-fallback none --mtp-draft-vocab <HOME>\Strata\Models\mtp\rt\draft_vocab.bin
  --spec 6 --draft-conf 0.35 --vision
```

Layer split chosen by the engine from free VRAM: layers 0–38 on the 6800 XT, 39–47 plus the head on the
6600, one hand-off per verification window.

## Method

`strata bench run --url http://127.0.0.1:8080 --suite standard-v1 --skip-model-hash --track custom`
against the already-running server. Protocol `strata-qwen38-thinking-v1-draft.1`: 1 warm-up (excluded),
3 repetitions each of the short (~80 tokens), 4K (~4,093) and 16K (~16,387) prompts, then 12 sanity
checks; streaming over loopback, one request at a time, no retries. Prompt and decode speeds are the
engine's own `timings`; TTFT is the client's monotonic clock to the first non-empty delta. Output cap
512 tokens including reasoning — every 4K and 16K request reached it, so their totals are time-to-cap,
not time-to-answer.

The run followed a fresh reboot with the engine as the only workload. The engine log was checked before
the suite for `expert arena: large pages are resident` — the arena was on 2 MB pages. The model was
started by the machine's scheduled task, not by the benchmark, and had answered no request before.

Not measured: power draw, GPU clocks, PCIe link width actually negotiated (only the engine's bandwidth
probe was recorded).

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short, 128k ctx, IQ3_S | ~77 | 0 | ~358–378 | 3 | n/a (fixed cost dominates) | 48.7 (45.1–51.2) | 1.84 (1.73–1.89) |
| prompt-4k | 4,093 | 0 | 512 (cap) | 3 | 430.2 (430.0–430.7) | 44.1 (39.6–46.0) | 9.56 (9.55–9.58) |
| prompt-16k | 16,387 | 0 | 512 (cap) | 3 | 741.7 (738.9–742.4) | 43.5 (40.8–48.2) | 22.14 (22.12–22.22) |

Total request time (median): 9.2 s short, 21.1 s 4K, 33.8 s 16K. Sanity checks **12 of 12**
(4 instruction, 4 retrieval, 4 tool-call).

Per-run data: [results.json](results.json) (every request, raw usage and timings), [outputs.jsonl](outputs.jsonl).

### Same machine, 4 KB pages instead of 2 MB (model-switch state)

Before the reboot the same suite ran on the same settings except that the arena had fallen back to
4 KB pages (`expert arena: ... locked 40290 MiB ... large pages refused ... VirtualAlloc error 1450;
using 4 KB pages`) — the normal state right after switching models on this machine, because the freed
block is smaller than the new expert set.

| Case | 2 MB pages (this report) | 4 KB pages | Δ |
| --- | --- | --- | --- |
| short decode tok/s | 48.7 | 46.7 | −4% |
| prompt-4k prompt/decode | 430.2 / 44.1 | 429.8 / 46.3 | 0% / +5% |
| prompt-16k prompt/decode | 741.7 / 43.5 | 744.1 / 40.0 | 0% / −8% |
| TTFT 1.8k / 4k / 16k | 1.84 / 9.56 / 22.14 | 1.88 / 9.56 / 22.08 | ~0 |

All differences are inside the run-to-run spread (decode ranges span 40–48 tok/s at 16K), so on this
machine **the page size did not matter while the arena stayed full size**. The 3×-style collapse other
reports mention appears when the arena *shrinks* and experts start streaming from the SSD, which did not
happen here.

## Correctness and limitations

- Sanity checks passed 12/12 in both runs (instruction following, retrieval, tool calls). No answer-quality
  claim beyond that; speed alone says nothing about quality.
- **Nonconforming by the runner's definition:** no model file could be hashed (`--skip-model-hash`), because
  the weights are a custom native pack plus two GGUF shards; the quantization repository revision is unknown.
- The tool reported the GPU as "AMD Radeon(TM) Graphics (2.0 GiB dedicated)" — that is the *disabled*
  integrated adapter, not either card used for the run. Two separate upstream issues could follow from this
  report: the GPU name/VRAM probe picking the iGPU, and nothing else.
- Expert cache stays warm across requests and evolves; these are not cold-model measurements. Prompt-prefix
  reuse is the shared template prefix only (reused tokens 0 in the table).
- Every 4K/16K request hit the 512-token cap including reasoning, so decode medians are measured under a
  reasoning-heavy workload, not on short answers.
- 842 MiB / 810 MiB of VRAM were free with the model loaded — near the engine's 1,024 MiB warning
  threshold (#2065). A larger context or a second concurrent request could tip a card into paging.
- Single machine, single boot; the 4 KB comparison run is one run, not three.
