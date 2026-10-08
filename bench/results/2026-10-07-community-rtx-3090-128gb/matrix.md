# Benchmark matrix (v0.1.40.3) — RTX 3090 24 GB + 128 GB RAM, Ryzen 9 5900X

Three Strata models, same machine, GPU dedicated to Strata (ComfyUI + openclaw-gateway
stopped, the Wayland compositor kept running; server bound to loopback, zero-foreign
guard passed). Context 262,144,
INT8 KV, greedy, 256-token output cap, three runs per length. Prefill and decode
are engine tok/s medians; TTFT is client-side streaming median. Each cell is the
median [minimum–maximum] of three runs. IQ3_S was also rerun with elastic VRAM
reserving 10,240 MiB for Flux/image generation; ComfyUI stayed idle and the
openclaw-gateway was stopped.

## IQ3_S (dedicated)

| Context | Prompt tok/s | Decode tok/s | TTFT s |
|---:|---:|---:|---:|
| 4,096   | 2,034 [1,918–2,036] | 88.0 [83.5–95.4] | 2.04 [2.04–2.16] |
| 32,768  | 2,607 [2,595–2,608] | 92.7 [91.5–97.9] | 12.63 [12.63–12.70] |
| 128,000 | 2,448 [2,442–2,457] | 90.4 [87.6–96.6] | 52.48 [52.30–52.61] |

## IQ3_S with 10 GiB reserved (elastic)

Same IQ3_S model and harness after `POST /v1/vram {"reserve_mib":10240}`. Expert
cache: 2,870 slots / 5,632 MiB instead of 7,935 slots / 15.04 GiB. GPU peak
14,291 MiB; minimum free VRAM sample 10,285 MiB; engine PSS peak 53.11 GiB.

| Context | Prompt tok/s | Decode tok/s | TTFT s |
|---:|---:|---:|---:|
| 4,096   | 1,735 [1,650–1,735] | 50.5 [46.0–52.0] | 2.41 [2.40–2.52] |
| 32,768  | 2,516 [2,515–2,525] | 48.7 [48.5–51.3] | 13.11 [13.11–13.12] |
| 128,000 | 2,355 [2,346–2,360] | 51.9 [49.0–52.2] | 54.56 [54.46–54.76] |

Decode hit rate was 56.4–62.2%, with 4.4–5.3% of routed experts over PCIe. Needles:
6/6.

## UD-IQ4_XS (dedicated)

| Context | Prompt tok/s | Decode tok/s | TTFT s |
|---:|---:|---:|---:|
| 4,096   | 1,268 [970–1,278]   | 53.1 [39.8–53.9] | 3.27 [3.24–4.25] |
| 32,768  | 1,991 [1,987–2,049] | 54.3 [53.7–55.0] | 16.55 [16.08–16.57] |
| 128,000 | 1,962 [1,959–1,975] | 56.9 [56.2–57.0] | 65.45 [65.01–65.54] |

## UD-Q4_K_XL (dedicated, experimental)

| Context | Prompt tok/s | Decode tok/s | TTFT s |
|---:|---:|---:|---:|
| 4,096   | 979 [755–1,028]     | 38.4 [30.9–39.3] | 4.22 [4.03–5.47] |
| 32,768  | 1,768 [1,718–1,776] | 36.5 [36.3–38.2] | 18.62 [18.54–19.97] |
| 128,000 | 1,712 [1,703–1,725] | 37.5 [37.3–39.3] | 74.98 [74.42–75.37] |

The 4,096 minimums are the first run after load (cold expert cache); the medians
are the steady value.

## Side by side (median)

| Context | Prefill IQ3_S / IQ4_XS / Q4_K_XL | Decode IQ3_S / IQ4_XS / Q4_K_XL | TTFT IQ3_S / IQ4_XS / Q4_K_XL |
|---:|---:|---:|---:|
| 4,096   | 2,034 / 1,268 / 979   | 88.0 / 53.1 / 38.4 | 2.04 / 3.27 / 4.22 |
| 32,768  | 2,607 / 1,991 / 1,768 | 92.7 / 54.3 / 36.5 | 12.63 / 16.55 / 18.62 |
| 128,000 | 2,448 / 1,962 / 1,712 | 90.4 / 56.9 / 37.5 | 52.48 / 65.45 / 74.98 |

The files are ordered by size; no quality gain was measured. On this card decode is
roughly 88 → 54 → 37 tok/s and engine PSS 53 → 95 → 118 GiB.

### IQ3_S dedicated vs 10 GiB reserved (median)

| Context | Prefill dedicated / reserved | Decode dedicated / reserved | TTFT dedicated / reserved |
|---:|---:|---:|---:|
| 4,096   | 2,034 / 1,735 | 88.0 / 50.5 | 2.04 / 2.41 |
| 32,768  | 2,607 / 2,516 | 92.7 / 48.7 | 12.63 / 13.11 |
| 128,000 | 2,448 / 2,355 | 90.4 / 51.9 | 52.48 / 54.56 |

## Memory (engine PSS is the model's own footprint)

`monitor2.py` reads the engine process's `smaps_rollup` (PSS) plus `/proc/meminfo`.
PSS is the defensible per-model footprint; `MemTotal − MemAvailable` (what the
first run reported) discounts reclaimable file-backed pages and understates the
Unsloth models.

| Model | Engine PSS peak GiB | Pss_Anon | Pss_File | Shmem | MemTotal−MemAvailable | Cached |
|---|---:|---:|---:|---:|---:|---:|
| IQ3_S | 53.2 | 49.1 | 0.2 | 4.0 | 60.2 | 66.8 |
| UD-IQ4_XS | 95.4 | 3.1 | 46.3 | 46.1 | 55.2 | 115.3 |
| UD-Q4_K_XL | 118.4 | 3.2 | 53.4 | 62.7 | 72.1 | 116.9 |

IQ3_S holds its experts as anonymous RAM (49 GiB, non-reclaimable). The two
Unsloth models hold them as file-backed `mmap` of the GGUF plus a shared arena, so
`MemTotal − MemAvailable` hides the file-backed part and PSS is the real figure.
All three fit 128 GB; UD-Q4_K_XL is the tightest (PSS 118 GiB, swap peaked 620 MiB
during the model swap). The IQ3_S 10 GiB elastic run peaked at 53.11 GiB engine PSS
and 14,291 MiB GPU used; its PSS breakdown was not sampled separately.

## Expert cache at startup

- **IQ3_S:** `auto` cache 7,935 slots / 15.04 GiB VRAM; decode hit 82.8–90.3%,
  0.5–1.1% of routed experts over PCIe.
- **UD-IQ4_XS:** `auto` cache 6,362 experts / 14.34 GiB VRAM; `--resident-budget-gib 55`
  (the full 59.5 GB expert set, in GiB); decode hit 83.3–89.6%, 6.6–10.4% over
  PCIe; no experts read from the SSD (`file_blobs` 0).
- **UD-Q4_K_XL:** `auto` cache 4,844 experts / 14.13 GiB VRAM; `--resident-budget-gib 71`
  (the full 71.7 GiB expert set); decode hit 77.2–84.8%, 10.2–15.0% over PCIe;
  no experts read from the SSD (`file_blobs` 0).

## Correctness

| Model | Needles 32k / 128k, depths 10 / 50 / 90 | Result |
|---|---|---|
| IQ3_S | 6 checks | 6/6 found |
| IQ3_S with 10 GiB reserved | 6 checks | 6/6 found |
| UD-IQ4_XS | 6 checks | 6/6 found |
| UD-Q4_K_XL | 6 checks | 6/6 found |

## Qwen3.8-27B under llama.cpp (not Strata)

Separate club-3090 check, not a controlled comparison: dense Qwen3.8-27B
UD-IQ4_XS, llama.cpp, 262,144 ctx, q4_0 KV, MTP n=2, vision, one slot, loopback.
Prefill/decode are llama.cpp server `timings`; TTFT is client-side streaming.

| Context | Prompt tok/s | Decode tok/s | TTFT s |
|---:|---:|---:|---:|
| 4,096   | 1,227 [1,216–1,228] | 75.0 [74.1–76.0] | 3.58 [3.52–3.59] |
| 32,768  | 1,128 [1,127–1,128] | 61.7 [60.6–65.5] | 29.77 [29.31–29.84] |
| 128,000 | 781 [780–781]       | 41.1 [40.7–41.6] | 166.47 [164.65–166.57] |

![IQ3_S versus llama.cpp Qwen3.8-27B speed](charts/qwen38-27b-vs-iq3s-speed.svg)

Needles: 6/6. GPU peak 22,333 MiB; container memory peak 10.21 GiB; MTP draft
acceptance 1,366/1,847 = 0.740.

Against IQ3_S on the same card, decode is **88.0 / 92.7 / 90.4** vs **75.0 / 61.7 /
41.1** tok/s at 4K/32K/128K. Quality was not measured here, but published full-model
rows favor Flash-Next, and published IQ3_S tracks its BF16 base (93.26 vs 93.12 task
average).
