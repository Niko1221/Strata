# Community benchmark: RTX 5060 Ti 16 GB + RTX 3060 12 GB layer split, IQ3_S

Measured on 2026-10-11 by @mmbeanz. Speed suite plus expert-cache
telemetry; no correctness check in this run. **Headline:** on an unequal 16+12 GB pair with
unequal weak links, the tuned `--layer-split 40` holds 8,842 of 24,576 experts and measures
**85.9% median expert-cache hit rate** — between the planner's old power-law coverage curve
(97.0% predicted) and its new `H(f)` fit (73.8%). Details in *Expert-cache coverage* below.

## Hardware and software

Hand-written from `nvidia-smi`, `lspci -vv` **under load**, `dmidecode` and `nvcc` on the
host — do not use the auto-generated `machine` block (see *Tool findings*).

- GPUs: **RTX 5060 Ti 16,311 MiB** on PCIe **5.0 x8** (`LnkSta` under load; measured H2D
  **28.2-28.7 GB/s**) + **RTX 3060 12,288 MiB** on PCIe **3.0 x4** (`LnkSta`; measured H2D
  **3.15-3.2 GB/s**), the 3060 behind the 600-series chipset. Layer split across both,
  explicit `--layer-split 40`, `--trim-stage-weights`. The board cannot negotiate better
  widths; idle `lspci` shows gen1 (ASPM), so only loaded readings are meaningful.
- CPU: AMD Ryzen 9 7900X (12 cores / 24 threads). RAM: **64 GB DDR5-6400** dual-channel
  (61.9 GiB visible to the container). Storage: Samsung 990 EVO Plus 4 TB NVMe
  (**DRAM-less**, HMB); serial redacted.
- OS: Linux 7.2.9-1-cachyos, bare metal. The engine runs in a **systemd-nspawn** container
  (shared host kernel: real PCIe links, real CPU topology, no virtio artifacts).
- GPU driver 615.71.09; CUDA 13.4 (nvcc V13.4.92).
- Strata 0.1.43, engine built from upstream `7695859a` (tag v0.1.43); the install carries
  one local commit changing **only `setup.py` (pip -> uv)** — no engine code changes.
- Background workloads: none (dedicated inference box); no power-limit changes.

## Model and configuration

- Model: Qwen3.8-Flash-Next **IQ3_S**, ISTA-DASLab GSQ-RCO pack (sha256 in
  `sha256sums.txt` in the report root), 512K context via yarn factor 2.0.
- Expert profile: the install's `data/expert-profile.bin`, **no `--expert-profile-save`**
  (pinned boot profile, no mutation between runs). Engine log confirms
  `profile ...: 24576 ranked pairs, built for 24576 slots`.
- Engine facts at boot (`GET /metrics`): `expert_slots 8842` (17,910 MiB) across both
  cards, `expert_slots_primary 4984` (9,474 MiB), `arena_mib 47962`, `pool_workers 11`,
  `kv int8`, `kv_resident 20480`, `pcie_frac 0.00`, `spec_min_p 0.70`, `vram_free_mib 324`.

```text
strata --serve --pack /home/llm/Strata-data/packs/iq3_s \
  --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
  --expert-profile /home/llm/Strata/data/expert-profile.bin \
  --expert-cache auto --prefill auto --spec 4 --mtp rt \
  --max-context 524288 --rope-scaling yarn --rope-scale 2.0 \
  --kv int8 --kv-resident 20480 --vision --remote-expert-opt --trim-stage-weights \
  --pcie-frac 0.00 --spec-min-p 0.70 --layer-split 40
```

(`--layer-split 40` is appended by the server from the config's `layer_split` key.
`--remote-expert-opt` is present but **inert** with a layer split — it feeds the
helper-cache path only, and setup adds it automatically, #578.)

## Method

`python3 tools/strata_bench.py --url <server> --config <live config> --long --runs 3`
against a **server started by the operator** (cold boot, 0 requests before the suite;
`GET /metrics` snapshots in `A-before.json` / `A-after.json` / `A-after-full.json`).
Model loading is not timed. One warm-up chat and a 1K prompt are discarded first; every
fresh prompt carries a unique tag so the conversation cache cannot help it. Thinking off,
temperature 0.

The bench client ran on a **different machine** on the same LAN: RTT 0.25 ms, HTTPS TTFB
~7 ms, so TTFT inflation is negligible (<0.2% of the smallest TTFT, 0.40 s) — but see
*Tool findings* for what the tool records in that mode.

Suite contents and sizes are the tool's standard ones (short chat, 4K/16K/32K fresh
prompts with a 24-token cap, 6-turn agent sessions x3). Total suite time 205 s.

## Results

| Configuration | Prompt tokens | Reused | Generated | Runs | Prompt tok/s (median, range) | Decode tok/s (median, range) | TTFT s (median, range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 | 9 | n/a | 74.4 (73.2-92.9) | 0.40 (0.37-0.45) |
| 4K prompt, fresh | 4019 | 0 | 24 cap | 3 | 868.1 (866.5-874.6) | 70.7 (70.6-73.1)* | 4.66 (4.62-4.67) |
| 16K prompt, fresh | 16058 | 0 | 24 cap | 3 | 1412.7 (1412.1-1414.7) | 58.3 (56.4-60.3)* | 11.40 (11.38-11.40) |
| 32K prompt, fresh | 32049 | 0 | 24 cap | 3 | 1720.8 (1720.5-1721.7) | 53.2 (48.3-56.9)* | 18.66 (18.65-18.67) |
| agent follow-up turns | 14421 at last turn | see below | up to 64 | 3 | see below | 80.7 (53.1-93.6) | 4.28 (3.31-4.67) |

\* decode for the capped prompt tests is not in the tool's table (only 24 tokens); these
come from the engine's per-request `decode_tok_s` in `GET /metrics?requests=all`. A 24-token
window is short, so treat as indicative — the long-context tax (70.7 -> 53.2 tok/s from 4K
to 32K) is the point, not the decimals.

Draft acceptance over the whole suite: **992 / 1,439 = 69.0%**.

### Expert-cache coverage (the interesting part)

Per-request `hit_rate` from `/metrics?requests=all`, all 40 requests:
**median 0.859, mean 0.842, range 0.550-0.969.** By prompt size (median):

| prompt read | hit_rate median (range) | decode tok/s median |
| ---: | --- | ---: |
| 32,049 | 0.926 (0.806-0.958) | 53.2 |
| 16,058 | 0.944 (0.907-0.969) | 58.3 |
| 4,019 | 0.866 (0.704-0.922) | 70.7 |
| 609 | 0.812 (0.752-0.842) | 71.6 |

This config holds **8,842 of 24,576 experts (f = 0.360)**. The layer-split planner's
coverage curve predicts the routed mass those 8,842 capture:

| curve | predicted | 
| --- | ---: |
| old `(r+1)^-1.2` (still the default at <=3 GPUs) | **97.0%** |
| new `H(f) = 1-(1-f)^b`, b=3 (0.1.43, fit on a 4-way rig) | **73.8%** |
| **measured hit_rate (median)** | **85.9%** |

Neither curve is right on this box; **b = 4.4** would fit the measurement. For scale, the
same comparison for the `auto` K=3 placement (6,135 pairs, f=0.25) predicts 95.7% (old) vs
57.8% (new) — that placement's measured hit_rate is the next data point (run B1).

Also confirmed by telemetry, all 40 requests: `pcie_share = 0`, `ram_blobs = 0`,
`file_blobs = 0`, `file_mb = 0` — with `--pcie-frac 0.00` no expert is copied over PCIe and
nothing is read from the SSD in steady use; every miss is computed by the CPU pool from the
pinned arena. Hit rate rises with prompt length (0.81 at 609 tokens -> 0.93 at 32K),
consistent with the long prompt warming the cache before its decode phase.

### Reproducibility against the earlier on-box run

The same suite run the day before, executed **on** bertha, gave: 32K prompt 1721.8 vs **1720.8** here (0.06%), short-chat
decode 75.1 vs **74.4**, agent cache hit 68.8% vs **68.7%**. Remote measurement from a LAN
client is equivalent for these numbers.

## Tool findings (bugs, with the above as evidence)

1. **`gpu_slug()` names multi-GPU folders after the first card only**: `f"{len(names)}x-{s}"`
   with `s` from `names[0]` — on this box it produced `2x-rtx-5060-ti` and silently dropped
   the 3060. (Related to #2001 / #1985.) The remote run named it correctly,
   because with no local `nvidia-smi` it fell back to the server's combined `gpu_name`
   string.
2. **`collect_machine()` blends client and server fields when `--url` points at a remote
   server**: `gpu` name, `cpu`, `strata_commit` come from the server (correct), but
   `ram_gib` (recorded **3.8 GiB** — the client!), `os`, `gpu_driver` and `cuda`
   (null, no local `nvidia-smi`) come from the client. A remote-run report is silently
   wrong unless hand-patched.
3. The generated README references `engine-log-excerpt.txt` even when the tool did not
   start the server (no such file is written in `--url` mode).
4. `/metrics` returns only the last **12** requests by default; the full history needs
   `?requests=all`. Worth a line in `docs/COMMUNITY_BENCHMARKS.md`, which asks for
   per-request data but does not mention the flag.

## Correctness and limitations

- Speed only: no needle test, no task-quality check in this run (the needle suite: `../needles/`).
- Decode for the capped prompt tests is a 24-token window (see footnote).
- No concurrency row (`--clients` not used), no vision latency, no mmap/low-RAM tier,
  Windows/HIP not tested, single server config (the `auto` split variants are separate runs).
- `vram_free_mib 324` at boot is tight; the engine's stall line is 256 MiB — worth noting
  for anyone copying this config on a 16+12 GB pair.
