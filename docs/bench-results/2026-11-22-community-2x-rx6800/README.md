# Community benchmark on 2x AMD Radeon RX 6800 (gfx1030)

Measured on 2026-11-22 by sudoagent1. Tested decode throughput, 67.7K-token fresh prefill, and 68.3K-token cached follow-up turns on Qwen3.8-Flash-Next IQ3_XXS at 256K context in rootless podman on Linux (`0.1.64-linux-x86_64-hip-rx6000`).

**Main limitations:**
- Nonstandard protocol (predates / does not run `strata bench run` `standard-v1`); uses a fixed 80-token decode prompt (5 runs x 256 tokens) and a custom ~67.7K-token synthetic prompt/follow-up harness (`bench_prompt.py`) rather than the 12-case sanity suite or `strata bench run`.
- Asymmetric PCIe topology on the test host: GPU 1 (`main`) is on PCIe 4.0 x16 (28.1-28.2 GB/s H2D probe), while GPU 0 (`helper`) is on PCIe 4.0 x4 (6.4 GB/s H2D probe).
- `bench_decode.py` computes decode tok/s from client wall-clock duration (`completion_tokens / dt`) with `temperature=0`, `max_tokens=256`, thinking on by default.

## Hardware and software

- **GPU and VRAM:** 2x AMD Radeon RX 6800 16 GiB (`gfx1030`, 32 GiB total dedicated VRAM; `GPU_ROLES=1=main,0=helper`; GPU 1 PCIe 4.0 x16 @ 28.2 GB/s, GPU 0 PCIe 4.0 x4 @ 6.4 GB/s)
- **CPU / RAM:** AMD Ryzen 9 5900X (12 cores), 62 GiB DDR4 RAM (hugepages unavailable; 39 GiB expert arena via file-cache read + `cudaHostRegister`)
- **OS / Runtime:** Ubuntu 24.04, rootless podman 4.9.3 (`group_add: keep-groups`, `ulimits.memlock: -1`)
- **Strata / Engine:** Strata 0.2.18 (`rx6000` HIP package), engine `0.1.64` (`0.1.64-linux-x86_64-hip-rx6000`, `build_id: f609f862fa9e+rocm`, release binary, `TUNE=skip`)
- **Background workloads:** Host media/VPN containers idle; dedicated inference runs, one arm at a time

## Model and configuration

- **Model:** `Qwen3.8-Flash-Next` ISTA-DASLab `IQ3_XXS` (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` + `00002-of-00002.gguf`), native pack `iq3_xxs`
- **Drafting:** DFlash2 (`qwen3.8-flash-next-dflash2.q4_0-selq4.pack`, `--draft-fallback none --spec 6 --draft-conf 0.35`)
- **Vision:** CPU vision tower (`--vision`, `qwen3vl_merger`, 27 layers, 12 threads)
- **Context / KV / Split:** `--max-context 262144`, `--kv int8 --kv-resident 32768`, `--expert-cache auto --prefill auto --layer-split auto`, `GPU_ROLES=1=main,0=helper`
- **Tuned env switches (winner-of-record):** `STRATA_DECODE_TIMING=1 STRATA_HC_PERSIST=1 STRATA_SPLIT_OVERLAP=1`

```text
/opt/strata/components/engine/0.1.64-linux-x86_64-hip-rx6000/strata \
  --serve --pack /data/Strata-data/packs/iq3_xxs \
  --native /data/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /data/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --expert-profile /opt/strata/terminal/data/expert-profile.bin \
  --expert-cache auto --prefill auto --max-context 262144 \
  --kv int8 --kv-resident 32768 --vision \
  --draft dflash2 --draft-model /data/Strata-data/dflash2/qwen3.8-flash-next-dflash2.q4_0-selq4.pack \
  --draft-fallback none --mtp-draft-vocab /data/Strata-data/dflash2/draft_vocab.bin \
  --spec 6 --draft-conf 0.35 --layer-split auto
```

## Method

- **Decode (`bench_decode.py`):** 5 sequential requests (`temperature=0`, `max_tokens=256`) on an ~80-token prompt, keeping run 1 in the median (cache warm-up included). Post-load cold-start passes showed a run-1/run-2 transient dip and were followed by a steady-state confirmation pass; both are reported.
- **Long-prompt & cached follow-up (`bench_prompt.py`):**
  - `P1` (fresh prefill): 5 fresh conversations with disjoint `[case NNN]` IDs at token 0 (~67.7K prompt tokens each, `0 reused` verified in engine logs), reporting engine `timings.prompt_ms` and `read_tokens / prompt_s`.
  - `P2` (cached follow-up): continuation requests on the same ~67.7K prefix with a 268-395-token tail (~68.3K total prompt tokens), verifying prefix reuse (`usage.prompt_tokens_details.cached_tokens` / `timings.cache_n`) and mid-payload needle recall.
- Engine-side per-window decode breakdown was captured via `STRATA_DECODE_TIMING=1`.

## Results

### Decode throughput (80-token prompt, 5 x 256 tokens)

| Configuration | Prompt tokens | Generated tokens | Runs | Decode tok/s median and range | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| Baseline (no `HC_PERSIST`, no `SPLIT_OVERLAP`) | ~80 | 256 | 5 | **33.8** (33.3-35.2) | T0-c control; original baseline 34.2 (30.2-35.7) |
| + `STRATA_HC_PERSIST=1` | ~80 | 256 | 5 | **35.1** (30.1-37.8) | +3.8% vs control; GPU-reach wait 29-40 -> ~20.8 ms |
| + `STRATA_HC_PERSIST=1` + `STRATA_SPLIT_OVERLAP=1` (E5 confirm) | ~80 | 256 | 5 | **43.0** (32.9-44.5) | +27.2% vs 33.8 control; window 56.6 -> ~47.5 ms |
| Winner-of-record (final confirm, continuation ladder) | ~80 | 256 | 5 | **42.3** (42.0-42.8) | Steady-state after post-load pass (41.8); LP final confirm 43.0 (34.3-44.2) |
| Model trade-down reference: `IQ2_XS` (same winner flags) | ~80 | 256 | 5 | **45.9** (36.4-47.2) | +6.7% vs 43.0; 3/3 spot gates pass; not adopted |
| Model trade-down reference: `Q2_0` (same winner flags) | ~80 | 256 | 5 | **44.8** (36.1-45.4) | +4.2% vs 43.0; 3/3 spot gates pass; not adopted |

### Long-prompt prefill and cached follow-up (winner-of-record, IQ3_XXS @ 256K)

| Phase | Actual prompt tokens | Reused tokens | Runs | Prompt ms median and range | Prompt tok/s | Notes |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `P1` fresh prefill (LP control) | ~67,700 | 0 | 5 | **48,891 ms** (48,721-49,190) | 1,380-1,394 | 9 x 8192 auto chunks; LP final confirm 49,576 ms (48,866-49,877) |
| `P2` cached follow-up (LP control) | 68,341 | 65,536 (95.9%) | 4 | **4,516 ms** (first follow-up; chain 4,514-5,039) | — | ~10.8x faster than fresh; needle recall 4/4 HIT (final confirm 2/2 HIT) |

### Notable null / blocked / reverted findings on this stack

- **Role flip (`GPU_ROLES=1=helper,0=main`):** regressed decode to **30.9 tok/s** (14.5-35.4); VRAM expert hits dropped 35-55 -> 23-33/layer-window. Reverted.
- **Env/run-config arms tested and reverted (null or regressed):** `STRATA_PCIE_AUTO_SPLIT=1` (41.6), `STRATA_SH_STREAM=1` (41.3), `STRATA_SELECT_SIMT=1` (42.7 short / 49,563 ms P1), `STRATA_PLE_AUTO_MMAP=1` (41.6 short / 49,468 ms P1; direct read probe already ~90-98k rows/s), `STRATA_PREFILL_PIPE=3072` (41.1 short / 49,487 ms P1), `--spec 4` (40.9), `--prompt-cache-spread` (42.0), `--conversation-cache-rewind-keep 2048` (P2 4,874 ms worse, decode 39.7-40.5), `KV=k8v4` (40.5 tok/s, -4.5%).
- **`TUNE=thorough` (E16):** completed in 31.6 min (11/23 planned runs) with `default_confirmed` ("already fastest", zero settings adopted; decode 42.3 tok/s). Restored `TUNE=skip`.

## Correctness and limitations

- All runs had zero window-timeout errors and zero `oomd` events.
- Under `P2` cached follow-ups on the winner-of-record config, mid-payload needle recall passed **4/4** in LP control and **2/2** in LP final confirm.
- For `IQ2_XS` and `Q2_0`, 3/3 spot gates passed (12,129-token needle recall, `is_balanced` code task, and CPU-tower vision check).
- `bench_decode.py`, `bench_prompt.py` and `strata_quality_gates.py` (the 3-gate battery: ~12K needle recall, bracket-balance coding task, in-memory PNG vision check) are included in this directory for reproducibility (`STRATA_URL` / `STRATA_API_KEY` env-configurable, no credentials included).
