# Community benchmark: Tesla V100-PCIE-32GB, Turbo Brilliance abliterated Flash-Next IQ3_S at 262K (sm_70, CUDA 12)

Submitted 2026-10-26 by **lh02577**. Daily-driver production box; first report in
this library for the V100-PCIe + abliterated IQ3_S + native 262,144-token context
combination. Confirms `STRATA_SM70_TABLE=1` on a **PCIe** V100 with the **IQ3_S**
quant (so far only verified on SXM2 with UD-IQ4_XS), reports `--ple-io direct`
(ngram PLE table streamed from NVMe) on a 262K serve, and posts 12/12 needle recall
at fills up to **254,775 tokens** — the library's deepest recorded recall probe.

## Hardware

- **GPU:** NVIDIA Tesla V100-PCIE-32GB (sm_70, PCI bus 02:00), solo for inference;
  a Tesla P40 (sm_61, bus 03:00) sits in the same box serving an unrelated
  llama.cpp router and was idle during the runs. PCIe 3.0 (Broadwell-EP platform
  ceiling; this driver's `nvidia-smi -q -d PCI` exposes no Link Gen/Width fields
  on Windows — queried, empty)
- **CPU:** Intel Xeon E5-2696 v4 @ 2.20 GHz (22C/44T, Broadwell-EP, AVX2, **no AVX-512**), single socket
  — confirmed by the engine's own startup log, not just the spec sheet:
  `strata generate: this CPU has no AVX-512: the expert kernels run on AVX-2 (multi-token for the i-quant gate/up rows)`;
  the memory guard line reads `limit 127.9 GiB (source: meminfo)`
- **RAM:** 128 GB usable; 4x 32 GB Micron 36ASF4G72PZ-2G3A1 DDR4-2400 RDIMM, quad-channel
  (CPU0-DIMM1/3/6/8)
- **Storage:** drive letter **I: = WDC WDS240G2G0C-00AJM0, 240 GB NVMe** — holds the PLE/ngram
  table, read with `--ple-io direct` (verified NVMe, so the spinning-disk warning in
  Strata issue #605 does not apply); rest of the box: 3x SATA SSD + 2x Seagate HDD
- **OS:** Windows 11 Pro build 22631, driver 576.57, CUDA 12.9 runtime
- **Engine health confirmed at report time:** `{"status":"ok","max_context":262144,"model":"Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S","images":true,"loaded":true}`

## Software

- **Strata:** engine 0.1.48.1, **source build** at commit `7ddc17a38fe7f31637e895ecfba58c2cd43d65ec`
  (auto-detected by the bench runner — the engine is built from this source tree on this box,
  `build-cuda12/` + `engine-cuda12/`, exe timestamp 2026-10-25 00:13). The engine has no
  `--version` flag or `/version` endpoint; version comes from its own log line
  `strata generate: session is up (engine 0.1.48.1)`. Build options not exposed by the running
  server (CUDA 12 target, sm_70; recorded as null by the runner, noted here for completeness)
- **Machine-level env:** `STRATA_SM70_TABLE=1` (Volta decode kernel table; production use since 2026-10-24,
  measured +8.6% decode with the engine's own repeated measurement; output tokens bit-identical)
- Server config attached as [`strata-iq3_s.json`](strata-iq3_s.json) (api_key removed;
  install paths shortened: `<install>` = the Strata source tree, `<models>` = model/pack data)

## Model

- **Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S** (2 shards, ~78.1 GB total
  incl. the ngram PLE table shard) — **source repo: `DavidAU/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF`**
  (Turbo Brilliance 50x build; the 220K-char chat template carries DavidAU's REASON-mode system —
  this is what the engine loads, not the plain SC117 file), revision
  `b1b82c36770c826c41e1874f250228048cdb3f5c` @ main, verified via the HF API 2026-10-26
  (both shards + `mmproj-Qwen3.8-Flash-Next-BF16.gguf` confirmed at this revision)
- Model tree (upstream chain): `Qwen/Qwen3.8-Flash-Next` → `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`
  (GSQ-RCO quantization) → `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF` (abliteration)
  → DavidAU Turbo Brilliance 50x (tokenizer/chat-template transplant + AI Constructs)
- MTP draft: `mtp/rt` pack (`--mtp`), vision: `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (`--vision`, on-GPU)
- Expert profile `data/expert-profile.bin`, `--expert-cache auto`
- Shard SHA-256 (hashed by the runner during the run):
  `00001-of-00002` = `78706315c2fe912777c006b4ae18be1024c9f4353bd25d33b8a306c9c55f9e34`,
  `00002-of-00002` (ngram PLE) = `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`

## Settings (server args, verbatim)

```text
--pack packs/iq3_s --native <IQ3_S-00001-of-00002.gguf> --ple-gguf <IQ3_S-00002-of-00002.gguf>
--expert-profile expert-profile.bin --expert-cache auto --prefill auto
--mtp mtp/rt --max-context 262144 --kv int8 --kv-resident 32768
--vision --vram-reserve-mib 700 --pcie-frac 0.00 --spec-min-p 0.75
--adapt-async 1 --ple-io direct
```

Server json also sets: `reasoning_effort: none`, `reasoning_budget_tokens: 8192`,
`fit_max_tokens: true`, `max_tokens: 4096`. (A Hermes client sends `reasoning_effort:
high` per request for chat use; the bench below runs on the protocol defaults.)

## Measured before the protocol run (from my own optimization logs, labeled as such)

| Date | Engine | Setting | Decode tok/s |
|---|---|---|---|
| 2026-10-26 | 0.1.48.1 | `--ple-io ram` (ngram table in RAM) | 64.9 |
| 2026-10-26 | 0.1.48.1 | `--ple-io direct` (ngram table on NVMe, current) | **66.9** |

Both with `STRATA_SM70_TABLE=1`, 262K context, daily workload; single-request
decode rate, not the protocol prompt set. `--pcie-frac 0.00` because the box has
one GPU and the NVMe-direct PLE path; switching to `direct` also freed ~29 GB RAM.

## Protocol run (strata-qwen38-thinking-v1-draft.1, `standard-v1`)

Run `20261026T073044Z-b33207f9`, 2026-10-26, fresh engine load (the runner recorded
`server_requests_before_run: 0`), no other GPU work, single slot (`server_capacity: 1`).
Status: **complete**; required stages in 154 s after model hashing. Sampling exactly per
model card: temp 1.0 / top_p 0.95 / top_k 20 / min_p 0 / penalties off, thinking on
(`enable_thinking` + `preserve_thinking`, `reasoning_effort: xhigh`). The draft's frozen
prompt tiers are short (77 tok) / **4,096** / **16,384** (not 32K/128K — long-context
depth is covered by the recall stage and the needle appendix below).

Medians of 3 repeats, single concurrency:

| Stage | Prefill tok/s | Decode tok/s | TTFT |
|---|---|---|---|
| short (77 tok)  | 88.5   | **74.7** | 0.90 s |
| 4,096 tok       | 1,221  | **73.5** | 3.39 s |
| 16,384 tok      | **1,625** | 65.9  | 10.1 s |

Quality: sanity **12/12** (instruction 4/4, retrieval 4/4, tool-call 4/4).

Long-context appendix — `tools/needle_bench.py`, greedy, thinking off, code words
hidden at 10/50/90% depth (`needles-2026-10-26.json` in this folder): **12/12 FOUND**,
including 250K and 262K fills:

| Fill (target) | Real prompt tokens | Depth 10% | 50% | 90% |
|---|---|---|---|---|
| 32K  | ~31,643  | FOUND (22 s) | FOUND (20 s) | FOUND (20 s) |
| 128K | ~125,870 | FOUND (93 s) | FOUND (85 s) | FOUND (52 s) |
| 250K | ~242,114 | FOUND (202 s) | FOUND (146 s) | FOUND (145 s) |
| 262K | ~254,775 | FOUND (217 s) | FOUND (217 s) | FOUND (218 s) |

The 262K tier is the tool's own 2%-under shave of the native window (254,775 tokens
in + 40 out), reached with the KV cache long past its 32,768-token resident line and
streamed int8 to system RAM — recall does not degrade with depth at full window.
Note this appendix ran warm (right after the protocol run), unlike the cold-start
protocol numbers.

Field notes honest-as-written (from `results.json` `missing`): the already-running
server does not expose its engine options, so kv/expert-cache/offload/prefill/MTP
fields are null in the machine-readable record — the `--track-notes` string and the
Settings section above carry them instead. The 16K decode staying at 65.9 (vs 73.5
at 4K) with int8 KV resident only to 32,768 tokens is the point of this configuration:
nothing spilled, nothing halved.

Protocol run command (for reproduction):

```sh
strata bench run --url http://127.0.0.1:8080 \
  --model-revision b1b82c36770c826c41e1874f250228048cdb3f5c \
  --custom-pack "prepared pack packs/iq3_s + custom expert-profile.bin" \
  --track-notes "engine 0.1.48.1 (2026-10-25 build, sm_70), STRATA_SM70_TABLE=1, ple-io direct on NVMe, MTP rt, kv int8 resident 32768, spec 4 min-p 0.75, DavidAU Turbo Brilliance 50x abliterated IQ3_S (tree: Qwen -> ISTA GSQ-RCO -> SC117 abliterated -> DavidAU)"
```

## Notes for readers

- Daily-use box: this serve answers an agent stack (tool calls, image input) all day;
  the numbers here are the idle-server bench path, not concurrent load.
- 262K is the model's native window; no YaRN. KV int8 streamed to RAM after 32,768
  resident tokens — the deep-context decode stays flat here where several dual-V100
  NVLink reports halved (their KV ate the VRAM).
- `--ple-io direct` requires the ngram table on NVMe; on a spinning disk keep `ram`
  (see Strata issue #605).
