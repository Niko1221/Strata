# Community benchmark on NVIDIA GeForce GTX 1080 Ti

Measured on 2026-10-15. A first Pascal **GTX 10 series** (sm_61) report — the earlier sm_61 answers in
[TEST_REQUESTS.md](../../../docs/TEST_REQUESTS.md) are Tesla P40s, and a consumer card has a different memory
subsystem (GDDR5X, higher clocks) on the same fp32 compute path. One model, one context, the standard
`tools/strata_bench.py` suite; the main limitation is that the server restarted part-way through and the 32K
prompt tests were never measured.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce GTX 1080 Ti (GP102, sm_61), 11264 MiB GDDR5X; CPU: 2x Intel Xeon E5-2696 v4
  (22 cores / 44 threads each, 44 cores / 88 threads total, Broadwell-EP, **no AVX-512** - the engine runs its
  expert kernels on AVX-2 and says so at start); installed RAM: 125.8 GiB; storage: NVMe SSD
  (model and pack files on `nvme0n1`, 466 GB)
- OS: Linux 7.2.9-2-cachyos (CachyOS); GPU driver: 580.178.04; CUDA: 13.0
- Strata: **0.1.44.1 official release engine**, the installed server (`serve/server.py --engine strata`) - a
  release binary, not a source build; the benchmark script ran from a source checkout of the same version
- **PCIe link: Gen3 x8** (the card negotiates x8 here although it supports x16 - nvidia-smi reports
  `pcie.link.gen.max 3, pcie.link.width.max 16` and `current 3 / 8`). GPU power limit 250 W. Background
  workloads: the machine was otherwise idle for the measured runs; power limits other than the stock 250 W: none.

## Model and configuration

- Model: Qwen3.8-Flash-Next, **GSQ-RCO IQ3_S**, the native expert pack `packs/iq3_s` with the two GGUF shards
  (`...-00001-of-00002.gguf` as `--native`, `...-00002-of-00002.gguf` as `--ple-gguf`); no vision encoder
- Custom packs or profiles: the pack's own `expert-profile.bin`, `--expert-cache auto`
- Context: **262,144 tokens** (the model's native window); KV type: **int8 with KV streaming**,
  `--kv-resident 32768` (3.09 GiB of the K/V kept in pinned RAM, 32,768 cells per QSA layer in VRAM); cache:
  `--conversation-cache-mib 8192`, `--conversation-cache-min-tokens 4096`; prefill: `--prefill auto`
- MTP: `--mtp` with the pack's draft layer, `--spec 4 --spec-min-p 0.5`; reasoning: the suite turns thinking off;
  sampling: temperature 0 (greedy); calibration: none; experimental speed projection: not enabled

```text
serve/server.py --engine strata --config strata-iq3_s.json --port 8080
  --pack /.../packs/iq3_s --native /.../IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
  --ple-gguf /.../IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
  --expert-profile /.../data/expert-profile.bin --expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5 --mtp /.../mtp/rt --max-context 262144
  --kv int8 --kv-resident 32768 --conversation-cache-mib 8192 --conversation-cache-min-tokens 4096
  --vram-reserve-mib 1024 --prompt-cache-tail --prompt-cache-every 4096
```

## Method

`python tools/strata_bench.py --long --runs 3` against the HTTP API of the already-running server, so **model
loading is not timed**. One warm-up chat and a 1K prompt are discarded first; every prompt carries a unique tag so
the fresh-prompt tests cannot be helped by the conversation cache. Speeds are the engine's own timings; first-token
times are measured by the client over HTTP. Total suite time 676 s. Per-request data in
[`results.json`](results.json).

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 | 9 | not measured | 27.4 (23.1-31.9) | 1.28 (1.08-1.32) |
| 4K prompt, fresh | 4,019 | 0 | 24 cap | 3 | **202.4 (201.1-202.8)** | not measured | 19.92 (19.88-20.05) |
| 16K prompt, fresh | 16,058 | 0 | 24 cap | 3 | **194.6 (193.5-196.7)** | not measured | 82.59 (81.70-83.05) |
| 32K prompt, fresh | not measured | not measured | 24 cap | 0 | not measured | not measured | not measured |
| agent session, follow-up turns | 13,070 (11,590-14,551) at the last turn | see below | up to 64 | 2 | see below | 23.6 (19.1-24.5) | 25.60 (12.78-29.16) |

Prompt throughput is flat from 4K to 16K on this card (~202 -> ~195 tok/s), and the two fresh-prompt tests are
reproducible to better than 1% across their three runs.

### Agent session (conversation cache)

Cache hit over the follow-up turns **66.0 % (63.2-68.8)**; earlier-turn tokens read again per session **3 (0-5)** -
the conversation cache holds almost all of the resent history, and the 10,808 (0-13,707) new tokens of the project
files are read once each. First turn (nothing cached yet): first token 7.41 s (7.35-7.46).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 82.4 | 7.461 | 19.9 |
| 1 | 3,756 | 633 | 3,123 | 115.0 | 27.235 | 23.2 |
| 2 | 5,289 | 3,819 | 1,470 | 115.7 | 12.777 | 23.6 |
| 3 | 8,710 | 5,352 | 3,358 | 115.8 | 29.065 | 24.0 |
| 4 | 11,625 | 8,763 | 2,862 | 114.3 | 25.114 | 24.5 |
| 5 | 14,551 | 11,652 | 2,899 | 113.6 | 25.596 | 19.1 |

Resent-history prompt tokens run at ~115 tok/s here - roughly 55% of the fresh-prompt rate, since the reused part
costs no read but the checkpoint restore and the new tail do.

### Failed and skipped

Five requests failed and **the 32K prompt test produced no numbers**:

- one agent session: no token came back
- one agent session and all three 32K prompts: `Connection refused` - the server had gone down

The engine log shows healthy requests up to the last successful one and then stops; the guard script
(`strata-guard.sh`) restarted the server, which is why the suite's remaining requests saw connection refused.
The most likely cause is memory pressure on an 11 GiB card: the agent sessions park their conversation snapshots
(the log's last parking line reports `bytes=3911781620`, 3.9 GiB, against the configured 8 GiB budget) and the 32K
fresh prompts then need their own large buffers on top. Not proven - no crash line was written to the engine log -
but the failure is reproducible in shape if not yet in cause, and it is the first thing an 11 GiB Pascal owner
should watch. The 32K prompt numbers on this card remain **not measured**.

## Correctness and limitations

No correctness check was run - this is a speed suite only (no needle test, no coding task). The conversation test
sends the model's own earlier replies back, so token counts differ slightly between runs. Machine facts come from
`nvidia-smi`, `lscpu` and the OS; other than the idle claim above, background load was not recorded and no thermal
or clock capture was taken. The engine used is the **release binary of 0.1.44.1**, so these numbers describe what a
user installing today gets on this hardware - not a source build with local patches. What was not tested: 32K
prompts (failed, above), contexts past 32K, `--batch` concurrency, the layer-split path, KV formats other than
`int8`, and any quality measurement.
