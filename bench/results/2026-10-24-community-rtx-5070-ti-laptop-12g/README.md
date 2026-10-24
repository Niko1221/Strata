# Community benchmark: RTX 5070 Ti Laptop 12 GB, Flash-Next IQ3_XXS

Measured 2026-10-24 by `kxyct` (report written by the machine's resident local AI assistant, on the owner's PC).
Covers one 12 GB laptop GPU with 64 GB system RAM at `vram-reserve 1000` and at `1536`, 2026-10-20.
Main limits: no 128K-tier run this day, no TTFT stream measurement, no recall check yet (needle_bench queued after submission).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5070 Ti **Laptop**, 12,227 MiB VRAM (nvidia-smi); CPU: AMD Ryzen 9 9955HX (16 cores / 32 threads); RAM: 2 x 32 GB DDR5-5600; storage: 2 x YMTC PC411 1 TB NVMe SSD; PCIe link speed/width: not measured; GPU power limit: not reported (`nvidia-smi` shows `[N/A]`, laptop).
- OS: Windows 11 (build 26300); NVIDIA driver 616.64; engine built for CUDA 13.0 (bundled cu13 DLLs shipped in the release folder).
- Strata engine 0.1.46 portable Windows release binary: commit `a7bdbfdabc9a`, source tag `f-021-rel`, archs sm_75/86/89/120, CUDA 13.0 (see `BUILD.json`, `engine-component.json` copied here from `D:\Strata\Strata-main\engine\`). Engine exe LastWrite 2026-10-20 21:51.

## Model and configuration

- Model repository: `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` on Hugging Face (downloaded 2026-10-18 via the hf-mirror.com endpoint; repository commit not recorded), quantization IQ3_XXS, two GGUF shards: `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf` (native, 41.57 GB) + `...-00002-of-00002.gguf` (PLE, 37.x GB). Shown to be abliterated on this machine: a refusal-probe set went from 66.7% refusals (original upstream GGUF) to 1.7% (≈0); capability spot-check 40/40/40 unchanged.
- Custom pack: expert pack generated locally with `tools/iq_pack.py` from the native GGUF (`packs/sc117_iq3_xxs`); dense.bin byte-identical to the stock pack. `--expert-profile data/expert-profile.bin` from the earlier stock IQ3_XXS pack (engine reports 24576 ranked pairs, accepted).
- MTP: `--mtp D:\strata\Strata-data\mtp\rt` (release default draft). Vision: `--vision` with `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (server starts `strata-vision`; image input not exercised by these benchmarks).
- `STRATA_ROUTE_RESIDENT=1` is set in the engine process environment (opt-in, enabled 2026-10-17 on 0.1.45; measured there at +20-23% decode and hit-rate 65% → 79-80% on the same machine, kept on 0.1.46 — **not** separately re-measured A/B on 0.1.46).

Launch command (as recorded in the engine's own startup log, 2026-10-24 19:19:29):

```text
D:\Strata\Strata-main\engine\strata.exe --serve \
  --pack D:\strata\Strata-data\packs\sc117_iq3_xxs \
  --native D:\Strata\Strata-data\models\sc117_iq3_xxs\Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf D:\Strata\Strata-data\models\sc117_iq3_xxs\Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf \
  --expert-profile D:\strata\Strata-main\data\expert-profile.bin \
  --expert-cache auto --prefill auto --spec 4 \
  --mtp D:\strata\Strata-data\mtp\rt \
  --max-context 262144 --kv int8 --vision \
  --vram-reserve-mib 1000 --spec-min-p 0.70 \
  --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --kv-resident 32768
```

Startup log (same run): `expert cache auto: 4.94 GiB free, 1000 MiB reserved (+184 MiB for the draft head) -> 1745 slots`.

## Method

- Script: `scripts/measure_parallel.py` (included; no third-party deps). Synthetic nonce-prefixed prompts at 300 / 4,096 / 32,768 target tokens, single request at a time (`--concurrent 1`), 256-token output cap, 3 runs per tier, run 2026-10-24 against the localhost OpenAI-compatible endpoint of the engine above.
- Prompt and decode throughput are read from the **engine's own timing lines** captured per request (kept verbatim in the JSON); client wall-time is also recorded separately. The script's follow-up rounds reuse the prefix: run 1 at each tier is a fresh read; later rounds at 4K/32K show `reused` counts and their small fresh read inflates the fresh-read-only t/s — the JSONs list every run, and the table reports fresh-read runs for prompt and all decode runs.
- Expert cache: long warm (this engine is the machine's daily driver; `--adapt-every 1` continuously re-ranks). Conversation state: the chat session that runs the benchmark is itself served by this engine, but engine-side timing excludes queueing; during the final clean runs nothing else issued requests.
- Memory: VRAM free at request start is logged by the engine; see Limitations for the 455 MiB warning. RAM: not measured this session.

## Results (engine 0.1.46, 2026-10-24, `vram-reserve-mib 1000`, 1745 slots)

| Tier | Prompt tokens | Reused | Generated | Runs | Prompt tok/s (fresh) | Decode tok/s median (range) | Draft accept |
|---|---:|---:|---:|---:|---:|---|---|
| short | 396 | 0 | 256 | 2 | 376.4 / 426.6 | 92.5 (range: 92.5-96.2, 2 fresh runs) | 185/191, 167/181 |
| 4K | 4,187 | 0 | 256 | 2 | 893.1 / 901.3 | 90.2 (86.1-94.2) | 170/180, 160/170 |
| 32K | 32,857 | 0 | 256 | 1 | 978.0 (single fresh run) | 97.2 (single fresh run) | 179/187 |

Prefix-reuse follow-up runs (from the same sessions, recorded in the JSONs): 4K `4333 = 4182 reused + 151 read` at 248.7 tok/s fresh-read, decode 109.9; 32K `~33,003 = ~32,985 reused + 18 read` at ~125 tok/s fresh-read, decode 102.9-105.8, drafts 84/84.

Client total latency (includes engine-side queueing between runs): short 3.7-4.1 s, 4K 7.4-8.3 s, 32K 36.3-36.8 s per request.

Earlier session at the same day's idle state (same engine, same command): decode 94-100 t/s, drafts 92-100% (see `runs-*` session 1 rounds 1-3: short 81.9-96.2, 4K 86.1-97.5, 32K 80.0-97.2) — engine-side numbers agree with the clean table above; the first session's client wall-times of 134-161 s at short/4K were interactive-chat queueing on the single-slot engine, not slowdown (engine timing lines in the same JSONs are normal). Per-run JSONs: `runs-short-20261024.json`, `runs-4k-20261024.json`, `runs-32k-20261024.json`.

## VRAM headroom ladder on this 12 GB card (same engine, measured 2026-10-20 to 2026-10-24)

- `reserve 1536`: full expert load with **1123 MiB** free (> the engine's 1024 MiB stall threshold, engine docs issue #2065), at ~1197 resident slots (measured 2026-10-20 22:29, startup log).
- `reserve 1000` (current): 1745 slots; full load with **875 MiB free once slots are written** (startup) and `455 MiB free at request start` (engine WARNING during load, recommends `--vram-reserve-mib 1600` or lower max-context / batch at this depth). 455 MiB < 1024 MiB threshold = in the zone the engine says can page GPU memory and drop decode to a few tok/s; observed decode this session stayed 80-110 t/s, so paging had not bitten at these workloads.
- Earlier (engine 0.1.44.1/0.1.45, same card): `700` reported stalling-prone (~150 MiB free + shrink warnings); `1061` was the 0.1.45-day stable floor. On a 12 GB card this model runs with a 4.94 GiB expert cache: the 64 GB RAM only bounds the mmap side, the 12 GB VRAM is the binding constraint.

## Correctness and limitations

- Recall/needle: **not measured yet** on this configuration at submission time; `scripts/needle_bench.py` (upstream tool, copied) will run next and results will be added to this report/PR thread. This machine's own 32K-context prefix-recall spot checks pass in daily use, not a substitute.
- No 128K-tier run today (the 262K config can hold it but the first fresh 128K prefill would take ~2 min; not attempted); no TTFT stream measurement; no streaming; output capped at 256 tokens (all runs finished `length`, i.e. hit the cap; the small 110-token follow-up runs finished naturally).
- Vision encoder loaded but never invoked in these runs. The model is a third-party abliterated GGUF; answer content may differ from upstream's original GGUF.
- The 12 GB card is roughly 4 GB below the VRAM class this model is normally run at; the practical findings are the headroom ladder above and the warning that the engine itself prints.

## 中文摘要（给报告作者的机主看的）

本机是 12G 显存的 5070 Ti 笔记本，模型是 orcarouter 的无审核 IQ3_XXS 版。今天的实测：短提示词解码 ~92-96、4K 提示词 ~90、32K ~97（引擎自身计时，草稿命中率 92-100%）。这台机器上最值钱的结论是**显存留量梯子**：reserve 1536 时全装完剩 1123 MiB（安全）、1000 时剩 1745 槽但请求开始只剩 455 MiB（引擎自己报警建议 1600）。引擎在 12G 上的瓶颈是显存不是内存。needle 检查还没跑，跑完补在这条 thread 里。
