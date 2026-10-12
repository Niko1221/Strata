# 2026-10-13: long-prompt prefill degradation traced to conversation-cache parking
### 4x Tesla V100-SXM2-16GB, fresh cold 81K prompts, Strata 0.1.43.1

Follow-up to `bench/results/2026-10-12-community-4x-v100-16gb`. That report noted a
long-context quirk it could not root-cause: *"81K traffic could leave the host in a
degraded memory state that survived engine restarts and cleared only on reboot"*.
It is now root-caused. The cause is not the host and not the layer split — it is
**`--conversation-cache-mib` / `--conversation-cache-slots` parking every long request**.

With the cache off, 81K prefill is stable at **3092–3098 tok/s (+10.5%)** instead of
dropping to **~2801** from the second request on. Nothing else was changed.

## Hardware and software

Same box as the 2026-10-12 run: 4x Tesla V100-SXM2-16GB (sm_70), 2x Xeon E5-2696 v4,
247 GiB RAM, asymmetric PCIe x8/x16/x8/x16. Ubuntu 26.04.1, driver 580.178.04, CUDA 12.9.

One change: the engine is now **0.1.43.1 source build** (was 0.1.42), still built with
`-DSTRATA_MMQ_KQUANTS=ON` and `-DCMAKE_CUDA_ARCHITECTURES=70`.

## The observed failure

Every 81K request after the first collapses ~10%, and it survives restarting the
service **and rebooting the whole machine**. A full reboot with freshly-cleared RAM
reproduces it exactly:

| Run order | 81K prefill tok/s | 81K decode tok/s |
| ---: | ---: | ---: |
| 1st | **3089.9** | 31.1 |
| 2nd | 2754.2 | 77.1 |
| 3rd | 2806.7 | 85.7 |
| 4th | 2795.3 | 71.0 |
| **median** | **2801.0** | 74.0 |

So it is not host memory, not thermal, not the disk: it is triggered by the *first
long request itself*, inside the engine process.

## Cause

Every 81K request ends with the engine parking its KV snapshot into the conversation
cache. Verbatim log lines (see `engine-log-excerpt.txt`):

```text
strata serve: conversation cache: parked 81077 tokens in 2841.2 ms; parked=2 bytes=5115104880 evictions=0 snapshot_bytes=2556971192 reused_kv_bytes=0
strata serve: conversation cache: parked 81276 tokens in 2967.0 ms; parked=3 bytes=7675080312 evictions=0 snapshot_bytes=2559975432 reused_kv_bytes=0
strata serve: conversation cache: parked 81440 tokens in 2752.3 ms; parked=4 bytes=10232235136 evictions=0 snapshot_bytes=2562483992 reused_kv_bytes=0
```

- **2.558 GiB parked per request**, taking 2.6–3.2 s each time.
- `parked` climbs 1,2,3,...,6 — the parked set reaches 6 x 2.558 GiB = **15.35 GiB,
  exactly filling the 16 GiB budget**.
- `reused_kv_bytes = 0` on every line. `evictions = 0` on every line.

`reused_kv_bytes = 0` is the whole point: these benchmarks use a **fresh random prompt
every run**, so a parked conversation is never read back. The engine pays 2.558 GiB of
parking work per request and gets nothing — the official default
(`--conversation-cache-mib` **0 = off**) is what a fresh-prompt workload wants.

## Fix and measurement

Set `--conversation-cache-mib 0 --conversation-cache-slots 0`, restart, re-run. Two
independent rounds, same seeds/method as the 2026-10-12 report:

| Config | 81K prefill, run 1..4 | median | 81K decode median |
| --- | --- | ---: | ---: |
| cache ON (rebooted) | 3089.9 / 2754.2 / 2806.7 / 2795.3 | 2801.0 | 74.0 |
| cache OFF, round A | 3083.5 / 3079.9 / 3107.6 / 3101.2 | **3092.3** | 74.8 |
| cache OFF, round B | 3108.6 / 3094.2 / 3099.9 / 3095.2 | **3097.6** | **83.2** |

**+10.5% on 81K prefill, and the second-request collapse is gone** — all 8 runs sit in
3080–3108. 2K is unaffected (prefill 469.4 median over 6 runs, decode 136.9).

Exclusions, so the effect is not mis-attributed:

- **KV streaming hit rate is not the cause.** Each request prints two "KV streaming"
  lines (a cumulative one and a per-request prefill one); the per-request figure is
  ~69-71% with the cache **on or off**. `engine-log-excerpt.txt` §D.
- **Expert cache is not the cause.** decode expert-cache hit rate stays 97.9-99.9%.
- **layer-split 9,25,34 is unchanged** and remains the bandwidth-tuned optimum; the
  2026-10-12 sweep still stands.

## Scope — and why this is not "the cache is bad"

This says nothing against the feature. It is about fresh prompts. The engine-side
purpose of the conversation cache is to reuse a *continuing* conversation's prefix, and
in that role it works: issue #2181 measures a real agent session at 68.7% cache hit
over follow-up turns (and its method notes that a fresh-prompt test cannot be helped by
it, since every prompt starts with a unique tag).

So the reading is: **on a workload of fresh long documents, parking is pure overhead**
(`reused_kv_bytes = 0`); on multi-turn agent traffic it pays for itself and should be
left on. If you want both, `--conversation-cache-min-tokens N` lets short side requests
through while keeping long parked conversations out of the slot budget.

## Reproduce

```text
engine-cuda12/strata --serve \
  --pack <pack>/unsloth-ud-q4_k_xl --native <model>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --expert-profile <profile>.bin --expert-cache auto --prefill auto \
  --spec 4 --spec-min-p 0.8 --suffix-draft 8 --mtp <mtp>/rt \
  --max-context 262144 --kv int8 --ple-io ram --kv-resident 32768 \
  --prompt-cache 24 --prompt-cache-every 8192 \
  --conversation-cache-mib 0 --conversation-cache-slots 0 \
  --trim-stage-weights --pipeline-windows 2 --layer-split 9,25,34
```

Then send four ~81K-token fresh prompts and read the engine log: with the cache on you
get `conversation cache: parked ... reused_kv_bytes=0` after every one of them, and the
prefill rate drops from run 2 onward; with the cache off the lines disappear and the
rate holds.

Per-run JSON: `bench_81k-convcache-on.json` (control), `bench_81k-convcache-off-a.json`,
`bench_81k-convcache-off-b.json`, `bench_2k-convcache-off.json`; exact config in
`launch-config.json`; log lines in `engine-log-excerpt.txt`.

## Limitations

- One machine, one model (Q4_K_XL). The mechanism (parking work proportional to context,
  never reused for fresh prompts) should generalize, but the 10.5% is this box's number.
- The two 81K decode medians differ (74.8 vs 83.2); decode is draft-acceptance dependent
  and not the point here — prefill is the stable signal.
- 4 requests per round; no output-quality check was re-run (the 2026-10-12 needle test
  is unaffected by a cache-size flag).
