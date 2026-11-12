# RTX 5090 32 GB in Docker, Q2_0 at 256K: a standard run, and where the memory goes at 64K-1M (2026-11-13)

`strata bench run --suite standard-v1`, protocol `strata-qwen38-thinking-v1-draft.1`, on the released **0.2.13**
(official `0.2.13-cuda13` image, engine **0.1.59**). Status **complete**: 21 of 21 required requests, 0 failed,
0 skipped, 12 of 12 sanity checks. The machine-readable result is [results.json](results.json), every request in
[outputs.jsonl](outputs.jsonl), the server's own defaults in [config.json](config.json).

This is also the first report from this particular box: a 5090 behind a **PCIe Gen3 x16** link, on a **two-socket
Xeon Gold 5218** with **no AVX-512**, where the expert pool runs on 30 workers over two NUMA nodes. Cards like this one
are usually reported on a desktop CPU with PCIe Gen4/5, so the prompt numbers below are the slower end of what a 5090
can do, and that is the point of sending them.

## Rig

- **GPU:** NVIDIA GeForce RTX 5090, 31.8 GiB dedicated (sm_120, driver 580.142, CUDA 13.0), one card, no layer split.
- **CPU:** 2x Intel Xeon Gold 5218 @ 2.30 GHz, 32 physical cores / 64 threads, 2 NUMA nodes, **no AVX-512**
  (the engine says so at start: `this CPU has no AVX-512: the expert kernels run on AVX-2`).
- **RAM:** 125.6 GiB. **OS:** Ubuntu 24.04.5, kernel 5.15.0-58. **Docker:** the official image, `--gpus device=1`,
  `--security-opt seccomp=unconfined`, `--shm-size 2g`.
- **PCIe:** Gen3 x16 — the engine's own probe reads **13.9 GB/s** host→device, which makes it pick `pcie_frac 0.38`
  instead of the 0.55 default.
- **Strata:** 0.2.13 (build 0.1.59), official `ghcr.io/niko1221/strata:0.2.13-cuda13`.

## Model and configuration

- Qwen3.8-Flash-Next **Q2_0** (ISTA-DASLab), the two-file pack, already installed in a Docker volume.
- **256K context** (`--max-context 262144`), `--kv int8` with **`--kv-resident 16384`**, `--vision gpu`,
  MTP draft layer with `--spec 4`, `--parallel 1`, no tuning run (`TUNE=skip`). Samples: the model card's
  (temperature 1.0, top_p 0.95, top_k 20), thinking on and preserved.
- Why `--kv-resident 16384` and not the 32,768 that setup writes at 64K: see *Where the memory goes* below. It is the
  flag that decides how much VRAM the session's KV paging takes, and on a 32 GB card at 256K it is worth ~8x the
  expert cache.

## Results (median, min-max, n=3)

| Case | Prompt tok/s | Decode tok/s | TTFT s | Total s | Capped |
| --- | --- | --- | --- | --- | --- |
| short | n/a | **231.5** (211.1-274.9) | 0.72 (0.45-0.83) | 2.1 (1.8-2.7) | 0/3 |
| prompt-4k | **2372.9** (2356.1-2379.7) | **242.7** (209.2-242.8) | 1.77 (1.74-1.79) | 3.9 (3.8-4.2) | 3/3 |
| prompt-16k | **5148.3** (5118.9-5165.7) | **214.4** (204.7-248.5) | 3.22 (3.20-3.24) | 5.6 (5.3-5.7) | 3/3 |

Sanity: **12 of 12** — instruction 4/4, retrieval 4/4, tool calls 4/4. Time after the model loaded: 0 min 55 s.

Decode sits at 210-250 tok/s across all three cases, which is the interesting part on this box: the expert pool has
enough cache here that prompt length barely moves decode speed. Prompt throughput scales almost linearly with length
(2,373 → 5,148 tok/s from 4K to 16K), i.e. the fixed per-request cost dominates the 4K case and disappears by 16K.

## Where the memory goes (measured on the same box, 0.2.2 / engine 0.1.47)

The standard suite above is what the protocol asks for. This second part is why I picked 256K and `--kv-resident 16384`,
measured before the 0.2.13 update on the same hardware and the same model. It is labelled separately because it is a
different build; the *shape* of it has not changed between 0.1.47 and 0.1.59 as far as the start line goes.

### The KV cache lives in pinned host RAM, and it costs ~13 KB/token

`--kv-resident N` keeps the whole KV in **pinned host RAM** and only the N-token window in VRAM. From the engine's own
start line (`the K/V in ... of pinned RAM`):

| context | `--kv` | `--kv-resident` | K/V in pinned RAM | KiB / token |
|---:|---|---:|---:|---:|
| 65,536 | int8 | 32,768 | 0.77 GiB | 12.3 |
| 262,144 | int8 | 16,384 | 3.09 GiB | 12.4 |
| 524,288 | q4_0 | 16,384 | 3.38 GiB | 6.9 |
| 1,048,576 | q4_0 | 32,768 | 6.75 GiB | 6.9 |
| 1,048,576 | int8 | 32,768 | 12.38 GiB | 12.4 |

`docs/DETAILS.md` says "~13 KB for the KV cache and ~0.26 KB for the rope tables per token". That is exactly what int8
measures here, and `q4_0` halves it. The RAM side is the cheap side.

### 1M context at `--kv int8` cannot start on a 32 GB card

The VRAM is what breaks:

```text
KV streaming: 32768 of 1048576 cells per QSA layer in VRAM, the K/V in 12.38 GiB of pinned RAM
expert cache auto: 0.52 GiB free, 700 MiB reserved (+184 MiB for the draft head) -> 0 slots
no VRAM is left for the expert cache: it needs at least 1336 slots (1761 MiB), about 2117 MiB more than this card has free
prefill: device buffers for a chunk of 1024 tokens do not fit ... trying a 512-token chunk
prefill: device buffers for a chunk of 512 tokens do not fit
the GPU has too little free VRAM for the prompt path: use a shorter context, or read prompts in smaller chunks
strata: The server did not become ready (failed: server_exited).
```

The session itself takes **2,326 MiB** of VRAM at 1,048,576 tokens (the engine prints this number), which leaves nothing
for the expert cache and then nothing for the prompt buffers.

### `--kv-resident` is the lever on a 32 GB card

Same model, same card, same context, only the KV window changed:

| `--kv-resident` | expert cache | decode (200-token answers) |
|---:|---:|---:|
| 32,768 | 1,147 slots | 76.7 tok/s |
| **16,384** | **1,209 slots** | **82.5 tok/s** |

and when another tenant's job on the same card stopped and freed 5.5 GB, with the same 16,384 window:

| expert cache | VRAM free after load | decode |
|---:|---:|---:|
| **7,302 slots** (10.27 GiB) | 408 MiB | **93.8 tok/s** |

With 897-1,147 slots the decode expert-cache hit rate sits at **48-58%**, and the engine keeps its 30 CPU workers busy
(1,011-3,000% CPU on this 2x16-core box); with 7,302 slots it reaches **92-93%**, and the same answers come out faster.
`--pool-cores` shows the trade is real but expensive: pinning the pool to 12 logical processors took CPU from 30.7 to
12.0 cores and decode from 76.7 to **34.8 tok/s** — halving the CPU halves the speed, because those threads are doing
the work the cache cannot hold.

### Long prompts through a 1M session (0.1.47, `--kv q4_0`)

| prompt | prompt tok/s | decode tok/s |
|---:|---:|---:|
| 1,176 tok | 596 | 94.5 |
| 11,176 tok | 831 | 85.9 |
| 44,506 tok | 825 | 116.0 |
| **100,628 tok** | **754** | **124.6** |

100K tokens through one 5090 in 133.5 s, 6 checkpoints, KV hit rate 80% with 205 MiB read from RAM over the whole
request. (That build used YaRN factor 4, past the model's 262,144 training context, so only the memory and speed there
are something I would quote.)

## Failures, exclusions and limitations

- No request failed in the standard run above.
- **One machine, one model, one card.** No second 5090, no 24 GB or 48 GB comparison, no IQ3_S/IQ3_XXS numbers.
- **`--parallel 1` throughout.** No batching or multi-request numbers in this report.
- **Card shared with other tenants for part of the session.** The GPU is in a shared box; the row that says "another
  5.5 GB free" is a co-tenant job stopping, not a flag. The standard run above was taken with **30,164 MiB / 32,607 MiB**
  in use by this server (15.8 GiB of it expert cache), so it is a normal, slightly contended state rather than an empty
  card.
- **PCIe Gen3 x16.** Everything prompt-side is slower than a Gen4/Gen5 host at the same card; the engine picked
  `pcie_frac 0.38` from its own 13.9 GB/s probe.
- **`0.2.2` numbers in the memory section** were taken before the 0.2.13 update; the standard results are 0.2.13.
- The second part is not a `strata bench run` report and does not claim the protocol's guarantees.

## Reproduce

```sh
docker run -d --name strata --gpus '"device=1"' --security-opt seccomp=unconfined --shm-size 2g \
  -p 8899:8080 -v <models>:/data \
  -e MODEL=Q2_0 -e CONTEXT=256k -e KV=int8 -e VISION=gpu -e PARALLEL=1 -e TUNE=skip \
  -e STRATA_SET="kv-resident=16384" -e API_KEY=<key> \
  ghcr.io/niko1221/strata:0.2.13-cuda13 run

docker exec -e STRATA_API_KEY=<key> strata \
  strata bench run --url http://127.0.0.1:8080 --suite standard-v1 \
    --out /data/bench --credential-env STRATA_API_KEY --track default \
    --track-notes "RTX 5090 32GB, 2x Xeon Gold 5218 (no AVX-512), PCIe Gen3 x16; Q2_0; 256K; int8 KV kv-resident 16384; vision gpu"
```

## Files

- `results.json` — the protocol result: hardware, versions, every summary and every request.
- `outputs.jsonl` — each request with its raw usage, timings and client event offsets.
- `config.json` — the server's own defaults as the runner read them.
- `provenance/` — the chat template, the corpus build and the corpus manifest hashes.
