# RTX 5090 32 GB in Docker, Q2_0 at 256K: a standard run, and where the memory goes at 64K-1M (2026-11-13)

`strata bench run --suite standard-v1`, protocol `strata-qwen38-thinking-v1-draft.1`, on the released **0.2.13**
(official `0.2.13-cuda13` image, engine **0.1.59**). **Two runs on the same box**, five minutes apart, with the card in
two different states; both **complete**: 21 of 21 required requests, 0 failed, 0 skipped, **12 of 12 sanity checks** each.

The machine is a 5090 behind a **PCIe Gen3 x16** link on a **two-socket Xeon Gold 5218** with **no AVX-512**, where the
expert pool runs on 30 workers over two NUMA nodes. Cards like this are usually reported on a desktop CPU with PCIe
Gen4/5, so the prompt numbers below are the slower end of what a 5090 can do — that is the point of sending them.

## Rig

- **GPU:** NVIDIA GeForce RTX 5090, 31.8 GiB dedicated (sm_120, driver 580.142, CUDA 13.0), one card, no layer split.
- **CPU:** 2x Intel Xeon Gold 5218 @ 2.30 GHz, 32 physical cores / 64 threads, 2 NUMA nodes, **no AVX-512**
  (the engine says so at start: `this CPU has no AVX-512: the expert kernels run on AVX-2`).
- **RAM:** 125.6 GiB. **OS:** Ubuntu 24.04.5, kernel 5.15.0-58. **Docker:** the official image, `--gpus device=1`,
  `--security-opt seccomp=unconfined`, `--shm-size 2g`.
- **PCIe:** Gen3 x16 — the engine's own probe reads **13.9 GB/s** host→device, so it picks `pcie_frac 0.38`
  instead of the 0.55 default.
- **Strata:** 0.2.13 (build 0.1.59), official `ghcr.io/niko1221/strata:0.2.13-cuda13`.

## Model and configuration

- Qwen3.8-Flash-Next **Q2_0** (ISTA-DASLab), the two-file pack, already installed in a Docker volume.
- **256K context** (`--max-context 262144`), `--kv int8` with **`--kv-resident 16384`**, `--vision gpu`,
  MTP draft layer with `--spec 4`, `--parallel 1`, no tuning (`TUNE=skip`). Samples: the model card's
  (temperature 1.0, top_p 0.95, top_k 20), thinking on and preserved.
- `--kv-resident 16384` rather than the 32,768 that setup writes at 64K: see *Where the memory goes*.

## Results — run A vs run B, same config, different card state

`results.json` is **run A** (the card shared with a co-tenant job: **3,090** expert-cache slots).
`results-idle-card.json` is **run B**, started 5 minutes later after that job exited: **14,570 slots**, and the engine
raised its own `--vram-reserve-mib` from 700 to **2,048**. `outputs.jsonl` and `outputs-idle-card.jsonl` are the
matching per-request files.

| Case | A: prompt tok/s | B: prompt tok/s | A: decode tok/s | B: decode tok/s | A: TTFT s | B: TTFT s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | n/a | n/a | 231.5 (211-275) | 242.5 (213-251) | 0.72 | 0.63 |
| prompt-4k | 2372.9 (2356-2380) | 2397.9 (2367-2402) | 242.7 (209-243) | 227.4 (201-241) | 1.77 | 1.73 |
| prompt-16k | 5148.3 (5119-5166) | 5167.8 (5154-5361) | 214.4 (205-249) | 233.8 (213-248) | 3.22 | 3.21 |

Sanity: **12 of 12 in both runs** (instruction 4/4, retrieval 4/4, tool calls 4/4). Time after the model loaded:
0 min 55 s and 0 min 54 s.

### The interesting part: 4.7x the expert cache moved decode by ±9%

Between the two runs the decode expert-cache hit rate went from **72-76%** (run A) to **95-99%** (run B), with
**1,147 → 14,570 slots**. Decode moved by **+4.8% on short, −6.3% on 4K, +9.0% on 16K** — i.e. inside the run-to-run
spread of a 3-sample median, in both directions.

So on this box, at 256K context, **the expert cache is not the lever for decode speed**. That surprised me and it
contradicts the headline I sent in #5852, where 8x the cache did buy 76.7 → 93.8 tok/s. The difference is the regime:
that measurement was at **1M context**, where the session's VRAM leaves the cache at ~1,100 slots and every miss is a
CPU-pool round trip. At 256K there are already enough slots that the misses the CPU covers are cheap. The honest
statement is **"cache size matters when it is starved, and at 256K on this card it is not starved"**, not "more cache
is faster".

What does move speed here is the **context length**: the same card does ~230 tok/s of decode at 256K and ~77-94 tok/s
at 1M (see below), because the KV streaming window and the session's VRAM share change what is left for the experts.

## Where the memory goes (same box, measured on 0.2.2 / engine 0.1.47)

The standard suite above is what the protocol asks for. This part is the memory accounting, measured before the
0.2.13 update on the same hardware and model, and **labelled separately** because it is a different build. The start
line it quotes has not changed between 0.1.47 and 0.1.59 as far as this report can see.

### The KV cache lives in pinned host RAM, at ~13 KB/token

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

```text
KV streaming: 32768 of 1048576 cells per QSA layer in VRAM, the K/V in 12.38 GiB of pinned RAM
expert cache auto: 0.52 GiB free, 700 MiB reserved (+184 MiB for the draft head) -> 0 slots
no VRAM is left for the expert cache: it needs at least 1336 slots (1761 MiB), about 2117 MiB more than this card has free
prefill: device buffers for a chunk of 1024 tokens do not fit ... trying a 512-token chunk
prefill: device buffers for a chunk of 512 tokens do not fit
the GPU has too little free VRAM for the prompt path: use a shorter context, or read prompts in smaller chunks
strata: The server did not become ready (failed: server_exited).
```

The session itself takes **2,326 MiB** of VRAM at 1,048,576 tokens (the engine prints this), which leaves nothing for
the expert cache and then nothing for the prompt buffers.

### `--kv-resident` at 1M, where it does matter

| `--kv-resident` | expert cache | decode (200-token answers) |
|---:|---:|---:|
| 32,768 | 1,147 slots | 76.7 tok/s |
| **16,384** | **1,209 slots** | **82.5 tok/s** |

and with the same 16,384 window, on a card that had 5.5 GB more free (a co-tenant stopping):

| expert cache | VRAM free after load | decode |
|---:|---:|---:|
| **7,302 slots** (10.27 GiB) | 408 MiB | **93.8 tok/s** |

`--pool-cores` shows the CPU side of the same trade: pinning the pool to 12 logical processors took CPU from 30.7 to
12.0 cores and decode from 76.7 to **34.8 tok/s** — halving the CPU halves the speed at 1M, because those threads are
doing the work the cache cannot hold.

### Long prompts through a 1M session (0.1.47, `--kv q4_0`)

| prompt | prompt tok/s | decode tok/s |
|---:|---:|---:|
| 11,176 tok | 831 | 85.9 |
| 44,506 tok | 825 | 116.0 |
| **100,628 tok** | **754** | **124.6** |

100K tokens through one 5090 in 133.5 s, 6 checkpoints, KV hit rate 80% with 205 MiB read from RAM over the whole
request. (That build used YaRN factor 4, past the model's 262,144 training context, so only the memory and speed there
are something I would quote.)

## Failures, exclusions and limitations

- No request failed in either run; 12 of 12 sanity checks in both.
- **One machine, one model, one card.** No second 5090, no 24 GB or 48 GB comparison, no IQ3_S/IQ3_XXS.
- **`--parallel 1` throughout.** No batching or multi-request numbers.
- **The two runs are not an interleaved A/B.** They are five minutes apart with the card's occupancy as the only
  difference, in one direction, once. The ±9% spread should be read as "no effect visible at this sample size",
  not as a measured zero.
- **A co-tenant job shares the card.** Run A had it loaded; run B is after it exited. The standard run is a normal
  contended state rather than an empty card in both cases.
- **PCIe Gen3 x16** makes the prompt side slower than a Gen4/Gen5 host at the same card.
- **The memory section is 0.2.2 / 0.1.47**, the standard results are 0.2.13 / 0.1.59. Separate sections, labelled.
- The memory section is not a `strata bench run` report and does not claim the protocol's guarantees.

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

| File | What |
| --- | --- |
| `results.json` | Run A, the card shared with a co-tenant job (3,090 expert-cache slots). |
| `outputs.jsonl` | Run A, every request with its raw usage, timings and client event offsets. |
| `results-idle-card.json` | Run B, the same config after that job exited (14,570 slots, `--vram-reserve-mib` 2,048). |
| `outputs-idle-card.jsonl` | Run B, per request. |
| `config.json` | The server's own defaults as the runner read them. |
| `provenance/` | The chat template, the corpus build and the corpus manifest hashes. |
