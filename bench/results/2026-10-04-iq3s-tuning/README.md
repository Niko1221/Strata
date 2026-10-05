# IQ3_S tuning on the RX 7700 XT container line (2026-10-04/05)

Card: RX 7700 XT (gfx1101, 12 272 MiB), Strata's share capped at **10 240 MiB** by contract -
the ~2 GiB left over is the desktop's and the GUI's, and it stays that way. All numbers below are
this machine, this image (`strata-hip:gfx1101-*` built from `3b128a0`+), `--prefill 2048`,
`--max-context 131072`, `--spec 4`, KV int8, pool auto. Guards: `docker/vram-guard.py` raw
(`--budget-mib 10240`) and in-container `--pid` share mode, 0.05 s sampling the whole window.

## The problem found by measurement

IQ3_S became the default quant (`11a6026`) with IQ3_XXS's pins (expert cache 800, later-allowance
768). Its blobs are ~17% larger, so the cache ate the prompt path's borrow room and the engine
halved the prompt chunk 2,048 -> 1,024. Baseline (ab2-base-*: the previous launcher default):

| line | fresh prefill tok/s | TTFT@1k / @4k s | decode tok/s | follow-up decode | share peak MiB |
|---|---|---|---|---|---|
| base: cache 800, later 768, ring default (chunk halved to 1,024; three runs) | 170.6-173.4 | 4.6-4.8 / 21.1 | 30.5 | 26.7-26.9 | 8,656 |
| cache auto alone (542 slots), ring untouched | 171.8 | 4.5 / 20.9 | 28.7 | - | - |
| later 512 alone, ring untouched | 173.2 | 4.8 / 20.9 | 29.8 | - | - |
| **ring 48** (with the old pins 800/768; three runs) | **231.1-234.3** | 4.6-4.8 / 16.2-16.3 | 30.1-30.4 | 27.4-28.0 | 8,723 |
| **shipping: ring 48 + cache auto (695) + later 700** (`baked`) | **235.1** | **4.75 / 16.27** | 30.0 | 26.2 | **8,723** |

The lever is the ring, on its own: the prompt path reserves `--prefill-ring` (default 384) cache
slots for itself, and 384 IQ3_S blobs did not fit beside the pinned cache, so it halved its chunk
to 1,024; the engine log prints "the prompt path runs 1024-token chunks" on the base lines and
"the prompt path borrows 534 CUDA0 cache slots" on the tuned ones. Sizing the cache to the room
the guard leaves (auto) and cutting the later-allowance to its 700 MiB floor do not change speed
by themselves (rows 2-3) - they are kept because they size the cache from measured free room
instead of a pinned slot count, which is what survives a busier desktop. Decode is flat across
all rows (within run noise); the gain is entirely on the prompt path.

## The shipping line, launched with no flags at all (`baked-*`)

`./run.sh` (IQ3_S, `STRATA_EXPERT_CACHE=auto`, `STRATA_VRAM_LATER_MIB=700`,
`STRATA_PREFILL_RING=48`): fresh prefill **235.1 tok/s**, decode 30.0, follow-up decode 26.2,
TTFT@1k/@4k 4.75 / 16.27 s, share peak **8,722.9 MiB** (6,037 samples during the arm; a further
19,364 during the gates below), coding smoke 205/205.
`--model IQ3_XXS` reproduces the pre-change docker line **byte for byte** (fixture diff in
`post-tune-iq3xxs.txt` vs `../2026-10-04-launcher-consolidation/pre-run-iq3xxs.txt`).

## Final gates on the shipping line (`baked-gate-*`, runner `gate-run.sh`)

| gate | result |
|---|---|
| 32 768 fresh + follow | 251.5 tok/s prefill; follow reused 32 761 tokens, TTFT 0.72 s |
| 65 536 fresh + follow | 246.5 tok/s; reused 65 529, TTFT 0.82 s |
| 130 944 fresh + follow | 241.6 tok/s; reused 130 937, TTFT 1.03 s - no long-context cliff |
| cancel mid-stream (~20 s of a 4,096-token stream) | connection dropped, next request `stop`, engine log clean (`cancel-probe.py`) |
| coding smoke after all of the above | PASS 205/205 |
| guards, whole gate window | raw PASS; share peak 8,723.9 MiB, 19,364 samples, zero violations |

## Swift 1.5 comparison (reference only - same card, same tuned knobs, not a switch)

`./run.sh --model IQ3_XXS -e STRATA_HF_REPO=ukisai/Swift-1.5-... -e STRATA_PACK_DIR=/work/packs/
swift-iq3_xxs -e STRATA_MODEL_NAME=swift-1.5-iq3_xxs` with auto/700/48 (`swift-*` files):
fresh prefill **252.9 tok/s**, decode **33.5**, TTFT@1k/@4k 4.16 / 14.84 s, share 8,696 MiB PASS,
smoke 205/205, expert-cache hit rate 12.4% vs IQ3_S's 9.7%. The smaller quant simply leaves more
room on this card (1,006-slot cache, 619 slots borrowed). The 128K ladder was not re-run for
Swift (comparison only). Its shared Qwen MTP draft still accepts (82/97 observed) - correctness
was verified by the smoke either way.

Starting Swift through `run.sh` exposed a real bug: Swift packs `per_layer_token_embd.weight` in
**shard 1**, the container always requested PLE as file 2, and the engine fatal-errored
("per_layer_token_embd.weight is not in ...00002"). Fixed in `docker/hfmodel.py` (`"ple": 1` on
the swift family + `--print ple` / `STRATA_PLE_FILE`) and `docker/entrypoint-hip.sh`;
`docker/test_hfmodel_ple.py` pins it, and the relaunch with **no** overrides loaded the 320,001,536-
row table from shard 1 and passed the smoke (`swift-nooverride-smoke.json`).

## Reproduce

```
./run.sh -p 19931 --fresh --detach                      # the shipping line
bash bench/results/2026-10-04-iq3s-tuning/arm-run.sh baked   # prefill+stream+smoke+guards
bash bench/results/2026-10-04-iq3s-tuning/gate-run.sh baked  # 128K ladder + cancel + guards
```

Contract tests (no GPU, no downloads): `python -m unittest discover -s docker` - 16/16.
