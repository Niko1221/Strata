# Evidence: the delta-restore work (phase 0 → A → B1 → B2 → pooled rows)

The raw output of every measurement behind `docs/nvme-delta-restore-handoff.md` and the numbers in
`docs/nvme-kv-cache-design.md` §6, kept here verbatim so nobody has to take a doc's word for them.

**How these were produced.** One device, the server STOPPED (§9 of `docs/nvme-delta-cache-handoff.md`),
`STRATA_DELTA_RESTORE_TIMING=1` set for the engine under test, a single conversation generated through the
engine's own stdin protocol (`GEN 16 <143,498 comma-separated token ids>`, then `QUIT`) against a scratch store
at `/tmp/restore-bench/store`, so every row restores THE SAME conversation and the same boundary:

- T = 143,495 tokens, 560 chunks, 2.15 GiB of chunk payloads, one 120,139,056-byte State record
- engine args: the oracle's set with `--max-context 262144` (production's value), `--kv-delta 1`
- the v3 control is the SAME engine's own cascade snapshot of that boundary, restored with `--nvme-restore`

`rss peak X MB (entry Y MB)` is `/proc/self/status` VmHWM against VmRSS, sampled by the restore itself: the
difference is the transient the promote added to a ~46 GiB engine.

## 1. The four phases, on device

| phase | line (verbatim, abbreviated only in the leading `strata serve:`) | wall | transient |
|---|---|---|---|
| before (Phase 0 baseline) | `kv-delta restore timing: manifest 0.5 ms, read+digest 3115.3 ms, assemble 2378.8 ms, apply 1887.8 ms, rss peak 50606 MB (entry 46202 MB), T 143495, 560 chunks, 2.15 GiB read` | 7.4 s | 4,404 MB |
| A (parallel read) | `kv-delta restore timing: manifest 0.5 ms, read+digest 3021.3 ms, assemble 1713.2 ms, apply 1811.1 ms, rss peak 48597 MB (entry 46265 MB), T 143495, 560 chunks, 2.15 GiB read` | 6.5 s | 2,332 MB |
| B2 (streaming) | `kv-delta restore timing: manifest 0.5 ms, read+digest+place 2635.4 ms, stage 0.0 ms, apply 24.5 ms, rss peak 46543 MB (entry 46202 MB), T 143495, 560 chunks, 0.11 GiB read` | 2.7 s | 341 MB |
| B2 + pooled rows per chunk (final) | `kv-delta restore timing: manifest 0.5 ms, read+digest+place 2569.7 ms, pooled rows 39.7 ms, apply 9.4 ms, rss peak 46334 MB (entry 46204 MB), T 143495, 560 chunks, 2.15 GiB read` | 2.6 s | **130 MB** |

The `apply` phase falls from 1,888 ms to 9 ms because the assembled image's second FNV pass is gone - the
streamed restore has no image to hash. The `read+digest+place` phase is what is left, and it is disk-bound: the
v3 tier pays the same throughput for its single 2.2 GB sequential file.

## 2. The KV line the tier itself printed (final build)

```
KV src=delta resume=143495 promote_ms=2621 promote_bytes=2309000692 staging_bytes=135802160 \
  dump_ms=2 dump_bytes=0 evict=0 evict_bytes=0 sweep=0 sweep_bytes=0 refused=0 transfer=0 ...
```

- `promote_ms=2621` for 143,495 tokens ≈ **55k tok/s**, at or better than the v3 tier's ~50k
- `staging_bytes=135802160` — see §4 for exactly what is in it

## 3. Byte-identity, at 142k, against the v3 tier's own restore of the same boundary

The two lines below are byte-identical, all nine fields. This is the oracle the streaming restore has to pass
(`docs/nvme-kv-cache-design.md` §5.2); it holds at 2355 tokens in `tools/nvme_delta_p0_test.sh` (its P1 check)
and here at 143,495.

```
v3 tier, --nvme-restore of the cascade snapshot of this boundary:
strata serve: STATE_HASH L=143495 gdn=2eb986e15c2f71b1 ple=ce108779e12069f7 tail=777f9eee10620fdd \
  pooled=0ac5d85128e1b9eb kv=11c7d46789b5cc6f mtp=c88b8c82dba69ad3 stale=d0f2657ae95e9ffb \
  dead=badb3297a4ccdeb6 ple_prev=248046,198

delta tier, streamed restore of its manifest for the same boundary:
strata serve: STATE_HASH L=143495 gdn=2eb986e15c2f71b1 ple=ce108779e12069f7 tail=777f9eee10620fdd \
  pooled=0ac5d85128e1b9eb kv=11c7d46789b5cc6f mtp=c88b8c82dba69ad3 stale=d0f2657ae95e9ffb \
  dead=badb3297a4ccdeb6 ple_prev=248046,198
```

(`stale` is the last partial page's unwritten cell, included here because these two ran against the same engine
binary; the oracle strips it because ITS pair comes from two processes.)

## 4. What `staging_bytes` is made of - and why ~100 MB is below the floor

The restore prints its own State's composition when the timing flag is on (verbatim):

```
kv-delta restore timing: state 120139056 bytes = gdn 117669888 + ple 368640 + kv tail pages 1723392 \
  + drafter tail pages 143616 + pooled tail rows 208896 + tails/dead/block_pos 24624
```

So the 135,802,160 bytes of staging are:

| term | bytes | note |
|---|---|---|
| gdn + ple | 118,038,528 | the apply pass's FIRST TWO device segments - a batch, so both are resident in host memory when it starts. `gdn` alone is 117,669,888 bytes. |
| 4 reader buffers | ~15.6 MiB | one chunk payload each, reused; Phase A's workers |
| pooled-rows buffer | 32,768 | one chunk's span of rows, reused |
| ~~pooled-row staging~~ | ~~219,547,308~~ | **gone**: the rows now cross the bus per chunk instead of being staged |

The handoff's "~100 MB (the bounded staging)" is therefore below what this geometry's `gdn` segment alone costs
at the 262,144-token context (112.2 MiB), and the apply pass's batch shape (frozen: `nvme_restore_apply`, R1)
means that segment has to be in host memory when the batch runs. What the work removed is the term that scaled
with the CONVERSATION - 219 MB of pooled rows that no longer exists - and with it the restore's scaling with
prompt length: the remaining staging is the model's own state plus a handful of read buffers, and the
transient peak is 130 MB against the assembled path's 4,404 MB.

Going below ~112 MiB would need either per-segment `gdn` applies (which changes the applies list's shape and
count, an owner decision) or reading `gdn` through a mapping so the bytes are reclaimable page cache rather than
anonymous memory (a real benefit for the OOM story, not a smaller number).

## 5. Gates on the final binary (`NVME_ENGINE=<build>/strata`, server stopped)

| gate | result |
|---|---|
| `ctest -R "conversation\|nvme\|delta\|transfer\|cache\|memory\|validation"` | 7/7 PASS |
| `kv_delta_host_test` | 109,707 checks, fuzz 25 seeds × 40 ops clean (the `nvme_restore(file) == delta_restore(store)` cross-check compares every array byte-for-byte) |
| `tools/nvme_steps123_test.sh` | ALL PASS |
| `tools/nvme_failure_contract_test.sh` | ALL PASS |
| `tools/nvme_p0_test.sh` | ALL PASS |
| `tools/nvme_delta_p0_test.sh` | ALL PASS, incl. `PASS bit-exact (P1)` |
| `serve.test_server` + `serve.test_mcp` | 104 PASS |
| `node --check serve/web/app.js` | OK |

Deployed: `build/strata` == `engine/strata`, md5 `04b4d228e8047372fbfd56309015e3aa`, and the zero-request
`/metrics` check (`enabled: true`, real `delta_entries`, `ram_tier` nulls) passed with that binary.

## 6. A tool caveat worth carrying forward

`tools/short_tests.py`'s S3 section flips one byte in EVERY chunk and State record of the store it is pointed
at and never repairs them - by design, as its negative control ("corrupt files that are never matched
legitimately remain until matched-and-refused or LRU-evicted"). A second run against a store an earlier run
poisoned therefore reports `S1 turn2 avoids prefill` / `S2 branch snapshots created` as failures that are the
tool's own control working. Reproduced both ways on 2026-09-30: the same binary scored `SOME FAILURES` against
the poisoned production store and `ALL PASS` against a fresh one.
