# The file tier of #149 on a machine whose model already fits the page cache: a negative result

Measured on 2026-10-20 by [Dmitry-B](https://github.com/Dmitry-B). Engine 0.1.45.3 (0.2.0-beta), one
RTX 4090, one question: does reading the experts **in place from the GGUF** (the new Linux file-tier
defaults: the zero-copy read, the I/O prefetch, the decode read-ahead gate) help on a machine where the
whole file set already fits the page cache? **No - it costs 21% to 57% of prompt reading and 19% to 24%
of decode.** The packed `experts.bin` stays. This is a boundary-of-feature result, not a bug report: the
tier is aimed at machines that cannot hold the model in memory, and this machine holds it.

## Machine and configuration

RTX 4090 24 GB (sm_89, driver 615.71.09, CUDA 13.4), AMD Ryzen 9 7950X, 46,464 MiB RAM, Ubuntu 26.04.1,
kernel 7.0.0-38, NVMe ADATA LEGEND 960. Engine built from source by `update.sh` (arch `[89]`, md5
`46db377627e32c2f71eb439d0f7d43ad`). Flash-Next IQ3_XXS, 204,800-token context, INT8 KV,
`--expert-cache auto --prefill auto --spec 4 --mtp --pool-workers 10`. Full config: [config.json](config.json),
machine details: [env.json](env.json).

The model on this machine is a 40.0 GiB expert set; the file tier is not what serves it:

```text
strata generate: native pack: packs/iq3_xxs experts (largest blob 2.33 MB), token embedding IQ3_S in mapped host memory (260 MiB, 0.3 s)
strata generate: experts via mmap (--mmap-experts; the A/B arm of R2.1)
strata: prefill plan: chunk 8192, ring 96 kept: file set 31.7 GiB fits the page cache 40.1 GiB -> about 0.0 GB read from the drive per 32768-token prompt (4 chunks)
```

i.e. the planner itself says nothing is read from the drive per prompt. The tier switches on only when the
experts are read from the GGUF in place, and that is visible in the same log:

```text
strata generate: experts via mmap (--mmap-experts; the GGUF shards in place, no experts.bin)
strata generate: file tier (Linux defaults, the GGUF read in place): page-cached experts computed from the mapping (STRATA_COLD_OVERLAP=4), io prefetch with whole-blob preads on 8 I/O threads, the decode's read-ahead only while the page cache has room for it and it pays (STRATA_IO_PF_GATE=1); off: STRATA_COLD_OVERLAP=0, STRATA_IO_PREFETCH=0, STRATA_IO_PF_GATE=0
```

## Method

Two interleaved pairs (`def` = the working config with `packs/iq3_xxs/experts.bin`, `gguf` = the same
config with that file renamed away, so the engine falls back to the GGUF offsets listed in
`packs/iq3_xxs/native_experts.txt`); nothing else differs, and the script puts the pack back afterwards.
Each arm runs a freshly started engine. Our own suite (synthetic prose prompts with a random marker, 3
runs per case, 4K / 32K / 128K / 194K and a generation-only case); medians. That suite is the one described
in the engine-A/B report of the same day; it is not comparable with other community reports, but both arms
of this comparison used it identically, and the auto-chunk is 8192 tokens / 96 slots in both arms.

Which files the engine really mapped is recorded per arm ([ab-gguf/](ab-gguf/), `*-mapped-files.txt`):

```text
def : /home/<user>/Strata-data/packs/iq3_xxs/experts.bin
gguf: /home/<user>/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
```

## Results

| case | prompt, pack | prompt, GGUF in place | x | decode, pack | decode, GGUF in place | x | TTFT x |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K | 1972.7 | 844.9 | 0.428 | 127.4 | 97.5 | 0.765 | 2.84 |
| 32K | 3309.7 | 1879.6 | 0.568 | 131.9 | 100.8 | 0.764 | 2.03 |
| 128K | 3285.5 | 2567.2 | 0.781 | 127.1 | 103.5 | 0.814 | 1.28 |
| 194K | 3249.9 | 2553.7 | 0.786 | 131.7 | 101.3 | 0.769 | 1.27 |
| generation-only | 297.6 | 130.5 | 0.439 | 157.4 | 124.8 | 0.793 | 2.27 |

**0 of 2 pairs in favour of the GGUF read in place, on every case.** The gap narrows with prompt length
(0.43 at 4K, 0.78 at 194K) but never closes. Prompt reading suffers more than decode, and TTFT grows
1.3-2.8x. Draft acceptance did not change (942/1393 vs 983/1329 in the 4K case, within the run-to-run
spread), so this is not a routing or quality effect: the same tokens are produced by a worse data path.

Our reading of why: with 40 GiB of experts and a 40 GiB page cache, every expert page is already resident
before the request, so the mechanisms of the tier have nothing to overlap, while the GGUF layout costs
more than the pack's: the pack stores experts contiguously (`experts.bin`, 42.9 GB, `largest blob 2.33
MB`), the GGUF spreads the same blobs across a shard with other tensors, so the mapping and the per-blob
preads touch more pages for the same work.

## Two related measurements on the same configuration

**`STRATA_IO_PREFETCH=1` on the packed `experts.bin` (the opt-in that does exist for a pack): off stays
off.** Three interleaved pairs, same config otherwise ([ab-opts-io-prefetch/](ab-opts-io-prefetch/)):
prompt 32K x1.003 (2 of 3 pairs), 4K x0.914, and decode is **slower in 3 of 3 pairs** - per pair,
generation-only: 159.1 -> 148.3, 149.6 -> 140.3, 148.6 -> 143.3 tok/s (median -6.2%).

**The drive assumption in the prefill plan is far from this machine.** The plan prints
`drive 1.50 GB/s (assumed)` (overridable with `STRATA_SSD_GBPS`). Sequential direct reads of
`experts.bin` on this NVMe gave 3.6-12 GB/s in six samples ([drive-read.txt](drive-read.txt)). On this
configuration the plan does not change either way (the file set fits the page cache), so it costs nothing
today - but the assumed value is 2-8x low for any modern NVMe, and on a machine where the file tier is
the bottleneck the plan would be wrong in the interesting direction.

## What would change this conclusion

The tier is for machines whose expert set does not fit the page cache. This machine has 46 GiB of RAM and
a 31.7 GiB file set, and our second machine (4x Tesla V100-SXM2 32 GB, the MS-01 baseboard) keeps 98% of
the experts resident, so it is not the target regime either. To test the tier where it is meant to work
one would need a smaller-RAM host (16-32 GB) with the same model - we do not have one.

## Not measured

Cold start with an empty page cache (that is where the tier should pay most, and it is the case this
report does not cover); other `STRATA_COLD_OVERLAP` / `STRATA_IO_PF_*` values than the defaults;
`--pcie-frac` variants; concurrency; a second machine.

## Files

| file | content |
| --- | --- |
| [ab-gguf/](ab-gguf/) | both arms of both pairs: `*-runs.json` (every request), `*-engine-start.txt` (what the engine said about the tier and the plan), `*-mapped-files.txt` (which file was really mapped) |
| [ab-opts-io-prefetch/](ab-opts-io-prefetch/) | the three `STRATA_IO_PREFETCH=1` pairs against the same-session baseline |
| [engine-start-lines-pack.txt](engine-start-lines-pack.txt) | the prefill-plan output of the working configuration |
| [drive-read.txt](drive-read.txt) | the direct-read measurements of the NVMe and the device name |
| [config.json](config.json), [env.json](env.json) | the configuration and the machine |
