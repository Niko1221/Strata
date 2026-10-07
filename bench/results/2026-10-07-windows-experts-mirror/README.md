# Reading each expert from two drives at once (Windows)

Measured October 6-7, 2026, on top of upstream `82f46a8` with #833's batched stager reads ported (the first commit
of this branch). Opt-in: without `STRATA_EXPERTS_MIRROR` and `STRATA_DIRECT_SPLIT_KIB` nothing changes.

## Why

With a RAM budget and unbuffered reads, a decode layer that misses an expert outside VRAM and RAM reads its ~2 MB
blob from `experts.bin` before it can go on. On the machine below that happened ~19 times per verify window, one
layer at a time, and `STRATA_DECODE_TIMING=1` put 5-11 ms of a 35-43 ms window on those reads. The drive was far
from busy (its temperature rose from 47 to 52 C during a 192K-token prompt); each read's own latency was the cost.

A small benchmark (`FILE_FLAG_NO_BUFFERING`, overlapped, 400 random blobs of 2,046,400 bytes from `experts.bin`, p50):

| How one blob is read | Latency |
| --- | ---: |
| one request, Samsung 990 PRO | 482 us |
| 2 / 4 / 8 / 16 slices in flight together, same drive | 434 / 422 / 438 / 447 us |
| half the blob, 990 PRO alone / KIOXIA EXCERIA PRO alone | 287 / 297 us |
| both halves at once, one from each drive | 288 us |

## What changes

- `STRATA_EXPERTS_MIRROR=<path>`: a byte-identical copy of `experts.bin` on another drive. Each unbuffered request
  is read as two halves, one from each file, both in flight together. Before it is used the copy is checked: same
  size, and 16 blocks of 64 KiB spread over the file (first and last included) compared with the original; a copy
  left from an older pack is refused and the reads stay on one drive (tested with a sparse file of the right size).
- `STRATA_DIRECT_SPLIT_KIB=K`: a request larger than K KiB is read as K-KiB slices issued together (with a mirror,
  alternating between the two drives).

Windows only, `experts.bin` only (not the GGUF read in place).

## Configuration

i7-12700K, 64 GB DDR5-6000, 2x RTX 5060 Ti 16 GB (layer split 30, `--pipeline-windows 2`), `experts.bin` on a
Samsung 990 PRO 2 TB (CPU lanes) and its mirror on a KIOXIA EXCERIA PRO 1 TB (chipset), IQ3_S with MTP,
`--resident-budget-gib 18 --resident-lru-gib 6`, `--spec 4 --spec-min-p 0.5`, `STRATA_UNBUFFERED_LOAD=1`, vision on.
Six 600-token answers on different topics per run.

## Results

| Reads | Decode tok/s (mean of 6 answers) | Prompt tok/s 8K / 32K |
| --- | ---: | ---: |
| one drive, one request per window (5 runs of the same build) | 53.7 +- 0.8 | ~750 / ~1,400 |
| one drive, 512 KiB slices | 55.9 | 757 / 1,389 |
| mirror, two halves | 57.2 | 790 / 1,449 |
| mirror, 512 KiB slices alternating drives | 58.4 | 791 / 1,449 |

Decode +8.7% with both; the slices on one drive alone are within about two standard errors. The mirror costs a
second copy of `experts.bin` (50 GB here) and has to be copied again whenever the pack changes.
