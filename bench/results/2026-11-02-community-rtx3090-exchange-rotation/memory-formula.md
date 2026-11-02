# Exchange-buffer memory cost

These equations describe the **exchange-buffer payload allocation** in the
measured official source, not total process RSS or all driver allocations.

Source: [ExchangeStorage::exchange_bytes_needed](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/include/strata/core/exchange_storage.hpp#L61-L76),
[reserve_exchanges](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/core/expert_source.cpp#L4242-L4257),
and the [async reservation](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/program/generate.cpp#L9229-L9241).

## General allocation geometry

- D = {d1, ..., dk}: **distinct positive expert block sizes** in bytes.
  Count each size once, not once per expert or layer.
- dmax = max(D), dsum = sum(D).
- N: total number of reserved exchange ordinals.
- b: number of trailing bounce-only ordinals, 0 <= b <= N.
  They stage copies and do not become resident buffers.

~~~text
copy path:     M_copy   = N * dmax
rotation path: M_rotate = (N - b) * dsum + b * dmax
extra payload: ΔM       = (N - b) * (dsum - dmax)
~~~

Rotation needs one buffer of **each size** for every committing ordinal.
Bounce-only ordinals need only the largest capacity. With one block size,
dsum == dmax, so the additional payload is zero.

## Ordinary and asynchronous adaptive tiers

For an active adaptive tier, s = min(--adapt-swaps, 96) is the positive
reserved swap capacity in this version.

| Mode | N | b | Copy path | Rotation path | Extra payload |
|---|---:|---:|---|---|---|
| --adapt-async 0 | s | 0 | s*dmax | s*dsum | s*(dsum-dmax) |
| --adapt-async 1 | 2s | s | 2s*dmax | s*dsum + s*dmax | s*(dsum-dmax) |

Both paths reserve more total space in async mode, but the **increment caused by
rotation** is the same for the same s. These formulas apply when rotation is
eligible and successfully enabled. If no exchange pool is reserved, its payload
is zero; an ineligible rotation request retains the copy path.

## This measured IQ2_XS pack

Distinct block sizes, verified against the measured pack metadata:

~~~text
D     = {1,177,600, 1,305,600, 1,510,400} bytes
dsum  = 3,993,600 bytes
dmax  = 1,510,400 bytes
s     = 96
~~~

| Mode | Copy bytes | Rotation bytes | Extra bytes |
|---|---:|---:|---:|
| async 0, measured | 144,998,400 | 383,385,600 | 238,387,200 |
| async 1, formula only | 289,996,800 | 528,384,000 | 238,387,200 |

For the **measured async-0 configuration**:

~~~text
ΔM = 96 * (3,993,600 - 1,510,400)
   = 238,387,200 bytes
   = 227.34375 MiB       (1 MiB = 1,048,576 bytes)
~~~

That increases the exchange pool from **138.28125 to 365.625 MiB**.
At s=48, the formula predicts an extra **113.671875 MiB**; that configuration
was not benchmarked here. The startup log prints the actual enabled block-size
count and pool budget.

## What this number includes

The extra 227.344 MiB is **pinned host exchange-buffer payload**, not additional
GPU expert-cache capacity. The measured GPU expert cache remained the same in
both arms. The resident RAM expert complement is an existing allocation and is
not charged again as new memory in this delta.

The allocation also has ownership tables, vectors and atomic metadata, and
CUDA/driver mapping costs. Those are not included in the payload equations and
were not independently measured. RSS, VmLck or total RAM-use differences need
not equal this byte count because shared mappings, file cache and other runtime
allocations also change.
