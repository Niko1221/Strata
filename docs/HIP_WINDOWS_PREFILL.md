# HIP/Windows long-prefill copy dependencies

The long-prefill issuer can group completion and reuse dependencies while
preserving the individual expert copies, their sources, destinations and order.
This is opt-in on HIP/Windows; all three defaults are `1`. Other platforms ignore
these overrides. Short-prefill staging and decode are unchanged.

Validated settings on one RX 9070 XT (`gfx1201`), Windows, HIP 7.2:

```powershell
$env:STRATA_HIP_COPY_BATCH = "8"
$env:STRATA_HIP_STAGE_BATCH = "16"
$env:STRATA_HIP_STAGE_COPY_BATCH = "16"
```

| Override | Dependency grouped | Limit |
|---|---|---|
| `STRATA_HIP_COPY_BATCH` | Direct pinned `copied` completion and `used` reuse wait | GPU ring depth, at most 32 |
| `STRATA_HIP_STAGE_BATCH` | Host staging `dma_done` completion | Host ring depth |
| `STRATA_HIP_STAGE_COPY_BATCH` | Pinned staged `copied` completion and `used` reuse wait | Both ring depths |

Values are clamped to at least `1`. The latter two settings are independent:
host DMA ownership and GPU slot reuse are separate dependencies. Pageable
staging sources keep individual GPU completion dependencies.

## Ordering and ownership

- A terminal event on an ordered stream covers all preceding operations in its
  group. The issuer records it before publishing that the group is available.
- Before copying into a reused GPU group, the issuer waits until every prior
  `used` record has been submitted, then waits on the terminal `used` event.
  The compute stream has already queued its prior `copied` waits before that
  event can be re-recorded for another GPU ring generation.
- A host worker cannot overwrite a staging buffer until the prior ownership
  group's terminal DMA event completes. Before that event can be re-recorded,
  every next-generation source in the group must be ready; those workers have
  therefore finished their previous-generation event waits.
- GPU completion groups stop at GPU wrap, host wrap and source-type boundaries.
  Host ownership groups stop at host wrap and the end of the job list.
- On partial abort, `finish()` fences the submitted portion at the group's
  terminal event before releasing workers. Repeated `finish()` adds no fence.
  The caller joins the issuer before finishing the stager and preserves its
  existing stream synchronization before the next request.

`STRATA_PREFILL_TIMING=1` also reports copy/group/fence counts and expert payload
bytes. These are application operation counts, not all HIP API calls or hardware
bus counters. Host reuse still waits once per reused buffer; batching reduces
event records, not that wait count.

## Regression fixture

Build and run `hip_prefill_stager` (also registered with CTest). It exercises the
production private stager with synthetic 64 KiB payloads: 1, 15, 16, 17, 31, 32,
33 and 65 jobs, host rings 16 and 17, batches 1/2/4/8/16, partial aborts,
repeated finish and repeated generations. It checks DMA readback byte for byte,
fence counts and HIP fence errors. No model fixture is required.

These settings have not been validated on other AMD GPUs, Linux HIP or CUDA.
They do not change kernels, quantization, routing, cache policy or MTP.
