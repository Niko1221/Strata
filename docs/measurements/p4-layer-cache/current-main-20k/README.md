# Current-main P4 20K follow-up

This answers the [requested 20K P4 check](https://github.com/Niko1221/Strata/pull/1704#issuecomment-6099698652).
Source tested: `eb393ba64cc89e9ed4da108b4fd72718228414eb`, merged with upstream
`61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`. Exact binary hashes are in
`provenance.json`. The shared active-layer guard and current-main CPU-share RAM
fallback are both retained by the merge.

One Tesla P4 (7.68 GiB usable VRAM), R730 host, ISTA IQ3_XXS, INT8 KV, single
device, 32K context, 27 CPU workers, PCIE fraction 0, adaptive swaps off, MTP
width 2, automatic prefill chunks and three fresh 20,000-token repetitions per
arm. Each engine first warms up with a 512-token prompt and 64 output tokens.
Both arms use automatic RAM staging and `STRATA_PREFILL_CPU_SHARE=0`; the
experimental scheduler disables CPU sharing internally, so matching this
setting is necessary. These timings do not compare against setup defaults with
CPU sharing enabled.

| Comparison | Off/main median prefill | Enabled median prefill | Time reduction |
|---|---:|---:|---:|
| Current main / candidate enabled | 199.9948s | 182.6055s | 8.695% |
| Candidate off / candidate enabled | 201.2198s | 182.7688s | 9.170% |

All twelve measured requests returned identical complete 64-token output
streams, with zero reused prompt tokens. Candidate-off also matches current
main in this controlled comparison. The scheduler activated in every enabled
request. Its 2048-token window target resolved to a 2560-token window of two
1280-token chunks; the first enabled request reported 480.6 MiB temporary weight
capacity and 100.0 MiB pinned residual rows. Expert staging adds its own pinned
buffers. These are logged allocation sizes, not independently sampled peak RAM
or VRAM. The request reported 193,152,204,800 prefetched bytes.

STOP during a 20K prompt finished with `cancel` in 2.661s wall time and the same
engine completed a following request. This tests cancellation/drain recovery,
not a full internal recurrent-state digest. The interrupted pre-restart campaign
was retained separately; the complete off/on/cancellation run uses a new scratch
directory and does not count interrupted samples.

Both current-main CUDA sm61 engines built. Refreshed HIP gfx1012 and standalone
SYCL builds also passed; no fresh HIP/SYCL 20K inference measurement is claimed.
The experiment remains opt-in and format/device constrained. This is one
synthetic workload, three observations per arm, fixed off-then-on ordering and
warm filesystem caches; it does not establish a general speedup on other GPUs.

## Reproduce

Build this branch and current main with identical compiler/CUDA flags. In a
local model configuration set 32K context/KV capacity, single-device execution,
`STRATA_PREFILL_LAYER_CACHE_TOKENS=2048` and the placement settings above. Use an
idle GPU and new result directories:

```sh
python tools/bench_prefill_layer_cache.py --config model.json --output main-vs-on \
  --tokens 20000 --chunk auto --repeats 3 --generate 64 --reference-exe /path/to/main/strata
python tools/bench_prefill_layer_cache.py --config model.json --output off-vs-on \
  --tokens 20000 --chunk auto --repeats 3 --generate 64 --cancel-test
```

JSON records include the full input/output IDs, per-request native statistics,
activation logs, staging profiles and cancellation recovery. The main/off folder
name in the private runner was misleading: its off arm was the reference main
binary, and this public record labels that comparison explicitly.

Unattended machine time: current-main comparison 1303.491s; completed candidate
off/on/cancellation campaign 1283.857s. CUDA configure/build operations totaled
442.052s. Engineering/setup time was not instrumented and is excluded. Prior
4K/8K/12K and cross-GPU results retain their original date/source.
