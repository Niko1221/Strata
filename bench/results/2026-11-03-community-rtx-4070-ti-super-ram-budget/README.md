# RTX 4070 Ti SUPER: bounded RAM expert cache at 128K

Custom prefill measurements by fm5454 on one **RTX 4070 Ti SUPER 16 GB**, Ryzen 7
7800X3D, 31.1 GiB usable RAM and a Samsung 990 PRO NVMe, on Windows. Runs were
conducted with assistance from **Sol 6.1 Sol High** (contributor-supplied assistant
label). The measured inference models were Swift 1.5 IQ2_XS and original
ISTA-DASLab Qwen3.8-Flash-Next IQ2_XS.

## Recommendation and scope

Investigate a **bounded resident-RAM candidate in opt-in tuning** for this
Windows/NVIDIA mmap workload, where the expert file set exceeds available cache.
Size candidates against available RAM, Windows commit headroom, configured
context and host allocations; compare repeated uncached prefill measurements
against the existing setting. Keep a candidate only when speed and memory
headroom are acceptable, including its effect on parked conversations.

This PR contributes evidence and a tested manual workaround. **It changes no
runtime code, defaults or tuner logic.** `--resident-budget-gib 14` is the measured
setting for these two weight sets on this PC, not a universal budget or an
implemented automatic-sizing policy. Maintainer scope agreement and further
validation would precede a separate implementation PR.

## Same-machine results

Both conditions kept **131,072-token capacity, INT8 KV, MTP, auto prefill and
1,536 MiB VRAM reserve**. The sole intentional setting change was adding
`--resident-budget-gib 14`. Each cell is the median of three requests.

| Model / measurement date | Nominal input | mmap baseline tok/s | 14 GiB budget tok/s | Throughput gain |
| --- | --- | ---: | ---: | ---: |
| Swift 1.5 IQ2_XS / Nov 3, 2026 | 4K | 188.2 | 1,051.1 | **+458.5%** |
| Swift 1.5 IQ2_XS / Nov 3, 2026 | 16K | 315.7 | 1,544.0 | **+389.1%** |
| Original Qwen IQ2_XS / Nov 4, 2026 | 4K | 196.4 | 1,073.8 | **+446.7%** |
| Original Qwen IQ2_XS / Nov 4, 2026 | 16K | 363.7 | 1,852.2 | **+409.3%** |

Gain is `(budget median / baseline median - 1) * 100`. All performance requests
reported zero reused prompt tokens. Each gain compares configurations of the
same model on the same day; cross-model absolute rates are not a controlled
model comparison. Both weight sets share the architecture and quantization;
this is not evidence across unrelated model families or GPUs.

## Evidence and limits

- [Swift report](swift-report.md): original 18 requests, raw measurements,
  sanitized settings/logs, baseline recheck and normal-app persistence check.
- [Original Qwen replication](qwen-iq2_xs/README.md): fresh-engine A/B, identical
  prompt hashes/counts within each pair, provenance, raw memory samples and logs.
- Near-capacity prefill completed for 124,529 tokens on Swift and
  **124,732 tokens** on Qwen (Qwen:
  **2,044.6 tok/s**).
- Qwen's separate near-capacity three-code retrieval returned
  **3/3** expected codes. This is one synthetic
  check, not a general model-quality evaluation.

The budget also selected unbuffered remaining file reads; these tests measure
the combined residency/I/O change, not either effect in isolation. Auto GPU slots
and cache sizing were not frozen. In the Qwen A/B, sampled available RAM reached
**6 MiB** in baseline and **5,908 MiB**
with the budget. Resident mode can disable parked-conversation caching; logs
preserve memory warnings and fallbacks. No universal safe/optimal budget follows.

The prompts are repetitive synthetic records, the performance output cap is
eight tokens, and the standard benchmark protocol was not run. Prefill speed is
separate from useful-answer latency, decode throughput and quality. Full-capacity
inputs, concurrent chats, natural documents, other hardware and budget sweeps
remain untested. See the detailed reports for different warm-up methods,
uncontrolled background load and missing provenance in the first run.

Related: [#5059](https://github.com/Niko1221/Strata/issues/5059),
[#3072](https://github.com/Niko1221/Strata/pull/3072), and the different-system
same-GPU benchmark [#4716](https://github.com/Niko1221/Strata/pull/4716).
