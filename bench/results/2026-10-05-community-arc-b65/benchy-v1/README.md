# Public benchy-v1 prompt fixtures, Arc Pro B65 Gen4

Measured 2026-10-05 using the promoted diagnostic-off IQ2_XS/8K/INT8KV/MTP4
profile, unchanged binary/model/driver/512 batch/3072 MiB reserve/four workers.
Exact 20/2185 raw input IDs are from maxfridbe/Strata_B70 at
79ad9d5aff716292602895b19860c7661bacc5ed, sycl/bench/v1. Our qualified native
serving protocol is used with those fixtures; this is not an unmodified
sycl/benchy.sh run. See [method and reproduction](METHOD.md) and
[build/model/hardware details](../README.md).

Three fresh native engine processes each run the long fixture first, then the
short fixture, both greedy with 256 outputs, followed by an excluded exact
canary. Startup is excluded; the first long decode includes native graph capture.
OS page cache remains warm/retained, prompt reuse/adaptation off. STRATA_TRACE=1
provides numeric path lines; no GPU profiler or device-event instrumentation.
All six measured outputs match their qualified reference token hashes,
and all repetitions match each other. All three native shutdowns exit0.

Each rate/time cell is median [minimum–maximum] of three repetitions; rates
tok/s and times seconds. Separate native prompt/decode timers are used.

|Input|Generated|Runs|Reused|Prompt tok/s|Decode tok/s|TTFT s|Total s|Draft acceptance|
|---:|---:|---:|---:|---|---|---|---|---:|
|20|256|3|0|54.22 [54.20–54.23]|48.53 [48.47–49.55]|0.40 [0.40–0.40]|5.64 [5.54–5.65]|72.6%|
|2185|256|3|0|317.53 [317.35–318.20]|48.05 [48.05–48.48]|6.93 [6.92–6.94]|12.21 [12.15–12.21]|77.4%|

[All records](results.json), [CSV](results.csv), [summary](summary.json),
[native timing lines](native-timings.log), [lifecycle](lifecycle.json),
[health](health-summary.json). All nine requests, including three excluded
canaries, are retained numerically. No generated text is published.

These prompts and output lengths differ from the five-task/640-output
benchmark, so their rate is not an equivalent-workload decode improvement.
Max's published B65 IQ2_XS values (51.42/53.93 tok/s at20/2185 inputs) have one
run per row, different context/prefill/compiler/host settings, and no published
artifact or output hashes in the checked public locations. The remaining
cross-system difference is not causally attributed; this report does not claim
identical weights or generated continuations on his machine. Source:
[Max's table](https://github.com/maxfridbe/Strata_B70/blob/79ad9d5aff716292602895b19860c7661bacc5ed/docs/INTEL_PERFORMANCE.md).
