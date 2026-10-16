# Primary Qwen validation: single-engine priority, 2026-10-16

Historical short-run evidence at `202f9fe`. The
[follow-up soak and corrected-runtime comparison](../2026-10-16-solo-priority-qwen-soak/README.md)
records the later prefill/failure-path fixes, one-hour workload and cancellation/recovery tests.

This is the primary model comparison for a prospective upstream contribution.
It uses Strata's default Qwen family, **non-abliterated Qwen3.8 Flash Next in
ISTA-DASLab GSQ-RCO IQ3_S quantization**. The separate
[SC117 deployment report](../2026-10-16-solo-priority-rtx3090/README.md)
is supplementary; its measurements are not pooled with these results.

## Reproducibility and control

- Hardware: RTX 3090 24 GiB, Ryzen 9 5900X, 128 GiB RAM, Windows 11.
- Native engine: official v0.1.44.1 release, unchanged.
- Python runtime commit: `202f9fe667cc02fa120f748795405b47919f9052`, based on
  upstream `296b59d2f73c8f04a805df29c2fa97655adace78`.
- Both arms use that same patched source. Control is `solo_priority: false`;
  treatment is `solo_priority: true`. This is a feature-flag comparison, not a
  second checkout of the upstream Python server. The previous report documents
  the unchanged-server comparison and default-loop AST check.
- Model family in upstream `setup.py`:
  `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, two-part IQ3_S.
  Installed filenames, byte sizes, first-MiB hashes and runtime hashes are in
  [results.json](results.json). The original download revision was not verified;
  these prefix hashes are identification aids, not whole-file integrity checks.
  Python source hashes use the Windows checkout's CRLF bytes; conversion from
  the commit's LF source reproduces both hashes.
- Model, pack, MTP vocabulary, expert profile, launch parameters and sampling
  stayed the same between arms. Effective engine settings also matched exactly.
  No model or native source was edited. No returned tool call was executed.

Launch parameters, excluding machine-specific paths:

```text
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
--mtp <existing vocabulary> --kv int8 --kv-resident 32768 --vision
--vram-reserve-mib 1536 --max-context 262144 --ple-io direct --adapt-async 1
--conversation-cache-mib 32768 --conversation-cache-slots 8
--conversation-cache-min-free-mib 16384
```

Effective spec 6, mtp_max 4, lookup 3, 11 pool workers, pcie_frac 0.55,
14,400 MiB / 7,417 expert slots, 47,962 MiB arena, speed projection off.
No parallel/batch slots. Three repetitions per arm, control then treatment on
the same day; not randomized or ABBA. Small throughput differences include
possible thermal/workload variance. Service/model loading is excluded.

## Warm ~128K coding workload

The synthetic archive distributes 20 facts across unrelated notes: exact IDs,
revision precedence, joins, sums, code contracts, citations and absent values.
After an initial audit warms the cache, the long request emits its audit followed
by Python code, capped at 3,072 output tokens to keep decode busy. This deliberate
cutoff does not establish that the generated module compiles or is complete.
Seven seconds later, the short client requests a 21-token JSON response with
`priority: 1` and a 128-token output cap. Timing uses the PC loopback API.

| Median, three pairs | Flag off | Flag on |
|---|---:|---:|
| Short time to first content | 19.921 s | 0.588 s |
| Both requests completed | 27.120 s | 27.776 s |
| Both outputs / pair wall time | 114.048 tok/s | 111.354 tok/s |

The short TTFTs with preemption were **0.922 / 0.588 / 0.567 s**.
Aggregate throughput decreased **2.36%**, and pair completion time increased
**2.42%** in this sample. Throughput here includes waiting, interruption and
restore overhead, and is distinct from native decode tok/s.

Both arms: all three long audits scored **20/20** and all three short replies
scored **3/3**. After these three interruptions, `resume_prompt_read=3` and
`resume_prefix_missed=0`: boundary-token work, with no processed prefix lost.

## Cold prefill: faster short reply, extra reread cost

One separate uncached ~128K prompt per arm; the short request arrives at +2 s.

| Single pair | Flag off | Flag on |
|---|---:|---:|
| Short time to first content | 53.108 s | 4.557 s |
| Both requests completed | 55.294 s | 61.122 s |
| Long factual audit | 20/20 | 20/20 |
| Short factual check | 3/3 | 3/3 |

Pair completion took **10.54% longer**. The interrupted cold continuation read
127,784 prompt tokens, of which **16,384 were an already-processed prefix that
had to be reread**. The remainder was unfinished input work. This cost must not
be described as zero-copy/full-prefix reuse. The final-source run confirms that
cold interruption costs more than an already-cached decode interruption.

## Multiple interruptions and tools

- One 8,192-token long response, three short clients submitted seven seconds
  apart: short TTFTs **0.371 / 0.395 / 0.435 s**; long wall time **73.258 s**.
  Its initial audit and the separate post-resume audit both scored **20/20**.
- Nine total preemptions; final counters: `resume_prompt_read=127792`,
  `resume_prefix_missed=16384`, no parked requests or waiters. All prefix misses
  were accounted for by the cold case above. Warm decode/tool/multiple-arrival
  continuations added no further prefix misses.
- Greedy and sampled (temperature 0.6, top_p 0.9, seed 42) tool streams each
  returned one call with all **450 ordered item strings correct**, in each arm.
  The returned argument strings were byte-identical across flag off/on, including
  the failure below. Short TTFT fell 19.183 → 0.154 s in the greedy pair, and
  16.368 → 0.198 s in the sampled pair. These are single pairs, not medians.
- **Schema failure retained:** `ready` was returned as the string `"False"`,
  rather than JSON boolean `false`, in all four tool cases. Thus these are not
  schema-correctness passes. The flag-off control reproduces the same issue;
  it cannot be attributed to preemption. This test does not localize whether
  the type originates in model output or tool-format conversion.

No new factual mistakes or altered tool arguments were observed in these matched
samples. This is not proof that the model is hallucination-free, that every
sampling distribution is unchanged, or that an overnight workload is reliable.

## Reproduce the primary warm comparison

Use the original Qwen IQ3_S files and matched settings above. The public archive
generator and HTTP client contain only synthetic data and perform no tool calls.
Credentials are read from `STRATA_API_KEY` and must not be included in results.

```sh
python bench/results/2026-10-16-solo-priority-rtx3090/make_fixture.py /tmp/solo-fixture.json
# Start the server with solo_priority=false, then:
python tools/bench_solo_priority.py --base http://127.0.0.1:8080 --fixture /tmp/solo-fixture.json --rounds 3 --out /tmp/original-control.json
# Restart with solo_priority=true and all other settings unchanged, then:
python tools/bench_solo_priority.py --base http://127.0.0.1:8080 --fixture /tmp/solo-fixture.json --rounds 3 --out /tmp/original-priority.json
```

The base fixture's initial request was 127,779 tokens on this model; the measured
warm coding requests were 128,182 tokens. A 262,144-token configured window does
not mean these requests filled that window. The client excludes role-only and
tool-name-only headers from first-content timing.

## Contribution status

The scheduler source did not change for this comparison. Previous automated
checks remain recorded in the supplementary report; they were not rerun or
counted as new passes here. Cancel isolation was previously tested on SC117;
this original-model run adds no separate cancellation result.

The feature remains opt-in. Original-model results are the primary upstream
evidence; SC117 covers a deployed derivative. Longer real multi-client use,
near-full-window/cache-pressure coverage, and non-Windows backends remain
unvalidated. No PR has been published on the strength of these short runs.
