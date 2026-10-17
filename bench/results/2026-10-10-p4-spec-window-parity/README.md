# P4 speculative-window token parity

Measured 2026-10-10 on a Dell R730xd with three Tesla P4s and two Xeon E5-2697 v3 CPUs, Linux, NVIDIA driver
580.178.04, existing CUDA 12.0 experimental sm_61 builds. Model: Qwen3.8-Flash-Next-GSQ-RCO IQ3_XXS, native pack,
int8 KV, MTP capacity 4, 27 CPU workers per process, 8,192-token allocation, greedy decoding. These are correctness
checks, not throughput measurements. No new engine patch was applied for this campaign.

All 92 requests produced exactly 128 tokens and matched the full IDs and finish reason of their respective
serial reference. **One coverage gate failed:** the repetitive-copy pipeline supplied zero suffix windows,
although its serial counterpart supplied 70. Equal tokens do not establish coverage of an unused draft source.
The failed gate is retained, with all eight outputs, rather than removed from the result set.

| Campaign | Requests | Token matches | Coverage result |
| --- | ---: | ---: | --- |
| In-process MTP/pipeline matrix, suffix flag off/on | 48 | 48/48 | Pass for MTP, speculation and forced rollback; suffix supplied no windows here |
| Matched resident experts, MTP local versus TCP loopback | 24 | 24/24 | Pass; actual drafts at both floors |
| Default IQ CPU arithmetic, serial draft-floor comparison | 12 | 12/12 | Pass on these three short prompts |
| Repetitive copy, suffix off/on, serial/pipeline | 8 | 8/8 | Fail: suffix-pipeline coverage is zero |

Each campaign used two repetitions, reversed case order in the second round, no excluded requests or warmups,
and zero reused prompt tokens. Code, prose and counting prompts contain 28, 26 and 31 input tokens; the copy
prompt contains 268. Exact input IDs are in the manifests. Outputs include truncated answers, since every
request used a 128-token generation limit. No answer-quality claim follows from token equality.

## What the controls exercised

The in-process split is `19,37` over devices 0/1/2. Its cases switch serial/pipeline on the same process, vary
`spec_min_p` between 0 and 0.95, force a wrong speculative first token on every launch, and disable the bonus
guess with `theta=1.1`. Initial expert placement is fixed within the process, PCIe sharing and adaptation are
off, and both caches are disabled. Startup reports 7,946 expert slots in both the suffix-off and suffix-on arms.
`STRATA_IQ_MT_MIN=1` controls expert arithmetic in this campaign.

- Serial floors 0 / 0.95 offered 708 / 486 drafts across their six requests.
- Pipeline floors 0 / 0.95 launched 244 / 314 speculative windows and rolled back 132 / 102.
- The forced-miss case launched and rolled back 236 windows; all six outputs matched serial.
- The no-guess case launched zero speculative windows and still matched serial.
- The initial suffix-enabled arm supplied zero suffix windows, so it does not establish suffix coverage.

The local/TCP comparison uses the same binary and cards, partition `16,32`, 512 filled slots per stage, and the
same 32 resident expert identities per layer. `profile.json` records those identities and the profile SHA-256,
`2be57e812108d9c7ba12272dd8cbfb2b5d78ed90fde0e5dee05b76c3de1c79f8`, checked again for this campaign. PCIe sharing,
adaptation, suffix drafting and both caches are off; `STRATA_IQ_MT_MIN=1`. Local runs finish before isolated
loopback workers start. Each transport offered 702 / 492 drafts at floor 0 / 0.95, with identical accepted counts
and token streams. Temporary workers and engines exited afterward; no permanent service was changed.

The default-arithmetic control removes `STRATA_IQ_MT_MIN` and runs only serial floors 0 / 0.95. It offered
714 / 486 drafts, with all outputs also matching the corresponding controlled-arithmetic reference. This finite
check did not reproduce the documented default arithmetic sensitivity in `native_gu_mt_min`; it does not remove
that sensitivity or establish independence for arbitrary drafts.

The copy campaign sets floor 1.0 to let suffix drafting supply serial windows: 193 drafts in 70 windows over
two serial suffix-enabled requests, with stable output despite different offered counts (101 / 92). Both
pipeline arms launched 236 speculative windows and rolled back two; the suffix-enabled pipeline supplied no
suffix windows. This is a missing exercised path, not evidence of a token divergence or a newly diagnosed
runtime regression.

## Relationship to the existing PRs

[PR #1773](https://github.com/Niko1221/Strata/pull/1773) is a Q8 TCP/local report with MTP disabled, not a runtime
fix or a test of draft independence. This campaign adds IQ3_XXS with MTP enabled, differing draft floors, forced
rollback, repeated references and a gate that refuses to certify missing coverage.

[PR #1748](https://github.com/Niko1221/Strata/pull/1748) currently refuses remote pipeline windows. The TCP results
here use serial windows. The in-process pipeline results do not validate the proposed remote-overlap follow-up
or deeper speculative pipelines. The one-token commit guard from [#1674](https://github.com/Niko1221/Strata/pull/1674)
is retained in both measured builds; this campaign does not isolate whether it was necessary for these outputs.

| Build | Source | Executable SHA-256 |
| --- | --- | --- |
| Pipeline, default CPU, copy | #1656 `06abdfa50aef6e5068869bc7552fafa5e99b7b9d` plus the two `!always_publish_` guards from #1674 | `73108e4ebc73a248d0ab769283fb686743f273c36e248c8bf23582d02b6eccea` |
| Matched local/TCP | `0f0406534ac6011ba4129b0d6688b3ee3f5da894` (#1748 at `809ff93` plus #1674) | `c094e28863390e2d3ca2b24d343fe999af9a3bb2b34e1f16c6b56e1801d52727` |

## Recheck and reproduce

The new Python gate is independent of these engine branches and uses only the standard library. It adds no
engine or default-path changes. Its 11 GPU-free tests exercise token and length divergence, unstable repeated
references, input mismatches, missing coverage, prompt reuse, count mismatches, ignored switches and process
failure. Existing engine binaries were reused; CUDA, HIP or SYCL source files are not touched by this PR.

```sh
python -m unittest tools.test_spec_window_parity
python tools/spec_window_parity.py --audit bench/results/2026-10-10-p4-spec-window-parity/pipeline-result.json
python tools/spec_window_parity.py --audit bench/results/2026-10-10-p4-spec-window-parity/tcp-result.json
python tools/spec_window_parity.py --audit bench/results/2026-10-10-p4-spec-window-parity/default-result.json
# Expected nonzero exit: missing suffix coverage, with zero token mismatches.
python tools/spec_window_parity.py --audit bench/results/2026-10-10-p4-spec-window-parity/suffix-result.json
```

To rerun, replace the `<...>` paths in a manifest with your matching artifacts, then use `--manifest` and a new
`--output` directory as described in [the gate documentation](../../../docs/SPEC_WINDOW_PARITY.md).
For TCP, run the local manifest first, stop that engine, then start two workers with the same model, profile,
512 slots, `--spec 4`, int8 KV and 8,192 context: use the included `profile.bin` for `<EXPERT_PROFILE>` in both
TCP manifests and both workers. Device 2 owns layers 32–47 on port 18860; device 1 owns 16–31 on
18859 and relays to 18860. Use one private `STRATA_STAGE_TOKEN` in all processes. Run the remote manifest, then
compare the concatenated local/remote rows with `audit(rows, 24, requirements)`; `tcp-result.json` contains that
combined audit. Full logs are retained with home paths replaced by `<HOME>`.

Not tested: long or multi-chunk prompts, dynamic adaptation, sampled decoding, other quantizations, HIP/SYCL
execution, another GPU architecture, real LAN transport, remote overlapped decode or arbitrary pipeline depth.
Full model/pack file hashes and artifact revision were not captured. The original 48-request capture predates
the suffix-counter field added to the runner; its retained logs confirm zero suffix windows. Those requests
still retain their full IDs, draft/rollback counts and switch traces. The dedicated copy run records suffix
counters explicitly and retains its failed coverage requirement.
