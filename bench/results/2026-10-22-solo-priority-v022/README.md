# Single-engine priority on Strata 0.2.2

The priority patch was rebased onto `011c9e600a93bbc9fd451c563501b5831c32bbca`
(the source tree of the 0.2.2 release). The conflict was in `Service.run`:
both paths now forward the new `declared_only` argument. Chat Completions with
no declared tools must not turn a model-emitted envelope into a tool call.
The Responses parser keeps its existing default. A regression exercises both
the enabled and disabled priority paths with the real fake-engine stream.

Tested patch revision: `03a8928d2ce2f5c26ea6fa5c4953533daeb34b60`.

## Service checks

- `python -m unittest serve.test_solo_priority serve.test_tool_call_envelope -v`:
  27 tests passed on macOS / Python 3.13.
- `python -m unittest discover -s serve -p 'test_*.py'` on Windows / Python 3.13:
  946 tests, 20 skipped, no failures or errors; 485.265 seconds.
- The same full suite on macOS: 946 tests, 15 skipped, four failures and one
  error. All five reproduce on the unmodified 0.2.2 source. They are
  `Roles.test_every_card_off`, three `Lifeline` process-watcher tests, and
  `VisionArgs.test_min_tokens_is_passed_only_when_set` (the mock process lacks
  `pid`). These are not counted as passes or repaired in this scoped patch.

No CUDA/HIP/SYCL/Metal source was changed. No native backend was rebuilt.
GPU results use the official Windows CUDA 13 release engine, version 0.1.47;
execution on other native backends remains unverified. The PR stays a draft.

## Hardware comparison

RTX 3090 24 GB / Ryzen 9 5900X / 128 GB RAM, Windows, original non-abliterated
Qwen3.8 Flash Next GSQ-RCO IQ3_S. Both arms enable the same solo-priority
policy and use 262144 context, 32768 MiB conversation cache, eight cache slots,
spec 4, int8 KV, 32768 resident KV cells, 1536 MiB VRAM reserve and an initial
copy of the same expert profile. The older arm uses the deployed 0.1.44.1
native engine and server snapshot. The newer arm uses the official 0.2.2
CUDA 13 package (engine 0.1.47) and the rebased Python server above.

This compares versions with the feature ON in both. It does not replace the
previous beta flag-off/on experiment or establish the isolated cost of this
patch on 0.2.2. SC117 measurements are kept separate and are not used here.

Three long/short request pairs, with a warm 63.7K audit archive and a 2048-token
long-response cap, gave these medians:

| Measure | 0.1.44.1 + priority | 0.2.2 + priority |
|---|---:|---:|
| Short request first output | 0.400 s | 0.415 s |
| Pair wall time | 20.326 s | 18.031 s |
| Aggregate generated tokens / pair wall time | 101.790 tok/s | 114.744 tok/s |
| Long request native decode rate | 110.9 tok/s | 126.2 tok/s |
| Correct fact audits | 3/3 | 3/3 |

The short JSON answer was also correct in every pair. Throughput rose 12.7%
in this sample; pair time fell 11.3%. A few repetitions on one card do not
establish a universal speedup. Input preparation and waiting are included in
pair wall time; the native decode rate excludes prefill.

Commands (same synthetic fixture and workload in each arm):

```sh
python tools/bench_solo_priority.py --base http://127.0.0.1:18082 --fixture fixture-64000-117.json --rounds 3 --out pairs.json
python tools/check_solo_priority_lifecycle.py --base http://127.0.0.1:18082 --out lifecycle
```

The fixture uses `fixture(seed=117, rows=391)` from
`bench/results/2026-10-16-solo-priority-rtx3090/make_fixture.py`, with
`max_tokens` set to 2048. Its user-prompt SHA-256 is
`b002a6322706e4a8f759cb8fbe4faf4fe93c0d1ae7b83ab72e20582dabfc788a`. The full expected 20-field audit precedes
the intentionally capped code. Synthetic model tools/code are not executed.

On both versions the lifecycle check passed queued, active and paused
cancellation, intact 450-item tool arguments, interrupted Responses tool
calls, and tool-result acknowledgement: 12 requests, nine completions and
three intentional cancellations per arm, zero unexpected transport errors.

The release engine archive SHA-256 was verified as
`59833d875b2f3807791533f5dd077a0cb0ed08d4c1cab3136ad2ceb165871cd1`.
The production configuration was not upgraded for these isolated trials.
