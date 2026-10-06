# Report active prefill separately from queued requests

A request waiting for admission can make the monitor appear to be reading a
prompt even while all engine slots decode. The last prompt throughput can also
remain visible after prefill ends. This patch adds explicit engine phase events
and uses them to report the actual phase rather than inferring it from waiters.

The engine emits `PFSTATE 1` before prompt work and `PFSTATE 0` afterwards,
including around individual chunked-prefill calls. The verify-window prompt
path uses a scope guard so ordinary early returns also close the phase. These
events describe prompt work, not slot ownership, generated tokens or TTFT.

The server consumes these lines in its stdout pump and adds two live metrics:

- `engine_prefill_active`: true/false, or null before a phase event arrives
  (including when running an older engine).
- `engine_prefill_epoch`: the number of prefill starts observed since engine
  initialization, useful for detecting an intervening phase between polls.

When explicit phase information exists, the live state uses reading for active
prefill, generating for decoding slots, and waiting instead of a stale reading
state. The previous prompt-rate mean is hidden outside prefill. It remains a
mean of the reported prompt measurement, not a newly calculated instantaneous
rate. The epoch counts chunks/events, not requests, and resets on engine restart.
An old engine's pump cannot overwrite its successor's phase; EOF clears active
prefill. Older engines retain the pre-existing fallback status behavior.

No scheduling or sampling behavior changes. PFSTATE is additive engine stdout
metadata; consumers of the raw engine protocol should ignore unrecognized
metadata lines. The Python server's request routing queues never receive these
phase lines. Local PowerShell monitors and personal launchers are not included.

Run `python -m unittest serve.test_prefill_phase` from the repository root.
The tests check phase consumption without disturbing DONE routing, epoch updates,
four decoding slots plus a fifth queued request, stale prompt-rate suppression,
and older-engine fallback. The integration with v0.1.40.1 also passed the
64-test tool-parser, restart, parallel-request, prefill and hotfix-version suite.
Those use synthetic engines, not a new hardware throughput benchmark.
