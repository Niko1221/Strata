# Single-GPU MTP and a dual-GPU service with one or two requests

These are different layouts. One request in a layer-split service still uses both
cards. It is not a physical single-GPU measurement.

**Experimental:** the [same-day evidence](https://github.com/cennac/Strata/tree/078146edb81c809722884e8e21dcc98e41fba4ed/bench/results/2026-11-02-community-dual-rx7900xtx-mtp-switch)
contains complete HIP/CUDA builds and 34 HIP request completions, but only 3/14
paired rendered-text hashes match. The control itself also varies. The cause
is unresolved; this is not a demonstrated lossless acceleration.

| Layout | Engine path | New in this patch |
| --- | --- | --- |
| One physical GPU, one request | Existing solo MTP | No algorithm change; control to measure separately |
| One physical GPU, concurrent requests | Existing `--batch-mtp`, one group | No algorithm change; measure separately if memory allows |
| Two layer stages, one request | Solo pipeline with MTP and optional penalty histories | Can stay enabled beside the supported batch configuration |
| Same two stages, two requests | Existing pipelined-group MTP | Reused; admissions cannot enter the solo pipeline |

## Experimental dual-GPU configuration

Use a separate config, log, and port. Keep the original executable and config for
rollback. Keep the same model, expert placement, context, sampling settings and
power limits for comparisons. This patch does not raise hardware power limits.

Add these engine arguments to a working **two-stage layer split**:

```text
--batch 4 --batch-groups 2 --pipeline-windows 2 --batch-mtp --spec 3
```

The existing `--mtp` must point to the matching model's drafter. Set these in the
engine environment (a run config's `env` dictionary also works):

```json
{
  "STRATA_HYBRID_WINDOWS": "1",
  "STRATA_PIPED_MTP": "2",
  "STRATA_PIPELINE_PENALTIES": "1",
  "STRATA_BATCH_SELF_COMMIT": "0"
}
```

`STRATA_BATCH_SELF_COMMIT=0` prevents the separate self-commit option from disabling
pipelined MTP, including when server defaults would otherwise select it. Keep
`--adapt-async` off. The supported configuration is explicit: four internal slots,
two groups, two layer stages, two pipeline windows, and two piped proposals. This
is not a claim about four simultaneous clients: the comparison below uses two.
Other topologies retain the existing startup gates. If slot allocation shrinks
the batch, or piped MTP cannot run, the exception is disabled and logged.

Unlike #3095, this option keeps the drafter on the last card while batch slots
exist. It does not enable the first-card drafter placement for batches. The
per-request guard runs after diagnostic `pw=` overrides. Both BGEN admissions
(including admission before any slot is active) and prompt reads beside active
slots use the serial verifier. The batch windows retain their own MTP path.
Existing drain and slot-copy code transfers the committed state; it is not a new
checkpoint format. The request-level HIP smoke test does not establish output
parity or quality equivalence; see the evidence above.

Check the startup log for `STRATA_PIPED_MTP:` and `STRATA_HYBRID_WINDOWS:`. Keep the
matching engine log with measurements. API throughput alone cannot establish
whether pipeline windows or batch MTP actually ran. Penalty histories are a
separate opt-in from #4909; coupled-draft and other existing fallback rules stay
in force. Unsetting `STRATA_HYBRID_WINDOWS` restores the old batch/pipeline gate.

## Physical single-GPU control

Start an isolated copy with only the intended card visible. Remove the layer
split and split-device arguments. For one request, remove batch options and use
`--pipeline-windows 0` with the existing matching `--mtp` and `--spec 3`. Unset
`STRATA_HYBRID_WINDOWS`, `STRATA_PIPED_MTP`, and `STRATA_PIPELINE_PENALTIES`. Keep
model and sampling settings equal; record any memory/cache changes needed to fit
one card. This is the existing single-card MTP implementation.

For a separate single-card concurrent control, use `--batch 2 --batch-groups 1
--batch-mtp`, if memory permits. Do not report a dual-card solo-request result as
a single-card result. Separate physical single-card observations are included in
the evidence above; they exercise the existing implementation.

## Reproducible request sequence

Use a public synthetic prompt in `prompt.txt`. Save its token count from server
usage along with the result; file length is not a token count. On an already
running isolated **physical single-GPU** server:

```sh
python tools/bench_mtp_switch.py --endpoint http://127.0.0.1:58159 \
  --model MODEL_ID --build-id EXACT_COMMIT_AND_BINARY_SHA256 \
  --topology single-gpu --scenario solo --prompt-file prompt.txt \
  --output single-gpu.json
```

On the isolated **dual-GPU** server:

```sh
python tools/bench_mtp_switch.py --endpoint http://127.0.0.1:58159 \
  --model MODEL_ID --build-id EXACT_COMMIT_AND_BINARY_SHA256 \
  --topology dual-gpu --scenario switch --prompt-file prompt.txt \
  --output dual-gpu.json
```

The switch sequence runs solo requests, synchronized request pairs, a long request
joined by a shorter request after streaming starts, then solo requests again.
It records whether the long stream emits more deltas after the short stream
finishes. Early EOS can prevent the intended overlap; that is reported as a
failed transition observation, not a successful switch. Repeat with a suitable
prompt or larger `--tokens`. Stream deltas are not token counts.

The tool retains server cache state, all runs, failures, usage, text hashes and
timing. It refuses to overwrite a previous result. Output rates use server
completion-token counts over the full HTTP interval, including prefill and
queueing. Missing usage yields no rate. The pair aggregate divides total output
by the shared pair interval; it does not add each request's average rate. These
numbers are not engine-only decode rates or an MTP correctness proof. Retain
engine logs and compare greedy outputs separately for parity.

Run the same sequence on the same day's baseline executable, changing only the
named opt-ins. Run a separate baseline with all new opt-ins unset before checking
default-output parity. Historical custom-build measurements are available in the
[complete experiment branch](https://github.com/cennac/Strata/tree/community/dual-7900xtx-repro-20261101);
they do not validate the submitted mainline integration.

## Related work

- [#3095](https://github.com/Niko1221/Strata/pull/3095): solo pipeline beside batch
  slots, including first-card drafter placement. This patch uses a narrower
  topology and adds a per-request admission guard.
- [#4121](https://github.com/Niko1221/Strata/pull/4121): pipelined-group MTP. The
  current base already contains that mechanism; this patch does not reimplement
  or claim it.
- [#4909](https://github.com/Niko1221/Strata/pull/4909): exact per-window penalty
  histories. This integration branch includes that pending patch and depends on
  it for the penalty-enabled solo pipeline. It is not an independent MTP
  algorithm.
