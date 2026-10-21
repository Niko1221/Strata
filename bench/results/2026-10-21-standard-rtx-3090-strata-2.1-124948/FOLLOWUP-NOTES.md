# Follow-up run notes — 2026-10-21 standard-v1, 2x RTX 3090, Strata 2.1

## Command (exact)

```
STRATA_NO_HANDOVER=1 strata bench run \
  --profile ~/strata-q6-021/strata-swift-q6-final7.json \
  --no-start --suite standard-v1 \
  --out bench/results/2026-10-21-standard-rtx-3090-strata-2.1-124948 \
  --track custom \
  --track-notes "Strata 2.1 stack (release commit 90767252f420e25223c739d604145ad15911d48b, tag v0.2.1; engine binary 0.1.46 signed prebuilt cuda13 package), pack packs/swift-q6, config strata-swift-q6-final7.json, 2x RTX 3090 layer split; follow-up of run 20261020T211303Z-ca1ea1e8; engine instance restarted 2026-10-21T11:34:32Z so no standard-v1 payload reuse from the earlier run" \
  --model-revision 0142554c963a8653912c0c8283f05f13f63653f3 \
  --custom-pack packs/swift-q6 \
  --background "strata agent sessions sharing this server during the run window"
```

Run from ~/strata-q6-021 (git HEAD 516ce85, branch
bench-2026-10-20-standard-rtx-3090-strata-2.1-220232). Launched
2026-10-21T11:49:48Z, detached runner PID 1242461; server was already
running and was used, not started and not stopped (`--no-start`; the log
confirms "using the server that is already running ... it will not be
stopped"). Runner exited 0; results.json written 2026-10-21T12:04 (UTC) ;
wall clock 11:49:48Z -> ~12:04:55Z (~15 min, most of it hashing the 168 GB
model set while agent traffic shared the GPU).

Launch note: the first launch attempt (11:46:54Z, log at
/tmp/bench-fresh-run.log) was killed at shard 2/6 by the driving agent
tool's command timeout (the runner shared the tool's process group; nohup
alone does not detach from it). No partial results dir and no server
effects; the relaunch above is the recorded run.

## CLI versions

- Handover CLI: 0.1.46.3 (established by preflight).
- NO_HANDOVER CLI (used, `STRATA_NO_HANDOVER=1 strata --version`):
  `strata 0.2.1 (build 0.1.46.2; python 3.12.15)`; launcher tree
  COMMIT file = 90767252f420e25223c739d604145ad15911d48b (tag v0.2.1).

## Recorded metadata (from results.json)

| Field | Value |
| --- | --- |
| Protocol | strata-qwen38-thinking-v1-draft.1 (profile qwen3.8-flash-next-thinking) |
| Run id | 20261021T120047Z-66e200f7 |
| Conformance | protocol is a draft; `conformance` field is null; status: complete (21 of 21 required, 0 failed, 0 skipped) |
| software.strata_commit | 90767252f420e25223c739d604145ad15911d48b |
| software.runner_commit | 90767252f420e25223c739d604145ad15911d48b |
| protocol.git_commit | 90767252f420e25223c739d604145ad15911d48b |
| software.engine_version | 0.1.46 |
| software.build_kind | release |
| hardware.gpus | 2 (NVIDIA GeForce RTX 3090 device_index 0; NVIDIA GeForce RTX 3090 device_index 1) |
| Live GPU inventory | 2x NVIDIA GeForce RTX 3090 (UUIDs stripped from this note by policy) |
| Sanity | 12 of 12 passed (instruction 4/4, retrieval 4/4, tool_call 4/4) |
| Phase counts | warmup 1 attempted: 1 truncated; performance 9: 3 completed / 6 truncated; sanity 12: 12 completed; failed 0; skipped 0 |
| Truncated total | 7 (1 warmup + 6 performance; output caps include reasoning, hitting the cap is expected and only fails a check in the sanity stage) |
| model.revision | 0142554c963a8653912c0c8283f05f13f63653f3 |
| model.source_revision | not given (not locally recoverable) |
| corpus_sha256 | e75eff8b9feea04f745520b392c4c17d97aefc1ba185e222aa6740efa8f68407 |

## Missing map (full, verbatim from results.json)

- /protocol/model_card_revision: the profile's model-card revision is not pinned yet (protocol draft)
- /run/optional_duration_s: the optional concurrency stage was not selected
- /software/build_options: the server does not report build options; pass --build-options for a source build
- /model/source_revision: not given: pass --source-revision
- /model/tokenizer_revision: not given: pass --tokenizer-revision
- /requests/0..21 /round: not applicable outside the concurrency stage
- /requests/0..21 /error: none: the request ended normally
- /requests/0..21 /token_accounting_note: none: the API and engine counts agree or only one exists
- /requests/0..11 /evaluation: not applicable outside the sanity stage (per-request; /requests/11..21 carry no /evaluation key)

(The /requests/N keys repeat per request index 0-21 with exactly the texts
above; no request-specific missing entries beyond them.)

## Model revision — how established

Trees files present: 0142554c963a8653912c0c8283f05f13f63653f3.json and
99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d.json under
~/.lmstudio/models/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF/.cache/huggingface/trees/.
0142554c963a8653912c0c8283f05f13f63653f3.json matched: its Q6_K entries
equal the five loaded shards on disk, in order
(545566016, 54400261312, 44411907360, 44558394624, 24197675936 bytes).
99bb8f7f... has no matching Q6_K entries. The trees filename is the
revision, so --model-revision was passed verbatim.

Base-Qwen source revision: NOT locally recoverable → --source-revision
was not passed (recorded in the missing map).

## Protocol cache-reuse

The engine instance serving the suite (server PID 1105082, engine PID
1105108) restarted 2026-10-21T11:34:32Z and has never served the
standard-v1 corpus before this run, so no prompt/payload cache from the
2026-10-20 run (20261020T211303Z-ca1ea1e8) was carried into these
timings.

## Track and interpretation

Swift 1.5 is a derivative model needing its own profile; this is a
--track custom run (custom pack packs/swift-q6). It must NOT be
presented as a standard-profile conforming result. The protocol is a
draft (model-card revision not pinned), and conformance is not asserted.

## Needle artifacts

The 6/6 needle artifacts from this series exist and were NOT rerun
during this run: 2026-10-20 run dir long-context/needles.json and
long-context/needles.txt; repo-root needles.json and needles.log in both
~/strata-q6-021 and ~/strata-q6.

## Background load

Declared via --background: strata agent sessions (including the agent
driving this run) shared the server for the whole window; the server had
served 24 requests before the run began (results.json extensions).
Speeds are therefore shared-box numbers, and the short-prompt spread
(18.2-38.0 tok/s prompt) reflects that.
