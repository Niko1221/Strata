# REPRODUCE.md - how this run was produced (RTX 3090 / CachyOS)

Fill the bracketed values from the log after the run; the commands are exact as run.

## Prerequisites on this host

- Keeper-side only. `~/.config/strata/profiles/` is `drwx------ <user>` and
  `Strata-data/models/` is `0700`, so a non-owner process cannot even resolve the profile:
  it fails with `strata: profile 'launch-iq3_s-128k' is not saved`.
- The saved profile `launch-iq3_s-128k` was created by `strata setup` on 2026-10-25 and then
  adjusted (131072 context, `--spec 4 --spec-min-p 0.5`, `--kv int8`, `--vram-reserve-mib 700`,
  `--batch 3`, MTP pack `Strata-data/mtp/rt`). A reproducer rebuilds it with `strata setup`
  plus those settings; the effective engine argv is in `NOTES.md` and in `config.json`.
- Model files verified by sha256 against the quantization repository's LFS oids before the run
  (`.done` markers in `Strata-data/models/IQ3_S/`, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`).

## Command, as run

```sh
# 0. host snapshot before anything is stopped
sudo /usr/local/bin/ldb-preop-snapshot

# 1. stop the managed runtime so the runner owns a fresh engine
strata runtime stop --profile launch-iq3_s-128k --wait 330 --output json

# 2. the run (cwd is where bench-results/ appears)
cd /home/<user>/ldb-bench-3090
strata bench run --profile launch-iq3_s-128k --suite standard-v1 --optional-concurrency \
  --track custom \
  --track-notes 'custom saved profile launch-iq3_s-128k: CachyOS host, 131072 ctx, --kv int8, --expert-cache auto, --prefill auto, --spec 4 --spec-min-p 0.5, --mtp rt pack, --vram-reserve-mib 700, --batch 3, server --parallel 3; release binaries (terminal 0.1.49.2 build 0.2.4.2, control 0.1.51, engine 0.1.50 cuda13, vision engine 0.1.48)' \
  --model-revision ed59f92082b1e93c0e96d60a8b11aab089b52f09 \
  --source-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --tokenizer-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --quantization IQ3_S \
  --custom-pack 'Strata-data/packs/iq3_s prepared by tools/iq_pack.py (native experts) from the IQ3_S shard; terminal 0.1.49.2/data/expert-profile.bin; Strata-data/mtp/rt MTP draft pack - hashed in provenance/pack-provenance.sha256' \
  --background 'KDE Plasma (kwin/plasmashell) held about 1.1 GiB VRAM the whole run' \
  --background 'agent harness and its supervisor stopped with the runtime; strata monitor closed' \
  --background 'no game, browser or other GPU work; zram swap of 62.3 GiB present, nothing paged' \
  --load-timeout 1800
# bench exit 0 (complete)   run folder: bench-results/2026-11-01-standard-rtx-3090-4b08ba5c
# wall clock: 4.9 minutes (BEGIN 03:20:12Z -> DONE 03:25:08Z); model hashing included;
# engine load 61 s (engine.log "engine started" 03:20:18Z -> first protocol request
# run.started_at 03:21:19Z; protocol itself 2 min 3 s required + 1 min 25 s concurrency)

# run 2, tuned keeper arm: identical command, cwd /home/<user>/ldb-tune-3090/t1, plus
# user_args ["--kv","k8v4","--spec-min-p","0.70","--pcie-frac","0.17"] in the run config
# (merged over the profile args at every engine start; the merged effective options are in
# the folder's config.json). Applied by compute/ldb-tune-3090.sh t1.
# bench exit 0 (complete)  folder: bench-results/2026-11-01-standard-rtx-3090-c11a6683
# started 08:28:26Z, 2 min 4 s after the model loaded, sanity 12/12, same backgrounds.

# 3. restart the serving runtime (kill a #4574 husk first if the start reports the profile locked)
strata runtime start --profile launch-iq3_s-128k --wait 300 --output json
```

Wrapper used: `compute/ldb-bench-3090.sh --go` (same steps, logs to
`/home/<user>/ldb-bench-3090/<UTC-stamp>.log`, husk check between steps 2 and 3, then
publishes the folder to the agent-readable corpus). `--restart` is the husk-recovery path.
Run-state file: `/home/<user>/Desktop/Agentic Harness LDB-X/BENCH-RUN-STATE.txt` recorded
`BEGIN 03:20:12Z` and `DONE bench_rc=0 ... serving=yes 03:25:08Z`.

## Conformance checks after the run

- `strata bench report <folder>` and the verdict the runner printed: **complete**
  ("21 of 21 required requests finished, 0 failed, 0 skipped", re-run 2026-11-01 by the agent
  against the published folder; identical to the runner's own output).
- `results.json`: `server_requests_before_run` = 0 (verified in `extensions`); the only
  revision-related `missing` entry is `/protocol/model_card_revision` (the protocol draft's
  own unpinned field; model/source/tokenizer revisions all recorded); sanity 12/12;
  the five "excluded from speed summaries" lines are `first_answer_s` only (requests still
  reasoning at the 512 cap) - all nine performance requests carry prompt/decode medians.
- `provenance/chat_template.jinja` sha256 measured `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`
  - equals the pack template. Match.
- Publish-ACL defect found after the run: the Desktop copy under
  `.../LDB-X/bench/` inherits `default:user:<agent>:rw-` (no `x`), so `chmod -R a+rX`
  cannot make the copied folder traversable by the cage; the agent read the results from
  `/home/<user>/ldb-bench-3090/bench-results/...` (made `a+rX` by the wrapper). Fix in the
  wrapper: `setfacl -R -m u:<agent>:rwx` on the published copy, plus
  `setfacl -d -m u:<agent>:rwx` on the publish directory.

## What goes into the pull request

Two folders, named exactly as the runner wrote them (protocol "Submit and review" step 2:
`bench/results/<UTC-date>-standard-<gpu>-<run-id>/`; step 1 explicitly wants a separate
run per configuration):

```
bench/results/2026-11-01-standard-rtx-3090-4b08ba5c/   stock profile arm
bench/results/2026-11-01-standard-rtx-3090-c11a6683/   keeper arm (+ three user_args)
  README.md            the runner's own, unchanged
  results.json
  outputs.jsonl
  config.json
  engine.log
  NOTES.md             from bench/NOTES-3090.md (host facts + the four-arm tuning table)
  REPRODUCE.md         this file
  provenance/pack-provenance.sha256   from bench/pack-provenance.sha256
```

The t2a/t2b/t2c arms were measured (one fresh engine and one protocol run each) and are
reported in the NOTES table but not submitted as folders; the `engine.log` lines quoted
there are their evidence. User names and home paths in NOTES.md and REPRODUCE.md are
redacted to `/home/<user>` in the submitted copies; the runner already writes
`/home/<user>` into `engine.log` and `results.json`.

Excluded on purpose: the GGUFs, `Strata-data/packs/**`, `Strata-data/mtp/**` (large, reproducible
from the hashes), any API key or host name beyond what the protocol asks for.
