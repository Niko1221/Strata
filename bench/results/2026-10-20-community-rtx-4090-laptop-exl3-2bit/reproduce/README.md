# Reproduction inputs

These files document the measured fork; they are evidence, not changes to upstream's engine. Rebuilding or running inference was **not** repeated during report editing. Use an isolated test machine with an NVIDIA GPU; the scripts launch the engine and write results, logs and caches.

## Source and build

Check out `sociolog/Strata` at `fa21db8bb1a9017e875717acb2b7c546f109b941`. Apply **only** `measured-source-with-new-files.patch`: it includes the tracked changes and the two new planner/test files. `measured-tracked-source.patch` is the exact tracked diff retained for comparison with the measurement's `source_diff_sha256`; do not apply both patches.

The patches include ngram/tokenizer compatibility fixes, telemetry, an experimental miss-sharing path (disabled for these results), quality/calibrator adapters and the benchmark port fix. They are not proposed for merging by this report.

Build settings: Release, GCC 15.2.0, CUDA 13.1.115, SM89, `STRATA_ENABLE_CUDA=ON`, `GGML_NATIVE=ON`, `STRATA_NATIVE_EXPERTS=ON`, `STRATA_MMQ_KQUANTS=ON`, `STRATA_Q6K_EXPERTS=ON`, `STRATA_PREFILL_MMQ=OFF`, `STRATA_BUILD_TESTS=ON`. The fork-pinned llama.cpp revision is in `../provenance.json`; use it for `STRATA_GGML_DIR`. The reused dependency directory has no git metadata, so its checkout identity was not independently recovered. Refer to that checkout's `docs/EXL3_LANE.md` for build prerequisites.

Copy `local_tune.py`, `local_quality.py` and `local_quality_compare.py` into the checkout's `tools/`, and `corpus.txt` into `tools/local_tuning/`. They rely on that fork's `serve.server`, `strata_tokenizer`, `quant_quality` and Python dependencies; they are not standalone scripts. The fixed harness hash matches the measurement record.

## Runtime inputs

- Download the pinned checkpoint and variant in the main report. Preserve its ngram table. Convert its tokenizer with that checkout's `tools/exl3/tokenizer_to_pack.py`; put the resulting tokenizer directory in each configuration.
- Use the included `expert-profile.bin`; its SHA-256 matches the pre-measurement preservation record.
- Both runs reused an existing MTP Q2_0 runtime. Its current file fingerprints are in `../provenance.json`. **The original draft-weight revision and complete preparation command were not retained**, so a byte-for-byte independent reconstruction of that input is not yet established. No model weights or generated MTP packs are included here.
- Copy `control.json` and `winner.json` outside this evidence folder. Replace all `<DATA_DIR>` paths, including `exe`, `cwd`, tokenizer, model, MTP and cache paths. They are literal placeholders, not environment-variable expansion. Keep the launch settings unchanged.

From the prepared fork, fixed-suite commands are:

```sh
python tools/local_tune.py --config CONTROL.json --label control-fresh --out RESULTS/control-fresh.json
python tools/local_tune.py --config SELECTED.json --label selected-fresh --out RESULTS/selected-fresh.json
```

Run the HTTP suite with `python tools/strata_bench.py --config CONFIG.json --port PORT --runs 3 --long --out NEW_FOLDER`, once for each configuration. The fixed harness calls the engine directly; its configuration's HTTP port is unused. Use fresh output names: the fixed harness refuses overwrites.

## Reading the evidence

`raw/*confirmation.json` contains `protocol`, per-request `runs`, summaries and memory samples. Exclude rows with `warmup=true`; throughput is tok/s, TTFT/wall time seconds, device memory MiB and host memory GiB. `output_ids` permits exact continuation comparisons. `stock-*.json` uses Strata benchmark schema 1 and includes HTTP errors and agent cache counts.

`quality-approved-worker.json` contains the fixed-set teacher-forced and full-vocabulary endpoint comparisons. One compressed teacher-forced TSV is included; the original baseline, control and selected TSVs were byte-identical. Their original hashes are recorded in provenance. Full-vocabulary logit arrays are not included; only their comparison summaries and the byte-equivalence result are shared. This limits independent auditing of the prompt-logit claim without rerunning the quality adapter.
