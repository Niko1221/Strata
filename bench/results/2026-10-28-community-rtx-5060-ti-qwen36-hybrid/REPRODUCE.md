# Reproduction and data fields

This directory is self-contained for input/output/statistical audit. It contains synthetic text/token inputs,96-token histories, expected task values, all40 fixed-history outputs, all20 pairs, numerical summaries and engine timing logs. It does not include weights or full-logit dumps.

## Offline audit

Python3.11+ is enough:

```sh
python verify.py
python replay.py --help
python build_adapter.py --help
```

`manifest.json` records SHA-256 for all report files except itself. `.gitattributes` preserves the archived bytes so line-ending conversion does not invalidate fixture hashes. Performance input `.tokens` and `.forced` files are comma-separated token IDs, without an added newline. JSON input hashes in the task specifications use compact JSON encoding of the integer list; those hashes are not the same encoding as the CSV token-file hashes.

`pairs.csv` / `fixed-history-pairs.csv` have one row per input/repetition. Milliseconds are native engine timings. `request_ms` is prefill+generation. Ratio columns divide2048 by4096, or kept by released. The reported median ratio is the median of the five ratios. `recorded-runs.json` keeps each actual97-ID output and protocol checks; numerical and task details are separate from formal timing data. Task prompts/IDs/expected values are in `fixtures/tasks.json`; the archived judgments include the three actual inputs for each executed function in the manifest and the original measured results in data summaries. Re-scored historical task answers retain their original judgments too.

## Build the measured engines

Use separate clean source/build directories for:

- Baseline `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`.
- #4335 `bdf0bce2e5520680524efd3a5cf79db984c3902c`.
- #4347 `7be0a3a17aeea0a7c0bf7bc63ae56c70e53311d4`.

The measured Windows build used VS2022 MSVC19.42, CUDA13.2, Ninja Release, `-DSTRATA_ENABLE_CUDA=ON`, `-DCMAKE_CUDA_ARCHITECTURES=120`, `-DSTRATA_BUILD_TESTS=ON`, `-DSTRATA_BUILD_CONVERSATION_TESTS=ON`, and ggml/llama.cpp at `3cf03257f219afbe7334045ff7c6a06ac68c627d` via `STRATA_GGML_DIR`. Use explicit `CUDAToolkit_ROOT`, `CMAKE_CUDA_COMPILER` and `CMAKE_CUDA_HOST_COMPILER` pointing to your toolchain. Build `strata-q36` and `strata_kq`; selected test names and per-test results are preserved in `logs/builds`. This is the recorded build configuration, not a claim that another toolkit produces the same binary or identical CPU/GPU rounding.

Windows cannot pass the16K ID list within its command-line limit. `file_entry.cpp` only reads `--tokens-file`/`--force-file`, converts them to existing in-memory argv strings and calls the existing engine entry. It changes no kernel or model code. From an x64 VS2022 Developer Command Prompt, link it to each already-built engine, for example:

```powershell
python build_adapter.py --source-dir C:/work/pr4335 --build-dir C:/work/build4335 --cuda C:/toolchains/cuda-13.2 --evidence C:/work/adapter-evidence4335
```

This produces `strata-q36-report-file.exe` alongside the original exe/DLL. It refuses to overwrite an existing report adapter, records its hash, and checks that original exe/DLL/library hashes do not change. The original numerical experiments also verified the original file entry against archived direct-argument logits; a newly linked binary is not assumed byte-identical to that historical executable.

## Replay examples: opt-in GPU work

The portable replay below was added after the measurements. Examples are proposed replay invocations, not additional published measurements. Keep the tested CUDA DLL directory on PATH, unload other models, pause downloads, and perform your own before/after transfer checks. This utility does not manage or stop other applications. It creates a new output directory and refuses to reuse one.

```powershell
# Prints20 planned runs; does not load the model.
python replay.py --experiment 4335 --exe C:/work/build4335/strata-q36-report-file.exe --model C:/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --output C:/work/replay4335

# Execute only after the environment is ready. The full model hash is checked.
python replay.py --experiment 4335 --exe C:/work/build4335/strata-q36-report-file.exe --model C:/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --output C:/work/replay4335 --execute

# Separate PR4347 executable; toggles KEEP_PACKED at fixed chunk2048.
python replay.py --experiment 4347 --exe C:/work/build4347/strata-q36-report-file.exe --model C:/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --output C:/work/replay4347 --execute
```

`--check-recorded` additionally requires the same expert counts and97 argmax IDs as the published machine. Other hardware or available VRAM can legitimately change both; report a new configuration rather than silently forcing the historical placement. The replay is fixed history only, not a replacement for full-logit gates or task-quality checking. Archived task sources and expected results are sufficient to repeat those checks through the pinned engine's `q36_harness`/Tokenizer; the portable performance driver does not run the task suite.

## Publication-time validation

Offline verification checks all fixture hashes,40 original measured runs,20 complete pairs and every median/range against source data. The newly added scripts were syntax-checked; both planning modes and CLI help were exercised. The adapter was compiled/linked against the existing #4335 build and its missing-file-argument error was checked without loading a model, with original artifact hashes unchanged. No new GPU replay, full-logit generation or performance rerun was performed for publication. Source measurement drivers and their original hash references remain in the package so this distinction can be audited.
