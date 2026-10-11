# Reproduce or inspect the recorded report

Use the pinned Strata 0.1.42 revision `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406` and matching Windows HIP release/runtime. This report does not ship model files, GPU DLLs or generated packs. `provenance.json` records the original model revisions, shard sizes/SHA256, engine binary, native pack, MTP pack, expert profile and tuning-table hashes.

The exact full prompts and response bodies are included so inspection does not require inference. Run from this folder:

```powershell
python report.py verify
python report.py summarize
```

`verify` checks the bundle checksums, all message references, 780 formal counts, 809 total records, complete 512-token uncached final comparison records and the reported aggregate score. It does not execute generated code or start an engine. `code_oracle.py` is the final grader used for the stored bounded code checks; the report retains its grading output and regrade evidence. Do not mistake an offline export verification for rerunning model quality.

## Prepare model packs

1. Download all three native Huihui `unsloth-UD-IQ4_XS` shards at the recorded HF revision. Verify their SHA256 against `provenance.json`.
2. From the pinned Strata root, prepare the native dense compatibility pack:

```powershell
python tools/iq_pack.py --gguf C:/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --out C:/models/huihui-pack --compat-bf16
```

3. Prepare the official model's 31 MTP tensors at the recorded revision with the pinned `tools/mtp_pack.py` and `tools/mtp_rt.py`, Q2_0 expert format, plus the pinned `data/draft_vocab.bin`. Use these tools' `--help` for source-format options. Compare the resulting runtime file hashes with the exported provenance before calling it a matching pack. The source MTP GGUF hash is also retained; this is the official target's MTP, not a Huihui-abliterated draft layer.
4. Use the pinned `data/expert-profile.bin` and the attached `gfx1151-hipblaslt-100500.txt`. Use the matching gfx1151 hipBLASLt library; Linux version-100401 tuning IDs must not be substituted into this Windows version-100500 table.

The deployment layout in the JSON files is an example; edit paths to your actual installation. `report.py render-config` only replaces `${WORKSPACE}` and writes a new JSON file. It neither changes the original configuration nor starts/stops processes:

```powershell
python report.py render-config --workspace C:/my-deployment --output C:/my-deployment/text-test.json
# From that installation's Strata application directory:
python -X utf8 -u -m serve.server --engine strata --config C:/my-deployment/text-test.json --host 127.0.0.1 --port 8080
```

Use the selected daily text config by default. For an exact final scored arm, render its recorded config from `profiles.json`; that retains prompt cache 0 and comparison port 8081:

```powershell
python report.py render-config --workspace C:/my-deployment --source generalization-general-current.json --output C:/my-deployment/selected-test.json
python report.py render-config --workspace C:/my-deployment --source generalization-general-original.json --output C:/my-deployment/original-test.json
```

Run each arm in a separate engine process, with the identical model/packs, driver, power mode and background conditions. Confirm full expert coverage, slot-zero verification, effective prefill equals the requested prefill, no partial-kernel flags, and zero missing-expert reads. Do not run two copies concurrently. Normal OS/SSD caches were retained in the original runs.

## Replay a recorded arm

Against the corresponding already-running engine:

```powershell
python report.py replay --source generalization-general-current.json --url http://127.0.0.1:8081 --output selected-new.jsonl
python report.py replay --source generalization-general-original.json --url http://127.0.0.1:8081 --output original-new.jsonl
```

Replay warms up with the recorded greedy and xhigh prompts at the first recorded floor (0 for the initial screen) and a 64-token cap, then sends the selected source's requests in their original order. It does not include warmups in the output file. Recorded request messages, floor, sampling, seed and cap are used without copying recorded answers into the response. The server must already match that source's configuration; the replay script does not verify the native command or reset OS caches. Replaying a quality/full-context source skips these throughput warmups. Conversation replay is deliberately rejected because it requires constructing each subsequent request from the *newly generated* prior replies; the included reconstructed transcript is for inspection, not a claim that feeding old replies reruns the conversation experiment.

Record fresh native logs and process-start/ready boundaries alongside new runs. Use at least three repetitions if extending the comparison. The original final generalization arm only has two repetitions per case. Prefix throughput runs usually stop at the token cap and must not be graded as complete answers. Use the completed quality arm separately, with its explicit thinking budget. Non-streaming replay measures total client latency and engine timings; it does not measure TTFT.

## Data format and attribution

`runs.jsonl` contains one JSON object per recorded request, with `Id`, `Stage`, `Formal`, `Source`, `ConfigurationId` and the original per-request fields. `profiles.json` maps configuration IDs to portable configs and retains each source's profile/manifests. `messages.json` keys are SHA256 of the exact message list serialized as UTF-8 JSON with sorted keys, compact separators and `ensure_ascii=False`. Response answer/reasoning texts and timing counters are unmodified; local path prefixes are replaced by `${WORKSPACE}`. Source hashes identify the private original measurement files, not a promise that sanitized files have the same hash. `SHA256SUMS` covers the exported files.

Long C++ fixtures contain public llama.cpp source at `3cf03257f219afbe7334045ff7c6a06ac68c627d` ([source](https://github.com/ggml-org/llama.cpp/tree/3cf03257f219afbe7334045ff7c6a06ac68c627d/src), [MIT license](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/LICENSE)). Existing copyright/license comments in the included source fragments are retained; the upstream MIT notice is included in `llama-cpp-LICENSE.txt`. Synthetic records and small algorithm prompts are in the exact messages export. No private user chat, credentials, model weights or toolchain binaries are included.
