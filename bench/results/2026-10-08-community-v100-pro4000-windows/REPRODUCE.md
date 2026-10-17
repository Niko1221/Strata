# Reproduction and data fields

Use the recorded checkout/engine/model and an existing verified pack. The CUDA 12 build is required for Volta; this report does not prescribe CUDA 13 for the mixed language engine. Replace `<STRATA_ROOT>`, `<DATA_ROOT>`, `<OPS_ROOT>` and `<GPU_PRO_UUID>/<GPU_V1_UUID>/<GPU_V2_UUID>` in a copied config. Also map `gpu` indices to those UUIDs. JSON paths need doubled backslashes or forward slashes. Preserve the specified order/reserves; auto allocation can change with free VRAM. Paths and device identifiers are the only config redactions. Do not overwrite an existing working configuration.

Start from the Strata checkout with the README command. Wait for health loaded=true, images=true, max_context=262144. Config env binds language GPUs, while vision.cuda_device binds the separate encoder. The head/MTP is on the final language stage. No tensor split or pipeline-windows override was used. For C, the later reserve affects V2 as well as P.

In another terminal, set `$Report`, `$Config`, `$Pack`, `$Output`, `$Arm` to the report directory, materialized config, IQ2_XS pack, a NEW output directory and the arm name:

```powershell
.\.venv\Scripts\python.exe "$Report\scripts\benchmark.py" --root $PWD --pack $Pack --config $Config --engine-provenance "$PWD\engine-cuda12\BUILD.json" --url http://127.0.0.1:8081 --model qwen3.8-flash-next-iq2_xs --label $Arm --out "$Output\speed" --targets 4096,32768,128000,258000 --runs 3 --max-tokens 256
.\.venv\Scripts\python.exe "$Report\scripts\api_checks.py" --url http://127.0.0.1:8081 --out "$Output\api-checks"
```

Use Python 3.12, psutil 7.2.2 and Pillow 12.3.0 as measured; card rendering uses Windows Arial. Restart between arms; stop each server before loading another. Both speed and correctness generators use only synthetic content. The exported benchmark differs only in its default checkout path and line endings; always pass --root. Its generic 'adaptive' comment is superseded by the actual profile/no-eviction policy in the engine logs. The API script omits the machine-specific Claude launcher; its tool/image/queue tests are unchanged.

On B only, from the recorded Strata checkout:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe tools/needle_bench.py --url http://127.0.0.1:8081 --lengths 32k,128k,250k --depths 10,50,90 --out "$Output\needles.json"
```

The official needle script builds a haystack from checkout docs/source; contents and actual token counts matter. Record all skips/errors, not only found verdicts. It may reuse prefixes; needle wall-clock times are correctness observations, excluded from speed medians.

Optional coding reproduction: `prepare_code_task.py --model IQ2_XS --out <NEW_DIR>` creates `code-task/`. Use Claude Code 2.1.252 against the local Anthropic endpoint with effort low, max 12 turns, max output 16384, and Read/Edit/Write/Glob/Grep only (no shell). Prompt: “Read TASK.md and the three Python files in this directory. Implement all requested repairs. Only edit settings.py, cache.py, and records.py. Do not run commands or create files.” Then `check_code_task.py --model IQ2_XS --out <NEW_DIR>` scores it. To inspect supplied outputs, pass `--out <REPORT>/coding/<arm>` in a copied scratch report; the checker writes results. No task repair was performed after scoring. The original wording is retained for faithful reproduction; an improved future evaluation should explicitly define empty strings and trimming.

Data:

- results.csv: one row per successful speed request. prompt_ms/decode_ms are measured engine milliseconds; client times seconds; rates tokens/s; memory MiB or GiB as named; temperature °C, clocks MHz, power watts. Prompt count includes the chat template; fresh_prompt_tokens equals prompt_tokens here. Client elapsed includes frontend overhead. No reasoning text was emitted. GPU0=P, GPU1=V1, GPU2=V2.
- summary.csv: 16 configuration/length groups, median/min/max, including client total latency. run-details.json keeps all generated text, warm-ups, usage and engine timing. `timings.predicted_*` means measured decode timing, not speed projection. Request hashes are SHA-256 over json.dumps(request, sort_keys=True); preserved file hashes use saved bytes.
- speculative_acceptance = drafts_accepted/drafts_offered. The recorded generate.cpp adds both MTP and suffix drafts to these counters. mtp_accepted/offered subtract logged suffix and lookup-chain counts; mtp_only_acceptance is the derived ratio. This run had lookup-chain off. Log rows were matched by prompt/output counts, accepted/offered counts and rounded prompt/decode timing. This is a derivation, not a separately exposed API metric.
- Resource columns in results.csv cover speed request windows; resolved.json separately contains complete-arm summaries (B includes extra needles), measured cache allocation, startup duration and outcome. GPU0 includes desktop processes. Throttle values are counts of sampled Active flags, not precise durations. Per-device power maxima cannot be summed into simultaneous whole-system power. Whole-arm resource intervals differ, so they are not a controlled encoder-memory comparison.
- Runtime exit success is separate from correctness score: A/D correctness scripts returned 1 for the coding edge cases, not an API/engine crash. All failed assertions are preserved in correctness.json. Code scaling checks are small timing probes, not proofs of complexity.
- Per-arm manifests record engine/tokenizer/script hashes measured during the experiment. Provenance includes earlier same-day pack/engine verification receipts; compiler flags and release source attestation are unavailable. Scripts are exported with portable default paths; manifests retain their original hashes.

The submission excludes the three one-shot initial screens, earlier afternoon dual-card results, large raw telemetry/SSE, generated image copies (the generator is included), operational launchers and private Claude CLI transcripts. No inference was rerun while preparing this package.
