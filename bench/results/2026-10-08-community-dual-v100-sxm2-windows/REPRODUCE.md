# Reproduction and file notes

Use the recorded frontend commit, CUDA 12 engine 0.1.40, model revision and pack/profile hashes. The release build targets sm_70 among other architectures; full release compiler/CMake flags and binary-to-source attestation are unavailable. Large GGUF hashes in provenance.json are installation verification receipts; pack/profile/engine hashes were collected after the benchmark.

Copy config.json to a new scratch location. Replace `<STRATA_ROOT>`, `<DATA_ROOT>` and `<RESULTS_ROOT>` with your checkout, model-data and new-results paths; forward slashes work inside JSON. Adjust only paths and the reproducing machine's V100 UUIDs/indices. The recorded config must not select the desktop GPU. Use a new log/output directory and keep numerical settings unchanged.

From the Strata checkout, launch using the README command and wait for `/health` to show `loaded=true`, `images=true`, `max_context=262144`. The copied configuration supplies CUDA_DEVICE_ORDER=PCI_BUS_ID and the two selected CUDA_VISIBLE_DEVICES UUIDs. Use the private Python environment with psutil installed (measured version 7.2.2).

In a second PowerShell terminal, set `$Report` to this report directory, `$Config` to your materialized config, `$Pack` to the IQ2_XS pack and `$Output` to a new results directory. From the Strata checkout:

```powershell
.\.venv\Scripts\python.exe "$Report\scripts\benchmark.py" --root $PWD --pack $Pack --config $Config --engine-provenance "$PWD\engine-cuda12\BUILD.json" --url http://127.0.0.1:8080 --model qwen3.8-flash-next-iq2_xs --label dual-v100-sxm2-windows-iq2-xs-vision-256k --out "$Output\speed" --targets 4096,32768,128000,258000 --runs 3 --max-tokens 256
.\.venv\Scripts\python.exe tools/needle_bench.py --help
.\.venv\Scripts\python.exe tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k,250k --depths 10,50,90 --timeout 1800 --out "$Output\needles.json"
```

The speed harness contains the complete prompt generator, excluded warm-up and timing/telemetry collection; verify every actual token count and zero prefix reuse. Its exported version only sanitizes installation paths and normalizes line endings. The generic harness comment about expert adaptation does not override the measured engine's profile/no-eviction policy. Generated text/draft acceptance may vary.

The official needle haystack reads source/docs from the checkout, including optional third_party docs; token counts depend on those contents. In this run a local tokenizer preflight checked output reservation before calling the unmodified scorer. Always check actual counts, misses, errors and skips; this report's extension had about 19K tokens of remaining capacity. Do not force an overflowing length on another checkout.

Optional correctness: copy `scripts/correctness.py` into a scratch `scripts/` directory and run it (Pillow 12.3.0 used); it creates its own card and records tool/image requests. `prepare_code_task.py --model IQ2_XS --out <scratch>/correctness` creates the coding fixture; have the local Claude Code launcher read TASK.md and repair only its three modules, then run `check_code_task.py --model IQ2_XS --out <scratch>/correctness`. The supplied repaired fixture can be checked with `--out "$Report\correctness"`. The measured CLI was 2.1.252, effort low, Read/Edit/Write/Glob/Grep only, max 12 turns; no shell tool. Stop the test server afterwards.

CSV units: tokens; `_ms` milliseconds; `_s` seconds; rates tokens/second; RAM bytes; GPU memory MiB, temperature Celsius, clocks MHz, power watts. GPU1/2 are physical indices, mapped to CUDA0/1. Fractions are 0–1. Peaks are sampled, not guaranteed instantaneous maxima. `run-details.json` retains all measured engine durations and generated outputs; its `timings.predicted_*` fields mean measured decode timing, not speed estimates. Full SSE/telemetry, earlier quantization benchmarks and private-source logs remain in the local audit archive.
