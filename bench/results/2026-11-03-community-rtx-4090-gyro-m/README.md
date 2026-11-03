# Community benchmark on RTX 4090 (Gyro-M TQ2_0, 262k, k8s)

Measured on 2026-11-03 by em3rik. Standard protocol `strata-qwen38-thinking-v1-draft.1` (draft) on an
RTX 4090 box serving the Gyro-M TQ2_0 quant of Qwen3.8-Flash-Next inside a Kubernetes pod.
One run, status **complete**, 21/21 required requests, sanity 12/12, 2 min 13 s after model load.
Main limitation: the server was already serving (6 prior requests) and its engine options are not
exposed over the API, so several `results.json` fields are `null` with a reason; operator-recorded
values are given below and attributed as such.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090, 24564 MiB dedicated, power limit 550 W, max SM clock 3120 MHz; PCIe link width/speed: not measured
- CPU: Intel Core i7-7700K (4 cores / 8 threads, Kaby Lake, no AVX-512), `powersave` governor (intel_pstate)
- RAM: 61.6 GiB DDR4 dual channel (platform maximum)
- Storage of the model files: PUSKILL 1 TB NVMe SSD (ext4), model fully on the expert page cache during the run
- Board: MSI Z170-Gaming K3
- OS: Ubuntu 26.04.1 LTS host, kernel 7.0.0-42-generic. `results.json` reports Ubuntu 24.04.5 LTS
  because the server and the bench runner run inside the official Strata Docker container (Ubuntu 24.04 base).
- Driver 580.178.04; CUDA 13.0; container runtime: microk8s containerd, `nvidia.com/gpu` device plugin
- Strata Terminal 0.2.8 (build 0.1.54), release build, official `cuda13` docker image (ghcr `niko1221/strata`, retagged into the cluster-local registry); engine binary `0.1.54-linux-x86_64-cuda-cuda13`
- Protocol/runner commit recorded in the result: `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`
- Background workloads: shared production box; light crew / Open WebUI agent traffic; the server had
  answered 6 requests before this run. No other GPU workloads ran during the suite.

## Model and configuration

- Model repository: `agentionai/Qwen3.8-Flash-Next-Gyro-GGUF` (repository head `05892758a67efff0895d23c6df10a6fdb6be8cf4` at report time; file downloaded 2026-10-17; the SHA-256 below equals the current LFS object, so the bytes are pinned regardless)
- Quantization: Gyro M / TQ2_0 (2-bit class); file `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` (91,965,774,272 bytes), SHA-256 `7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f`
- Vision encoder: `mmproj-F16.gguf`, GPU vision enabled (`VISION=yes`)
- Context 262,144 tokens; server capacity 3 slots; KV int8 with `--kv-resident 32768` (setup-chosen at 262k); expert resident budget 34 GiB (setup-chosen)
- MTP draft ON (`--spec 4` family, setup default for the preset); prefill `--prefill auto:32768`
  (winner of the setup-time thorough machine calibration on this box, engine 0.1.54)
- Sampling: model-card defaults (temperature 1.0, top_p 0.95, top_k 20, min_p 0, reasoning_effort xhigh, thinking on and preserved) - see `results.json`
- Track: recorded as **default** in `results.json`. Note: the preset carries the machine-calibrated
  arm from `TUNE=thorough`, which is a setup-generated (not release-stock) choice, so on a strict
  reading this is a tuned-default configuration. We flag it here instead of rewriting the recorded file.

```text
# Container entrypoint derives the launch from these deployment env vars (credentials removed):
MODEL_PATH=/data/Strata-data/models/Gyro-M/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf
CONTEXT=262144 VISION=yes PRESET=gyro PORT=8080 STRATA_AUX_CPUS=auto
# resolved engine (operator record from the setup log; a running server does not expose its args):
strata launch --preset gyro ... --gpu --vision --spec 4 --kv int8 --kv-resident 32768 --prefill auto:32768

# bench invocation (already-running server, so no engine log is attached):
strata bench run --url http://127.0.0.1:8080 --suite standard-v1 --credential-env STRATA_API_KEY \
  --track default --track-notes "release 0.2.8 (build 0.1.54), official docker image cuda13, gyro preset, batch 3, 262k ctx" \
  --background "shared production box; k8s pod; light crew/Open-WebUI agent traffic"
```

## Method

The frozen `bench/prompts/standard-v1` fixtures (corpus `e75eff8b9feea04f...`), loopback HTTP,
streaming, one request at a time, in protocol order: 1 warm-up (excluded), 3 repetitions of
short / 4K / 16K, twelve sanity checks. No retries, no optional concurrency stage. Prompt and decode
rates are the engine's own `timings`; TTFT is the client's monotonic clock to the first non-empty
delta. Expert cache warm (the server had loaded the model ~1 h earlier and answered 6 requests).
Full method text: [RUNNER.md](RUNNER.md). Machine-readable evidence: [results.json](results.json),
per-request raw data: [outputs.jsonl](outputs.jsonl).

## Results

Run `20261103T164154Z-4a85149c`, status complete, 0 failed, 0 skipped, suite 133.8 s after load.

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short (~80 tok) | n/a | 52.6 (45.6-59.4) n=3 | 0.77 (0.53-0.81) | 2.36 (1.74-3.20) | 6.6 (6.0-7.8) | 0/3 |
| prompt-4k (4,093 tok) | 1754.5 (1751.5-1810.9) | 52.7 (50.0-58.1) | 2.36 (2.29-2.37) | 5.91 (5.75-6.07) n=2 | 11.2 (10.4-12.5) | 2/3 |
| prompt-16k (16,386 tok) | 4476.3 (4442.6-4597.5) | 57.5 (55.3-57.9) | 3.71 (3.62-3.74) | 9.37 (8.98-9.52) | 12.6 (12.6-12.9) | 3/3 |

Median (min-max), n=3. Cells as defined by the runner (see [RUNNER.md](RUNNER.md) for the same table
plus phase/sanity tables). Memory: the pod pins ~35 GiB RAM for experts + KV at this configuration;
host had ~6 GiB free at steady state (operator observation, not instrumented during the suite).

Sanity checks: **12 of 12 passed** (instruction 4/4, retrieval 4/4, tool calls 4/4).

## Correctness and limitations

- All twelve sanity checks passed; no request failed or was skipped.
- TTFT here counts the first reasoning token (thinking is on), not the first answer token;
  first-answer times are in the table and in `results.json`.
- These are not cold-model measurements: prompt-template-prefix reuse is the allowed shared prefix;
  the expert cache was warm from prior serving.
- The engine's effective flags could not be read back over the API (`--url` mode); the operator
  values above are from the setup record, not from the running process. `results.json` says `null`
  with a reason for every field the runner could not read.
- Track: recorded `default`; the calibrated `--prefill auto:32768` arm is a setup-time machine
  measurement of this box (see the note under Model and configuration).
- The box is a 2017 4-core/8-thread DDR4 platform; this run's value for the community is precisely
  the weak-CPU/RAM-bandwidth end: a 2026-class desktop with the same GPU and quant will stream
  experts faster. Decode ~52-58 tok/s with MTP on is this CPU class's ceiling at 262k context.
- The same base model on an RTX 5090 box with the UD-Q4_K_XL quant is reported separately in
  [`../2026-11-03-community-rtx-5090-ud-q4-k-xl/README.md`](../2026-11-03-community-rtx-5090-ud-q4-k-xl/README.md).

## Files

- `README.md` - this report
- `RUNNER.md` - the runner's own report for this run
- `results.json` - schema-version-1 machine result (the evidence file)
- `outputs.jsonl` - every request: raw usage, engine timings, client event offsets, final output
- `config.json` - server defaults read back (context, slots, sampling)
- `provenance/` - effective chat template (+SHA-256), corpus manifest, corpus build record
