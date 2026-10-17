# Reproduction and artifact notes

## Hardware

2x Intel Arc Pro B50 (Battlemage G21, `8086:e212`, 16.2 GB each) on an AMD Ryzen 9 5950X host with 60.7 GiB RAM, Ubuntu 26.04.1 LTS, kernel `7.0.0-38-generic`, `xe` driver, Level Zero 1.28.2, driver packages 26.05.37020.3-1, Intel oneAPI 2026.1.1. GPU firmware `21.1182` on both cards. Bare metal — the engine runs directly on the host, not in a container or VM.

## Software

Strata 0.1.45, commit `477ac13cbe8e582c20484871ce459d20d82df09d`, native SYCL **Release build, JIT** (this host has no `intel-ocloc`, so no `bmg-g21` ahead-of-time build; the first request after a start pays kernel compilation).

## Model

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, Flash-Next GSQ-RCO **IQ3_XXS**. SHA-256 of both shards, the expert profile and the engine binary are in `host-and-run-facts.json`. Shard 2 (the per-layer n-gram lookup table) is byte-identical across quantizations and is stored once on the host, hard-linked between the IQ3_XXS and IQ2_XS builds.

Native pack built with upstream `tools/iq_pack.py` from the release's pinned llama.cpp revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Shipped `data/expert-profile-sycl.bin`, not a custom calibration.

## Configuration

```text
strata --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
       --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
       --expert-profile expert-profile-sycl.bin --expert-cache auto --stream-experts \
       --prefill 1024 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 \
       --kv int8 --vram-reserve-mib 2048 --kv-resident 32768 --layer-split 25
```

`STRATA_VERIFY_NO_HOST=0` is mandatory on this configuration: the second layer-split stage has 6,136 experts the host serves, and with the flag on the engine refuses to start.

## Running the benchmark

With the service already running on its configured port and its active config selected, from the Strata checkout using its Python environment:

```sh
python tools/strata_bench.py --config active-model.json --url http://127.0.0.1:8085 --runs 3 --long --out community-output
```

The speed suite ran against the already-running service, so model loading is not part of any timing. One arithmetic/tool smoke check and the tool's own warm-ups preceded the measured runs.

The needle tool accepts its key only as a command-line argument, so a wrapper invoked its Python entry point in-process to keep the real key out of the OS process list:

```python
import json, sys
from pathlib import Path
sys.path.insert(0, 'tools')
import needle_bench
cfg = json.loads(Path('active-model.json').read_text())
sys.argv = ['needle_bench.py', '--url', 'http://127.0.0.1:8085', '--api-key', cfg['api_key'],
            '--lengths', '32k,128k,262k', '--depths', '10,50,90', '--out', 'needles.json']
raise SystemExit(needle_bench.main())
```

A watchdog polled `sycl-ls` for both `level_zero` devices every three seconds during the run and stopped the suite if either card disappeared. Neither card was lost.

## Files

- `results.json` — upstream sanitized speed-suite output: per-request tokens, engine timings, client latency, cache state and machine fields. The collector's GPU description, RAM and CPU strings are preserved as emitted; the corrected hardware facts are in the README and `host-and-run-facts.json`.
- `needles.json` — upstream synthetic recall output. Times are total client seconds, not decode throughput.
- `host-and-run-facts.json` — hand-collected allowlist: CPU, kernel, OS, GPU model, driver package versions, firmware versions, layer split, engine arguments, engine environment, model revision and SHA-256 digests. No credentials.

## Privacy

Host names, the office LAN address, user name and API key are excluded; the automatic redactor replaced them before anything was written, and an exact API-key check ran over every generated file. The server bind address and port in the automatic config view were replaced with placeholders — this is the only manual edit to the upstream speed JSON. Model names, revisions, hashes and firmware versions are public reproducibility metadata. No private conversations, project files, full run configs, models or generated packs are included.