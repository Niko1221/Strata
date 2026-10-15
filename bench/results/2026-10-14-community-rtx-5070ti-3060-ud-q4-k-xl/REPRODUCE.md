# Reproduce the workload

Use Strata commit `296b59d2f73c8f04a805df29c2fa97655adace78`, a CUDA build for sm_86/sm_120, the pinned four GGUF shards and the same MTP, expert profile and English draft vocabulary. Check hashes against provenance.json. Obtain model/runtime assets through the normal Strata setup; binaries and weights are not redistributed here. The pack was prepared with:

```bash
python tools/iq_pack.py --gguf "$MODEL_DIR/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" --out "$PACK_DIR" --compat-bf16
```

Run `tools/iq_pack.py --help` from the pinned checkout before preparation; pack conversion metadata is included. Preserve existing model/pack directories and use a new output directory. The supplied configuration templates use explicit @PATH@ markers; they are not launchable until rendered. Example from the report folder:

```bash
python3 render_config.py --arm DUAL-BASE --strata /absolute/Strata   --engine /absolute/strata --cuda /absolute/cuda-13.4   --model /absolute/model-shards --pack /absolute/native-pack   --mtp /absolute/mtp/rt --expert-profile /absolute/expert-profile.bin   --output /absolute/new-run
```

The renderer resolves paths, checks required assets and creates a fresh config.json. Change only `--arm` for the four configurations. Physical GPU IDs in templates assume 5070 Ti=0 and 3060=1; verify your enumeration. Keep both cards installed at the measured PCIe x8 layout to repeat this comparison. Retain existing OC; changing it creates a different experiment. The package does not apply OC settings.

Before each run: stop other models/training, check ≥100 GiB MemAvailable, verify no inference listener uses port 8081 and retain a separate monitoring terminal. Start one manual, bounded service from the pinned Strata checkout, using its configured Python environment:

```bash
systemd-run --user --unit=strata-community-bench --collect   --property=MemoryHigh=100G --property=MemoryMax=106G   --property=MemorySwapMax=0 --property=RuntimeMaxSec=7200   --working-directory=/absolute/Strata   /absolute/Strata/.venv/bin/python serve/server.py --engine strata   --config /absolute/new-run/config.json --port 8081
```

Wait for `http://127.0.0.1:8081/health` to be ready. The following helper performs the two original smoke requests, the fixed-nonce official three-repeat/long suite (with its warm-ups), then the nine calibration-prompt probes:

```bash
/absolute/Strata/.venv/bin/python run_workload.py --strata /absolute/Strata   --config /absolute/new-run/config.json --output /absolute/new-run/workload
systemctl --user stop strata-community-bench.service
```

Capture service RAM/swap/events, global MemAvailable/swap, GPU temperature/VRAM/power/clocks/link and physical block-device counters about every two seconds. Resolve the current model mount with `findmnt -T "$MODEL_DIR"` and its parent device with `lsblk`; do not copy a historical NVMe number. The measured harness stopped below 12 GiB available RAM, above 512 MiB additional global swap, at GPU ≥85 °C, repeated CPU ≥95 °C, service swap/OOM or repeated telemetry failures. The portable workload helper does not implement that original local monitor; use equivalent monitoring and these stop thresholds. Cgroup limits are enforced by the launch command above.

Stop the service after each arm, then repeat block 1 DUAL-BASE/SINGLE-BASE/SINGLE-TUNED/DUAL-TUNED and block 2 in reverse. Use separate fresh output directories, preserve OS page cache, retain the same nonce and do not run arms simultaneously. Loading must be timed separately from the official suite and extra probes. No automatic service enable/autostart is needed.

These portable helpers adapt path handling only; the suite and probe algorithms come from the pinned upstream tools. CLI/compilation and mock-based protocol checks were performed offline. They were not rerun on GPUs to create a second set of results. `verify.py` validates the exported original data without GPU access.
