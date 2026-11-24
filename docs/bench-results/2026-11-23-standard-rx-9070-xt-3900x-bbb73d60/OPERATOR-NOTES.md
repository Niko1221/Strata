# Operator notes for run `20261123T042737Z-bbb73d60`

Facts the runner could not record (`missing` in results.json), checked on the machine by the operator.
The generated `README.md`, `results.json` and `outputs.jsonl` are unedited.

## Hardware

- GPU: AMD Radeon RX 9070 XT 16 GB (gfx1201). The runner reports only "AMD Radeon"; `lspci` gives
  `1002:7550 (rev c0)` and the card's hwmon power cap is 317 W (`power1_cap`), which identifies the XT.
  PCIe 4.0 x16 (Strata hardware view).
- CPU: Ryzen 9 3900X (Zen 2, AVX2, no AVX-512).
- RAM: 4 x 32 GB DDR4-3200, dual channel (`dmidecode`: configured 3200 MT/s, two `CMK64GX4M2E3200C16` kits).
- Model storage: Crucial T500 1 TB NVMe, PCIe 4.0 x4, CPU-attached.

## Software

- Ubuntu 26.04.1 LTS, kernel 7.0.0-44-generic, in-kernel `amdgpu`; `linux-firmware` 20260319.
- No system ROCm installed. Strata 0.2.20 terminal with the prebuilt `hip-rx9070` engine 0.1.66.

## Model and configuration

- Model: Qwen3.8-Flash-Next, ISTA-DASLab GSQ-RCO IQ3_S pack, downloaded by Strata setup. Repository revision not
  recorded; file SHA-256 hashes are in results.json.
- The run config was written by `strata launch` setup on 2026-11-22 and not edited afterwards. Setup choices,
  as shown by setup: `strata launch --model iq3_s --context 256k --vision cpu --tune skip --access local --port 8080`
  (drafting Auto/MTP, parallel 1, KV int8, no tuning, no advanced settings changed). Engine arguments are in `config.json`.

## How this run was made

- Command: `strata bench run --profile launch-iq3_s-256k --no-start`, against the server that was already running.
- `--track` was not passed, so the report says `custom`. The configuration is the unmodified setup output described
  above; no custom pack or setting change was used.
- **Not a freshly loaded engine.** Before this run the server had answered 95 requests from interactive Claude Code
  coding sessions (about 1.5 hours), so the expert cache was warmed by a coding workload.
- During the run: the Strata terminal monitor and an idle Claude Code session were open; a desktop session with a
  browser was running. No other requests reached the server: its lifetime count went from 95 before this run to 117
  before the next one (95 + 22 benchmark requests).
- Engine log: not captured. The runner did not start the engine (`--no-start`) and no engine log file was kept.

## Related run

`../2026-11-23-standard-rx-9070-xt-3900x-1e463ce6/` is a second run on the same engine about five minutes later,
with `--track default`. It is `nonconforming` because the 4K and 16K prompts were still cached from this run.
