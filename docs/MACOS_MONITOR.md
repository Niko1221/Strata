# macOS Monitor

The existing macOS IOAccelerator reader remains the default. To opt into the
extra Apple Silicon charts, install [macmon](https://github.com/vladkens/macmon)
and start the server with `STRATA_MACOS_TELEMETRY=1`:

```sh
brew install macmon
STRATA_MACOS_TELEMETRY=1 python -m serve.server --engine strata --config strata.json
```

Use your normal server arguments and run config. The setting only affects native
Apple Silicon macOS processes; NVIDIA, AMD and Intel Mac readers keep their
existing behavior. Unset the variable to return to the original Monitor.

No sudo, background service, extra port, Python binding or GPU workload is needed.
Strata starts one `macmon pipe --soc-info -i 1000` process and closes it when the
server exits. The helper is optional: CPU, RAM, disk and IORegistry readings still
work with the opt-in enabled and no macmon installed. CPU/RAM/disk use `psutil`,
as on the other platforms.

The eight cards show speed, GPU load, GPU allocation, GPU temperature, GPU power,
GPU frequency, CPU load and disk reads. The Apple-specific meanings are:

- **GPU load:** whole-device active residency from macmon, falling back to the
  driver's `Device Utilization %`. It includes other applications.
- **GPU allocation:** `Alloc system memory` from the GPU driver, across all apps,
  against total physical unified memory. These are allocated buffers, not dedicated
  VRAM, system RAM use, Strata's own allocation, or the memory available to a model.
  Model sizing continues to use the existing Metal reader and working-set budget;
  it never subtracts these driver allocations from physical RAM.
- **GPU temperature:** average GPU die temperature reported by macmon.
- **GPU power:** GPU domain watts, not whole-system/CPU power or charger draw.
- **GPU frequency:** current active GPU clock in MHz. Apple Silicon uses shared
  memory rather than a discrete GPU PCIe link, so this replaces the PCIe card.

IORegistry and macmon's IOReport/SMC sensors depend on the macOS version and chip.
A missing sensor stays unavailable; a helper reading older than five seconds is
discarded. A stopped or stalled helper is closed and retried after 30 seconds.
With the opt-in enabled, charts leave gaps for unavailable data and keep real zero
readings. The default chart paths and NVIDIA/AMD telemetry retain their existing behavior.

With strata-mlx, the expert-cache bar shows the configured `expert_memory_gb`
budget (decimal GB converted to the dashboard's binary GB). It is labeled as a
budget rather than a measured count of resident experts.

Validation on an M3 Pro (18 GPU cores, 36 GiB unified memory) with macmon 0.10.0
returned all five Apple GPU readings during a streamed Flash Next generation using
the earlier v0.1.40-based server fork. The current-main port is also tested with
real sensors and the mock HTTP engine. No native Metal/CUDA/HIP/SYCL engine source
or build changes are included. Tests without hardware:

```sh
python -m unittest serve.test_apple_telemetry serve.test_server.AmdTelemetry
node serve/test_charts.cjs
```
