# MI50 tuning layer (prerequisite for the platinum numbers)

The platinum performance figures assume this layer. It is **not** part of the
engine — a box without it runs, just slower and less predictably.

## What and why

- **Memory clock 1000 → 1080 MHz** (`safe-mi50.yaml`), everything else at
  vendor stock: 225 W cap, 1725 MHz core, **no HBM sub-timing changes**.
  Upstream testing showed the gains come from UCLK; sub-timing tightening is
  where the hangs live (documented failures: one timing value → 156k UMC
  errors in a single run, another → GPU hang + host reboot). Our cards are
  mixed Samsung/Hynix, so sub-timings stay at each vendor's own stock.
- Toolchain: `amd-tuning` (YAML profile orchestration) + `upp` + `amdmemtweak`
  from the community project [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906)
  (needs Mike Farah `yq` v4). The profile keys validate against the
  tool's allowlists (`amd-tuning validate --profile safe-mi50.yaml`).

## Install

1. Install `amd-tuning` from ML-gfx906 (`.deb`, or build per its repo).
2. `amd-tuning backup --gpu N --output gpuN-stock.yaml` for every card, and
   record RAS counters (`rocm-smi --showrasinfo UMC GFX SDMA`) as a baseline.
3. Check each card's VBIOS (`/sys/bus/pci/devices/*/vbios_version`) — this
   profile targets `113-D1631711-100`; keys are VBIOS-specific.
4. `amd-tuning diff --gpu N --profile safe-mi50.yaml`, then `apply`, per GPU.
5. **PPTable writes do not survive reboot by themselves** — the driver
   re-latches the factory VBIOS table at init, silently reverting. Install
   `mi50-tuning.service` (edit `PROFILE_PATH`, `systemctl enable`) so the
   profile re-applies at every boot.
6. Verify: `journalctl -u mi50-tuning` shows the Uclk 1000→1080 lines, and
   `pp_dpm_mclk` lists 1080 MHz. Under load, not idle (idle reads the sleep
   state). `rocm-smi --showuse` always reads 0 % on MI50 — use
   `/sys/bus/pci/devices/<BDF>/gpu_busy_percent` for real utilization.

Pitfall we hit: a non-amdgpu display card breaks `amdmemtweak` enumeration
and the service fails silently at boot (cards revert to stock clocks).
Symptom/fix: `journalctl -u mi50-tuning` + `cat /sys/kernel/debug/dri/*/name`.
