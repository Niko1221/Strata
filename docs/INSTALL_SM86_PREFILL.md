# Experimental SM86 prefill choices in setup

Setup offers these choices on NVIDIA compute-capability **8.6** cards, including
the RTX 3060. They are **off by default** and require a local CUDA build. Setup
already detects the GPU architecture; no manual CMake command or environment edit
is needed when choosing through the installer.

The available values depend on the checkout. Run `python setup.py --help` to see
the choices compiled into that checkout's installer:

| Choice | Purpose | Runtime setting setup writes |
|---|---|---|
| `off` | Normal engine, optimization disabled | Both SM86 runtime switches set to `0`; previous fused-path choice restored |
| `prefetch-one` | One-superblock weight prefetch (prefetch PR) | `STRATA_PF_PREFETCH_ONE=1` |
| `iq3-stage2` | Two activation stages for the measured IQ3 pairs (stage PR) | `STRATA_PF_IQ3_STAGE2=1` |

Each independent PR initially supplies only its own enabled choice. When both are
available, choose one per model; the two build variants cannot be combined.

## Windows and Linux

Run the usual setup and select the experimental choice when asked. Enter keeps a
new installation off. `--yes` alone never enables an experimental choice, and an
existing choice is remembered.

For scripts, add `--sm86-prefill` to a normal setup command. For example, in the
prefetch checkout:

```sh
# Linux
./setup.sh --setup --model IQ3_XXS --sm86-prefill prefetch-one

# Windows
START-HERE.bat --setup --model IQ3_XXS --sm86-prefill prefetch-one
```

Use `iq3-stage2` instead in the stage-buffering checkout. Other setup options such
as `--context`, `--vision`, `--yes` and `--no-start` work as usual. These are setup
choices, so rerunning setup follows the normal model/settings questions.

An enabled choice:

1. Requires a selected NVIDIA SM86 GPU and bypasses ready-made engines.
2. Uses the normal build-tool installer, passes the matching CMake option, and
   clears the other mutually exclusive option in that variant's CMake cache.
3. Writes `STRATA_PF_FUSED=1` and the selected runtime opt-in into this model's
   configuration; the other SM86 runtime switch is explicitly off.
4. Records the choice in both the model config and the engine's `BUILD.json`.

Enabling native fused prefill can change rounding and answers compared with MMQ.
The measured optimization gain is relative to the existing fused path, not the
whole MMQ-to-fused difference. See the corresponding report in `docs/benchmarks/`
and [DETAILS.md](DETAILS.md). Measurements are on Linux/RTX 3060; other SM86 cards
and Windows performance are not established by those numbers.

## Reuse, updates and turning it off

Variants live separately, for example:

```text
engine-sm86-prefetch-one/
build-sm86-prefetch-one/
engine-sm86-iq3-stage2/
build-sm86-iq3-stage2/
```

CUDA 12 uses `engine-cuda12-sm86-...` and corresponding build directories. Images
use a variant-specific encoder/build cache as well. The normal `engine/` and other
models' variants are not replaced. The build cache checks the selected variant,
source fingerprint, architectures and CPU ISA floor before reusing a binary.

Normal starts and `UPDATE.bat` / `update.sh` retain the saved variant and rebuild
its local engine when needed. An update does not replace it with a plain prebuilt.
If a selected GPU changes, setup adds the needed architecture while retaining SM86
code; the kernel uses its original fallback on non-SM86 CUDA devices. This does not
make it an AMD or Intel optimization.

To disable it, rerun setup for that model:

```sh
./setup.sh --setup --model IQ3_XXS --sm86-prefill off
# Windows: START-HERE.bat --setup --model IQ3_XXS --sm86-prefill off
```

The model returns to the normal engine selection. Setup disables both SM86 runtime
switches and restores the fused-path setting it saved before enabling the feature,
preserving unrelated environment entries. Cached variant binaries remain available
for reuse. Config changes retain the usual `.json.bak` backup.

Use `--setup` for changes rather than combining this choice with `--update`,
`--rollback-engine`, `--calibrate` or `--inspect`. Calibration can be run afterward.
An enabled choice is rejected for HIP/SYCL before installing anything. Their normal
setup paths and all default configuration choices remain unchanged.

## Installer validation

`python -m unittest tools.test_setup_sm86_prefill` exercises interactive and CLI
selection, `--yes` defaults, simulated Windows/Linux policy, dedicated builds,
cache reuse/invalidation, explicit mutually exclusive CMake definitions, saved
choices, opt-out restoration, starts, updates and failed-build config preservation.
It uses mocks and temporary directories: no GPU, compiler or model download.
The existing setup golden tests check that default configs remain byte-identical.
