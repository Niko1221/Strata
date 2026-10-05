# Strata on Bazzite (and other immutable Fedoras): two AMD cards

For Bazzite and the other read-only-image Fedoras - Silverblue, Kinoite, Aeonite, and community spins like Bluefin:
`/usr` and `/opt` are read-only, `dnf install` does not work, and system packages are added with `rpm-ostree`. This
page covers the one case where that matters for Strata: **two or three AMD Radeon cards of different GPU families
sharing one model** (for example an RX 7900 XTX, gfx1100, with an RX 9070 XT, gfx1201). The AMD backend itself is
[AMD_HIP.md](AMD_HIP.md), sharing a model across cards is [MULTI_GPU.md](MULTI_GPU.md), the ordinary install steps
are [INSTALL.md](INSTALL.md#amd-cards).

**One card needs nothing from this page.** Setup then installs ROCm into `.venv` from AMD's wheels (~10 GB, no sudo,
no compiler packages, nothing layered) - `./setup.sh --backend hip --gpu 0`.

## Why two cards of two families need ROCm on the PC

Setup's AMD path installs ROCm into `.venv` from AMD's TheRock wheels, and each wheel index carries **one GPU
family's** kernels: `gfx110X-dgpu` (gfx1100, gfx1101), `gfx120X-all` (gfx1200, gfx1201), `gfx103X-all` (gfx1030).
One engine binary for two families (`-DCMAKE_HIP_ARCHITECTURES="gfx1100;gfx1201"`) therefore cannot be built from
them, so setup asks for a **system ROCm 7**, which contains both. It looks for one in `/opt/rocm` (AMD's own
packages) or in `/usr` (a distro's packages: hipcc in `/usr/bin`, HIP in `/usr/lib64` and `/usr/include`, ROCm's
clang in `/usr/lib64/rocm/llvm`); `$ROCM_PATH` names one install instead.

## What has to be on the PC

Most immutable Fedoras already have the ROCm *runtime* - check first, because what is missing is the *development*
side, and setup checks exactly these files (`rpm -qa | grep -i rocm` on this PC listed `rocm-hip`, `hipcc`,
`rocm-runtime`, `rocm-comgr`, `rocm-device-libs`, `rocm-smi` and no `-devel` package):

| package | what it owns (Fedora 44 layout) | what needs it |
| --- | --- | --- |
| `rocm-hip-devel` | `/usr/include/hip/`, `/usr/lib64/cmake/hip*` (incl. `hip-lang-config.cmake`), `libamdhip64.so` | CMake's `enable_language(HIP)` (#446) |
| `hipblas-devel` | `/usr/lib64/libhipblas.so*`, its headers, `cmake/hipblas/` | setup's gate for a system ROCm; the engine's matrix products |
| `hipblaslt-devel` | `libhipblaslt.so*`, `/usr/include/hipblaslt/hipblaslt-version.h` | the tuned dense-prefill path (see the tuning note below) |
| `rocm-core-devel` (optional) | `/usr/include/rocm_version.h`, `/usr/lib64/cmake/rocm-core/` | lets setup read the ROCm version; without it setup still uses the install |
| `gcc-c++`, `cmake`, `git` | - | compiling the engine (setup says so when they are missing) |

## Install them (package layering)

```sh
rpm-ostree status | head -20                                    # what is already layered
rpm-ostree install rocm-hip-devel hipblas-devel hipblaslt-devel --reboot
```

- `rpm-ostree` resolves the dependencies itself (`rocm-comgr-devel`, `rocm-runtime-devel`, `binutils`, `gawk`, ...) -
  no `dnf builddep`, and `dnf install` is not available anyway.
- It finalizes a new boot deployment and reboots. `rpm-ostree status` then lists the layered packages; undo with
  `rpm-ostree uninstall rocm-hip-devel hipblas-devel hipblaslt-devel`.
- Bazzite's own docs put package layering after Flatpak, Homebrew, containers/Distrobox and AppImage, and advise
  avoiding it where possible because "layered packages can break system upgrades until removed" ([Package Layering
  - Bazzite Documentation](https://docs.bazzite.gg/Installing_and_Managing_Software/rpm-ostree/)). These are ordinary
  Fedora RPMs that the engine links against at every start, so layering is the normal choice here; if an update ever
  complains, `rpm-ostree uninstall` them and re-run setup (it goes back to the wheels path for one card).

After the reboot, check the four files setup looks for:

```sh
ls /usr/lib64/libhipblas.so* /usr/lib64/cmake/hip-lang/hip-lang-config.cmake \
   /usr/include/hip/hip_runtime.h /usr/include/hipblaslt/hipblaslt-version.h
```

## Run setup

Strata's own files stay in your home directory (`/var/home/<you>`), which is writable: clone the repository there,
and setup writes `.venv`, `build-hip`, `engine/` and the start script into that folder and the model into
`Strata-data` next to it. Nothing goes into `/usr` or `/opt`.

```sh
./setup.sh --check                                    # GPU(s), driver, RAM, disk
./setup.sh --yes --backend hip --gpus 1,0 --no-start  # the first number is the main card
```

Step 4 should then say

```
  [ok] ROCm: /usr (the ROCm installed on this PC)
  Compiling the Strata engine for your AMD GPUs (gfx1100, gfx1201; 10-20 minutes, once) ...
```

instead of the "cards of two GPU families" error. The rest is the same as on any Linux (model download, start
script, server): [INSTALL.md](INSTALL.md), [AI_SETUP.md](AI_SETUP.md). Start it and check it:

```sh
nohup ./run-iq3_xxs.sh > strata-server.out 2>&1 &     # the name setup printed
curl http://127.0.0.1:8080/health
```

`engine/BUILD.json` records what the engine was built for - after a two-card setup its `"archs"` list holds both
cards (`["gfx1100", "gfx1201"]`) and its `"lib_dirs"` point at `/usr/lib64` instead of the `.venv` wheel folders.
An engine built for one card is rebuilt when you add a card of another family, since it does not cover it.

## The tuning table may not apply

Setup uses `tools/hip/<arch>-hipblaslt-<version>.txt` only when it matches both the card and the installed hipBLASLt
version, where the version is `major*100000 + minor*100 + patch` read from
`/usr/include/hipblaslt/hipblaslt-version.h` (1.2.0 -> `100200`, 1.5.0 -> `100500`):

```sh
grep HIPBLASLT_VERSION_ /usr/include/hipblaslt/hipblaslt-version.h
```

The tables in this repository are `gfx1100-hipblaslt-100100`, `gfx1100-hipblaslt-100200`,
`gfx1200-hipblaslt-100202`, `gfx1201-hipblaslt-100202`, `gfx1201-hipblaslt-100500`. Fedora 44 ships hipBLASLt from
the ROCm 7.1.1 release (`hipblaslt-7.1.1-7.fc44`); if its version has no table for one of your cards, setup says so
(`no hipBLASLt tuning table for gfx1100 with hipBLASLt 100500 (have: ...)`) and that card's prompt-time dense
matrix products use plain hipBLAS - slower prompts, same answers. Making a table for your card and version:
[AMD_HIP.md](AMD_HIP.md#tuning-table) (`tools/hip/tune_hipblaslt.cpp`).

## If you would rather not layer anything

- **One card:** `./setup.sh --yes --backend hip --gpu 0` - the wheels path, nothing added to the image. A split only
  pays when no single card holds the model's experts ([AMD_HIP.md](AMD_HIP.md#rdna4-gfx1201)).
- **A container:** build and run Strata inside Distrobox or Toolbox (both ship with Bazzite, and Bazzite's docs
  recommend containers for Linux packages and development workflows) with the `-devel` packages installed inside and
  the cards passed through (`distrobox create --name strata --additional-flags "--device /dev/kfd --device
  /dev/dri"`). The engine then only runs in that container, so the server, the web app and the API live there too.
  Not measured by the maintainers.
- **`rpm-ostree usroverlay`:** a temporary `/usr` overlay you can drop downloaded RPMs into to try a build; it is
  gone at the next reboot.

## When setup still stops

| What setup says | What it means |
| --- | --- |
| `cards of two GPU families (gfx1100, gfx1201) need a system ROCm 7` | no system ROCm 7 found in `/opt/rocm` or `/usr`; the message names what the one on the PC is missing. Install the three packages above, or use one card. |
| `the ROCm in /usr has hipcc but no hipBLAS library` | `hipblas-devel` (and `hipblaslt-devel`) missing. |
| `the ROCm in /usr has no HIP development files (cmake/hip-lang/..., include/hip/hip_runtime.h)` | `rocm-hip-devel` missing (#446). |
| `the ROCm in /usr is 6.x; Strata needs 7.0 or newer` | the distro's ROCm is too old for this backend; setup falls back to the wheels (one family). |
| `a C++ compiler and git are needed to compile the AMD engine` | `rpm-ostree install gcc-c++ cmake git --reboot`. |
| the engine names your card and says it is not in the build's list | `engine/strata` was compiled for another architecture; re-run setup with the cards you want (`--gpus 1,0`). |
| the HIP runtime sees no card | `ls -l /dev/kfd` and `groups`; if your user is not in `render`/`video`, `sudo usermod -aG render,video $USER` (works on Bazzite - `/etc` is writable) and log out and in. |

## What is measured and what is not

- The Fedora layout itself is not new to this project: [AMD_HIP.md](AMD_HIP.md#build) documents a hand-built engine
  on a Fedora-family test host with `-DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++`.
- The package names and file paths above are from Fedora 44's package database and the ROCm spec files
  (`rocclr`, `hipblas`, `hipblaslt`, `rocm-core`), not from a Bazzite machine.
- **setup's automatic use of a `/usr` ROCm (this page) has not been measured end to end on a Fedora Atomic install
  yet.** Please report a run - cards, `rpm-ostree status`, the setup output, the end of `strata-<model>.log`, and the
  speed lines of a first answer - at https://github.com/Niko1221/Strata/issues.
