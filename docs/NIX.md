# Nix (Linux, AMD ROCm)

Strata's Nix files live in `tools/nix`. Its flake's default package builds the HIP engine from source and
includes the Python API server, web app and model preparation tools. It does not run the installer, create a
virtualenv, download models, or use CUDA. The existing CUDA development shell is the same flake's `devShell` and
is unchanged; you do not need to enter it.

```sh
nix build ./tools/nix#strata-rocm --cores 4
./result/bin/strata-device --list-devices
./result/bin/strata --help
./result/bin/strata-server --help
```

The build needs no GPU. Running the engine needs the kernel's amdgpu/KFD driver and access to
`/dev/kfd` and the appropriate `/dev/dri/renderD*` device. On NixOS, enable `hardware.graphics.enable`
and give the account access to the `render` and `video` groups. Do not install another ROCm into a
virtualenv or export `LD_LIBRARY_PATH`: the engine and ROCm libraries use Nix store RPATHs. The package
also works outside `nix develop`.

## What is built

- `strata`, `strata-device`, `strata-plan`, `strata-gguf`, `strata-dequant`.
- `tune_rocblas` and `tune_hipblaslt`.
- `strata-server`, `strata-iq-pack`, `strata-mtp-pack`, `strata-mtp-rt`, `strata-mtp-fetch`,
  `strata-tokenizer`.
- Strata's own HIP prompt kernels and native IQ CPU experts, on the ggml vendored in the repository at
  `engine/third_party/strata-ggml`. The packages fetch nothing at build time.
- An AVX2 CPU baseline, not `-march=native`. The engine's AVX-512 kernels retain runtime dispatch.

ROCm comes from the locked nixpkgs (currently 7.2.3). The default engine contains code for
`gfx1030`, `gfx1100`, `gfx1101`, `gfx1102`, `gfx1151`, `gfx1200` and `gfx1201`.
This is a build target list, not a claim that every card has been tested with this package. See
[AMD_HIP.md](AMD_HIP.md) for upstream hardware validation and limits. The separate wave64 gfx906
backend is not packaged here.

The Python-only `nix build ./tools/nix#strata-server` avoids the ROCm dependency. Both packages use Python
libraries from the same locked nixpkgs, not the installer's pip requirements.

## Choose GPU targets

Compiling fewer architectures saves build time and disk space. In a consuming flake:

```nix
inputs.strata.url = "github:Niko1221/Strata?dir=tools/nix";

# Inside outputs, for example:
packages.x86_64-linux.strata =
  inputs.strata.packages.x86_64-linux.strata-rocm.override {
    rocmGpuTargets = [ "gfx1100" ];
    # hipblasltSupport = false; # optional, e.g. for RDNA2-only builds
  };
```

For a local checkout without another flake:

```sh
nix build --cores 4 --impure --expr '
  let f = builtins.getFlake (toString ./tools/nix);
  in f.packages.x86_64-linux.strata-rocm.override {
    rocmGpuTargets = [ "gfx1100" ];
  }'
```

This customizes the engine only; it leaves nixpkgs' cached ROCm libraries intact. Unsupported targets
are rejected by upstream CMake. Multi-architecture builds use a single binary, not runtime compilation
for whichever GPU happens to be attached to the build machine.

## Models and API server

Keep models, packs, config, logs and caches in a writable directory outside the store. Follow
[AMD_HIP.md](AMD_HIP.md#model-and-serving-configuration) for model preparation and engine arguments;
the packaged tools replace the corresponding `python tools/*.py` commands. Pass explicit output
paths to the packing tools. Model downloads are deliberate user operations, never part of `nix build`.

For an existing server JSON, set `exe` to the installed engine's **absolute path**, `backend` to `hip`,
`cwd` to your writable model directory, and keep your model's own tokenizer and engine arguments.
For example, after `nix build`, obtain the executable path with:

```sh
readlink -f result/bin/strata
```

Do not copy the installer's `lib_dirs` or loader-path entries from `env`: Nix already links ROCm.
An absolute engine path is important: the server resolves relative `exe` paths against `cwd`, not PATH.

```sh
./result/bin/strata-server --engine strata --config /path/to/strata-hip.json \
  --host 127.0.0.1 --port 8080
```

Keep loopback binding unless you also set `--api-key`. Optional vision/image-generation binaries are
not built by this package. No performance or numerical switches are enabled in the environment.

Tuning tables are installed under `result/share/strata/hip/`. They are not enabled automatically:
solution IDs must match the GPU and BLAS version. If you opt in, use `STRATA_HIPBLASLT_TUNING` or
`STRATA_ROCBLAS_TUNING` in the JSON's `env`, following [AMD_HIP.md](AMD_HIP.md#tuning-table).

## Checks

```sh
nix flake check ./tools/nix --no-build # evaluate all outputs, including the existing dev shell
nix flake check ./tools/nix --cores 4 # build HIP, run Python tests and installed-package smoke tests
```

The checks do not need a GPU or model. They check installed ELF loading without `LD_LIBRARY_PATH`,
CLI startup, and the mock server's health, web assets and model listing from a directory outside the
checkout. They do not validate inference, GPU kernels, throughput or model quality.
