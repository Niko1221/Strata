{
  lib,
  rocmPackages,
  cmake,
  ninja,
  python3,
  runCommand,
  src,
  server,
  # Explicit targets: never probe the build machine's GPU in a sandbox.
  rocmGpuTargets ? [
    "gfx1030"
    "gfx1100"
    "gfx1101"
    "gfx1102"
    "gfx1151"
    "gfx1200"
    "gfx1201"
  ],
  hipblasltSupport ? true,
}:

assert lib.assertMsg (rocmGpuTargets != [ ]) "Strata needs at least one ROCm GPU target";

rocmPackages.rocmClangStdenv.mkDerivation (finalAttrs: {
  pname = "strata-rocm";
  version = "0.1.66";
  inherit src;

  # The engine build is engine/ (0.2.16 layout); its tune tools compile from ../terminal/tools/hip.
  # The cmake hook builds in cmakeBuildDir and configures from its parent, i.e. cmake -S engine.
  cmakeBuildDir = "engine/build";

  strictDeps = true;
  nativeBuildInputs = [
    cmake
    ninja
    python3
  ];
  buildInputs =
    with rocmPackages;
    [
      clr
      hipblas
      rocblas
    ]
    ++ lib.optional hipblasltSupport hipblaslt;

  cmakeFlags = [
    (lib.cmakeBool "STRATA_ENABLE_HIP" true)
    (lib.cmakeBool "STRATA_ENABLE_CUDA" false)
    (lib.cmakeBool "STRATA_ENABLE_SYCL" false)
    (lib.cmakeBool "STRATA_BUILD_TESTS" false)
    (lib.cmakeBool "STRATA_NATIVE_EXPERTS" true)
    # Upstream otherwise forces GGML_NATIVE=ON. AVX2 is the release baseline;
    # Strata's separate AVX-512 kernels still dispatch at runtime.
    (lib.cmakeBool "STRATA_PORTABLE" true)
    # ggml comes from the vendored engine/third_party/strata-ggml (the upstream default; no STRATA_GGML_DIR).
    (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
    (lib.cmakeFeature "CMAKE_HIP_COMPILER" "${rocmPackages.clr.hipClangPath}/clang++")
    (lib.cmakeFeature "CMAKE_HIP_COMPILER_ROCM_ROOT" (toString rocmPackages.clr))
    (lib.cmakeFeature "CMAKE_HIP_ARCHITECTURES" (lib.concatStringsSep ";" rocmGpuTargets))
  ]
  ++ lib.optional (!hipblasltSupport) (lib.cmakeBool "CMAKE_DISABLE_FIND_PACKAGE_hipblaslt" true);

  # Upstream has no install rules and its ALL target includes GPU diagnostics.
  # Build/install only the engine, useful CLI tools and available BLAS tuners.
  buildPhase = ''
    runHook preBuild
    cmake --build . --parallel "$NIX_BUILD_CORES" --target \
      strata strata-device strata-gguf strata-dequant strata-plan tune_rocblas \
      ${lib.optionalString hipblasltSupport "tune_hipblaslt"}
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/bin" "$out/share/strata/hip"
    for program in strata strata-device strata-gguf strata-dequant strata-plan tune_rocblas \
      ${lib.optionalString hipblasltSupport "tune_hipblaslt"}; do
      install -Dm755 "$program" "$out/bin/$program"
    done
    ln -s ${server}/bin/strata-* "$out/bin/"
    install -m644 "$src"/terminal/tools/hip/*.txt "$out/share/strata/hip/"
    install -Dm644 "$src/LICENSE" "$out/share/licenses/strata/LICENSE"
    runHook postInstall
  '';

  # No GPU access in a Nix build. Exercise installed CPU tools and ELF loading,
  # not hipInit or inference. ROCm's own store RPATHs locate its kernel libraries.
  passthru = {
    inherit rocmGpuTargets server;
    tests.smoke = runCommand "strata-rocm-smoke" { nativeBuildInputs = [ python3 ]; } ''
      test -z "''${LD_LIBRARY_PATH-}"
      ${finalAttrs.finalPackage}/bin/strata-plan --help > /dev/null
      ${finalAttrs.finalPackage}/bin/strata-server --help > /dev/null
      python3 ${./test-package.py} ${finalAttrs.finalPackage}
      touch "$out"
    '';
  };

  meta = {
    description = "Strata mixture-of-experts inference engine for AMD GPUs (ROCm)";
    homepage = "https://github.com/Niko1221/Strata";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "strata";
  };
})
