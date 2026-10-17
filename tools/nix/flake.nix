{
  description = "Strata ROCm package and development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
      ...
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        # Allow NVIDIA's CUDA toolkit (nvcc, cudart, cuBLAS, ...).
        config.allowUnfree = true;
        config.cudaSupport = true;
      };

      # setup.py wants CUDA 13.0: the ready-made engine is built with 13.0.88, and 13.2.0/13.2.1 miscompile for sm_120.
      cudaPackages = pkgs.cudaPackages_13_0;
      # One merged toolkit root (bin/nvcc, include/, lib/ with cudart and cuBLAS) for CUDA_HOME / CUDAToolkit_ROOT.
      cuda = cudaPackages.cudatoolkit;
      # The gcc that this nvcc supports as its host compiler.
      stdenv = cudaPackages.backendStdenv;

      # Package evaluation is independent of the CUDA-enabled development shell.
      rocmPkgs = import nixpkgs { inherit system; };
      # The engine (engine/, 0.2.16 layout) builds from the ggml vendored in engine/third_party; the server
      # (terminal/) imports its tools and the vendored strata-gguf-py from the same tree. No downloads.
      src = rocmPkgs.lib.fileset.toSource {
        root = ../..;
        fileset = rocmPkgs.lib.fileset.unions [
          ../../LICENSE
          ../../engine/CMakeLists.txt
          ../../engine/cmake
          ../../engine/include
          ../../engine/src
          ../../engine/tests
          ../../engine/third_party/ggml
          ../../engine/third_party/strata-ggml
          ../../engine/third_party/strata-gguf-py
          ../../engine/third_party/stb
          ../../engine/third_party/zstd-dec
          ../../terminal/serve
          ../../terminal/setup.py
          ../../terminal/strata_cli
          ../../terminal/strata_control
          ../../terminal/tools
        ];
      };
      server = rocmPkgs.callPackage ./server.nix { inherit src; };
      strata = rocmPkgs.callPackage ./package.nix { inherit src server; };
    in
    {
      packages.${system} = {
        default = strata;
        strata-rocm = strata;
        strata-server = server;
      };
      apps.${system} = {
        default = {
          type = "app";
          program = "${strata}/bin/strata";
          inherit (strata) meta;
        };
        strata-server = {
          type = "app";
          program = "${server}/bin/strata-server";
          inherit (server) meta;
        };
      };
      checks.${system} = {
        inherit server;
        rocm = strata;
        package = strata.tests.smoke;
      };
      formatter.${system} = rocmPkgs.nixfmt;

      devShells."${system}" = {
        default = (pkgs.mkShell.override { inherit stdenv; }) {
          packages = with pkgs; [
            # ./setup.sh: makes .venv and runs setup.py
            python3

            # setup.py's engine build. cmake/ninja from PATH: the pip wheels' binaries do not run on NixOS.
            cuda
            cmake
            ninja
          ];

          shellHook = ''
            export CUDA_HOME=${cuda}
            export CUDA_PATH=${cuda}
            export CUDAHOSTCXX=${stdenv.cc}/bin/c++
            # The pip wheels in .venv (numpy, pillow, ...) need libstdc++ and zlib; the engine needs the driver's libcuda.
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:${
              pkgs.lib.makeLibraryPath [
                stdenv.cc.cc.lib
                pkgs.zlib
              ]
            }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          '';
        };
      };
    };
}
