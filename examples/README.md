# This fork's configs

The configs this fork runs with on a PC with an RTX 3090, an RTX 5070 Ti and 160 GB of RAM, for `serve/server.py`
(their speeds: [COMPARISON.md](../docs/COMPARISON.md)):

- `iq3_s.json`: ISTA-DASLab's GSQ-RCO IQ3_S (~53 GB of RAM taken).
- `ud-q4_k_xl.json`: unsloth's UD-Q4_K_XL (~81 GB of RAM taken).

Their paths are relative to the Strata folder, where the server starts: the engine in `build/`; the model files, the
packs and the MTP draft layer in setup's data folder, `../Strata-data`.

1. Set up IQ3_S once: `START-HERE.bat --setup --model IQ3_S --no-start` downloads the model and prepares its pack and
   the MTP draft layer.
2. Build the engine from this branch with code for both cards (the ready-made engine setup puts in `engine/` is
   upstream's, without this fork's options), with MSVC, CUDA 13, CMake and Ninja (setup puts the last two in
   `.venv\Scripts`):

       cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF ^
         "-DCMAKE_CUDA_ARCHITECTURES=86;120"
       cmake --build build --target strata

   `86;120` is an RTX 30 card and an RTX 50 card. The engine needs the CUDA runtime DLLs (`cudart64_13.dll`,
   `cublas64_13.dll`, `cublasLt64_13.dll`) beside it, on the PATH, or their folder in the config's `lib_dirs`.
3. For UD-Q4_K_XL, download unsloth's four `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf` files into
   `../Strata-data/models/UD-Q4_K_XL` and pack them (seconds):

       .venv\Scripts\python.exe tools\iq_pack.py --out ../Strata-data/packs/ud-q4_k_xl ^
         --gguf ../Strata-data/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

4. Set the GPU numbers, as `nvidia-smi` numbers the cards (here the 5070 Ti is 0 and the 3090 is 1). `--main-gpu`
   runs the model, and nothing else may use that card: a display or another program on it slows generation to a few
   tokens per second. `--second-gpu` holds more experts and may drive the display.
5. Start a config from the Strata folder:

       .venv\Scripts\python.exe serve\server.py --engine strata --config examples\iq3_s.json --port 8081

   (`START-HERE.bat` starts setup's own config.)

On Linux the engine is `./build/strata` and Python `.venv/bin/python`.
