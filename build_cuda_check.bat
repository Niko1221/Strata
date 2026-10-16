@echo off
rem Build the CUDA engine on this machine: MSVC BuildTools + CUDA 13.3 + the vendored llama.cpp.
rem CMake would otherwise clone llama.cpp from GitHub (no network here), so STRATA_GGML_DIR points
rem at the checkout that ships with the repo.  Arch 120 = the RTX 5060 Ti.
setlocal
set BT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
set MSVC=%BT%\VC\Tools\MSVC\14.44.35207
set SDK=C:\Program Files (x86)\Windows Kits\10
set SDKV=10.0.26100.0
set CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set PATH=%MSVC%\bin\Hostx64\x64;%SDK%\bin\%SDKV%\x64;%BT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%CUDA%\bin;%PATH%
set INCLUDE=%MSVC%\include;%SDK%\Include\%SDKV%\ucrt;%SDK%\Include\%SDKV%\um;%SDK%\Include\%SDKV%\shared
set LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDKV%\ucrt\x64;%SDK%\Lib\%SDKV%\um\x64
set CMAKE=%BT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
"%CMAKE%" -S "%~dp0." -B "%~dp0build-cuda" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 ^
  -DSTRATA_GGML_DIR="%~dp0third_party/llama.cpp" ^
  -DCMAKE_MAKE_PROGRAM="%BT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" || exit /b 1
"%CMAKE%" --build "%~dp0build-cuda" --target %1 || exit /b 1
echo BUILD OK
