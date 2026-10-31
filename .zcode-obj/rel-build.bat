@echo off
rem The release-shaped engine build (packaging/ENGINE-COMPONENT.md, windows cuda13 variant):
rem CUDA 13.0 archs 75;86;89;120 real + 120 PTX, STRATA_PORTABLE=ON (the CUDA DLLs land beside the exe).
call "D:\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" >nul
set CMAKE=D:\VisualStudio\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
%CMAKE% -S F:\Strata-dev -B F:\Strata-dev\build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75-real;86-real;89-real;120-real;120-virtual" -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR=F:/Strata/third_party/llama.cpp 1>F:\Strata-dev\.zcode-obj\rel-cfg.log 2>&1
if errorlevel 1 exit /b 1
%CMAKE% --build F:\Strata-dev\build-release --target strata strata_kq -j 16 1>F:\Strata-dev\.zcode-obj\rel-build.log 2>&1
exit /b %errorlevel%
