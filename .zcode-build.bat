@echo off
call "D:\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" >nul
set CMAKE=D:\VisualStudio\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
%CMAKE% -S F:\Strata-dev -B F:\Strata-dev\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DSTRATA_BUILD_TESTS=ON -DSTRATA_GGML_DIR=F:/Strata/third_party/llama.cpp 1>F:\Strata-dev\.zcode-cfg.log 2>&1
if errorlevel 1 exit /b 1
%CMAKE% --build F:\Strata-dev\build --target strata sampler_parity grammar_test grammar_registry_test logit_bias_test -j 16 1>F:\Strata-dev\.zcode-build.log 2>&1
