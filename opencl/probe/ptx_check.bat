@echo off
rem Does nvcc fuse `acc += d * c` into an FFMA at the flags the engine builds with?  The OpenCL
rem row kernel has to round the same way, so the answer decides FP_CONTRACT here.
setlocal
set BT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
set MSVC=%BT%\VC\Tools\MSVC\14.44.35207
set SDK=C:\Program Files (x86)\Windows Kits\10
set SDKV=10.0.26100.0
set CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set PATH=%MSVC%\bin\Hostx64\x64;%SDK%\bin\%SDKV%\x64;%CUDA%\bin;%PATH%
set INCLUDE=%MSVC%\include;%SDK%\Include\%SDKV%\ucrt;%SDK%\Include\%SDKV%\um;%SDK%\Include\%SDKV%\shared
set LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDKV%\ucrt\x64;%SDK%\Lib\%SDKV%\um\x64
"%CUDA%\bin\nvcc.exe" -O3 -arch=sm_120 -ptx "%~dp0fmac_tmp.cu" -o "%~dp0fmac_tmp.ptx" || exit /b 1
echo ---- k (natural `acc += d * c`) ----
findstr /C:"mul.f32" /C:"add.f32" /C:"fma.rn.f32" "%~dp0fmac_tmp.ptx" > "%~dp0lines.txt"
echo ---- k2 (explicit __fmul_rn / __fadd_rn) ----
"%CUDA%\bin\nvcc.exe" -O3 -arch=sm_120 -ptx -fmad=false "%~dp0fmac_tmp.cu" -o "%~dp0fmac_nofmad.ptx" || exit /b 1
echo BUILD OK
