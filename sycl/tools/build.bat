@echo off
rem Build the SYCL port on Windows with Intel oneAPI (icx/icpx) + Ninja. Run from the Strata checkout:
rem   .\sycl\tools\build.bat [target...]
rem Needs the conda env (or oneAPI) with icx/icpx on PATH plus the VS dev environment
rem (vsdevcmd.bat) for link.exe. sycl\tools\build.sh is the Linux equivalent.
setlocal
set REPO=%~dp0..\..
if defined BUILD_DIR (set B=%BUILD_DIR%) else (set B=%REPO%\build-sycl)
if not exist "%B%\build.ninja" (
  rem conda's own activation uses icx for both C and CXX; icpx defaults to GNU-like flags on Windows
  rem while CMake drives MSVC-like flags, so CXX must be icx too. Release: a Debug device link
  rem (per-kernel modules through the spir64 backend) exhausts the offload wrapper's memory.
  cmake -S "%REPO%\sycl" -B "%B%" -G Ninja "-DCMAKE_C_COMPILER=icx" "-DCMAKE_CXX_COMPILER=icx" "-DCMAKE_BUILD_TYPE=Release" "-DSTRATA_SYCL_AOT=%AOT%"
  if errorlevel 1 exit /b 1
)
if "%~1"=="" (
  cmake --build "%B%" -j %JOBS%
) else (
  cmake --build "%B%" -j %JOBS% --target %*
)
echo BUILD EXIT %ERRORLEVEL%
