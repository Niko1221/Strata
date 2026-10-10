@echo off
rem Build the SYCL port on Windows with Intel oneAPI (icx/icpx) + Ninja. Run from the Strata checkout:
rem   .\sycl\tools\build.bat [target...]
rem Needs the conda env (or oneAPI) with icx/icpx on PATH plus the VS dev environment
rem (vsdevcmd.bat) for link.exe. sycl/tools/build.sh is the Linux equivalent.
setlocal
set REPO=%~dp0..\..
if defined BUILD_DIR (set B=%BUILD_DIR%) else (set B=%REPO%\build-sycl)
rem The MSVC linker + Windows SDK (link.exe, kernel32.lib): import once per shell, silently when present.
where link >NUL 2>NUL
if errorlevel 1 if exist "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\vsdevcmd.bat" call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\vsdevcmd.bat" -no_logo -arch=x64 >NUL 2>&1
rem Intel compiler runtime (libircmt.lib) beside icx: on LIB for the link, or the link fails with LNK1104.
rem The compiler comes from PATH when the shell has it (an activated conda env) and from the build dir's own
rem CMakeCache.txt otherwise, so this works from a plain prompt too.
set ICX_DIR=
for /f "delims=" %%i in ('where icx 2^>NUL') do if not defined ICX_DIR set "ICX_DIR=%%~dpi"
if not defined ICX_DIR if exist "%B%\CMakeCache.txt" for /f "tokens=2 delims==" %%v in ('findstr /b /c:"CMAKE_CXX_COMPILER:" "%B%\CMakeCache.txt"') do if not defined ICX_DIR set "ICX_DIR=%%~dpv"
if defined ICX_DIR if exist "%ICX_DIR%..\lib\libircmt.lib" set "LIB=%ICX_DIR%..\lib;%LIB%"
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
