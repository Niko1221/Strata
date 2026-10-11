@echo off
rem Strata's SYCL (Intel Arc) engine for Windows: strata.exe, strata-device.exe and the probes, compiled with oneAPI's
rem icx and put into <repo>\engine with the oneAPI runtime DLLs they load and a BUILD.json (docs/INTEL_WINDOWS.md).
rem
rem   sycl\tools\build_windows.bat        build (ahead-of-time for the Arc Pro B70, bmg-g31) and package into engine\
rem
rem Needs: the Intel oneAPI Base Toolkit 2026.0.0 or newer (the DPC++/C++ compiler and oneMKL), Visual Studio 2022 or
rem 2026 Build Tools with the C++ workload (cl.exe compiles the C files; icx uses its linker, CRT and Windows SDK),
rem CMake 3.25+ and Ninja (the Build Tools' "C++ CMake tools for Windows" component has both), Python 3.10+ (setup.py
rem fetches the pinned llama.cpp without git; package_windows.py writes BUILD.json).  To run the result: the Intel Arc
rem graphics driver 32.0.101.9034 or newer (Level Zero loader 1.32; with an older one a oneAPI 2026.1.1 engine
rem crashes on its first submit).  No admin rights.  Linux keeps sycl/tools/build.sh.
rem
rem Settings (environment variables, all optional):
rem   STRATA_SYCL_AOT    bmg-g31 (the B70's die; "none" = SPIR-V only, JIT-compiled at run time)
rem   STRATA_SYCL_DOORBELL  fenced | atomic: another reading of the host flags (default: the uncached read hint; sycl_doorbell.hpp)
rem   ONEAPI_ROOT        C:\Program Files (x86)\Intel\oneAPI        BUILD_DIR   <repo>\build-sycl-win
rem   JOBS               RAM in GB / 12, at least 2 (an icx + ocloc pair takes up to about 10 GB at its peak)
rem   STRATA_GGML_DIR    a llama.cpp checkout at the pinned commit (default: setup.get_llama_cpp(), third_party\llama.cpp)
rem   STRATA_BUNDLE_ZE_LOADER=1   also copy oneAPI's ze_loader.dll beside the exes (for a driver older than 101.9034)
setlocal EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "SRC=%%~fI"
if not defined STRATA_SYCL_AOT set "STRATA_SYCL_AOT=bmg-g31"
if not defined BUILD_DIR set "BUILD_DIR=%SRC%\build-sycl-win"
if not defined ONEAPI_ROOT set "ONEAPI_ROOT=%ProgramFiles(x86)%\Intel\oneAPI"
set "AOT=%STRATA_SYCL_AOT%"
if /i "%AOT%"=="none" set "AOT="
set "MISSING="
set "STAGE=1"

rem ---- 1. Visual Studio (vswhere, then vcvars64.bat) and oneAPI (setvars.bat, unless icx is on PATH already)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS="
if exist "%VSWHERE%" for /f "delims=" %%V in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VS=%%V"
if not defined VS set "MISSING=!MISSING! vs"
set "SETVARS="
where icx >nul 2>nul
if errorlevel 1 (
  if exist "%ONEAPI_ROOT%\setvars.bat" (set "SETVARS=%ONEAPI_ROOT%\setvars.bat") else (set "MISSING=!MISSING! oneapi")
)
if defined MISSING goto :refuse
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || (echo vcvars64.bat failed & exit /b 1)
if defined SETVARS call "%SETVARS%" intel64 --force >nul || (echo "%SETVARS%" failed & exit /b 1)
set "STAGE=2"

rem ---- 2. what that environment has: cmake 3.25+, ninja, icx, python 3.10+
set "CMAKE_VER="
for /f "tokens=3" %%V in ('cmake --version 2^>nul ^| findstr /b /c:"cmake version"') do set "CMAKE_VER=%%V"
if not defined CMAKE_VER (
  set "MISSING=!MISSING! cmake"
) else (
  for /f "tokens=1,2 delims=." %%a in ("!CMAKE_VER!") do (
    if %%a lss 3 set "MISSING=!MISSING! cmake"
    if %%a equ 3 if %%b lss 25 set "MISSING=!MISSING! cmake"
  )
)
where ninja >nul 2>nul || set "MISSING=!MISSING! ninja"
where icx >nul 2>nul || set "MISSING=!MISSING! oneapi"
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=python"
if not defined PY set "MISSING=!MISSING! python"
if defined MISSING goto :refuse

rem the Intel GPU driver on this PC: a note only, a build box without the card is fine
set "DRV="
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "$g = (Get-CimInstance Win32_VideoController).Where({ $_.PNPDeviceID -like 'PCI\VEN_8086*DEV_E22*' }); if (-not $g) { 'none' } elseif ([version]$g[0].DriverVersion -lt [version]'32.0.101.9034') { 'old ' + $g[0].DriverVersion } else { 'ok ' + $g[0].DriverVersion }"`) do set "DRV=%%D"
set "DRV_STATE=unknown"
set "DRV_VER="
for /f "tokens=1,2" %%a in ("!DRV!") do (set "DRV_STATE=%%a" & set "DRV_VER=%%b")
if /i "!DRV_STATE!"=="none" echo note: no Intel BMG-G31 GPU - Arc Pro B70 or B65 - on this PC; building anyway
if /i "!DRV_STATE!"=="old" echo WARNING: the Intel graphics driver is !DRV_VER!, older than 32.0.101.9034: its Level Zero loader is older than 1.32 and the engine built with oneAPI 2026.1.1 crashes on its first submit. Install the Arc Pro driver 32.0.101.9034 or newer, or set STRATA_BUNDLE_ZE_LOADER=1
if /i "!DRV_STATE!"=="ok" echo Intel graphics driver !DRV_VER!

rem ---- 3. the pinned llama.cpp (ggml for the CPU experts): setup.py unpacks its zip into third_party\llama.cpp, no git
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%" || exit /b 1
if not defined STRATA_GGML_DIR (
  pushd "%SRC%"
  %PY% -c "import pathlib, setup; pathlib.Path(r'%BUILD_DIR%\llama_dir.txt').write_text(str(setup.get_llama_cpp()), encoding='utf-8')" || (popd & echo setup.get_llama_cpp failed & exit /b 1)
  popd
  set /p STRATA_GGML_DIR=<"%BUILD_DIR%\llama_dir.txt"
)
if not exist "!STRATA_GGML_DIR!\ggml\CMakeLists.txt" (echo no ggml in "!STRATA_GGML_DIR!" - set STRATA_GGML_DIR to a llama.cpp checkout at the pinned commit & exit /b 1)
set "GGML_F=!STRATA_GGML_DIR:\=/!"

rem ---- 4. configure + build (STRATA_PORTABLE: ggml for an AVX2 baseline, as the NVIDIA and AMD zips are built)
if not defined JOBS (
  set "RAM="
  for /f "usebackq delims=" %%M in (`powershell -NoProfile -Command "[int][math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)"`) do set "RAM=%%M"
  if not defined RAM set "RAM=24"
  set /a JOBS=RAM/12
  if !JOBS! lss 2 set "JOBS=2"
)
echo configuring %BUILD_DIR%: AOT "%AOT%", ggml from !STRATA_GGML_DIR!, %JOBS% jobs
cmake -G Ninja -S "%SRC%\sycl" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx "-DSTRATA_SYCL_AOT=%AOT%" "-DSTRATA_GGML_DIR=!GGML_F!" ^
  -DSTRATA_PORTABLE=ON -DSTRATA_SYCL_PROBES=ON "-DSTRATA_SYCL_DOORBELL=%STRATA_SYCL_DOORBELL%" || exit /b 1
set "TARGETS=strata strata-device"
if exist "%SRC%\sycl\probe\CMakeLists.txt" set "TARGETS=!TARGETS! probes"
cmake --build "%BUILD_DIR%" --target !TARGETS! -j %JOBS% || exit /b 1

rem ---- 5. engine\: the programs, the oneAPI runtime DLLs beside them, BUILD.json (sycl\tools\package_windows.py)
set "BUNDLE="
if "%STRATA_BUNDLE_ZE_LOADER%"=="1" set "BUNDLE=--bundle-ze-loader"
%PY% "%SRC%\sycl\tools\package_windows.py" --build "%BUILD_DIR%" --oneapi "%ONEAPI_ROOT%" --aot "%AOT%" %BUNDLE% || exit /b 1
echo.
echo Done: %SRC%\engine\strata.exe - "engine\strata-device.exe --list-devices" shows the cards it sees
endlocal
exit /b 0

:refuse
echo.
echo Cannot build Strata's SYCL engine: this PC lacks what is marked MISSING below. Install it, open a new command
echo prompt and run sycl\tools\build_windows.bat again. ("?": not checked yet, it needs the items above it.)
call :item vs     "Visual Studio 2022 or 2026 Build Tools with the C++ workload (Desktop development with C++; its C++ CMake tools component brings CMake and Ninja)"
call :item oneapi "Intel oneAPI Base Toolkit 2026.0.0 or newer (the DPC++/C++ compiler and oneMKL) at %ONEAPI_ROOT%, or where ONEAPI_ROOT points"
call :item cmake  "CMake 3.25 or newer on PATH (the Build Tools' C++ CMake tools component, or cmake.org)"
call :item ninja  "Ninja on PATH (the same component, or ninja-build.org)"
call :item python "Python 3.10 or newer (python.org; the Microsoft Store alias that only prints an install hint does not count)"
call :item driver "Intel Arc Pro graphics driver 32.0.101.9034 or newer, to run the engine (Level Zero loader 1.32)"
exit /b 1

:item
set "MARK=   ok   "
for %%K in (!MISSING!) do if "%%K"=="%~1" set "MARK=MISSING "
if "!STAGE!"=="1" for %%K in (cmake ninja python) do if "%%K"=="%~1" set "MARK=   ?    "
if "%~1"=="driver" set "MARK=  note  "
echo   [!MARK!] %~2
exit /b 0
