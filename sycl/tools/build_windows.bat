@echo off
setlocal
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo Visual Studio C++ build tools are required.
    exit /b 1
)
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%I"
if not defined VS (
    echo Install the Visual Studio Desktop development with C++ workload.
    exit /b 1
)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
if not defined STRATA_ONEAPI_ROOT if exist "%LOCALAPPDATA%\Strata-tools\oneapi\Library\bin\icx.exe" set "STRATA_ONEAPI_ROOT=%LOCALAPPDATA%\Strata-tools\oneapi\Library"
if defined STRATA_ONEAPI_ROOT (
    set "PATH=%STRATA_ONEAPI_ROOT%\bin;%PATH%"
    set "INCLUDE=%STRATA_ONEAPI_ROOT%\include;%INCLUDE%"
    set "LIB=%STRATA_ONEAPI_ROOT%\lib;%LIB%"
    set "MKLROOT=%STRATA_ONEAPI_ROOT%"
) else (
    where icx >nul 2>nul
    if errorlevel 1 (
        if not defined ONEAPI_ROOT set "ONEAPI_ROOT=%ProgramFiles(x86)%\Intel\oneAPI"
        call "%%ONEAPI_ROOT%%\setvars.bat" intel64 vs2022
        if errorlevel 1 exit /b 1
    )
)
where icx >nul 2>nul
if errorlevel 1 (
    echo Intel oneAPI DPC++ and oneMKL are required. See docs\INTEL_ARC.md.
    exit /b 1
)
cmake -S "%ROOT%\sycl" -B "%ROOT%\build-sycl-win" -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 %*
if errorlevel 1 exit /b 1
cmake --build "%ROOT%\build-sycl-win" --parallel 4
exit /b %errorlevel%
