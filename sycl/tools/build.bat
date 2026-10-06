@echo off
rem Build the SYCL port on Windows with Intel oneAPI (icx/icpx) + Ninja. sycl\tools\build.bat [target...]
rem Usage: call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" ^|^| call setvars.bat, then sycl\tools\build.bat
setlocal
set REPO=%~dp0..\..
if defined BUILD_DIR (set B=%BUILD_DIR%) else (set B=%REPO%\build-sycl)
if not exist "%B%\build.ninja" (
  cmake -S "%REPO%\sycl" -B "%B%" -G Ninja "-DCMAKE_C_COMPILER=icx" "-DCMAKE_CXX_COMPILER=icpx" "-DSTRATA_SYCL_AOT=%AOT%" 2>&1 | tail -15
  if errorlevel 1 exit /b 1
)
cmake --build "%B%" -j %JOBS% %* 2>&1 | findstr /R "FAILED error: build stopped Linking ^\[[0-9]*\/[0-9]*\] Linking"
echo BUILD EXIT %ERRORLEVEL%
