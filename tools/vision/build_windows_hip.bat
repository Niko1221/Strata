@echo off
rem Build the optional Windows HIP image encoder. Does not replace a running installation.
rem   set STRATA_HIP_ARCHS=gfx1030
rem   set LLAMA_DIR=C:\path\to\llama.cpp-at-Strata-pinned-commit
rem   tools\vision\build_windows_hip.bat
rem Requires VS C++ Build Tools, Python, CMake and Ninja. ROCm is private to .rocm-win.
rem Use the same STRATA_ROCM_VERSION as engine\BUILD.json; never mix runtime releases.
setlocal EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "SRC=%%~fI"
if not defined STRATA_HIP_ARCHS (echo Set STRATA_HIP_ARCHS to the target card, e.g. gfx1030 for RX 6900 XT. & exit /b 1)
if not defined STRATA_ROCM_VERSION set "STRATA_ROCM_VERSION=10.2.0a20260930"
if not defined STRATA_ROCM_INDEX set "STRATA_ROCM_INDEX=https://nightly.repo.amd.com/rocm/whl-next/"
if not defined ROCM_VENV set "ROCM_VENV=%SRC%\.rocm-win"
if not defined BUILD_DIR set "BUILD_DIR=%SRC%\build-vision-hip"
if not defined LLAMA_DIR set "LLAMA_DIR=%SRC%\third_party\llama.cpp"
if not defined BUILD_JOBS set "BUILD_JOBS=8"
if not exist "%LLAMA_DIR%\tools\mtmd\CMakeLists.txt" (
  echo LLAMA_DIR must point to llama.cpp at the LLAMA_CPP_COMMIT in setup.py.
  exit /b 1
)
set "PY="
py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys" >nul 2>nul && set "PY=python"
if not defined PY (echo Python 3.10+ is needed & exit /b 1)
if not exist "%ROCM_VENV%\Scripts\python.exe" %PY% -m venv "%ROCM_VENV%" || exit /b 1
set "EXTRAS=libraries,devel"
for %%A in (%STRATA_HIP_ARCHS:;= %) do set "EXTRAS=!EXTRAS!,device-%%A"
set "STAMP=%ROCM_VENV%\strata-rocm.txt"
set "WANT=%STRATA_ROCM_VERSION% %EXTRAS%"
set "HAVE="
if exist "%STAMP%" set /p HAVE=<"%STAMP%"
if not "!HAVE!"=="!WANT!" (
  "%ROCM_VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check --index-url "%STRATA_ROCM_INDEX%" "rocm[%EXTRAS%]==%STRATA_ROCM_VERSION%" || exit /b 1
  "%ROCM_VENV%\Scripts\rocm-sdk.exe" init || exit /b 1
  >"%STAMP%" echo !WANT!
)
for /f "delims=" %%R in ('"%ROCM_VENV%\Scripts\rocm-sdk.exe" path --root') do set "ROCM=%%R"
if not exist "%ROCM%\lib\llvm\bin\clang++.exe" (echo ROCm compiler not found & exit /b 1)
set "ROCM_F=%ROCM:\=/%"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS="
if exist "%VSWHERE%" for /f "delims=" %%V in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VS=%%V"
if not defined VS (echo Visual Studio C++ Build Tools are needed & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "HIP_PLATFORM=amd"
set "HIP_PATH=%ROCM%"
set "ROCM_PATH=%ROCM%"
set "HIP_DEVICE_LIB_PATH=%ROCM%\lib\llvm\amdgcn\bitcode"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
rem Pinned ggml uses HIP's C++ driver on Windows, rather than CMake's HIP language.
cmake -G Ninja -S "%SRC%\tools\vision" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release ^
  "-DLLAMA_DIR=%LLAMA_DIR:\=/%" -DSTRATA_VISION_HIP=ON -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=ON ^
  "-DGPU_TARGETS=%STRATA_HIP_ARCHS%" "-DCMAKE_PREFIX_PATH=%ROCM_F%" ^
  "-DCMAKE_C_COMPILER=%ROCM_F%/lib/llvm/bin/clang.exe" "-DCMAKE_CXX_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" ^
  -DGGML_HIP_NO_VMM=ON || exit /b 1
cmake --build "%BUILD_DIR%" --target strata-vision --parallel %BUILD_JOBS% || exit /b 1
echo Built %BUILD_DIR%\bin\strata-vision.exe
echo Deploy beside the matching HIP runtime; see docs/AMD_HIP.md, Windows image encoder.
endlocal
