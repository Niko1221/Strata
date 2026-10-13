@echo off
rem ============================================================
rem  Strata Vulkan probe - build, then run both programs.
rem  Double-click this file. Output stays on screen at the end.
rem ============================================================
setlocal
cd /d "%~dp0"

echo.
echo ############################################################
echo #  Strata Vulkan backend study - feasibility probe
echo #  Machine: RTX 5060 Ti + Radeon VII (MI50 / gfx906)
echo ############################################################
echo.

call build_probe.bat
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  STEP 1/3  Device enumeration
echo  Expect: both cards visible in ONE process, MI50 shows
echo          32 GB and a compute queue, subgroups 32 vs 64.
echo ============================================================
echo.
build\vk_probe.exe
echo.

echo ============================================================
echo  STEP 2/3  Expert GEMV on every device
echo  Same SPIR-V runs on both cards and is checked against a
echo  CPU reference. Expect max_err=0.0000, bad=0/4096.
echo ============================================================
echo.
build\vk_gemv.exe
echo.

echo ============================================================
echo  STEP 3/3  IQ3_XXS expert row (the format the model holds)
echo  Checked against the engine's own integer path. Expect
echo  max_err=0.000000, bad=0/64 - bit-exact, not approximate.
echo ============================================================
echo.
if not exist iq3_test.bin (
  echo  iq3_test.bin is missing. Generate it first:
  echo      python iq3_ref.py
  echo.
) else (
  build\vk_iq3.exe
  echo.
)

echo ============================================================
echo  Done. Scroll up to see the tables.
echo ============================================================
echo.
pause
exit /b 0

:fail
echo.
echo  BUILD FAILED - the compiler error is above this line.
echo.
pause
exit /b 1
