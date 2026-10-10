@echo off
rem strata-sycl.bat: the SYCL-built engine as a drop-in `exe` for serve/server.py (--engine strata) on Windows.
rem Runs the native build directly (no Docker): stdin/stdout are the serve protocol lines, stderr goes to the log.
rem   STRATA_SYCL_BIN   engine binary relative to the repo (default: build-sycl-aot\strata.exe)
rem   STRATA_SYCL_NAME  unused on Windows (kept for parity with strata-sycl.sh)
rem   ONEAPI_DEVICE_SELECTOR  passed through when set (level_zero:0 default; level_zero:* for a two-card split)
setlocal
set HERE=%~dp0..\..
set BIN=%STRATA_SYCL_BIN:build-sycl-aot/strata=build-sycl-aot\strata.exe%
if "%BIN%"=="%STRATA_SYCL_BIN%" set BIN=%STRATA_SYCL_BIN%
if not defined STRATA_SYCL_BIN set BIN=build-sycl-aot\strata.exe
rem allow forward-slash default from configs: normalize to backslash
set BIN=%BIN:/=\%
if not exist "%HERE%\%BIN%" (
  rem fall back to the non-AOT build
  set BIN=build-sycl\strata.exe
)
rem the port's run-time switches: the device-built verify plan without host handshakes (docs/INTEL.md)
set STRATA_VERIFY_DEVICE_PLAN=1
set STRATA_VERIFY_NO_HOST=1
rem No SYCL command graphs on the OpenCL backend (begin_recording fails: verify: begin capture failed):
rem replay each window's body instead of its captured graph (slower per round, same tokens).
rem Level Zero (level_zero:0, with ze sysman) can unset this again once the driver exposes it.
if not defined STRATA_VERIFY_EAGER set STRATA_VERIFY_EAGER=1
if not defined STRATA_STAGER_THREADS set STRATA_STAGER_THREADS=12
rem The spin bound. Upstream picks it per device at run time from the Linux kernel driver - 20,000 reads under xe (the
rem B-series on Linux), 2,000,000 everywhere else - and intel_gpu_driver() reads /sys/class/drm, so on Windows every
rem card gets the long bound (sycl/include/strata/sycl_queue.hpp, sycl_doorbell.hpp, #1397). On this port's OpenCL
rem backend a window's per-layer waits are expected to run out and give up, so with the MTP drafter on the short bound
rem is worth most of decode: Arc Pro B70, the same binary one STRATA_SPIN_MAX apart, 256 greedy tokens after a
rem 2,049-token prompt, medians of 3 interleaved runs - 44.1 against 24.9 tok/s (+77%), 2.15 against 1.18 tokens per
rem round, and with the long bound 204 of 217 rounds accepted no draft at all. Without --mtp the suffix drafter's own
rem waits want the long bound instead (21.8 against 68.8 tok/s with --spec 2), so the default is left alone there.
rem An A-series card wants the long bound even with --mtp: there the GPU genuinely waits on the host's per-layer CPU
rem expert work and a short bound made it go on with the experts' outputs missing - hence the B-series check.
rem STRATA_SPIN_MAX already in the environment always wins. docs/INTEL.md has both measurements.
if defined STRATA_SPIN_MAX goto :spin_bound
echo %* | findstr /c:"--mtp" >NUL || goto :spin_bound
for /f "usebackq delims=" %%v in (`%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -Command "if ((Get-CimInstance Win32_VideoController | Where-Object { $_.Status -ne 'Error' }).Name -match 'Arc.* B\d') {'yes'} else {'no'}"`) do set BSERIES=%%v
if not "%BSERIES%"=="yes" goto :spin_bound
set STRATA_SPIN_MAX=20000
>&2 echo [strata-sycl] B-series card with --mtp: STRATA_SPIN_MAX=20000. On the OpenCL backend the window's waits are meant to run out, and the long bound cost 44%% of decode here (docs/INTEL.md). Set STRATA_SPIN_MAX yourself to override.
:spin_bound
"%HERE%\%BIN%" %*
