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
rem The B-series spin bound (sycl/CMakeLists.txt: 20,000 on Windows) is what the window's waits want, and with the MTP
rem drafter on it is worth 73% of decode (measured on an Arc Pro B70: 45.4 against 26.2 tok/s, 4.19 against 2.28
rem tokens per round). Without --mtp the suffix drafter's waits run out inside that bound, so it drafts nothing and
rem every round verifies one token: 21.8 against 68.8 tok/s with --spec 2. Say so rather than let it be a mystery.
echo %* | findstr /c:"--mtp" >NUL || >&2 echo [strata-sycl] no --mtp in the arguments: the build's 20,000 spin bound leaves the suffix drafter without a draft (measured 3.7x slower decode). Rebuild with -DSTRATA_SYCL_SPIN_MAX=2000000 for this configuration.
"%HERE%\%BIN%" %*
