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
rem #1549: the Level Zero loader looks for the ICD in the registry only unless told it may also search the
rem executable's path.  Without it the loader reports "uninitialized", sycl-ls segfaults (0xC0000005), and the
rem engine is left with the OpenCL adapter as the only GPU it can see - which is how this port ended up on
rem OpenCL in the first place.  With it the loader finds the card's ICD and answers one driver.  Measured on
rem sycl-ls, conda oneAPI 2026.1.1, driver 32.0.101.8976; LEVEL_ZERO_FINDINGS.md has the whole diagnosis.
rem Harmless when an adapter is selected by other means.
if not defined ZE_ENABLE_LOADER_DRIVER_SEARCH set ZE_ENABLE_LOADER_DRIVER_SEARCH=1
rem #1549: the UR adapters beside the exe do not carry their own dependencies.  sycl/CMakeLists.txt stages
rem ur_loader.dll and the three ur_adapter_*.dll, but ur_adapter_level_zero.dll also imports UMF.dll and the
rem VCRUNTIME140* set, and without those it fails to load - silently.  The UR loader then reports only the
rem OpenCL adapter, the Level Zero device never appears, and ONEAPI_DEVICE_SELECTOR=level_zero:0 kills the
rem process at device selection (0xC0000409 before main prints anything).  Putting the toolchain's bin on PATH
rem first is the workaround until the copy list is fixed.  Override STRATA_SYCL_RUNTIME_DIR to move it.
if not defined STRATA_SYCL_RUNTIME_DIR set STRATA_SYCL_RUNTIME_DIR=C:\Users\demet\miniforge3\envs\strata-sycl-261\Library\bin
if exist "%STRATA_SYCL_RUNTIME_DIR%\ur_loader.dll" set PATH=%STRATA_SYCL_RUNTIME_DIR%;%PATH%
rem #1549: Level Zero, OPT-IN (STRATA_SYCL_LEVEL_ZERO=1).  Off by default because it is not reliable here
rem yet: with this port's device-plan verify path the window occasionally never gets its doorbell and the
rem engine stalls ("waiting for the GPU to finish the window; the GPU rang 0"), which a long prompt run or a
rem server request hits within a few minutes.  When it does run it is much faster - 108.8 tok/s decode
rem against 66.6 on OpenCL, and the verify graph captured in 276 ms where the eager path skips it - so the
rem settings are kept here and the default is left alone until that is fixed.
rem The Arc Pro driver line's latest (32.0.101.8976) does not expose Level Zero through any SYCL adapter at
rem all; the generic Arc line's 32.0.101.9034 does, with both ext_intel_free_memory and command graphs.
rem ONEAPI_DEVICE_SELECTOR already set in the environment always wins, so a two-card split still works.
if not defined STRATA_SYCL_LEVEL_ZERO goto :lz_done
if not defined ONEAPI_DEVICE_SELECTOR set ONEAPI_DEVICE_SELECTOR=level_zero:0
rem The v1 Level Zero adapter, not v2.  On an Arc Pro B70 / driver 9034 / 2047-token prompt, measured
rem against v2 in the same run: decode 108.8 vs 87.9 tok/s, and the verify graph captured in 276 ms instead
rem of 9495 ms.  v2 is pinned on for graph replay elsewhere (intel/llvm#23073), but on this card that trade
rem is the wrong one.  LEVEL_ZERO_FINDINGS.md has both numbers.
if not defined SYCL_UR_USE_LEVEL_ZERO_V2 set SYCL_UR_USE_LEVEL_ZERO_V2=0
rem sysman: it is what makes ext_intel_free_memory answer, so the expert cache sizes itself from the
rem card instead of a constant.  Harmless where there is no sysman.
if not defined ZES_ENABLE_SYSMAN set ZES_ENABLE_SYSMAN=1
rem NO COMMAND GRAPHS on Level Zero.  With them the verify window's doorbell flag is written by kernels
rem running inside a replayed graph and the host polling for it never sees it: the engine stalls with
rem "waiting for the GPU to finish the window; the GPU rang 0" at KV position ~84, which is ~20 windows
rem into a request and reproducible every time.  With the eager replay (no graph, the window's body
rem re-run on the queue) the same workload is stable: 1000+ windows over 35 requests with no stall,
rem against the graph configuration stalling in under 20.
rem The cost is real and small - decode 63.3 vs 74.5 tok/s in the CLI, and the graph capture (276 ms)
rem is skipped - and the alternative is a server that stops answering.  So this is forced ON here, not
rem left to the launcher's default: an explicit STRATA_VERIFY_EAGER=0 in the environment is the only way
rem back to the graph path.
set STRATA_VERIFY_EAGER=1
>&2 echo [strata-sycl] STRATA_SYCL_LEVEL_ZERO=1: Level Zero, v1 adapter, sysman on, NO graphs (LEVEL_ZERO_FINDINGS.md)
:lz_done
rem No SYCL command graphs on the OpenCL backend (begin_recording fails: verify: begin capture failed):
rem replay each window's body instead of its captured graph (slower per round, same tokens).  Level Zero
rem has graphs, so there this stays unset - which is why the decode figure is +63% over the eager replay
rem (108.8 vs 66.6 tok/s, LEVEL_ZERO_FINDINGS.md).
if not defined STRATA_VERIFY_EAGER echo %ONEAPI_DEVICE_SELECTOR% | findstr /b /c:"level_zero" >NUL || set STRATA_VERIFY_EAGER=1
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
rem #1549: only on the OpenCL backend.  The measurement above is an OpenCL one - there a window's per-layer
rem waits are expected to run out and give up, so a short bound wins.  On Level Zero the doorbell actually
rem fires and the GPU genuinely waits, so the short bound makes a window give up before the GPU has rung:
rem with --mtp on a B70 / driver 9034 the engine stalls with "the GPU rang 0" at layer step 1 and dies with a
rem thread-stack dump.  The long bound (the default, 2,000,000) is the right one there - the Level Zero CLI
rem runs above do not set it and decode at 108.8 tok/s, which is the whole +63% over OpenCL.
echo %ONEAPI_DEVICE_SELECTOR% | findstr /b /c:"level_zero" >NUL && goto :spin_bound
echo %* | findstr /c:"--mtp" >NUL || goto :spin_bound
for /f "usebackq delims=" %%v in (`%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -Command "if ((Get-CimInstance Win32_VideoController | Where-Object { $_.Status -ne 'Error' }).Name -match 'Arc.* B\d') {'yes'} else {'no'}"`) do set BSERIES=%%v
if not "%BSERIES%"=="yes" goto :spin_bound
set STRATA_SPIN_MAX=20000
>&2 echo [strata-sycl] B-series card with --mtp: STRATA_SPIN_MAX=20000. On the OpenCL backend the window's waits are meant to run out, and the long bound cost 44%% of decode here (docs/INTEL.md). Set STRATA_SPIN_MAX yourself to override.
:spin_bound
"%HERE%\%BIN%" %*
