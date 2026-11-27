# build_hip_gfx906_windows.ps1 — gfx906 engine on Windows, 2026-10-23 (ported to the engine/ tree 2026-11-26)
# The gfx906 engine (Instinct MI50 / Radeon Pro VII / Radeon VII) for Windows, built against HIP 5.x.
#
# Why HIP 5.x: on Windows only the HIP 5.x runtime (amdhip64.dll, installed by the Radeon Pro driver) lists gfx906 cards;
# the 6.x and 7.x runtimes (amdhip64_6.dll, amdhip64_7.dll) do not. So headers, device libraries and import libraries come
# from HIP SDK 5.7. The compiler is the clang of HIP SDK 6.4 (clang 20), because the STL of MSVC 14.43+ refuses clang < 19
# and HIP SDK 5.7 has clang 17. Code object v5: the newest the 5.7 device libraries have, and it runs on the 5.x runtime.
#
# Needs: HIP SDK 5.7 and 6.4 (C:\Program Files\AMD\ROCm\5.7, \6.4), Visual Studio 2022 Build Tools with the C++ workload,
# cmake and ninja on PATH (pip install cmake ninja). The ggml sources are vendored in engine\third_party\strata-ggml.
# Keep the source and build folders on a local disk with short paths (not a synced drive: builds fail there).
#
#   powershell -ExecutionPolicy Bypass -File engine\scripts\build_hip_gfx906_windows.ps1 [-Build C:\src\b906c]
param(
    [string]$Build = "C:\src\b906c",
    [string]$Rocm57 = "C:\Program Files\AMD\ROCm\5.7",
    [string]$Rocm64 = "C:\Program Files\AMD\ROCm\6.4",
    [string]$VsTools = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
)
$ErrorActionPreference = "Stop"
$src = Split-Path -Parent $PSScriptRoot
# The CMake from pip, found before vcvars64 runs: vcvars64 puts Visual Studio's own CMake first on PATH, and that one does not
# see C:\Windows\System32\amdhip64.dll (if(EXISTS) answers no), so the hip-lang package below fails its file check.
$cmakeExe = (Get-Command cmake -ErrorAction SilentlyContinue | Where-Object { $_.Source -notmatch 'Microsoft Visual Studio' } | Select-Object -First 1).Source
if (-not $cmakeExe) {
    $cmakeExe = Get-ChildItem "$env:APPDATA\Python\Python3*\Scripts\cmake.exe", "$env:LOCALAPPDATA\Programs\Python\Python3*\Scripts\cmake.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $cmakeExe) { throw "cmake not found: pip install cmake ninja" }
$scriptsDir = Split-Path -Parent $cmakeExe
$fso = New-Object -ComObject Scripting.FileSystemObject
# 8.3 names: paths with spaces break the HIP flags CMake writes into its generated files
$r57 = $fso.GetFolder($Rocm57).ShortPath -replace '\\', '/'
$r64 = $fso.GetFolder($Rocm64).ShortPath -replace '\\', '/'

# HIP SDK 5.7 ships hip-lang-targets.cmake with AMD's build-machine path (C:/constructicon/...) and expects amdhip64.dll in its
# bin folder (the runtime is the driver's, in System32). A corrected copy goes in the build folder; the SDK is left untouched.
$fix = Join-Path $Build "hip-lang-fix"
$hl = Join-Path $fix "cmake\hip-lang"
New-Item -ItemType Directory -Force $hl | Out-Null
Copy-Item "$Rocm57\lib\cmake\hip-lang\*.cmake" $hl -Force
$f = Join-Path $hl "hip-lang-config.cmake"
(Get-Content $f) -replace '^get_filename_component\(PACKAGE_PREFIX_DIR .*', "set(PACKAGE_PREFIX_DIR `"$r57`")" `
                 -replace '^get_filename_component\(_IMPORT_PREFIX "\$\{_DIR\}/\.\./\.\./\.\./\.\./" ABSOLUTE\)', "set(_IMPORT_PREFIX `"$r57`")" |
    Set-Content $f -Encoding ascii
$f = Join-Path $hl "hip-lang-targets.cmake"
(Get-Content $f) -replace '^set\(_IMPORT_PREFIX "C:/constructicon[^"]*"\)', "set(_IMPORT_PREFIX `"$r57`")" | Set-Content $f -Encoding ascii
$f = Join-Path $hl "hip-lang-targets-release.cmake"
(Get-Content $f) -replace '"\$\{_IMPORT_PREFIX\}/bin/amdhip64\.dll"', '"C:/Windows/System32/amdhip64.dll"' | Set-Content $f -Encoding ascii

$fixFwd = $fix -replace '\\', '/'
$srcFwd = $src -replace '\\', '/'
$buildFwd = $Build -replace '\\', '/'
$cmd = @"
call "$VsTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "PATH=$scriptsDir;$($r57 -replace '/', '\')\bin;%PATH%"
rem hip-config.cmake of HIP SDK 5.7 stops on an empty HIP_PLATFORM
set HIP_PLATFORM=amd
set "HIP_PATH=$($r57 -replace '/', '\')\"
set "ROCM_PATH=$($r57 -replace '/', '\')"
"$cmakeExe" -S $srcFwd -B $buildFwd -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_BUILD_TESTS=OFF -DCMAKE_C_COMPILER=$r64/bin/clang.exe -DCMAKE_CXX_COMPILER=$r64/bin/clang++.exe -DCMAKE_HIP_COMPILER=$r64/bin/clang++.exe -DCMAKE_HIP_COMPILER_ROCM_ROOT=$r57 -DCMAKE_HIP_COMPILER_ROCM_LIB=$fixFwd -DCMAKE_PREFIX_PATH=$r57 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 "-DCMAKE_HIP_FLAGS=-mcode-object-version=5 --rocm-path=$r57 --rocm-device-lib-path=$r57/amdgcn/bitcode" || exit /b 1
ninja -C $buildFwd strata || exit /b 1
"@
$bat = Join-Path $Build "build-gfx906.cmd"
Set-Content $bat $cmd -Encoding ascii
& cmd /c $bat
if ($LASTEXITCODE -ne 0) { throw "gfx906 build failed (exit $LASTEXITCODE)" }
Write-Host "engine: $Build\strata.exe  (run it with $Rocm57\bin on PATH for hipblas.dll / rocblas.dll)"