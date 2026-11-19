#Requires -Version 5.1
<#
  package_windows_component.ps1 - build this branch's Windows SYCL engine as a Strata engine
  component, and put it where Control and the app look for one.

  What it produces, under <Out>\<version>-windows-x86_64-sycl\:
    strata.exe, the oneAPI runtime DLLs it loads, the MSVC runtime, BUILD.json, and
    engine-component.json (schema 2: every file's SHA-256 and size, the arch tag, the build id).
  That folder is the same thing the release assets carry for Linux SYCL, and the same shape the
  app's bundled-engine installer accepts (apps/host/server/bundled-engine.mjs).

  Usage, from the repo root:
    powershell -NoProfile -ExecutionPolicy Bypass -File sycl\tools\package_windows_component.ps1
    ... -Out D:\somewhere\engine          # keep it out of the app's folders
    ... -Verify                            # hash-check what is already there and stop

  Needs: a built engine (build-sycl-win\strata.exe), Python, the oneAPI install, and the Visual
  Studio redistributable (for MSVCP140/VCRUNTIME140). Nothing is signed: the component keeps an
  empty signature field, as a build box's package does.

  What this does NOT do: make the app's own engine picker choose SYCL on Windows. That needs
  `"sycl": ("linux-x86_64", "windows-x86_64")` in strata_control/targets.py together with the three
  tests that assert the opposite - the product decision in #5236. What it does do: Control lists the
  engine (`strata components engines`), a run config that names it starts, and the app's bundled
  installer accepts the same folder if it is placed in the installer's engine-bundle directory.
#>
param(
    [string]$Engine = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path 'build-sycl-win\strata.exe'),
    [string]$Out = (Join-Path $env:LOCALAPPDATA 'Strata\components\engine'),
    [string]$OneApiRoot = 'C:\Program Files (x86)\Intel\oneAPI',
    [string]$Python = 'python',
    [switch]$Verify,
    [switch]$KeepStage
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$packager = Join-Path $repo 'packaging\package_engine.py'

function Say([string]$m) { Write-Host "  $m" }
function Die([string]$m) { Write-Host "  $m"; exit 1 }

if (-not (Test-Path $packager)) { Die "not a Strata checkout: $packager is missing" }

if ($Verify) {
    if (-not (Test-Path $Out)) { Die "nothing at $Out" }
    $pkg = Get-ChildItem $Out -Directory | Where-Object { $_.Name -like '*-windows-x86_64-sycl' } |
        Sort-Object Name -Descending | Select-Object -First 1
    if (-not $pkg) { Die "no *-windows-x86_64-sycl folder under $Out" }
    & $Python $packager --verify $pkg.FullName
    exit $LASTEXITCODE
}

if (-not (Test-Path $Engine)) {
    Die "no engine at $Engine - build it first (sycl\tools\build_windows.bat)"
}
$versionLine = $null
$setvars = Join-Path $OneApiRoot 'setvars.bat'
if (Test-Path $setvars) {
    # the engine needs the oneAPI runtime DLLs on PATH just to print its version
    $versionLine = (& cmd /c "call `"$setvars`" intel64 >nul 2>&1 && `"$Engine`" --version" 2>&1 | Select-Object -First 1)
} else {
    $versionLine = (& $Engine --version 2>&1 | Select-Object -First 1)
}
if (-not $versionLine) { Die "the engine printed no version - is the oneAPI runtime installed?" }
if ("$versionLine" -notmatch '(?<v>\d+\.\d+\.\d+(?:\.\d+)?)') { Die "cannot read the engine version from '$versionLine'" }
$version = $Matches['v']
Say "engine: $Engine  ($versionLine)"

$git = (git -C $repo rev-parse --short=8 HEAD 2>$null)
if (-not $git) { $git = 'local' }
$buildId = "$git-oneapi2026.2"

$stage = Join-Path ([IO.Path]::GetTempPath()) ("strata-win-component-" + [guid]::NewGuid().ToString('n').Substring(0, 8))
New-Item -ItemType Directory -Force -Path $stage | Out-Null
try {
    Copy-Item $Engine (Join-Path $stage 'strata.exe') -Force
    @{ source = 'local'; version = $version; archs = @('intel-arc'); vision = 'none' } |
        ConvertTo-Json | Set-Content (Join-Path $stage 'BUILD.json') -Encoding utf8
    Say "staged: $stage"

    # Where the DLLs the engine imports live. The packager copies only what is missing from the
    # staged folder, and refuses a package whose imports nothing can satisfy.
    $search = @()
    foreach ($part in @('compiler\latest\bin', 'mkl\latest\bin', 'tbb\latest\bin')) {
        $p = Join-Path $OneApiRoot $part
        if (Test-Path $p) { $search += $p } else { Say "warning: not found - $p" }
    }
    $crt = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio' -Recurse -Filter 'MSVCP140.dll' -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like '*Redist*x64*' } | Select-Object -First 1
    if ($crt) { $search += $crt.Directory.FullName } else { Say 'warning: no Visual Studio redistributable found; MSVCP140/VCRUNTIME140 must be in the package another way' }

    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    $args = @($packager, '--src', $stage, '--backend', 'sycl', '--platform', 'windows-x86_64',
              '--version', $version, '--arch', 'intel-arc', '--build-id', $buildId, '--out', $Out,
              '--no-picture-reader')
    foreach ($s in $search) { $args += @('--search', $s) }
    Say ("running: python " + ($args -join ' '))
    & $Python @args
    if ($LASTEXITCODE -ne 0) { Die "the packager refused (see its output above)" }

    $pkg = Join-Path $Out ("{0}-windows-x86_64-sycl" -f $version)
    Say ""
    Say "component: $pkg"
    Say "  verify: powershell -NoProfile -ExecutionPolicy Bypass -File sycl\tools\package_windows_component.ps1 -Verify"
    Say "  use   : strata components engines    (Control lists it), then a run config whose exe is the package's strata.exe"
    Say "  note  : the app's own engine picker still needs the BACKEND_MATRIX row in #5236"
}
finally {
    if (-not $KeepStage) { Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue }
}
