param(
    [Parameter(Mandatory=$true)][string]$IgclIncludeDir,
    [string]$OutputPath
)
$ErrorActionPreference = 'Stop'
if (!$OutputPath) { $OutputPath = Join-Path $PSScriptRoot '..\..\build-sycl-windows\strata-intel-telemetry.exe' }
if (!(Test-Path -LiteralPath "$IgclIncludeDir\igcl_api.h")) { throw 'igcl_api.h not found' }
if (!(Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw 'Run from an x64 Visual Studio developer environment' }
$destination = [System.IO.Path]::GetFullPath($OutputPath)
New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($destination)) | Out-Null
& cl.exe /nologo /EHsc /std:c++17 /O2 /W4 "/I$IgclIncludeDir" "$PSScriptRoot\..\probe\windows_telemetry.cpp" "/Fo$destination.obj" "/Fe$destination"
if ($LASTEXITCODE -ne 0) { throw "Telemetry build failed: $LASTEXITCODE" }
