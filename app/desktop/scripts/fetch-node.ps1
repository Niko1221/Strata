# Fetch and verify the private Node runtime the Windows desktop bundle ships (win-x64).
#
# The PowerShell twin of fetch-node.sh: same pinned version, same SHA-256 pin, same refusal to
# install anything whose digest does not match. Not committed output: node.exe is a ~90 MB binary.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File app\desktop\scripts\fetch-node.ps1
#   ... -File fetch-node.ps1 -Check                  # verify what is already installed
#   ... -File fetch-node.ps1 -FromFile node.zip      # verify and install a zip you already have (offline)
#
# The Node version must satisfy the App Host's own engines.node (>= 24.22.3).
[CmdletBinding()]
param(
    [switch]$Check,
    [string]$FromFile = '',
    [string]$Dest = ''
)
$ErrorActionPreference = 'Stop'

$NodeVersion = if ($env:STRATA_NODE_VERSION) { $env:STRATA_NODE_VERSION } else { '24.22.3' }
if ($NodeVersion -ne '24.22.3') { throw 'Node version has no recorded checksum; update both fetch scripts before installing it.' }
$Target = 'win-x64'
# SHA-256 of node-v24.22.3-win-x64.zip from nodejs.org's SHASUMS256.txt; keep equal to fetch-node.sh.
$Expected = 'e612af99f626ab415ea225af6bce5c971feb448cb5cae250bb1c3fa50c5f7d9f'
if (-not $Dest) { $Dest = Join-Path $PSScriptRoot '..\src-tauri\resources\node' }
$Installed = Join-Path $Dest 'node.exe'

function Fail([string]$Message) { [Console]::Error.WriteLine("fetch-node: $Message"); exit 1 }

if ($Check) {
    if (-not (Test-Path -LiteralPath $Installed)) { Fail "not installed at $Installed" }
    if (-not (Test-Path -LiteralPath (Join-Path $Dest 'LICENSE'))) { Fail 'the official Node licence is missing' }
    $got = (& $Installed --version).Trim()
    Write-Output "fetch-node: $Installed is $got"
    if ($got -ne "v$NodeVersion") { Fail "expected v$NodeVersion" }
    exit 0
}

$Archive = "node-v$NodeVersion-$Target.zip"
$Work = Join-Path ([IO.Path]::GetTempPath()) ("strata-node-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Work | Out-Null
try {
    $Zip = Join-Path $Work $Archive
    if ($FromFile) {
        Copy-Item -LiteralPath $FromFile -Destination $Zip
    } else {
        $Url = "https://nodejs.org/dist/v$NodeVersion/$Archive"
        Write-Output "fetch-node: $Url"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Zip
    }
    $hash = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Zip)
    try { $Actual = [BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $hash.Dispose() }
    if ($Actual -ne $Expected) {
        Fail "REFUSING - $Archive hashes $Actual, expected $Expected"
    } else {
        Write-Output 'fetch-node: sha256 verified against the recorded digest'
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zipFile = [IO.Compression.ZipFile]::OpenRead($Zip)
    try {
        $entry = $zipFile.GetEntry("node-v$NodeVersion-$Target/node.exe")
        $license = $zipFile.GetEntry("node-v$NodeVersion-$Target/LICENSE")
        if (-not $entry -or -not $license) { Fail "node.exe or LICENSE is not inside $Archive" }
        New-Item -ItemType Directory -Force -Path $Dest | Out-Null
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $Installed, $true)
        [IO.Compression.ZipFileExtensions]::ExtractToFile($license, (Join-Path $Dest 'LICENSE'), $true)
    } finally { $zipFile.Dispose() }
    Write-Output "fetch-node: installed at $Installed"
} finally {
    Remove-Item -LiteralPath $Work -Recurse -Force -ErrorAction SilentlyContinue
}
