param(
  [Parameter(Mandatory=$true)][ValidateSet('Check','Install')][string]$Mode,
  [Parameter(Mandatory=$true)][string]$TempRoot,
  [Parameter(Mandatory=$true)][string]$ProgramsRoot,
  [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ZipSha256,
  [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ManifestSha256,
  [Parameter(Mandatory=$true)][int64]$ZipBytes,
  [Parameter(Mandatory=$true)][int64]$UnpackBytes,
  [Parameter(Mandatory=$true)][int64]$ProgramBytes,
  [Parameter(Mandatory=$true)][int64]$ComponentBytes
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$reserve=[int64]6*1024*1024*1024
function No-Reparse([string]$path) {
  $current=[IO.Path]::GetFullPath($path)
  while ($null -ne $current) {
    if ([IO.Directory]::Exists($current) -or [IO.File]::Exists($current)) {
      if (([IO.File]::GetAttributes($current) -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Reparse path refused' }
    }
    $parent=[IO.Directory]::GetParent($current)
    $current=if ($null -eq $parent) { $null } else { $parent.FullName }
  }
}
function Digest([string]$path) {
  $stream=[IO.File]::OpenRead($path);$sha=[Security.Cryptography.SHA256]::Create()
  try { return [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
  finally { $sha.Dispose();$stream.Dispose() }
}
$TempRoot=[IO.Path]::GetFullPath($TempRoot)
$ProgramsRoot=[IO.Path]::GetFullPath($ProgramsRoot)
$components=[IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'Strata\components'))
foreach ($path in @($TempRoot,$ProgramsRoot,$components)) { No-Reparse $path }
if (-not [IO.Directory]::Exists($TempRoot)) { throw 'Installer scratch directory missing' }
if ([IO.Directory]::Exists($ProgramsRoot) -or [IO.File]::Exists($ProgramsRoot)) { throw 'Install folder already exists. Choose a new folder.' }
foreach ($bytes in @($ZipBytes,$UnpackBytes,$ProgramBytes,$ComponentBytes)) {
  if ($bytes -le 0) { throw 'Invalid installer resource budget' }
}
$budgets=@{}
function Add-Budget([string]$path,[int64]$bytes) {
  $drive=[IO.Path]::GetPathRoot($path)
  if (-not $budgets.ContainsKey($drive)) { $budgets[$drive]=0L }
  $budgets[$drive]+=$bytes
}
# Before payload publication include the ZIP copy, expanded package and final
# files. The running setup EXE is already included in the free-space snapshot.
Add-Budget $TempRoot ($UnpackBytes + $(if ($Mode -eq 'Check') { $ZipBytes } else { 0L }))
Add-Budget $ProgramsRoot $ProgramBytes
Add-Budget $components $ComponentBytes
foreach ($drive in $budgets.Keys) {
  if ([IO.DriveInfo]::new($drive).AvailableFreeSpace -lt ($budgets[$drive]+$reserve)) {
    throw "Insufficient space on $drive for setup staging, installation and the 6 GiB reserve."
  }
}
if ($Mode -eq 'Check') { Write-Output 'Setup resource admission passed.'; exit 0 }
$zip=Join-Path $TempRoot 'full-package.zip'
No-Reparse $zip
if (([IO.FileInfo]$zip).Length -ne $ZipBytes -or (Digest $zip) -ne $ZipSha256) { throw 'Embedded package hash mismatch' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$extract=Join-Path $TempRoot ('p-'+[guid]::NewGuid().ToString('N').Substring(0,8))
if ([IO.Directory]::Exists($extract) -or [IO.File]::Exists($extract)) { throw 'Extraction target occupied' }
[IO.Compression.ZipFile]::ExtractToDirectory($zip,$extract)
$manifest=Join-Path $extract 'full-windows.json'
No-Reparse $manifest
if ((Digest $manifest) -ne $ManifestSha256) { throw 'Extracted manifest hash mismatch' }
$data=Get-Content -LiteralPath $manifest -Raw -Encoding UTF8 | ConvertFrom-Json
$installer=Join-Path $extract 'Install-Strata.ps1'
No-Reparse $installer
if ((Digest $installer) -ne [string]$data.public.installer_sha256) { throw 'Extracted installer hash mismatch' }
$shortcutRoot=Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
& $installer -ExpectedManifestSha256 $ManifestSha256 -ProgramsRoot $ProgramsRoot -ComponentsRoot $components -ShortcutRoot $shortcutRoot
if (-not $?) { throw 'Full package installation failed' }
Write-Output 'Strata installed. Use its Start Menu preview shortcut.'
