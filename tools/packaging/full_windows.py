"""Build and verify a source-pinned, fresh-install Windows Strata ZIP.

The private spec enumerates every source file and its SHA-256. The public
manifest contains only package-relative names, sizes, hashes, and source
commits. Installed user state, model weights, and old program trees are never
inputs to the installer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

from native_preview import check_source, no_reparse, relative_name, sha256
import binary_deps

MANIFEST = "full-windows.json"
INSTALLER = "Install-Strata.ps1"
STARTER = "START-HERE.cmd"
WEBVIEW2 = "prerequisites/MicrosoftEdgeWebView2RuntimeInstallerX64.exe"
WEBVIEW2_SHA256 = "9b0142e22e8daa92855d746ae4d1e9eeef561ccaf67d1dd452165c4af3df3191"
SCHEMA = "strata-full-windows-v1"
RESERVE_BUILD = 10 * 1024**3
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
FOLDER = re.compile(r"[A-Za-z0-9_+.-]+\Z")
FORBIDDEN_SUFFIXES = {".gguf", ".safetensors", ".dmp", ".pdb", ".pyc", ".log", ".pfx", ".p12", ".key", ".sqlite"}
FORBIDDEN_PARTS = {"__pycache__", ".git", ".cache", "examples", ".ssh", ".aws"}
FORBIDDEN_BASENAMES = {".env", ".npmrc", ".pypirc", "id_rsa", "id_ed25519"}
WINDOWS_PLATFORM_IMPORTS = {
    "bcryptprimitives.dll", "combase.dll", "cabinet.dll", "d3dcompiler_47.dll",
    "dwrite.dll", "usp10.dll", "d2d1.dll", "icu.dll",
}
CONTROL_TOOLS = {
    "_paths.py", "ab_engine.py", "agent_turns_bench.py", "calibrate.py", "canonical_xcheck.py",
    "dflash2_pack.py", "draft_vocab.py", "dsv41/pack_gguf.py", "dsv41/smoke.py", "dsv41/tune.py",
    "gguf_reader.py", "gguf_writer.py", "iq_pack.py", "mtp_fetch.py", "mtp_pack.py", "mtp_rt.py",
    "nvfp4_convert.py", "nvfp4_local_pack.py", "nvfp4_verify.py", "pack_index.py", "pack_layer.py",
    "quantscope.py", "serve_load.py", "strata_bench.py", "strata_inspect.py", "strata_mcp.py",
    "strata_pack.py", "strata_tokenizer.py",
}
CONTROL_METAL_TOOLS = {"metal/tools/metal_repack.py", "metal/tools/stream_experts.py"}

# This script runs on stock Windows PowerShell 5.1. It accepts only an empty
# program target and previously absent component folders. It never consults
# PATH for a package manager, Python, Node, or an engine.
#
# Product folder name comes from repository-root VERSION (Hakua product SemVer).
def product_version() -> str:
    return (Path(__file__).resolve().parents[1] / "VERSION").read_text(encoding="utf-8").strip()


def programs_folder() -> str:
    return f"Strata-{product_version()}"


INSTALLER_SCRIPT_TEMPLATE = r'''param(
  [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedManifestSha256,
  [string]$ProgramsRoot = (Join-Path $env:LOCALAPPDATA 'Programs\__PROGRAMS_FOLDER__'),
  [string]$ComponentsRoot = (Join-Path $env:LOCALAPPDATA 'Strata\components'),
  [string]$ShortcutRoot = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$reserve = [int64]6 * 1024 * 1024 * 1024
$package = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$manifestPath = Join-Path $package 'full-windows.json'
function Require-Plain([string]$value) {
  if ($value -notmatch '^[A-Za-z0-9_+.-]+$' -or $value -eq '.' -or $value -eq '..') { throw 'Unsafe component folder' }
}
function Require-Relative([string]$value) {
  if ($value -notmatch '^[^\\/:<>"|?*\r\n]+(?:/[^\\/:<>"|?*\r\n]+)*$') { throw "Unsafe relative path: $value" }
  foreach ($part in $value.Split('/')) {
    if ($part -eq '.' -or $part -eq '..' -or $part.EndsWith('.') -or $part.EndsWith(' ') -or
        $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') { throw "Unsafe relative path: $value" }
  }
}
function Require-NoReparse([string]$value) {
  $current = [IO.Path]::GetFullPath($value)
  while ($null -ne $current) {
    if ([IO.File]::Exists($current) -or [IO.Directory]::Exists($current)) {
      if (([IO.File]::GetAttributes($current) -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Reparse point: $current"
      }
    }
    $parent = [IO.Directory]::GetParent($current)
    $current = if ($null -eq $parent) { $null } else { $parent.FullName }
  }
}
function Digest([string]$value) {
  $stream = [IO.File]::OpenRead($value)
  $algorithm = [Security.Cryptography.SHA256]::Create()
  try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
  finally { $algorithm.Dispose(); $stream.Dispose() }
}
function Child([string]$root, [string]$name) {
  Require-Relative $name
  $path = $root
  foreach ($part in $name.Split('/')) { $path = Join-Path $path $part }
  Require-NoReparse $path
  return $path
}
function Verify-File([string]$path, $item) {
  Require-NoReparse $path
  if (-not [IO.File]::Exists($path)) { throw "Missing file: $($item.path)" }
  if (([IO.FileInfo]$path).Length -ne [int64]$item.bytes -or (Digest $path) -ne [string]$item.sha256) {
    throw "Hash or size mismatch: $($item.path)"
  }
}
function Require-EmptyTarget([string]$path) {
  Require-NoReparse $path
  if ([IO.File]::Exists($path) -or [IO.Directory]::Exists($path)) { throw "Target already exists: $path" }
}
function Owned-Folder([string]$path, [string]$markerName, [string]$markerText) {
  Require-NoReparse $path
  if (-not [IO.Directory]::Exists($path)) { return $false }
  $marker = Join-Path $path $markerName
  Require-NoReparse $marker
  return [IO.File]::Exists($marker) -and [IO.File]::ReadAllText($marker) -ceq $markerText
}
function Free-Bytes([string]$path) {
  $root = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($path))
  return [IO.DriveInfo]::new($root).AvailableFreeSpace
}
function Has-WebView2 {
  $id = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
  foreach ($hive in @('HKLM','HKCU')) {
    foreach ($base in @('SOFTWARE\Microsoft\EdgeUpdate\Clients','SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients')) {
      $key = "$hive`:\$base\$id"
      try {
        $pv = [string](Get-ItemProperty -LiteralPath $key -ErrorAction Stop).pv
        if ($pv -match '^\d+\.\d+\.\d+\.\d+$' -and $pv -ne '0.0.0.0') { return $true }
      } catch { }
    }
  }
  return $false
}

Require-NoReparse $package
Require-NoReparse $manifestPath
if ((Digest $manifestPath) -ne $ExpectedManifestSha256.ToLowerInvariant()) { throw 'Manifest SHA-256 differs from release pin' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.schema -ne 'strata-full-windows-v1') { throw 'Unsupported package manifest' }
Require-Plain ([string]$manifest.public.native_folder)
Require-Plain ([string]$manifest.public.generic_folder)
Require-Plain ([string]$manifest.public.programs_folder)
if ($manifest.public.programs_folder -ne '__PROGRAMS_FOLDER__') { throw 'Unexpected program version' }
if ((Digest (Join-Path $package 'Install-Strata.ps1')) -ne [string]$manifest.public.installer_sha256) { throw 'Installer hash mismatch' }
$starterExpected = "-ExpectedManifestSha256 $($ExpectedManifestSha256.ToLowerInvariant())"
if (-not ([IO.File]::ReadAllText((Join-Path $package 'START-HERE.cmd')).Contains($starterExpected))) { throw 'Starter does not pin this manifest' }
$seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
$groups = @{'programs'=0L;'engine'=0L;'generic-gguf'=0L}
foreach ($item in $manifest.files) {
  Require-Relative ([string]$item.path)
  if (-not $seen.Add([string]$item.path)) { throw 'Duplicate package path' }
  $name = [string]$item.path
  if ($name.StartsWith('programs/')) { $groups['programs'] += [int64]$item.bytes }
  elseif ($name.StartsWith("components/engine/$($manifest.public.native_folder)/")) { $groups['engine'] += [int64]$item.bytes }
  elseif ($name.StartsWith("components/generic-gguf/$($manifest.public.generic_folder)/")) { $groups['generic-gguf'] += [int64]$item.bytes }
  elseif ($name.StartsWith('licenses/') -or $name -eq 'README.md' -or $name -eq 'prerequisites/MicrosoftEdgeWebView2RuntimeInstallerX64.exe') { }
  else { throw "Unexpected installed payload: $name" }
  Verify-File (Child $package $name) $item
}
$actual = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($file in Get-ChildItem -LiteralPath $package -Recurse -Force -File) {
  Require-NoReparse $file.FullName
  $rel = $file.FullName.Substring($package.Length + 1).Replace('\','/')
  if ($rel -ne 'full-windows.json' -and $rel -ne 'Install-Strata.ps1' -and $rel -ne 'START-HERE.cmd') { [void]$actual.Add($rel) }
}
if ($actual.Count -ne $seen.Count) { throw 'Package membership differs from manifest' }
foreach ($dir in Get-ChildItem -LiteralPath $package -Recurse -Force -Directory) { Require-NoReparse $dir.FullName }
foreach ($name in $actual) { if (-not $seen.Contains($name)) { throw "Unlisted package file: $name" } }

$programs = [IO.Path]::GetFullPath($ProgramsRoot)
$components = [IO.Path]::GetFullPath($ComponentsRoot)
$defaultPrograms = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'Programs\__PROGRAMS_FOLDER__'))
$defaultComponents = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'Strata\components'))
if ($programs -eq $components -or $programs.StartsWith($components + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
    $components.StartsWith($programs + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Install roots overlap' }
$native = Join-Path (Join-Path $components 'engine') ([string]$manifest.public.native_folder)
$generic = Join-Path (Join-Path $components 'generic-gguf') ([string]$manifest.public.generic_folder)
$pointer = Join-Path (Join-Path $components 'generic-gguf') 'active.json'
foreach ($target in @($programs,$native,$generic,$pointer)) { Require-EmptyTarget $target }
$bytesByDrive = @{}
foreach ($entry in @(@($programs,$groups['programs']),@($native,$groups['engine']),@($generic,$groups['generic-gguf']))) {
  $drive = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath([string]$entry[0]))
  if (-not $bytesByDrive.ContainsKey($drive)) { $bytesByDrive[$drive] = 0L }
  $bytesByDrive[$drive] += [int64]$entry[1]
}
foreach ($drive in $bytesByDrive.Keys) {
  if ((Free-Bytes $drive) -lt ($bytesByDrive[$drive] + $reserve)) { throw "Insufficient free disk space on $drive" }
}
if (-not (Has-WebView2)) {
  $webview = Child $package 'prerequisites/MicrosoftEdgeWebView2RuntimeInstallerX64.exe'
  $webviewItem = @($manifest.files | Where-Object { $_.path -eq 'prerequisites/MicrosoftEdgeWebView2RuntimeInstallerX64.exe' })
  if ($webviewItem.Count -ne 1) { throw 'Offline WebView2 runtime is missing' }
  Verify-File $webview $webviewItem[0]
  $process = Start-Process -FilePath $webview -ArgumentList @('/silent','/install') -Wait -PassThru -WindowStyle Hidden
  if ($process.ExitCode -ne 0 -or -not (Has-WebView2)) { throw "WebView2 installation did not complete (exit $($process.ExitCode))" }
}
foreach ($drive in $bytesByDrive.Keys) {
  if ((Free-Bytes $drive) -lt ($bytesByDrive[$drive] + $reserve)) { throw "Insufficient free disk space after WebView2 setup on $drive" }
}
$id = [guid]::NewGuid().ToString('N')
$programStage = "$programs.install-$id"
$nativeStage = "$native.install-$id"
$genericStage = "$generic.install-$id"
$stages = @{'programs'=$programStage;'engine'=$nativeStage;'generic-gguf'=$genericStage}
foreach ($target in @($programStage,$nativeStage,$genericStage)) { Require-EmptyTarget $target }
$receiptPath = Join-Path $components "install-full-$id.json"
$published = New-Object System.Collections.ArrayList
$markerName = ".strata-install-owner-$id"
$markerText = "strata-full-windows-v1:$($ExpectedManifestSha256.ToLowerInvariant()):$id"
$shortcutStatus = 'not-created'
$shortcut = ''
try {
  foreach ($target in @($programStage,$nativeStage,$genericStage)) {
    New-Item -ItemType Directory -Path $target -Force | Out-Null
    [IO.File]::WriteAllText((Join-Path $target $markerName),$markerText,[Text.UTF8Encoding]::new($false))
  }
  foreach ($item in $manifest.files) {
    $pieces = ([string]$item.path).Split('/')
    if ($pieces[0] -eq 'licenses' -or $pieces[0] -eq 'prerequisites' -or [string]$item.path -eq 'README.md') { continue }
    if ($pieces[0] -eq 'programs') { $destination = $programStage; $offset = 1 }
    else { $destination = $stages[$pieces[1]]; $offset = 3 }
    for ($i=$offset; $i -lt $pieces.Length; $i++) { $destination = Join-Path $destination $pieces[$i] }
    $parent = Split-Path -Parent $destination
    if (-not [IO.Directory]::Exists($parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Copy-Item -LiteralPath (Child $package ([string]$item.path)) -Destination $destination -ErrorAction Stop
    Verify-File $destination $item
  }
  foreach ($target in @($programs,$native,$generic,$pointer)) { Require-EmptyTarget $target }
  [IO.Directory]::Move($nativeStage,$native); [void]$published.Add($native)
  [IO.Directory]::Move($genericStage,$generic); [void]$published.Add($generic)
  [IO.Directory]::Move($programStage,$programs); [void]$published.Add($programs)
  $pointerTemp = "$pointer.install-$id"
  Require-EmptyTarget $pointerTemp
  $content = '{"folder":"' + [string]$manifest.public.generic_folder + '","install_owner":"' + $id + '"}' + "`n"
  [IO.File]::WriteAllText($pointerTemp,$content,[Text.UTF8Encoding]::new($false))
  Require-EmptyTarget $pointer
  [IO.File]::Move($pointerTemp,$pointer); [void]$published.Add($pointer)
  $shortcutStatus = 'custom-root'
  if ($ShortcutRoot -ne '' -or ($programs -eq $defaultPrograms -and $components -eq $defaultComponents)) {
    $shortcutBase = if ($ShortcutRoot -ne '') { [IO.Path]::GetFullPath($ShortcutRoot) } else { Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs' }
    Require-NoReparse $shortcutBase
    if (-not [IO.Directory]::Exists($shortcutBase)) { [IO.Directory]::CreateDirectory($shortcutBase) | Out-Null }
    $shortcut = Join-Path $shortcutBase 'Strata Preview 0.3.2 (SM120).lnk'
    Require-NoReparse $shortcut
    if ([IO.File]::Exists($shortcut) -or [IO.Directory]::Exists($shortcut)) { $shortcutStatus = 'existing-kept' }
    else {
      try {
        $wsh = New-Object -ComObject WScript.Shell
        $link = $wsh.CreateShortcut($shortcut)
        $link.TargetPath = Join-Path $programs 'strata-desktop.exe'
        $link.WorkingDirectory = $programs
        $link.Description = 'Strata Windows preview'
        $link.Save()
        $shortcutStatus = 'created'
      } catch { $shortcutStatus = "failed:$($_.Exception.Message)" }
    }
  }
  $receipt = @{status='success';schema='strata-full-windows-install-v1';manifest_sha256=$ExpectedManifestSha256.ToLowerInvariant();programs=$programs;components=$components;native_folder=$manifest.public.native_folder;generic_folder=$manifest.public.generic_folder;shortcut=$shortcutStatus;inference_tested=$false}
  [IO.File]::WriteAllText($receiptPath,($receipt | ConvertTo-Json -Compress),[Text.UTF8Encoding]::new($false))
  $receipt | ConvertTo-Json -Compress
} catch {
  $failure = $_.Exception.Message
  $quarantined = @()
  if ($shortcutStatus -eq 'created' -and [IO.File]::Exists($shortcut)) {
    try {
      Require-NoReparse $shortcut
      $check = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcut)
      if ([IO.Path]::GetFullPath($check.TargetPath) -eq [IO.Path]::GetFullPath((Join-Path $programs 'strata-desktop.exe'))) {
        [IO.File]::Move($shortcut,"$shortcut.failed-$id")
        $quarantined += "$shortcut.failed-$id"
      } else { $quarantined += "RETAINED-CONCURRENT:$shortcut" }
    } catch { $quarantined += "RETAINED:$shortcut" }
  }
  if ($published.Contains($pointer) -and [IO.File]::Exists($pointer)) {
    try {
      Require-NoReparse $pointer
      if ([IO.File]::ReadAllText($pointer) -eq $content) {
        [IO.File]::Move($pointer,"$pointer.failed-$id")
        $quarantined += "$pointer.failed-$id"
      } else { $quarantined += "RETAINED-CONCURRENT:$pointer" }
    } catch { $quarantined += "RETAINED:$pointer" }
  }
  foreach ($entry in @(@($programs,$programStage),@($native,$nativeStage),@($generic,$genericStage))) {
    foreach ($path in $entry) {
      try {
        if (Owned-Folder $path $markerName $markerText) {
          $failed = "$path.failed-$id"
          Require-EmptyTarget $failed
          [IO.Directory]::Move($path,$failed)
          $quarantined += $failed
        }
      } catch { $quarantined += "RETAINED:$path" }
    }
  }
  $receipt = @{status='failed';schema='strata-full-windows-install-v1';manifest_sha256=$ExpectedManifestSha256.ToLowerInvariant();error=$failure;quarantined=$quarantined;inference_tested=$false}
  try { [IO.File]::WriteAllText($receiptPath,($receipt | ConvertTo-Json -Compress),[Text.UTF8Encoding]::new($false)) } catch { }
  throw "Installation failed: $failure. Quarantined paths: $($quarantined -join ', ')"
}
'''

STARTER_TEMPLATE = r'''@echo off
setlocal
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-Strata.ps1" -ExpectedManifestSha256 {manifest_sha256} %*
exit /b %ERRORLEVEL%
'''


def installer_script() -> str:
    folder = programs_folder()
    if not FOLDER.fullmatch(folder):
        raise ValueError(f"unsafe programs folder from VERSION: {folder}")
    return INSTALLER_SCRIPT_TEMPLATE.replace("__PROGRAMS_FOLDER__", folder)


def _folder(value: str) -> str:
    if not isinstance(value, str) or not FOLDER.fullmatch(value) or value in (".", ".."):
        raise ValueError("unsafe component folder")
    return value


def _payload_name(value: str, native: str, generic: str, neutral_assets: set[str]) -> str:
    name = relative_name(value)
    parts = name.split("/")
    lower = [part.lower() for part in parts]
    if any(part in FORBIDDEN_PARTS or part.startswith(".env") for part in lower):
        raise ValueError(f"forbidden payload path: {name}")
    if parts[-1].lower() in FORBIDDEN_BASENAMES or Path(name).suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"forbidden payload file: {name}")
    allowed = (
        name == "programs/strata-desktop.exe"
        or name.startswith("programs/host/server/")
        or name.startswith("programs/host/dist/")
        or name == "programs/host/package.json"
        or name == "programs/host/hakua-chatbot.enabled"
        or name == "programs/node/node.exe"
        or name.startswith("programs/control/")
        or name.startswith("programs/harness/")
        or name.startswith("programs/licenses/")
        or name == "programs/README.md"
        or name.startswith(f"components/engine/{native}/")
        or name.startswith(f"components/generic-gguf/{generic}/")
        or name.startswith("licenses/")
        or name == WEBVIEW2
        or name in ("README.md",)
    )
    if not allowed:
        raise ValueError(f"unexpected payload path: {name}")
    if name.startswith("programs/control/"):
        rel = name[len("programs/control/"):]
        top = rel.split("/")[0]
        if top not in {
            "BUNDLE.json", "COMMIT", "VERSION", "README.txt", "strata.py", "setup.py",
            "requirements-monitor.txt", "requirements.txt", "bin", "data", "python", "ref",
            "serve", "strata_cli", "strata_control", "tools", "metal",
        }:
            raise ValueError(f"unapproved Control file: {name}")
        if top == "tools" and rel[len("tools/"):] not in CONTROL_TOOLS:
            raise ValueError(f"unapproved Control tool: {name}")
        if top == "metal" and rel not in CONTROL_METAL_TOOLS:
            raise ValueError(f"unapproved Control Metal tool: {name}")
        if top == "ref" and rel != "ref/load.py":
            raise ValueError(f"unapproved Control reference file: {name}")
        if top == "data" and "experimental-speed-projection" in rel.split("/"):
            raise ValueError(f"experimental model payload is not distributable: {name}")
    if name.startswith("programs/harness/"):
        rel = name[len("programs/harness/"):]
        top = rel.split("/")[0]
        if top not in {"BUNDLE.json", "LICENSE", "package.json", "README.i18n.yaml", "THIRD_PARTY_NOTICES.md", "lib", "node_modules"}:
            raise ValueError(f"unapproved Harness file: {name}")
    if any(p.startswith("gtwalsheim") for p in lower):
        raise ValueError(f"unlicensed static font: {name}")
    if name.startswith("programs/host/dist/") and (
        "brand" in lower
        or parts[-1].lower().startswith("strata-logo")
        or parts[-1].lower() == "sprite.svg"
        or parts[-1].lower().startswith("favicon") or parts[-1].lower().startswith("icon-")
        or parts[-1].lower() in {"icon.png", "icon.ico", "apple-touch-icon.png"}
        or parts[-1].lower() == "site.webmanifest"
    ) and name not in neutral_assets:
        raise ValueError(f"unlicensed static asset: {name}")
    return name


def _manifest(spec: dict) -> dict:
    public = spec["public"]
    if not COMMIT.fullmatch(public["source_commit"]):
        raise ValueError("source_commit must be a full SHA")
    native = _folder(public["native_folder"])
    generic = _folder(public["generic_folder"])
    programs = _folder(public["programs_folder"])
    if programs != programs_folder():
        raise ValueError("unexpected program folder")
    neutral = public.get("neutral_assets", {})
    if not isinstance(neutral, dict) or any(not SHA.fullmatch(value) for value in neutral.values()):
        raise ValueError("invalid neutral asset pins")
    neutral_names = set(neutral)
    if any(not name.startswith("programs/host/dist/") or relative_name(name) != name for name in neutral_names):
        raise ValueError("invalid neutral asset path")
    files = []
    seen = set()
    for item in spec["files"]:
        name = _payload_name(item["path"], native, generic, neutral_names)
        key = name.casefold()
        if key in seen or name in (MANIFEST, INSTALLER, STARTER):
            raise ValueError("duplicate or reserved payload path")
        seen.add(key)
        if not SHA.fullmatch(item["sha256"]) or not isinstance(item["bytes"], int) or item["bytes"] < 0:
            raise ValueError(f"invalid file pin: {name}")
        files.append({"path": name, "sha256": item["sha256"], "bytes": item["bytes"]})
    names = {item["path"] for item in files}
    pins = {item["path"]: item["sha256"] for item in files}
    if pins.get(WEBVIEW2) != WEBVIEW2_SHA256:
        raise ValueError("offline WebView2 payload differs from verified Microsoft installer")
    if any(pins.get(name) != digest for name, digest in neutral.items()):
        raise ValueError("neutral asset does not match its public pin")
    required = {
        "programs/strata-desktop.exe", "programs/host/server/index.mjs", "programs/host/dist/index.html",
        "programs/README.md",
        "programs/host/package.json",
        # Hakua fork: ship the enable marker next to package.json so ROOT/hakua-chatbot.enabled
        # is present after NSIS/full install (capabilities.hakuaChatbot default-on for this fork).
        "programs/host/hakua-chatbot.enabled",
        "programs/node/node.exe", "programs/control/BUNDLE.json",
        "programs/control/python/python.exe", "programs/control/strata.py", "programs/harness/BUNDLE.json",
        "programs/harness/lib/bin.js",
        WEBVIEW2,
        f"components/engine/{native}/engine-component.json",
        f"components/engine/{native}/strata.exe",
        f"components/generic-gguf/{generic}/llama-server.exe",
    }
    if not required <= names:
        raise ValueError(f"missing required runtime files: {sorted(required - names)}")
    if not any(n.startswith("programs/licenses/") for n in names):
        raise ValueError("installed third-party notices are required")
    if re.search(r"[A-Za-z]:[/\\]|/Users/|/home/", json.dumps(public)):
        raise ValueError("private path in public metadata")
    return {"schema": SCHEMA, "public": public, "files": sorted(files, key=lambda x: x["path"])}


def audit_auxiliary(stage: Path) -> dict[str, list[str]]:
    """Require every shell, Node, Python and Harness PE import in the payload or Windows."""
    program = stage / "programs"
    groups = {
        "shell": [program / "strata-desktop.exe"],
        "node": [program / "node" / "node.exe"],
        "python": [p for p in (program / "control" / "python").rglob("*")
                   if p.is_file() and p.suffix.lower() in {".exe", ".dll", ".pyd"}],
        "harness": [p for p in (program / "harness").rglob("*")
                    if p.is_file() and p.suffix.lower() in {".exe", ".dll", ".node"}],
    }
    if not groups["python"] or not groups["harness"]:
        raise ValueError("missing private Python or Harness binary inventory")
    python_home = program / "control" / "python"
    node_home = program / "node"
    numpy_root = python_home / "Lib" / "site-packages" / "numpy"
    numpy_libs = python_home / "Lib" / "site-packages" / "numpy.libs"
    numpy_init = (numpy_root / "__init__.py").read_text(encoding="utf-8") if (numpy_root / "__init__.py").exists() else ""
    numpy_adds_libraries = "os.add_dll_directory(libs_dir)" in numpy_init and "'numpy.libs'" in numpy_init
    directory_cache = {}

    def names(directory: Path) -> set[str]:
        if directory not in directory_cache:
            directory_cache[directory] = {p.name.casefold() for p in directory.iterdir() if p.is_file()}
        return directory_cache[directory]

    graph = {}
    for group, paths in groups.items():
        for path in sorted(paths):
            no_reparse(path)
            available = names(path.parent).copy()
            if group == "python":
                available |= names(python_home)
                if numpy_adds_libraries and path.is_relative_to(numpy_root):
                    available |= names(numpy_libs)
            if group == "harness":
                available |= names(node_home)
            imports = binary_deps.pe_imports(path)
            missing = sorted({name for name in imports if name.casefold() not in available
                              and name.casefold() not in WINDOWS_PLATFORM_IMPORTS
                              and not binary_deps._match(name, binary_deps.WINDOWS_OS + binary_deps.WINDOWS_DRIVER, True)})
            if missing:
                raise ValueError(f"uncovered {group} PE imports in {path.relative_to(stage)}: {missing}")
            graph[path.relative_to(stage).as_posix()] = imports
    return graph


def verify(stage: Path, *, audit_native=None, audit_generic=None, audit_other=None) -> dict:
    """Reject drift, extras, reparse points, and damaged PE dependency sets."""
    stage = Path(stage)
    no_reparse(stage)
    manifest_path = stage / MANIFEST
    no_reparse(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported full Windows schema")
    expected = _manifest(manifest)
    if expected != manifest:
        raise ValueError("manifest is not canonical")
    for item in manifest["files"]:
        path = stage / item["path"]
        check_source(path, item["sha256"])
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"size mismatch: {item['path']}")
    actual = set()
    for path in stage.rglob("*"):
        no_reparse(path)
        if path.is_file():
            actual.add(path.relative_to(stage).as_posix())
    names = {item["path"] for item in manifest["files"]}
    if actual != names | {MANIFEST, INSTALLER, STARTER}:
        raise ValueError("full package membership differs from manifest")
    if sha256(stage / INSTALLER) != manifest["public"]["installer_sha256"]:
        raise ValueError("installer hash mismatch")
    expected_starter = STARTER_TEMPLATE.format(manifest_sha256=sha256(manifest_path)).encode("ascii")
    if (stage / STARTER).read_bytes() != expected_starter:
        raise ValueError("starter does not pin manifest")
    if audit_native is None:
        from native_preview import audit as audit_native
    if audit_generic is None:
        from gui_overlay import audit_generic
    native = stage / "components" / "engine" / manifest["public"]["native_folder"]
    generic = stage / "components" / "generic-gguf" / manifest["public"]["generic_folder"]
    audit_native(native)
    audit_generic(generic)
    (audit_other or audit_auxiliary)(stage)
    component = json.loads((native / "engine-component.json").read_text(encoding="utf-8"))
    if component.get("id") != "strata-engine" or component.get("entrypoint") != "strata.exe":
        raise ValueError("invalid native component")
    component_names = {f"components/engine/{manifest['public']['native_folder']}/{x['path']}" for x in component["files"]}
    component_names.add(f"components/engine/{manifest['public']['native_folder']}/engine-component.json")
    if component_names != {n for n in names if n.startswith(f"components/engine/{manifest['public']['native_folder']}/")}:
        raise ValueError("native component membership differs")
    for item in component["files"]:
        full = native / relative_name(item["path"])
        if sha256(full) != item["sha256"] or full.stat().st_size != item["bytes"]:
            raise ValueError("native component pin differs")
    return manifest


def build(spec: dict, stage: Path, archive: Path, *, audit_native=None, audit_generic=None, audit_other=None) -> dict:
    """Build once from pinned sources. Output paths must be fresh siblings."""
    stage, archive = Path(stage), Path(archive)
    if stage.exists() or archive.exists() or stage.parent.resolve() != archive.parent.resolve():
        raise ValueError("fresh stage and ZIP archive must share a parent")
    if archive.suffix.lower() != ".zip":
        raise ValueError("full package must be a ZIP")
    no_reparse(stage.parent)
    public = dict(spec["public"])
    public["installer_sha256"] = hashlib.sha256(installer_script().encode("utf-8")).hexdigest()
    local = {"public": public, "files": spec["files"]}
    manifest = _manifest(local)
    total = sum(item["bytes"] for item in manifest["files"])
    if shutil.disk_usage(stage.parent).free < total * 2 + RESERVE_BUILD:
        raise ValueError("insufficient build disk reserve")
    stage.mkdir()
    by_name = {item["path"]: item for item in spec["files"]}
    for item in manifest["files"]:
        name = item["path"]
        source = Path(by_name[name]["source"])
        check_source(source, item["sha256"])
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        check_source(target, item["sha256"])
    (stage / INSTALLER).write_text(installer_script(), encoding="utf-8", newline="")
    (stage / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (stage / STARTER).write_bytes(STARTER_TEMPLATE.format(manifest_sha256=sha256(stage / MANIFEST)).encode("ascii"))
    verify(stage, audit_native=audit_native, audit_generic=audit_generic, audit_other=audit_other)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            z.write(path, path.relative_to(stage).as_posix())
    return {"archive": str(archive), "archive_sha256": sha256(archive), "manifest_sha256": sha256(stage / MANIFEST), "files": len(manifest["files"]), "bytes": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "verify"])
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.mode == "build":
        if not args.spec or not args.archive:
            parser.error("build requires --spec and --archive")
        print(json.dumps(build(json.loads(args.spec.read_text(encoding="utf-8")), args.stage, args.archive), indent=2))
    else:
        print(json.dumps({"manifest_sha256": sha256(args.stage / MANIFEST), "files": len(verify(args.stage)["files"])}, indent=2))


if __name__ == "__main__":
    main()
