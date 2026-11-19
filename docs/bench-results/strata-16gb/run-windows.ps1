# PR-A baseline diagnostic capture for RTX 5060 Ti / Windows 11.
# Records git SHAs, dedicated VRAM, shared GPU memory, CPU/RAM. Never kills other GPU apps.
$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$OutDir = Join-Path $PSScriptRoot 'results'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$OutFile = Join-Path $OutDir "baseline-$Stamp.json"

Set-Location $Root
$forkHead = (git rev-parse HEAD).Trim()
$originMain = (git rev-parse origin/main 2>$null)
if (-not $originMain) { $originMain = $null } else { $originMain = $originMain.Trim() }
$upstreamMain = (git rev-parse upstream/main 2>$null)
if (-not $upstreamMain) { $upstreamMain = $null } else { $upstreamMain = $upstreamMain.Trim() }
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
$statusShort = @(git status --short)

$nvidia = $null
try {
  $nvidia = (nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free,memory.reserved,pcie.link.gen.current,pcie.link.width.current,power.limit,clocks.current.graphics,temperature.gpu --format=csv) | Out-String
} catch { $nvidia = $null }

$shared = @()
try {
  $shared = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory |
    Select-Object Name, DedicatedUsage, SharedUsage, TotalCommitted)
} catch { $shared = @() }

$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed
$procs = @(Get-Process | Where-Object {
  $_.ProcessName -match 'llama|strata|hermes|lmstudio|LM Studio|ollama|vllm'
} | Select-Object ProcessName, Id, WorkingSet64)

# Node capture through host modules (null-safe dedicated/shared split)
$pins = [ordered]@{
  capturedAt = (Get-Date).ToUniversalTime().ToString('o')
  branch = $branch
  forkHead = $forkHead
  originMain = $originMain
  upstreamMain = $upstreamMain
  gitStatusShort = $statusShort
  planForkHead = 'b52a198300f11437b1ed2d2cc373aa73f60ea055'
  planUpstreamHead = 'c62d5e3a15b40ea51c96aae9e8d36928965f53ab'
  hikariRevision = 'c42fdf9debd1ba8daf4ac171d81b2052f841058f'
  prismSource = '73bfe6cc485554822f80c473a3c0c4f25e80f41b'
  nvidiaSmiCsv = $nvidia
  windowsGpuAdapterMemory = @($shared | ForEach-Object {
    [ordered]@{
      Name = $_.Name
      DedicatedUsage = [int64]$_.DedicatedUsage
      SharedUsage = [int64]$_.SharedUsage
      TotalCommitted = [int64]$_.TotalCommitted
    }
  })
  os = [ordered]@{
    Caption = $os.Caption
    Version = $os.Version
    TotalVisibleMemorySizeKiB = [int64]$os.TotalVisibleMemorySize
    FreePhysicalMemoryKiB = [int64]$os.FreePhysicalMemory
  }
  cpuInfo = $cpu
  processes = @($procs | ForEach-Object {
    [ordered]@{ name = $_.ProcessName; pid = $_.Id; workingSetBytes = [int64]$_.WorkingSet64; owner = $null }
  })
  ram = [ordered]@{
    totalMib = [math]::Round($os.TotalVisibleMemorySize / 1024.0, 1)
    availableMib = [math]::Round($os.FreePhysicalMemory / 1024.0, 1)
  }
  cpu = [ordered]@{ loadPct = $null; privateCommitMib = $null }
}

$tmpPins = Join-Path $env:TEMP "strata-16gb-pins-$Stamp.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($tmpPins, ($pins | ConvertTo-Json -Depth 8), $utf8NoBom)
$captureJs = Join-Path $PSScriptRoot 'capture-baseline.mjs'
try {
  node $captureJs $tmpPins $OutFile
  if ($LASTEXITCODE -ne 0) { throw "capture-baseline.mjs exited $LASTEXITCODE" }
} finally {
  Remove-Item -Force $tmpPins -ErrorAction SilentlyContinue
}

Write-Host "Wrote $OutFile"
Write-Host "fork=$forkHead upstream=$upstreamMain"
