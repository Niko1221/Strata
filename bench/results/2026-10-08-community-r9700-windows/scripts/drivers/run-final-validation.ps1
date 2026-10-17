# Final validation driver (measurement executor only - touches nothing under src/).
#
# Task 1: base -> shstream0 -> shstream0-minp0 at --prompt-tokens 24576 (~99.7k real tokens),
#         plus a trailing base re-anchor so the drift is bracketed instead of extrapolated.
# Task 2: long context, base and shstream0 both, at $LongPrompt requested tokens
#         (~4.06 real tokens per requested unit -> $LongPrompt 49152 ~= 200k real tokens),
#         again with a trailing base anchor.
#
# Discipline, same as run-w2-sweep.ps1 / run-verify-minp0.ps1:
#   * double pre-check before every engine start: strata.exe count == 0 AND no :8080 listener
#     (serve/server.py binds only once the engine is ready, so the port alone is not enough)
#   * after every arm strata.exe is re-proven 0, and an ORPHAN serve/server.py of ours that
#     survived the harness (it re-execs and breaks its process tree) is cleared before the next arm
#   * never kills anything it does not own: a force-kill happens only when no live gpu-window lock
#     owner exists, and the orphan filter requires our own cfg-ab config on the command line
param(
    [int]$LongPrompt = 32768,
    [int]$Task1Prompt = 24576,
    [int]$Repeats = 3,
    [int]$MaxWaitMinutes = 60,
    [switch]$Task2Only,
    [switch]$Task1Only,
    [string]$Suffix = '',
    [int]$GpuLockWait = 60
)

$ErrorActionPreference = 'Continue'
$root = 'D:\Workstation\Strata'
$py   = "$root\.venv\Scripts\python.exe"
$lock = "$root\tools\opt\gpu-window.json"

function Get-EngineCount { (Get-Process strata -ErrorAction SilentlyContinue | Measure-Object).Count }

function Get-LockBusy {
    if (-not (Test-Path $lock)) { return $false }
    try { $l = Get-Content $lock -Raw | ConvertFrom-Json } catch { return $true }
    $alive = $null -ne (Get-Process -Id ([int]$l.pid) -ErrorAction SilentlyContinue)
    if (-not $alive) { return $false }
    $now = [double]([DateTimeOffset]::Now.ToUnixTimeMilliseconds() / 1000.0)
    return (($now - [double]$l.started_epoch) -le [double]$l.ttl_s)
}

function Get-PortBusy { (Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0 }

function Clear-OrphanServer {
    # Our own serve/server.py survivors only: cmdline names serve\server.py AND one of the
    # cfg-ab configs. Refuses while an engine is alive or a live lock owner holds the card.
    if ((Get-EngineCount) -gt 0) { return $false }
    if (Get-LockBusy) { return $false }
    $orphans = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                 Where-Object { $_.CommandLine -like '*serve\server.py*' -and $_.CommandLine -like '*cfg-ab*' })
    if ($orphans.Count -eq 0) { return $true }
    foreach ($o in $orphans) {
        Write-Output "[guard] orphan server pid $($o.ProcessId) from our cfg-ab - stopping it"
        Stop-Process -Id ([int]$o.ProcessId) -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 4
    return (-not (Get-PortBusy))
}

function Wait-Idle {
    param([int]$MaxMinutes = 60)
    $deadline = (Get-Date).AddMinutes($MaxMinutes)
    while ((Get-Date) -lt $deadline) {
        if ((Get-EngineCount) -eq 0 -and -not (Get-LockBusy) -and -not (Get-PortBusy)) {
            Write-Output "[idle] card free (engine=0, no lock, no :8080)"
            return $true
        }
        if (-not (Get-PortBusy) -and (Get-EngineCount) -eq 0) { [void](Clear-OrphanServer) }
        Start-Sleep -Seconds 20
    }
    Write-Output "[idle] TIMED OUT waiting for a free card"
    return $false
}

function Ensure-Clean {
    for ($i = 0; $i -lt 12; $i++) {
        if ((Get-EngineCount) -eq 0) { break }
        Start-Sleep -Seconds 5
    }
    if ((Get-EngineCount) -gt 0) {
        if (Get-LockBusy) { Write-Output "[guard] engine survived but a LIVE lock owner exists - not killing"; return $false }
        Write-Output "[guard] orphan strata.exe - taskkill /F /IM strata.exe"
        & taskkill /F /IM strata.exe | Out-Null
        Start-Sleep -Seconds 5
    }
    [void](Clear-OrphanServer)
    return ((Get-EngineCount) -eq 0 -and -not (Get-PortBusy))
}

function Invoke-Arm {
    param([string]$Name, [string]$Cfg, [int]$Prompt, [string]$OutFile, [string]$LogPath)
    Write-Output ""
    Write-Output "################ ARM $Name  start $(Get-Date -Format 'HH:mm:ss') ################"
    for ($try = 1; $try -le 2; $try++) {
        if (-not (Wait-Idle -MaxMinutes $MaxWaitMinutes)) { Write-Output "!!!! $Name : card never freed - SKIP"; return }
        Write-Output "[run] try $try  cfg=$Cfg  prompt=$Prompt  out=$OutFile"
        Push-Location $root
        & $py tools\opt\bench_decode.py --start-server "$py serve\server.py --engine strata --config `"$root\tools\opt\cfg-ab\$Cfg`" --port 8080" `
            --gpu-lock --gpu-lock-wait $GpuLockWait --log-path $LogPath --arm-label $Name `
            --prompt-tokens $Prompt --max-tokens 128 --repeats $Repeats --seed 1234 `
            --warm-cold "cold engine; rep1 fresh prompt+cold expert cache, rep2-$Repeats warm" `
            --out-json $OutFile --out-md ("tools\opt\results\" + [IO.Path]::GetFileNameWithoutExtension($OutFile) + ".md")
        $code = $LASTEXITCODE
        Pop-Location
        Write-Output "[run] $Name try $try exit=$code  $(Get-Date -Format 'HH:mm:ss')"
        if ($code -eq 0) { break }
        if ($code -eq 3) { Write-Output "[run] gpu window busy - back off 30s"; Start-Sleep -Seconds 30; continue }
        Write-Output "[run] $Name FAILED exit=$code - one retry after a 20s settle (known VRAM-release crash)"
        Start-Sleep -Seconds 20
    }
    if (-not (Ensure-Clean)) { Write-Output "!!!! card not clean after $Name" }
    Write-Output "################ ARM $Name end $(Get-Date -Format 'HH:mm:ss') ################"
}

# ---------------------------------------------------------------- task 1: 24k requested (~99.7k real)
if (-not $Task2Only) {
Invoke-Arm -Name "final-base$Suffix"              -Cfg 'base.json'            -Prompt $Task1Prompt -OutFile "tools\opt\results\final-base$Suffix.json"              -LogPath 'tools\opt\logs\ab-w2.log'
Invoke-Arm -Name "final-shstream0$Suffix"         -Cfg 'shstream0.json'       -Prompt $Task1Prompt -OutFile "tools\opt\results\final-shstream0$Suffix.json"         -LogPath 'tools\opt\logs\shstream0.log'
Invoke-Arm -Name "final-shstream0-minp0$Suffix"   -Cfg 'shstream0-minp0.json' -Prompt $Task1Prompt -OutFile "tools\opt\results\final-shstream0-minp0$Suffix.json"   -LogPath 'tools\opt\logs\shstream0-minp0.log'
}

# ---------------------------------------------------------------- task 2: long context (~4.06x requested)
if (-not $Task1Only) {
Invoke-Arm -Name "final-long-base$Suffix"         -Cfg 'base.json'            -Prompt $LongPrompt  -OutFile "tools\opt\results\final-long-base$Suffix.json"         -LogPath 'tools\opt\logs\ab-w2.log'
Invoke-Arm -Name "final-long-shstream0$Suffix"    -Cfg 'shstream0.json'       -Prompt $LongPrompt  -OutFile "tools\opt\results\final-long-shstream0$Suffix.json"    -LogPath 'tools\opt\logs\shstream0.log'
Invoke-Arm -Name "final-long-base-b$Suffix"       -Cfg 'base.json'            -Prompt $LongPrompt  -OutFile "tools\opt\results\final-long-base-b$Suffix.json"       -LogPath 'tools\opt\logs\ab-w2.log'
}

Write-Output ""
Write-Output "DRIVER COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"