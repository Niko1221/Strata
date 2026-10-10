# W2 env A/B sweep driver (measurement executor helper).
# Runs one arm at a time. Before every arm it waits until the card is genuinely free:
#   gpu-window.json unheld (or dead owner pid / past TTL)  AND  strata.exe count == 0  AND  no 8080 listener.
# After every arm it re-proves strata.exe == 0 before the next one starts.
# Never kills a process it does not own: an engine is force-killed only when no live
# lock owner exists (i.e. it is an orphan left by our own run).

param([string[]]$Only = @())

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
    $age = $now - [double]$l.started_epoch
    return ($age -le [double]$l.ttl_s)
}

function Get-PortBusy { (Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0 }

function Wait-Idle {
    param([int]$MaxMinutes = 120)
    $deadline = (Get-Date).AddMinutes($MaxMinutes)
    while ((Get-Date) -lt $deadline) {
        $eng = Get-EngineCount; $lb = Get-LockBusy; $pb = Get-PortBusy
        if ($eng -eq 0 -and -not $lb -and -not $pb) { Write-Output "[idle] card is free"; return $true }
        Start-Sleep -Seconds 20
    }
    Write-Output "[idle] TIMED OUT waiting for a free card"
    return $false
}

function Ensure-NoEngine {
    # Only escalates to a by-name kill when there is no live lock owner, i.e. the
    # engine is an orphan of our own run rather than somebody else's live arm.
    for ($i = 0; $i -lt 12; $i++) {
        if ((Get-EngineCount) -eq 0) { return $true }
        Start-Sleep -Seconds 5
    }
    if ((Get-LockBusy)) { Write-Output "[guard] an engine survived but a LIVE lock owner exists - not killing it"; return $false }
    Write-Output "[guard] orphan strata.exe after our arm - taskkill /F /IM strata.exe"
    & taskkill /F /IM strata.exe | Out-Null
    Start-Sleep -Seconds 5
    return ((Get-EngineCount) -eq 0)
}

$arms = @(
    @{ name = 'base-a';    arm = 'base';      log = 'tools\opt\logs\ab-w2.log'      },
    @{ name = 'shstream0'; arm = 'shstream0'; log = 'tools\opt\logs\shstream0.log' },
    @{ name = 'hcplain';   arm = 'hcplain';   log = 'tools\opt\logs\hcplain.log'   },
    @{ name = 'dbstore';   arm = 'dbstore';   log = 'tools\opt\logs\dbstore.log'   },
    @{ name = 'base-b';    arm = 'base';      log = 'tools\opt\logs\ab-w2.log'      },
    @{ name = 'coherent';  arm = 'coherent';  log = 'tools\opt\logs\coherent.log'  }
)
if ($Only.Count -gt 0) { $arms = $arms | Where-Object { $Only -contains $_.name } }

foreach ($a in $arms) {
    Write-Output ""
    Write-Output "################ ARM $($a.name)  start $(Get-Date -Format 'HH:mm:ss') ################"
    $done = $false
    for ($try = 1; $try -le 2 -and -not $done; $try++) {
        if (-not (Wait-Idle)) { Write-Output "!!!! $($a.name): card never freed - SKIP"; break }
        $out = "tools\opt\results\w2-$($a.name).json"
        Write-Output "[run] try $try : w2-$($a.name)"
        Push-Location $root
        & python tools\opt\bench_decode.py --start-server "$py serve\server.py --engine strata --config tools\opt\cfg-ab\$($a.arm).json --port 8080" `
            --gpu-lock --gpu-lock-wait 60 --log-path $a.log --arm-label "w2-$($a.name)" `
            --prompt-tokens 24576 --max-tokens 128 --repeats 3 --seed 1234 `
            --warm-cold "cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm" `
            --out-json $out
        $code = $LASTEXITCODE
        Pop-Location
        Write-Output "[run] exit=$code  $(Get-Date -Format 'HH:mm:ss')"
        if ($code -eq 0) { $done = $true }
        elseif ($code -eq 3) { Write-Output "[run] lock busy (another member won the window) - retrying"; Start-Sleep -Seconds 20 }
        else { Write-Output "!!!! ARM $($a.name) FAILED exit=$code"; break }
    }
    if (-not (Ensure-NoEngine)) { Write-Output "!!!! engine still alive after $($a.name)"; }
    Write-Output "################ ARM $($a.name) end $(Get-Date -Format 'HH:mm:ss') ################"
}
Write-Output ""
Write-Output "SWEEP COMPLETE $(Get-Date -Format 'HH:mm:ss')"