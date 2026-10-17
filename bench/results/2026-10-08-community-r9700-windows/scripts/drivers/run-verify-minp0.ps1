# W2 verification driver: correctness gate for shstream0, then the --spec-min-p 0 stack arm.
#
# Same discipline as run-w2-sweep.ps1: before EVERY engine start it proves the card is
# genuinely free -- gpu-window.json unheld (or dead owner pid / past TTL) AND strata.exe
# count == 0 AND no 8080 listener.  After every arm it re-proves strata.exe == 0, because
# serve/server.py re-execs and breaks the process tree it was launched in.
#
# Never kills a process it does not own: an engine is force-killed only when no live lock
# owner exists (i.e. it is an orphan left by our own run, not somebody else's live arm).
param(
    [int]$MaxWaitMinutes = 90,
    [switch]$SkipGate,     # skip stage 1 (correctness gate)
    [switch]$SkipControl,  # skip stage 1b (base-vs-base determinism control)
    [switch]$SkipMinp0,    # skip stage 2 (--spec-min-p 0 arm)
    [switch]$SkipLongCtx   # skip stage 3 (long-context validation)
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
    $age = $now - [double]$l.started_epoch
    return ($age -le [double]$l.ttl_s)
}

function Get-PortBusy { (Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0 }

function Wait-Idle {
    param([int]$MaxMinutes = 90)
    $deadline = (Get-Date).AddMinutes($MaxMinutes)
    $tick = 0
    while ((Get-Date) -lt $deadline) {
        $eng = Get-EngineCount; $lb = Get-LockBusy; $pb = Get-PortBusy
        if ($eng -eq 0 -and -not $lb -and -not $pb) {
            Write-Output "[idle] card is free (engine=0, no lock, no :8080)"
            return $true
        }
        $tick++
        # Heartbeat.  A silent multi-minute wait is not just noise: a background job with
        # no output for minutes gets torn down, taking the arm with it.  Report what we are
        # waiting on so the wait is visible and survivable.
        if ($tick % 3 -eq 1) {
            $who = 'unknown'
            if (Test-Path $lock) { try { $who = (Get-Content $lock -Raw | ConvertFrom-Json).label } catch { $who = 'unreadable' } }
            Write-Output "[idle] waiting $(Get-Date -Format 'HH:mm:ss') engine=$eng lock=$lb port8080=$pb (held by '$who')"
        }
        Start-Sleep -Seconds 20
    }
    Write-Output "[idle] TIMED OUT waiting for a free card"
    return $false
}

function Ensure-NoEngine {
    for ($i = 0; $i -lt 12; $i++) {
        if ((Get-EngineCount) -eq 0) { break }
        Start-Sleep -Seconds 5
    }
    if ((Get-EngineCount) -ne 0) {
        if (Get-LockBusy) { Write-Output "[guard] engine survived but a LIVE lock owner exists - not killing it"; return $false }
        Write-Output "[guard] orphan strata.exe after our arm - taskkill /F /IM strata.exe"
        & taskkill /F /IM strata.exe | Out-Null
        Start-Sleep -Seconds 5
    }
    # serve/server.py re-execs, so terminating the process token_check started leaves a
    # second server.py holding :8080 with a dead engine behind it.  Report it rather than
    # killing it: that server may belong to another member's arm, and this driver is not
    # allowed to kill a process it does not own.
    if (Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue) {
        Write-Output "[guard] NOTE: something still holds :8080 (an orphan serve\server.py from a re-exec, or another member's arm) - left alone"
    }
    return ((Get-EngineCount) -eq 0)
}

function Invoke-Stage {
    param([string]$Name, [string]$OutFile, [string]$Command)
    Write-Output ""
    Write-Output "################ STAGE $Name  start $(Get-Date -Format 'HH:mm:ss') ################"
    # exit 3 = "another member holds the GPU window".  That is contention, not a failure of
    # this arm, so it gets its own retry budget rather than eating a real try.
    $budget = 4
    for ($try = 1; $try -le $budget -and $try -le 4; $try++) {
        if (-not (Wait-Idle -MaxMinutes $MaxWaitMinutes)) { Write-Output "!!!! $Name : card never freed - SKIP"; return }
        Push-Location $root
        Invoke-Expression $Command
        $code = $LASTEXITCODE
        Pop-Location
        Write-Output "[stage] $Name try $try exit=$code  $(Get-Date -Format 'HH:mm:ss')"
        if ($code -eq 0) { break }
        if ($code -eq 3) { Write-Output "[stage] $Name lost the GPU window to another member - re-waiting"; Start-Sleep -Seconds 20; continue }
        Write-Output "[stage] $Name failed (exit=$code)"
        break
    }
    if (-not (Ensure-NoEngine)) { Write-Output "!!!! engine still alive after $Name" }
    Write-Output "################ STAGE $Name end $(Get-Date -Format 'HH:mm:ss') ################"
}

# ---------------------------------------------------------------- stage 1: correctness gate
# Same prompt, greedy (temperature 0 / top_k 1), one engine start per arm because env differs.
if (-not $SkipGate) {
    Invoke-Stage -Name 'gate-base' -OutFile 'ab-text-base.txt' `
        -Command "$py tools\opt\cfg-ab\token_check.py --config `"$root\tools\opt\cfg-ab\base.json`" --out `"$root\tools\opt\results\ab-text-base.txt`" --max-tokens 512 --gpu-lock --gpu-lock-wait 60"

    Invoke-Stage -Name 'gate-shstream0' -OutFile 'ab-text-shstream0.txt' `
        -Command "$py tools\opt\cfg-ab\token_check.py --config `"$root\tools\opt\cfg-ab\shstream0.json`" --out `"$root\tools\opt\results\ab-text-shstream0.txt`" --max-tokens 512 --gpu-lock --gpu-lock-wait 60"

    Write-Output ""
    Write-Output "################ GATE DIFF  $(Get-Date -Format 'HH:mm:ss') ################"
    Push-Location $root
    & $py tools\opt\cfg-ab\token_check.py --diff tools\opt\results\ab-text-base.txt tools\opt\results\ab-text-shstream0.txt
    Write-Output "[diff] exit=$LASTEXITCODE  (0 = identical text, 1 = different)"
    Pop-Location
}

# ------------------------------------------------- stage 1b: base-vs-base determinism control
# A base-vs-shstream0 diff cannot be attributed to shstream0 until the engine is shown to
# be reproducible at all: if base disagrees with ITSELF across two cold starts, then the
# engine is non-deterministic and this gate cannot separate "the arm changed the numbers"
# from "any two runs differ".
if (-not $SkipControl) {
    Invoke-Stage -Name 'control-base-2' -OutFile 'ab-text-base2.txt' `
        -Command "$py tools\opt\cfg-ab\token_check.py --config `"$root\tools\opt\cfg-ab\base.json`" --out `"$root\tools\opt\results\ab-text-base2.txt`" --max-tokens 512 --gpu-lock --gpu-lock-wait 60"

    Write-Output ""
    Write-Output "################ CONTROL DIFF (base vs base)  $(Get-Date -Format 'HH:mm:ss') ################"
    Push-Location $root
    & $py tools\opt\cfg-ab\token_check.py --diff tools\opt\results\ab-text-base.txt tools\opt\results\ab-text-base2.txt
    Write-Output "[diff] exit=$LASTEXITCODE  (0 = base is reproducible, 1 = engine is non-deterministic)"
    Pop-Location
}

# ---------------------------------------------------------------- stage 2: --spec-min-p 0
if (-not $SkipMinp0) {
    Invoke-Stage -Name 'minp0' -OutFile 'w2-shstream0-minp0.json' `
        -Command "$py tools\opt\bench_decode.py --start-server `"$py serve\server.py --engine strata --config tools\opt\cfg-ab\shstream0-minp0.json --port 8080`" --gpu-lock --gpu-lock-wait 60 --log-path tools\opt\logs\shstream0-minp0.log --arm-label w2-shstream0-minp0 --prompt-tokens 24576 --max-tokens 128 --repeats 3 --seed 1234 --warm-cold `"cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm`" --out-json tools\opt\results\w2-shstream0-minp0.json"
}

# ---------------------------------------------------------------- stage 3: long context
# --prompt-tokens is only a target: the generator emits ~4.06 tokens per requested unit,
# so 49152 asks for a ~200k-token prompt (max-context in the arm configs is 262144).
if (-not $SkipLongCtx) {
    Invoke-Stage -Name 'longctx' -OutFile 'w2-shstream0-longctx.json' `
        -Command "$py tools\opt\bench_decode.py --start-server `"$py serve\server.py --engine strata --config tools\opt\cfg-ab\shstream0.json --port 8080`" --gpu-lock --gpu-lock-wait 60 --log-path tools\opt\logs\shstream0-longctx.log --arm-label w2-shstream0-longctx --prompt-tokens 49152 --max-tokens 128 --repeats 3 --seed 1234 --warm-cold `"cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm`" --out-json tools\opt\results\w2-shstream0-longctx.json"
}

Write-Output ""
Write-Output "DRIVER COMPLETE $(Get-Date -Format 'HH:mm:ss')"
