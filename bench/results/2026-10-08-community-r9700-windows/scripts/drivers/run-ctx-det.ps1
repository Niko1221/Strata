# Context + determinism driver (Q1 four arms, Q2 shstream0 second run).
#
# Same discipline as run-final-validation.ps1 / run-verify-minp0.ps1:
#   * double pre-check before EVERY engine start: strata.exe count == 0 AND no :8080 listener
#   * after every arm strata.exe is re-proven 0, and an orphan serve/server.py of ours is cleared
#   * never kills anything it does not own: a force-kill only when no live gpu-window owner exists
#
# Difference from the other drivers: it polls the free-card window every 5 s (not 20 s) because
# two peer sessions are competing for the same card, so winning the next genuine gap matters.
param(
    [int]$MaxWaitMinutes = 75
)

$ErrorActionPreference = 'Continue'
$root = 'D:\Workstation\Strata'
$py   = "$root\.venv\Scripts\python.exe"
$lock = "$root\tools\opt\gpu-window.json"

function Get-EngineCount { (Get-Process strata -ErrorAction SilentlyContinue | Measure-Object).Count }
function Get-PortBusy { (Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0 }
function Get-LockBusy {
    if (-not (Test-Path $lock)) { return $false }
    try { $l = Get-Content $lock -Raw | ConvertFrom-Json } catch { return $true }
    if ($null -eq (Get-Process -Id ([int]$l.pid) -ErrorAction SilentlyContinue)) { return $false }
    $now = [double]([DateTimeOffset]::Now.ToUnixTimeMilliseconds() / 1000.0)
    return (($now - [double]$l.started_epoch) -le [double]$l.ttl_s)
}
function Clear-OrphanServer {
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
    param([int]$MaxMinutes = 75)
    $deadline = (Get-Date).AddMinutes($MaxMinutes)
    while ((Get-Date) -lt $deadline) {
        if ((Get-EngineCount) -eq 0 -and -not (Get-LockBusy) -and -not (Get-PortBusy)) {
            Write-Output "[idle] card free (engine=0, no lock, no :8080) at $(Get-Date -Format 'HH:mm:ss')"
            return $true
        }
        if (-not (Get-PortBusy) -and (Get-EngineCount) -eq 0) { [void](Clear-OrphanServer) }
        Start-Sleep -Seconds 5
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
    param([string]$Name, [string]$Cfg, [string]$PromptFile, [int]$PromptTokens, [string]$OutFile, [string]$LogPath, [string]$ExtraArgs = '')
    Write-Output ""
    Write-Output "################ ARM $Name  start $(Get-Date -Format 'HH:mm:ss') ################"
    # tokens-*.txt hold one TOKEN ID per line; sent as prompt text each id re-tokenizes to
    # ~4.18 tokens, so tokens-100k.txt is really ~444k tokens and the 262144 context rejects
    # it with HTTP 400.  The long arm therefore uses synth_prompt (--prompt-tokens), which is
    # the ~99,696-real-token prompt the historical shstream0 numbers were taken on.
    $promptArg = if ($PromptFile) { "--prompt-file $PromptFile" } else { "--prompt-tokens $PromptTokens" }
    for ($try = 1; $try -le 6; $try++) {
        if (-not (Wait-Idle)) { Write-Output "!!!! $Name : card never freed - SKIP"; return }
        Push-Location $root
        $cmd = "$py tools\opt\bench_decode.py --start-server `"$py serve\server.py --engine strata --config `"$root\tools\opt\cfg-ab\$Cfg`" --port 8080`" " +
                "--gpu-lock --gpu-lock-wait 180 --log-path $LogPath --arm-label $Name " +
                "$promptArg --max-tokens 128 --repeats 3 --seed 1234 " +
                "--warm-cold `"cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm`" " +
                "--out-json $OutFile --out-md (`"tools\opt\results\`" + [IO.Path]::GetFileNameWithoutExtension($OutFile) + `".md`") $ExtraArgs"
        & ([scriptblock]::Create($cmd))
        $code = $LASTEXITCODE
        Pop-Location
        Write-Output "[run] $Name try $try exit=$code  $(Get-Date -Format 'HH:mm:ss')"
        if ($code -eq 0) { break }
        if ($code -eq 3) { Write-Output "[run] gpu window busy - back off 15s"; Start-Sleep -Seconds 15; continue }
        Write-Output "[run] $Name FAILED exit=$code - retry after a 25s settle (known VRAM-release crash)"
        Start-Sleep -Seconds 25
    }
    if (-not (Ensure-Clean)) { Write-Output "!!!! card not clean after $Name" }
    Write-Output "################ ARM $Name end $(Get-Date -Format 'HH:mm:ss') ################"
}

# ---------------------------------------------------------------- Q1: long context (~99.7k real tokens)
Invoke-Arm -Name 'ctx-long-base'       -Cfg 'base.json'      -PromptTokens 24576 -OutFile 'tools\opt\results\ctx-long-base.json'      -LogPath 'tools\opt\logs\ab-w2.log'
Invoke-Arm -Name 'ctx-long-shstream0'  -Cfg 'shstream0.json' -PromptTokens 24576 -OutFile 'tools\opt\results\ctx-long-shstream0.json' -LogPath 'tools\opt\logs\shstream0.log'

# ---------------------------------------------------------------- Q2: shstream0 second greedy run
# base-vs-base already exists (ab-text-base.txt vs ab-text-base2.txt); this completes the pair.
Write-Output ""
Write-Output "################ Q2 shstream0 run 2  start $(Get-Date -Format 'HH:mm:ss') ################"
for ($try = 1; $try -le 3; $try++) {
    if (-not (Wait-Idle)) { Write-Output "!!!! Q2 shstream0#2 : card never freed - SKIP"; break }
    Push-Location $root
    & $py tools\opt\cfg-ab\token_check.py --config "$root\tools\opt\cfg-ab\shstream0.json" `
        --out "$root\tools\opt\results\ab-text-shstream0-2.txt" --max-tokens 512 --gpu-lock --gpu-lock-wait 120
    $code = $LASTEXITCODE
    Pop-Location
    Write-Output "[run] Q2 shstream0#2 try $try exit=$code  $(Get-Date -Format 'HH:mm:ss')"
    if ($code -eq 0) { break }
    if ($code -eq 3) { Start-Sleep -Seconds 15; continue }
    Start-Sleep -Seconds 25
}
if (-not (Ensure-Clean)) { Write-Output "!!!! card not clean after Q2 shstream0#2" }

Write-Output ""
Write-Output "DRIVER COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
