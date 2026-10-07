# tools/remote-vision/test-wrapper.ps1
#
# Checks the wrapper the way Strata drives it, without starting the model server.  It finishes on
# its own, so it is safe to run while a server is up (the server owns its own wrapper).
#
#   .\test-wrapper.ps1                         the host comes from STRATA_VISION_SSH
#   .\test-wrapper.ps1 -VisionHost my-host     or from this argument
#   .\test-wrapper.ps1 -KillToTestRecovery     also kills the remote encoder mid-session and
#                                              checks that the next image still succeeds
param(
    [string]$VisionHost = $env:STRATA_VISION_SSH,
    [string]$RemoteRoot = $env:STRATA_VISION_ROOT,
    [string]$Mmproj = 'mmproj-Qwen3.8-Flash-Next-BF16.gguf',
    [switch]$KillToTestRecovery
)
$ErrorActionPreference = 'Stop'
if (-not $VisionHost) { throw "no vision host: set STRATA_VISION_SSH (or pass -VisionHost)" }
if (-not $RemoteRoot) { $RemoteRoot = '~/Developer/strata-vision' }    # the wrapper's own default
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $Here '..\..')
# The wrapper is a Python script; the server runs it through strata-vision-remote.cmd.  Here it is
# started with the interpreter directly, because ProcessStartInfo cannot run a .cmd with
# UseShellExecute = false (the server's Popen can - that is what the shim exists for).
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { $Python = 'python' }
$Script = Join-Path $Here 'strata_vision_remote.py'
$Out  = Join-Path $env:TEMP 'strata-vision-test.sve'

if (-not (Test-Path $Script)) { throw "missing $Script" }

# a small picture, made here so this script needs nothing else
$img = Join-Path $env:TEMP 'strata-vision-test.png'
& $Python -c @"
from PIL import Image, ImageDraw
im = Image.new('RGB', (640, 480), (245, 245, 240))
d = ImageDraw.Draw(im); d.rectangle([40,40,600,440], outline=(20,20,20), width=6)
d.ellipse([240,200,400,360], fill=(200,60,60)); im.save(r'$img')
"@ | Out-Null

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $Python
$psi.WorkingDirectory = $Root
$psi.UseShellExecute = $false
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
$psi.StandardErrorEncoding  = [System.Text.Encoding]::UTF8
$psi.ArgumentList.Add($Script)
foreach ($a in @('--mmproj', $Mmproj, '--model', 'tools\remote-vision\text-vocab-stub.gguf',
                 '--gpu', '--max-tokens', '1024')) { $psi.ArgumentList.Add($a) }
$psi.EnvironmentVariables['STRATA_VISION_SSH'] = $VisionHost
# a named session: its scratch dir and pidfile are scratch/<session> on the vision host, so the
# kill below and the leak check after it touch ONLY this test's encoder - never the production one.
$Session = 'test-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$psi.EnvironmentVariables['STRATA_VISION_SESSION'] = $Session
$PidFile = "$RemoteRoot/scratch/$Session/encoder.pid"

$p = [System.Diagnostics.Process]::new(); $p.StartInfo = $psi; $null = $p.Start()
$errTask = $p.StandardError.ReadToEndAsync()
$fail = 0

function Read-Line([int]$TimeoutSec, [string]$What) {
    $t = $p.StandardOutput.ReadLineAsync()
    if (-not $t.Wait($TimeoutSec * 1000)) { Write-Host "  FAIL: no $What within $TimeoutSec s" -ForegroundColor Red; return $null }
    return $t.Result
}

$ready = Read-Line 120 'READY'
if ($ready -notlike 'READY*') { Write-Host "  FAIL: $ready" -ForegroundColor Red; $p.Kill(); exit 1 }
Write-Host "  READY: $ready" -ForegroundColor Green

function Send-Image([string]$Label) {
    $script:sw = [System.Diagnostics.Stopwatch]::StartNew()
    $p.StandardInput.WriteLine("ENC $img $Out"); $p.StandardInput.Flush()
    $r = Read-Line 300 'a reply'
    $script:sw.Stop()
    if ($r -like 'OK*') { Write-Host ("  {0}: {1}  ({2:N2} s)" -f $Label, $r, $script:sw.Elapsed.TotalSeconds) -ForegroundColor Green; return $true }
    Write-Host "  ${Label}: $r" -ForegroundColor Red; return $false
}

if (-not (Send-Image 'image 1')) { $fail++ }

if ($KillToTestRecovery) {
    Write-Host '  killing the remote encoder to test recovery ...'
    ssh -o BatchMode=yes $VisionHost "p=`$(cat $PidFile 2>/dev/null); [ -n `"`$p`" ] && kill -TERM `"`$p`"" 2>&1 | Out-Null
    Start-Sleep -Seconds 2
    if (-not (Send-Image 'image 2 (after the kill)')) { $fail++ }
}

$p.StandardInput.WriteLine('QUIT'); $p.StandardInput.Flush()
if (-not $p.WaitForExit(20000)) { $p.Kill() }
Start-Sleep -Seconds 3

Write-Host '  --- remote state after exit ---'
ssh -o BatchMode=yes $VisionHost "p=`$(cat $PidFile 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `"`$p`" 2>/dev/null; then echo '  test encoder STILL RUNNING (leak)'; else echo '  test encoder stopped'; fi; ls -d $RemoteRoot/scratch/$Session 2>/dev/null || echo '  test session cleaned'; echo '  other sessions on this host:'; pgrep -af 'build/bin/strata-vision --mmproj' | grep -v 'bash -c' || echo '    none'" 2>&1
Write-Host '  --- wrapper log ---'
($errTask.Result -split "`n" | Select-Object -Last 8) | ForEach-Object { "  $_" }

if ($fail -eq 0) { Write-Host 'PASS' -ForegroundColor Green } else { Write-Host "$fail FAILURE(S)" -ForegroundColor Red; exit 1 }
