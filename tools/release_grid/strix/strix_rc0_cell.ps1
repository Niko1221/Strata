# strix_rc0_cell.ps1: ONE Strix Halo (gfx1151, Windows) cell of the release grid on rc0 (4251d80bb), a DIRECT engine + serve run: the grid's own install path is Linux-only,
# so this starts the rc0 HIP Inkling engine behind rc0's serve/server.py (the pattern of INKOTHER's gate_hip_strix3.ps1) and runs the usage suite against it.
# Approved by the coordinator 2026-11-23 (script, the scp of it and of usage_suite.py, the rc0 terminal tree).
# jobq: python jobq.py submit strix --name grid-rc0-strix-q4kxl --timeout 3h --lock gpu -- powershell -NoProfile -File C:\strata-test\grid0221\strix_rc0_cell.ps1
# Our files only under C:\strata-test\ ; the model and tokenizer are read only; nothing else is listed or touched. Own PIDs only.
$ErrorActionPreference = "Continue"
$root   = "C:\strata-test\grid0221"
$py     = "C:\strata-test\jobq-python\python.exe"
$exe    = "C:\strata-test\rc0\inkling\strata-inkling.exe"
$model  = "C:\strata-test\models\inkling\Inkling-2-Mini-UD-Q4_K_XL.gguf"
$tok    = "C:\strata-test\models\inkling\tok"
$src    = "$root\term"                                   # rc0's terminal folder (tip 4251d80bb) was copied by scp -r to $src itself (not $src\terminal): its serve is $src\serve\server.py
$out    = "$root\results\rc0-4251d80bb"
New-Item -ItemType Directory -Force -Path $out | Out-Null
"DIRECT engine + serve run on Strix (rc0 HIP strata-inkling.exe behind rc0 serve/server.py); not the grid's strata launch path; UD-Q4_K_XL, drafter off" | Set-Content "$out\LABEL.txt"

$free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
"== $(Get-Date -Format T) RAM free $free GB"
if ($free -lt 14) { "NOT STARTED: only $free GB RAM free (floor 14 GB)"; exit 7 }
if (Get-Process strata* -ErrorAction SilentlyContinue | Where-Object { $_.Path -ne $exe }) { "NOT STARTED: another strata process is running"; exit 8 }

# server A: --parallel 1 (the default): chat, coding, parallel with 1 client.  server B: --parallel 32: parallel with 32 clients.
$runs = @(
  @{ name = "p1";  par = 1;  ctx = 32768; port = 18771; cases = "chat,coding,parallel"; levels = "1" },
  @{ name = "p32"; par = 32; ctx = 8192;  port = 18772; cases = "parallel";             levels = "32" }
)
foreach ($r in $runs) {
  $o = "$out\$($r.name)"
  New-Item -ItemType Directory -Force -Path $o | Out-Null
  $cfg = @{ arch = "inkling2_loop"; exe = $exe; args = @("--native", $model, "--max-context", "$($r.ctx)", "--parallel", "$($r.par)", "--draft", "off");
            cwd = $src; tokenizer = $tok; model_name = "inkling-2-mini-ud-q4_k_xl"; log = "$o\srv.log"; host = "127.0.0.1"; port = $r.port }
  ($cfg | ConvertTo-Json -Depth 5) | Set-Content "$o\cfg.json" -Encoding ascii
  $env:PYTHONUNBUFFERED = "1"
  $srv = $null
  $srv = Start-Process -ErrorAction Stop -FilePath $py -ArgumentList @("$src\serve\server.py", "--engine", "strata", "--config", "$o\cfg.json", "--port", "$($r.port)", "--no-api-key") `
         -WorkingDirectory "$src" -RedirectStandardOutput "$o\server.out" -RedirectStandardError "$o\server.err" -NoNewWindow -PassThru
  if (-not $srv) { "server $($r.name) could not be started: stopping"; break }
  "server $($r.name) pid $($srv.Id), --parallel $($r.par), context $($r.ctx)"
  $ready = $false
  $t0 = Get-Date
  while (((Get-Date) - $t0).TotalSeconds -lt 1500) {
    if ($srv.HasExited) { "server $($r.name) exited early, rc $($srv.ExitCode)"; break }
    try { Invoke-RestMethod -Uri "http://127.0.0.1:$($r.port)/v1/models" -TimeoutSec 3 | Out-Null; $ready = $true; break } catch { Start-Sleep -Seconds 4 }
  }
  "ready=$ready after $([math]::Round(((Get-Date) - $t0).TotalSeconds))s"
  if ($ready) {
    & $py "$root\usage_suite.py" --url "http://127.0.0.1:$($r.port)" --cases $r.cases --parallel-levels $r.levels --out "$o\usage.json" --md "$o\usage.md" 2>&1 | ForEach-Object { $_.ToString() }
    "usage suite $($r.name) rc=$LASTEXITCODE"
  }
  if ($srv -and -not $srv.HasExited) { Stop-Process -Id $srv.Id }
  Start-Sleep -Seconds 8
  Get-Process strata-inkling -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe } | ForEach-Object { "engine left, our exe, pid $($_.Id): stopping"; Stop-Process -Id $_.Id }
  Start-Sleep -Seconds 5
}
"== done $(Get-Date -Format T)"
