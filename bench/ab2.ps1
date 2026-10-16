# bench/ab2.ps1 - one binary, one base, arms that differ only in the flag/env under test.
# Base = bench/opt_run.sh's BASE plus --host-core last (what the live config uses), so the numbers
# stay comparable with bench/opt/*.log.  Two rounds, same order, so drift hits both rounds alike.
$ErrorActionPreference = 'Continue'
$ROOT = 'G:\01_Projects\Strata'
$DATA = 'D:\25_Strata\Strata-main\Strata-data'
$EXE  = "$ROOT\build-cuda\strata.exe"
$OUT  = "$ROOT\bench\ab2"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;$env:PATH"

$BASE = @('--pack',   "$DATA\packs\iq3_s",
          '--native', "$DATA\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
          '--ple-gguf',"$DATA\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
          '--expert-profile', "$ROOT\data\expert-profile.bin",
          '--expert-cache','auto', '--prefill','auto', '--spec','4',
          '--mtp', "$DATA\mtp\rt", '--max-context','262144', '--kv','int8', '--kv-resident','32768',
          '--spec-min-p','0.70', '--host-core','last',
          '--tokens-file', "$ROOT\bench\prompt_zh.txt", '--max-new','128')

$ARMS = @(
  @{n='pcie0';    a=@('--pcie-frac','0.00');            e=@{}},
  @{n='auto';     a=@();                                e=@{}},
  @{n='pf10';     a=@('--pcie-frac','0.10');            e=@{}},
  @{n='pw12';     a=@('--pcie-frac','0.00','--pool-workers','12'); e=@{}},
  @{n='pre0';     a=@('--pcie-frac','0.00');            e=@{'STRATA_IQ_PREFETCH'='0'}},
  @{n='pre8192';  a=@('--pcie-frac','0.00');            e=@{'STRATA_IQ_PREFETCH'='8192'}},
  @{n='tasks192'; a=@('--pcie-frac','0.00','--pool-tasks','192');  e=@{}}
)

function Stop-All {
  Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
  Get-Process strata -ErrorAction SilentlyContinue | Stop-Process -Force
  for ($i = 0; $i -lt 120; $i++) {
    if (-not (Get-Process strata -ErrorAction SilentlyContinue)) { break }
    Start-Sleep -Seconds 1
  }
  Start-Sleep -Seconds 3
}

$results = @()
foreach ($round in 1, 2) {
  foreach ($arm in $ARMS) {
    Stop-All
    foreach ($k in 'STRATA_IQ_PREFETCH','STRATA_NO_LARGEPAGES','STRATA_ROUTE_TAIL_SKIP','STRATA_PCIE_FRAC_DEFAULT') {
      Remove-Item "env:$k" -ErrorAction SilentlyContinue
    }
    foreach ($k in $arm.e.Keys) { Set-Item "env:$k" $arm.e[$k] }
    $tag = "r${round}_$($arm.n)"
    $log = "$OUT\$tag"
    Write-Host "=== $tag extra=[$($arm.a -join ' ')] env=[$($arm.e -join ' ')]"
    $p = Start-Process -FilePath $EXE -ArgumentList ($BASE + $arm.a) -NoNewWindow -Wait -PassThru `
         -WorkingDirectory $ROOT -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err"
    $dec = (Select-String -Path "$log.out" -Pattern '^decode ' | Select-Object -Last 1).Line
    $pre = (Select-String -Path "$log.out" -Pattern '^prefill ' | Select-Object -Last 1).Line
    $rows = (Select-String -Path "$log.err" -Pattern 'GB/s over the rows' | Select-Object -Last 1).Line
    $vw  = (Select-String -Path "$log.err" -Pattern '^verify window' | Select-Object -Last 1).Line
    $pe  = (Select-String -Path "$log.err" -Pattern 'pcie experts|PCIe probe' | Select-Object -Last 1).Line
    $slots = (Select-String -Path "$log.err" -Pattern 'expert cache \d+ slots' | Select-Object -Last 1).Line
    Write-Host "  exit=$($p.ExitCode)"
    Write-Host "  $dec"
    Write-Host "  $pre"
    Write-Host "  $rows"
    Write-Host "  $vw"
    Write-Host "  $pe"
    Write-Host "  $slots"
    $results += [pscustomobject]@{Tag=$tag; Decode=$dec; Prefill=$pre}
  }
}
Stop-All
$results | Format-Table -AutoSize -Wrap
