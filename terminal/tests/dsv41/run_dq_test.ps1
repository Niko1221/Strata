# run_dq_test.ps1 -Bin <ds4_dq_test.exe> [-Device 0] [-Extra '--quick']
# The Windows twin of run_dq_test.sh: one process per (type, kernel, path group), so a kernel that kills the device context ends only its own case.
param([string]$Bin, [int]$Device = 0, [string]$Extra = '')
$env:DS4_LAUNCH_CHECK = '1'
$types = 'Q6_K','Q5_K','Q4_K','Q3_K','Q2_K','IQ2_XXS','IQ2_XS','IQ2_S','IQ3_XXS','IQ3_S','IQ4_XS','IQ1_S','IQ1_M','IQ4_NL','Q8_0','BF16'
$table = @()
foreach ($t in $types) {
  $runs = @()
  foreach ($k in 'new','old','default') { foreach ($p in 'rows','gguf,gemm') { $runs += ,@($k, $p) } }
  $runs += ,@('default', 'mv,f16')
  foreach ($r in $runs) {
    $k = $r[0]; $p = $r[1]
    "=== $t kernel=$k path=$p"
    $a = @('--device', $Device, '--type', $t, '--kernel', $k, '--path', $p) + ($Extra -split ' ' | Where-Object { $_ })
    $out = & $Bin @a 2>&1
    $rc = $LASTEXITCODE
    $out | Where-Object { $_ -match '^(DEVICE|RESULT|  FAIL|  first|  note|STICKY|DEVICE ERROR|SUMMARY)' } | Select-Object -First 60
    if ($rc -ge 2) { "CRASH $t kernel=$k path=$p exit=$rc" }
    $table += $out | Where-Object { $_ -match '^RESULT' }
  }
}
'================ TABLE'
$table | Sort-Object
'ALLDONE'
