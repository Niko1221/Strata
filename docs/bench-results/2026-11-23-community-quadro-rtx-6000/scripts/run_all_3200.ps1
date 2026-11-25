# Re-runs every test of the community report with the RAM at 3200 MT/s (XMP on).
# Order: ladder, 16K, 256K screen, 256K confirmation, SM75 A/B, default-start tool suites, then
# restarts the user's tuned preset. Output goes to results\*-3200 and results\all3200.log.
$ErrorActionPreference = 'Continue'
$B = $PSScriptRoot
$S = "$env:LOCALAPPDATA\Strata\terminal\current"
$strata = "$S\bin\strata.cmd"
$log = "$B\results\all3200.log"
Set-Location $B

function Write-Step($t) { "=== $(Get-Date -Format 'HH:mm:ss') $t" | Tee-Object -FilePath $log -Append }

Write-Step 'stop any server'
& $strata launch --stop *>> $log

Write-Step 'ladder'
& .\run_strata_speed.ps1 -Mode ladder -ConfigFile .\configs\ladder3200.psd1 -OutputDirectory .\results\ladder-3200 *>> $log

Write-Step 'short 16K'
& .\run_strata_speed.ps1 -Mode short -ConfigFile .\configs\short3200.psd1 -OutputDirectory .\results\short-16k-3200 *>> $log

Write-Step 'screen 256K'
& .\run_strata_speed.ps1 -Mode short -ShortContext 262144 -ConfigFile .\configs\tune3200.psd1 -OutputDirectory .\results\tune-256k-3200 *>> $log

Write-Step 'confirm 256K'
& .\run_strata_speed.ps1 -Mode ladder -Contexts 262144 -ConfigFile .\configs\confirm.psd1 -OutputDirectory .\results\confirm-256k-3200 *>> $log

Write-Step 'SM75 A/B'
& $strata launch --stop *>> $log
Set-Location "$B\ab"
$env:STRATA_TERMINAL_ROOT = $S; $env:PYTHONUNBUFFERED = '1'
& "$S\python\python.exe" -I ab_engine_full.py ab-config.json 'A:' 'B:STRATA_SM75_PERF=1' 'C:STRATA_SM75_PERF=1,STRATA_EXP_MODE=8' --rounds 3 --port 8090 --log ab-3200.jsonl *>> $log
Set-Location $B

Write-Step 'default-start tool suites (preset iq2xs: 64K, vision gpu)'
& $strata launch --stop *>> $log
& $strata launch --preset iq2xs --restart --yes --no-tui *>> $log
& .\run_tool_benchmark.ps1 -ExternalBaseUrl http://127.0.0.1:8080 -ExternalName 'Strata-UD-IQ2_XS-default-64k-ddr3200' -Seed 20260927 -Suite core -OutputRoot "$B\results\tools-core-default-3200" *>> $log
& .\run_tool_benchmark.ps1 -ExternalBaseUrl http://127.0.0.1:8080 -ExternalName 'Strata-UD-IQ2_XS-default-64k-ddr3200' -Seed 20260927 -Suite cpp -OutputRoot "$B\results\tools-cpp-default-3200" *>> $log

Write-Step 'restart tuned preset'
& $strata launch --stop *>> $log
& $strata launch --preset tuned-256k --restart --yes --no-tui *>> $log
Write-Step 'CHAIN DONE'
