# Follow-up to run_all_3200.ps1: waits for it to finish, then re-runs the default ladder rows lost to the
# #2065 slowdown and a 3-run check of draft-conf 0.5 and q4_0 KV on the tuned start, and restarts the preset.
$ErrorActionPreference = 'Continue'
$B = $PSScriptRoot
$S = "$env:LOCALAPPDATA\Strata\terminal\current"
$strata = "$S\bin\strata.cmd"
$log = "$B\results\followup3200.log"
Set-Location $B
while (-not (Select-String -Path "$B\results\all3200.log" -Pattern 'CHAIN DONE' -SimpleMatch -Quiet)) { Start-Sleep -Seconds 30 }

function Write-Step($t) { "=== $(Get-Date -Format 'HH:mm:ss') $t" | Tee-Object -FilePath $log -Append }

Write-Step 'stop preset'
& $strata launch --stop *>> $log

Write-Step 'default ladder re-run'
& .\run_strata_speed.ps1 -Mode ladder -Contexts 131072,196608,262144 -ConfigFile .\configs\ladderfix3200.psd1 -OutputDirectory .\results\ladderfix-3200 *>> $log

Write-Step '3-run check at 256K'
& .\run_strata_speed.ps1 -Mode short -ShortContext 262144 -ConfigFile .\configs\check3200.psd1 -OutputDirectory .\results\check-256k-3200 *>> $log

Write-Step 'restart tuned preset'
& $strata launch --stop *>> $log
& $strata launch --preset tuned-256k --restart --yes --no-tui *>> $log
Write-Step 'FOLLOWUP DONE'
