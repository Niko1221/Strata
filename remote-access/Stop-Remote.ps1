param([switch]$RemoveStartup)
$ErrorActionPreference = 'Stop'
$remoteDir = Join-Path $env:LOCALAPPDATA 'StrataRemote'
New-Item -ItemType Directory -Path $remoteDir -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $remoteDir 'disabled') -Force | Out-Null
$tailscalePath = 'C:\Program Files\Tailscale\tailscale.exe'
& $tailscalePath funnel --https=443 off
if ($LASTEXITCODE -ne 0) { Write-Warning 'Tailscale did not disable Funnel. Stopping the gateway prevents model access.' }
$gatewayPath = Join-Path $PSScriptRoot 'gateway.mjs'
$pidPath = Join-Path $remoteDir 'gateway.pid'
if (Test-Path -LiteralPath $pidPath) {
    $gatewayPid = [int](Get-Content -LiteralPath $pidPath)
    $gatewayProcess = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $gatewayPid) -ErrorAction SilentlyContinue
    if ($gatewayProcess -and $gatewayProcess.Name -eq 'node.exe' -and $gatewayProcess.CommandLine.Contains($gatewayPath)) {
        Stop-Process -Id $gatewayPid
    }
    Remove-Item -LiteralPath $pidPath -ErrorAction SilentlyContinue
}
if ($RemoveStartup) {
    $startupPath = Join-Path ([Environment]::GetFolderPath('Startup')) 'Strata Remote API.vbs'
    Remove-Item -LiteralPath $startupPath -ErrorAction SilentlyContinue
}
Write-Output 'Public access stopped. The local Strata model is still running.'
