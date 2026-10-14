param([switch]$RemoveStartup)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'RemoteCommon.ps1')
$remoteDir = Join-Path $env:LOCALAPPDATA 'StrataRemote'
New-Item -ItemType Directory -Path $remoteDir -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $remoteDir 'disabled') -Force | Out-Null
$tailscalePath = 'C:\Program Files\Tailscale\tailscale.exe'
$configPath = Join-Path $remoteDir 'gateway.json'
try {
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $funnelOutput = & $tailscalePath funnel status --json 2>&1
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read Funnel status.' }
    $funnelConfig = ($funnelOutput -join "`n") | ConvertFrom-Json
    $ownership = Get-StrataFunnelOwnership -Configuration $funnelConfig -Port $config.port
    if ($ownership -eq 'owned') {
        & $tailscalePath funnel --https=443 off
        if ($LASTEXITCODE -ne 0) { Write-Warning 'Tailscale did not disable the owned Funnel endpoint.' }
    } elseif ($ownership -eq 'other') {
        Write-Warning 'Unrelated Tailscale handlers are unchanged. Stopping only the owned gateway.'
    }
} catch {
    Write-Warning 'Funnel ownership could not be verified. Stopping only the owned gateway.'
}
$gatewayPath = Join-Path $PSScriptRoot 'gateway.mjs'
$pidPath = Join-Path $remoteDir 'gateway.pid'
if (Test-Path -LiteralPath $pidPath) {
    $gatewayPid = [int](Get-Content -LiteralPath $pidPath)
    $gatewayProcess = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $gatewayPid) -ErrorAction SilentlyContinue
    if ($gatewayProcess -and $gatewayProcess.Name -eq 'node.exe' -and $gatewayProcess.CommandLine -and
        $gatewayProcess.CommandLine.Contains($gatewayPath) -and $gatewayProcess.CommandLine.Contains($configPath)) {
        Stop-Process -Id $gatewayPid
    }
    Remove-Item -LiteralPath $pidPath -ErrorAction SilentlyContinue
}
if ($RemoveStartup) {
    $startupPath = Join-Path ([Environment]::GetFolderPath('Startup')) 'Strata Remote API.vbs'
    Remove-Item -LiteralPath $startupPath -ErrorAction SilentlyContinue
}
Write-Output 'Public access stopped. The local Strata model is still running.'
