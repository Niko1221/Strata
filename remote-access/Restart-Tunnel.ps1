$ErrorActionPreference = 'Stop'
Restart-Service -Name Tailscale -Force
(Get-Service -Name Tailscale).WaitForStatus('Running', [TimeSpan]::FromSeconds(30))
Write-Output 'Tailscale service restarted. Persisted Funnel configuration is retained.'
