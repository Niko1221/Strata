param([switch]$SkipFunnel, [switch]$InstallStartup, [switch]$Recovery, [switch]$RefreshFunnel)
$ErrorActionPreference = 'Stop'

function Test-StrataGatewayCredential {
    param([int]$Port, [string]$ApiKey)
    # OPTIONS checks gateway authentication without depending on model readiness.
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Method Options -Uri ('http://127.0.0.1:' + $Port + '/v1/models') -Headers @{ Authorization=('Bearer ' + $ApiKey) } -TimeoutSec 3
        return ([int]$response.StatusCode -eq 405)
    } catch {
        return ($null -ne $_.Exception.Response -and [int]$_.Exception.Response.StatusCode -eq 405)
    }
}

function Start-StrataFunnel {
    [CmdletBinding(SupportsShouldProcess, ConfirmImpact='Low')]
    param([string]$TailscalePath, [int]$Port, [int]$TimeoutSeconds=120, [int]$RetryMilliseconds=1000, [string]$DisabledPath)
    if (!$PSCmdlet.ShouldProcess('Tailscale Funnel on HTTPS port 443', 'Publish Strata gateway')) { return }
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $lastProblem = 'Tailscale has not reported its state.'
    $reportedWaiting = $false
    do {
        if ($DisabledPath -and (Test-Path -LiteralPath $DisabledPath)) { return }
        $status = $null
        try {
            $statusOutput = & $TailscalePath status --json 2>&1
            if ($LASTEXITCODE -eq 0) {
                $status = ($statusOutput -join "`n") | ConvertFrom-Json
                $lastProblem = 'Tailscale state: ' + $status.BackendState
            } else {
                $lastProblem = $statusOutput -join "`n"
            }
        } catch {
            $lastProblem = $_.Exception.Message
        }
        if ($status.BackendState -in 'NeedsLogin', 'NeedsMachineAuth') {
            throw 'Tailscale requires approval: sign in or approve this PC in Tailscale, then start Strata again.'
        }
        if ($status.BackendState -eq 'Running' -and $status.Self.DNSName) {
            try {
                $funnelOutput = & $TailscalePath funnel --bg --yes ('http://127.0.0.1:' + $Port) 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $funnelOutput | ForEach-Object { Write-Information -MessageData $_ -InformationAction Continue }
                    return $status
                }
                $lastProblem = $funnelOutput -join "`n"
            } catch {
                $lastProblem = $_.Exception.Message
            }
        }
        if ([DateTime]::UtcNow -ge $deadline) { break }
        if (!$reportedWaiting) {
            Write-Information 'Waiting for Tailscale to connect; public API startup will retry automatically.' -InformationAction Continue
            $reportedWaiting = $true
        }
        Start-Sleep -Milliseconds $RetryMilliseconds
    } while ([DateTime]::UtcNow -lt $deadline)
    throw ('Public API startup timed out waiting for Tailscale. Last problem: ' + $lastProblem)
}

$remoteDir = Join-Path $env:LOCALAPPDATA 'StrataRemote'
$disabledPath = Join-Path $remoteDir 'disabled'
if ($Recovery -and (Test-Path -LiteralPath $disabledPath)) { return }
if (!$Recovery) { Remove-Item -LiteralPath $disabledPath -ErrorAction SilentlyContinue }
$configPath = Join-Path $remoteDir 'gateway.json'
$gatewayPath = Join-Path $PSScriptRoot 'gateway.mjs'
$tailscalePath = 'C:\Program Files\Tailscale\tailscale.exe'
$nodePath = (Get-Command node -ErrorAction Stop).Source
New-Item -ItemType Directory -Path $remoteDir -Force | Out-Null
$remoteIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
& icacls.exe $remoteDir /inheritance:r /grant:r "*${remoteIdentity}:(OI)(CI)F" '*S-1-5-18:(OI)(CI)F' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Cannot protect the credentials directory.' }
if (!(Test-Path -LiteralPath $configPath)) {
    $keyBytes = New-Object byte[] 32
    $keyRng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $keyRng.GetBytes($keyBytes) } finally { $keyRng.Dispose() }
    $key = 'strata_' + [Convert]::ToBase64String($keyBytes).TrimEnd('=').Replace('+','-').Replace('/','_')
    @{ apiKey=$key; port=8787 } | ConvertTo-Json | Set-Content -LiteralPath $configPath -Encoding ascii
}
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$existing = Get-NetTCPConnection -LocalPort $config.port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    $gatewayProcess = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $existing[0].OwningProcess)
    if ($gatewayProcess.Name -ne 'node.exe' -or !$gatewayProcess.CommandLine.Contains($gatewayPath)) {
        throw ('Port ' + $config.port + ' belongs to another service.')
    }
    if (!(Test-StrataGatewayCredential -Port $config.port -ApiKey $config.apiKey)) {
        Write-Information 'Restarting the owned gateway to load the saved credentials.' -InformationAction Continue
        Stop-Process -Id $gatewayProcess.ProcessId -ErrorAction Stop
        Wait-Process -Id $gatewayProcess.ProcessId -Timeout 10 -ErrorAction SilentlyContinue
        $existing = $null
    }
}
if (!$existing) {
    $gatewayArgs = '"' + $gatewayPath + '" "' + $configPath + '"'
    $gatewayProcess = Start-Process -FilePath $nodePath -ArgumentList $gatewayArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $remoteDir 'gateway.log') -RedirectStandardError (Join-Path $remoteDir 'gateway-error.log')
    $gatewayProcess.Id | Set-Content -LiteralPath (Join-Path $remoteDir 'gateway.pid')
    for ($readyAttempt=0; $readyAttempt -lt 30; $readyAttempt++) {
        $ready = Get-NetTCPConnection -LocalPort $config.port -State Listen -ErrorAction SilentlyContinue
        if ($ready -and (Test-StrataGatewayCredential -Port $config.port -ApiKey $config.apiKey)) { break }
        $ready = $null
        if ($gatewayProcess.HasExited) { throw 'Gateway failed. See gateway-error.log.' }
        Start-Sleep -Milliseconds 200
    }
    if (!$ready) { throw 'Gateway did not start.' }
}
if (!$SkipFunnel) {
    if ($RefreshFunnel) {
        $currentFunnel = (& $tailscalePath funnel status --json | ConvertFrom-Json)
        $ownedProxy = @($currentFunnel.Web.PSObject.Properties | Where-Object { $_.Name.EndsWith(':443') -and $_.Value.Handlers.'/'.Proxy -eq ('http://127.0.0.1:' + $config.port) })
        if ($ownedProxy.Count -eq 1) {
            & $tailscalePath funnel --https=443 off
            if ($LASTEXITCODE -ne 0) { throw 'Could not refresh the owned Funnel endpoint.' }
        }
    }
    $cancelPath = if ($Recovery) { $disabledPath } else { $null }
    $tailStatus = Start-StrataFunnel -TailscalePath $tailscalePath -Port $config.port -DisabledPath $cancelPath
    if ($Recovery -and (Test-Path -LiteralPath $disabledPath)) {
        & (Join-Path $PSScriptRoot 'Stop-Remote.ps1')
        return
    }
} else {
    $tailStatus = (& $tailscalePath status --json | ConvertFrom-Json)
}
$publicHost = $tailStatus.Self.DNSName.TrimEnd('.')
$settingsPath = Join-Path $remoteDir 'editor-settings.txt'
@"
Provider: OpenAI-compatible / Custom OpenAI
API base URL: https://$publicHost/v1
API key: $($config.apiKey)
Model: qwen3.8-flash-next-coder-iq1_m

Chat Completions: https://$publicHost/v1/chat/completions
Responses API: https://$publicHost/v1/responses
Anthropic base URL: https://$publicHost
Anthropic Messages: https://$publicHost/v1/messages

Keep this key private. This PC and Strata must stay running.
Start: powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\Start-Remote.ps1"
Stop: powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\Stop-Remote.ps1"
"@ | Set-Content -LiteralPath $settingsPath -Encoding utf8
if ($InstallStartup) {
    $startupPath = Join-Path ([Environment]::GetFolderPath('Startup')) 'Strata Remote API.vbs'
    $startPath = Join-Path $PSScriptRoot 'Start-Remote.ps1'
    $startupCommand = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $startPath + '"'
    ('CreateObject("WScript.Shell").Run "' + $startupCommand.Replace('"','""') + '", 0, False') | Set-Content -LiteralPath $startupPath -Encoding ascii
    Write-Output 'Gateway startup installed for Windows sign-in.'
}
Write-Output ('Gateway ready on localhost:' + $config.port)
Write-Output ('Editor settings: ' + $settingsPath)
if (!$SkipFunnel) { Write-Output ('Public API: https://' + $publicHost + '/v1') }
