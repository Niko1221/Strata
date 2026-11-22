param(
    [ValidateSet("98k", "65k")]
    [string]$Context = "98k",
    [int]$Port = 8092,
    [string]$BindHost = "127.0.0.1",
    [double]$LoopExitTol = 0.0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\downl\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}

$ConfigPath = Join-Path $RepoRoot "configs\strata-sakura-inkling2-$Context.json"
if (-not (Test-Path $ConfigPath)) {
    throw "Config file not found: $ConfigPath"
}

if ($LoopExitTol -gt 0.0) {
    $cfg = Get-Content -Raw -Encoding UTF8 $ConfigPath | ConvertFrom-Json
    $argsList = [System.Collections.ArrayList]@($cfg.args)
    [void]$argsList.Add("--loop-exit-tol")
    [void]$argsList.Add($LoopExitTol.ToString([System.Globalization.CultureInfo]::InvariantCulture))
    $cfg.args = $argsList
    $TempCfg = Join-Path $env:TEMP "strata-sakura-inkling2-$Context-elt.json"
    $cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $TempCfg
    $ConfigPath = $TempCfg
}

if (-not $env:STRATA_API_KEY) {
    throw "STRATA_API_KEY is not set. Hermes provider 'inkling2' authenticates with key_env: STRATA_API_KEY."
}

$env:PYTHONPATH = "$RepoRoot\terminal;$RepoRoot"
Write-Host "[strata-hermes] Starting Sakura-Inkling-2-Mini ($Context) on http://${BindHost}:${Port}/v1 (API key: STRATA_API_KEY) ..."
& $PythonExe "$RepoRoot\terminal\serve\server.py" `
    --engine strata `
    --config $ConfigPath `
    --host $BindHost `
    --port $Port
