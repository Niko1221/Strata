# Strata Dev: baut UI und Dev-App (tauri.dev-build.conf.json, Feature dev-build) und startet sie. Siehe docs/DEV_BUILD.md.
#
#   .\dev-build.ps1                      # UI bauen, Dev-App + Installer bauen, Dev-App aus target\release starten
#   .\dev-build.ps1 -Install             # zusaetzlich still nach %LOCALAPPDATA%\Programs\Strata Dev installieren und von dort starten
#   .\dev-build.ps1 -HostOnly            # nur Host + UI im Entwicklermodus (Host :8792, Vite :5200), ohne Rust-Build
#   .\dev-build.ps1 -NoStart             # nur bauen
#   .\dev-build.ps1 -ResourcesFromLive   # fehlende Bundle-Teile (node, control, harness) aus der installierten Live-App KOPIEREN (nur lesen)
#
# Das Skript installiert nichts systemweit. Fehlende Werkzeuge werden aufgelistet, dann bricht es ab.
# Live (Strata, %LOCALAPPDATA%\Strata, %APPDATA%\Strata, ~\.strata, Ports 47831/8791/8080) wird nie veraendert.
[CmdletBinding()]
param(
    [switch]$Install,
    [switch]$HostOnly,
    [switch]$NoStart,
    [switch]$SkipUi,
    [switch]$ResourcesFromLive,
    [string]$HarnessSrc = ''
)
$ErrorActionPreference = 'Stop'

$Repo = $PSScriptRoot
$HostDir = Join-Path $Repo 'apps\host'
$DesktopDir = Join-Path $Repo 'apps\desktop'
$Tauri = Join-Path $DesktopDir 'src-tauri'
$Resources = Join-Path $Tauri 'resources'
$Local = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $env:USERPROFILE 'AppData\Local' }
$DevRoot = Join-Path $Local 'StrataDev'
$LiveInstall = Join-Path $Local 'Programs\Strata'
$DevInstall = Join-Path $Local 'Programs\Strata Dev'
$Ports = @{ Window = 47832; Host = 8792; Engine = 8081 }

function Say([string]$Text) { Write-Host "dev-build: $Text" }
function Fail([string]$Text) { [Console]::Error.WriteLine("dev-build: $Text"); exit 1 }

# Dieselbe Liste wie apps/desktop/src-tauri/src/devbuild.rs (variables). Bereits gesetzte Variablen bleiben.
function Set-DevEnvironment {
    $vars = [ordered]@{
        STRATA_DEV                    = '1'
        STRATA_DEV_CONFIG             = (Join-Path $Repo 'dev.config.json')
        STRATA_HOME                   = (Join-Path $DevRoot 'home')
        STRATA_PROFILE_DIR            = (Join-Path $DevRoot 'profiles')
        STRATA_STATE_DIR              = (Join-Path $DevRoot 'state')
        STRATA_TUNE_DIR               = (Join-Path $DevRoot 'tuning')
        STRATA_TIMESKIP_DIR           = (Join-Path $DevRoot 'time-skip')
        STRATA_RUNTIME_DIR            = (Join-Path $DevRoot 'runtime')
        STRATA_DESKTOP_DATA_DIR       = (Join-Path $DevRoot 'desktop')
        STRATA_SETTINGS_FILE          = (Join-Path $DevRoot 'app-settings.json')
        STRATA_INSTALL_DIR            = (Join-Path $DevRoot 'install')
        STRATA_TERMINAL_SETTINGS_FILE = (Join-Path $DevRoot 'terminal-settings.json')
        STRATA_WORKSPACE              = (Join-Path $DevRoot 'workspace')
        STRATA_ENGINE_PORT            = "$($Ports.Engine)"
        STRATA_NO_REPAIR_ACCESS       = '1'
    }
    foreach ($name in $vars.Keys) {
        if (-not [Environment]::GetEnvironmentVariable($name, 'Process')) { [Environment]::SetEnvironmentVariable($name, $vars[$name], 'Process') }
    }
}

function Test-PortFree([int]$Port) {
    $listener = $null
    try { $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port); $listener.Start(); return $true }
    catch { return $false }
    finally { if ($listener) { $listener.Stop() } }
}

# --- Werkzeuge -------------------------------------------------------------------------------------------------
$missing = @()
$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) { $missing += 'Node.js >= 24.12 (https://nodejs.org)' }
else {
    $v = (& node --version) -replace '^v', ''
    if ([version]$v -lt [version]'24.12.0') { $missing += "Node.js >= 24.12 (gefunden: $v)" }
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { $missing += 'npm (kommt mit Node.js)' }
if (-not $HostOnly) {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { $missing += 'Rust/Cargo (https://rustup.rs)' }
}
if ($missing.Count) { Fail ("Es fehlen Werkzeuge, nichts wurde gebaut:`n  - " + ($missing -join "`n  - ")) }

if (-not (Test-Path (Join-Path $Repo 'dev.config.json'))) { Fail "dev.config.json fehlt im Repo-Ordner $Repo." }

# --- Nur Host + UI im Entwicklermodus ----------------------------------------------------------------------------
if ($HostOnly) {
    foreach ($p in @($Ports.Host, 5200)) { if (-not (Test-PortFree $p)) { Fail "Port $p ist belegt. Laeuft Strata Dev schon?" } }
    Set-DevEnvironment
    if (-not (Test-Path (Join-Path $HostDir 'node_modules'))) { Say 'npm ci in apps/host ...'; & npm --prefix $HostDir ci --no-audit --no-fund; if ($LASTEXITCODE) { Fail 'npm ci ist fehlgeschlagen.' } }
    Say "Host http://127.0.0.1:$($Ports.Host)  UI http://127.0.0.1:5200  Daten $DevRoot"
    & npm --prefix $HostDir run dev
    exit $LASTEXITCODE
}

# --- Bundle-Teile (node, control, harness) -----------------------------------------------------------------------
$parts = [ordered]@{
    node    = { Test-Path (Join-Path $Resources 'node\node.exe') }
    control = { (Test-Path (Join-Path $Resources 'control\strata.py')) -and (Test-Path (Join-Path $Resources 'control\python\python.exe')) }
    harness = { Test-Path (Join-Path $Resources 'harness\BUNDLE.json') }
}
if ($HarnessSrc -and -not (& $parts.harness)) {
    Say "Harness aus $HarnessSrc bereitstellen ..."
    & node (Join-Path $DesktopDir 'scripts\stage-harness.mjs') --src $HarnessSrc
    if ($LASTEXITCODE) { Fail 'stage-harness.mjs ist fehlgeschlagen.' }
}
$absent = @($parts.Keys | Where-Object { -not (& $parts[$_]) })
if ($absent.Count -and $ResourcesFromLive) {
    if (-not (Test-Path (Join-Path $LiveInstall 'strata-desktop.exe'))) { Fail "Keine Live-App in $LiveInstall gefunden." }
    foreach ($name in $absent) {
        $from = Join-Path $LiveInstall $name
        if (-not (Test-Path $from)) { continue }
        Say "Kopiere $name aus der Live-App (nur lesen): $from"
        $to = Join-Path $Resources $name
        New-Item -ItemType Directory -Force $to | Out-Null
        # robocopy liest die Quelle nur; /E Unterordner, keine Loeschung im Ziel ausser was wir kopieren
        & robocopy $from $to /E /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { Fail "Kopieren von $name ist fehlgeschlagen (robocopy $LASTEXITCODE)." }
    }
    $absent = @($parts.Keys | Where-Object { -not (& $parts[$_]) })
}
if ($absent.Count) {
    $how = @{
        node    = 'powershell -NoProfile -ExecutionPolicy Bypass -File apps\desktop\scripts\fetch-node.ps1'
        control = 'python apps\desktop\scripts\stage-control.py'
        harness = 'node apps\desktop\scripts\stage-harness.mjs --src <gebauter Harness-Checkout>   (oder dev-build.ps1 -HarnessSrc <Pfad>)'
    }
    Fail ("Diese Bundle-Teile fehlen in apps\desktop\src-tauri\resources:`n" + (($absent | ForEach-Object { "  - ${_}:  $($how[$_])" }) -join "`n") + "`nOder: dev-build.ps1 -ResourcesFromLive (kopiert sie aus der installierten Live-App, die dabei nur gelesen wird).")
}

# --- UI ----------------------------------------------------------------------------------------------------------
if (-not $SkipUi) {
    if (-not (Test-Path (Join-Path $HostDir 'node_modules'))) { Say 'npm ci in apps/host ...'; & npm --prefix $HostDir ci --no-audit --no-fund; if ($LASTEXITCODE) { Fail 'npm ci in apps/host ist fehlgeschlagen.' } }
    Say 'UI bauen (vite build) ...'
    & npm --prefix $HostDir run build
    if ($LASTEXITCODE) { Fail 'Der UI-Build ist fehlgeschlagen.' }
}

# --- Dev-App + Installer -----------------------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $DesktopDir 'node_modules\.bin\tauri.cmd'))) { Say 'npm ci in apps/desktop (Tauri CLI, nur im Projekt) ...'; & npm --prefix $DesktopDir ci --no-audit --no-fund; if ($LASTEXITCODE) { Fail 'npm ci in apps/desktop ist fehlgeschlagen.' } }
Say 'Dev-App bauen (tauri build --config tauri.dev-build.conf.json --features dev-build --bundles nsis) ...'
Push-Location $DesktopDir
try {
    & (Join-Path $DesktopDir 'node_modules\.bin\tauri.cmd') build --config src-tauri\tauri.dev-build.conf.json --features dev-build --bundles nsis
    if ($LASTEXITCODE) { Fail 'tauri build ist fehlgeschlagen.' }
} finally { Pop-Location }

$installer = Get-ChildItem (Join-Path $Tauri 'target\release\bundle\nsis') -Filter 'Strata Dev_*-setup.exe' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$exe = @('strata-dev.exe', 'strata-desktop.exe') | ForEach-Object { Join-Path $Tauri "target\release\$_" } | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($installer) { Say "Installer: $($installer.FullName)" } else { Say 'Kein Installer gefunden (target\release\bundle\nsis).' }

if ($Install) {
    if (-not $installer) { Fail 'Ohne Installer kann nicht installiert werden.' }
    if ([IO.Path]::GetFullPath($DevInstall).TrimEnd('\') -ieq [IO.Path]::GetFullPath($LiveInstall).TrimEnd('\')) { Fail 'Der Dev-Ordner waere der Live-Ordner. Abbruch.' }
    Say "Installiere still nach $DevInstall ..."
    $p = Start-Process -FilePath $installer.FullName -ArgumentList '/S', "/D=$DevInstall" -Wait -PassThru
    if ($p.ExitCode) { Fail "Der Installer endete mit $($p.ExitCode)." }
    $exe = Join-Path $DevInstall 'strata-dev.exe'
}

if ($NoStart) { exit 0 }
if (-not $exe -or -not (Test-Path $exe)) { Fail 'Die gebaute Dev-App wurde nicht gefunden.' }
if (-not (Test-PortFree $Ports.Window)) { Say "Port $($Ports.Window) ist belegt (laeuft Strata Dev schon?). Die App nimmt dann einen freien Port." }
Say "Starte $exe  (Fenster/Host :$($Ports.Window), Engine :$($Ports.Engine), Daten $DevRoot)"
Start-Process -FilePath $exe
