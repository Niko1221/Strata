# Strata Dev: alles pruefen, bevor gebaut oder uebernommen wird. Siehe docs/DEV_BUILD.md ("Pruefen").
#
#   .\dev-check.ps1                 # Validator, Typecheck, npm test, pytest, Smoketest; kurzer Bericht am Ende
#   .\dev-check.ps1 -SkipPytest     # ohne die Python-Tests (dauern ~35 min)
#   .\dev-check.ps1 -SkipSmoke      # ohne Smoketest (startet sonst die gebaute Dev-App kurz, falls sie nicht laeuft)
#   .\dev-check.ps1 -Python <exe>   # diese Python-Umgebung fuer pytest
#   .\dev-check.ps1 -SetupVenv      # legt einmal eine eigene venv an (%LOCALAPPDATA%\StrataDev\test-venv) und installiert dort
#                                   # requirements.txt + requirements-monitor.txt + pytest (laedt Pakete aus PyPI; das System-Python bleibt)
#
# Live (Strata, %LOCALAPPDATA%\Strata, Documents\Strata, Ports 47831/8791/8080) wird nur gelesen: der Smoketest prueft, dass dort
# kein Dev-Prozess lauscht. Kein Modell wird geladen, kein Lasttest.
#
# Vorher die Live-App ganz schliessen (ein Teil der Python-Tests startet und beendet Strata-Prozesse) und nur EIN dev-check zur Zeit.
# Laeuft Live oder schon ein dev-check, bricht das Skript ab (-AllowLive laesst dann nur pytest aus).
# Bericht, Fortschritt und Logs: Documents\Strata-Dev-Logs\dev-check\ (dort sichtbar, auch wenn das Skript aus einer Store-App
# laeuft, deren AppData-Zugriffe Windows umleitet). In PowerShell mit Punkt davor: .\dev-check.bat
[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipPytest,
    [switch]$SkipSmoke,
    [switch]$SetupVenv,
    [switch]$AllowLive,
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
# Node, npm und Python schreiben UTF-8; ohne das liest PowerShell 5.1 sie mit der OEM-Codepage (kaputte Umlaute in den Logs).
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = 'utf-8'

$Repo = $PSScriptRoot
$HostDir = Join-Path $Repo 'apps\host'
$Local = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $env:USERPROFILE 'AppData\Local' }
$DevRoot = Join-Path $Local 'StrataDev'
$OutDir = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Strata-Dev-Logs\dev-check'
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Venv = Join-Path $DevRoot 'test-venv'
$Ports = @{ Window = 47832; Host = 8792; Engine = 8081 }
$LivePorts = @(47831, 8791, 8080)
# Python-Tests, die auf diesem PC schon vor Strata Dev fehlschlugen (Hardware/Platte, siehe PHASE2_BERICHT.md) oder unter Last wackeln
# (Zeitmessung; einzeln laufen sie durch). Sie zaehlen als bekannt, jeder andere Fehler als neu.
$KnownFailures = @(
    'tests/terminal/test_bundle_imports.py::BundleImports::test_setup_py_and_each_of_its_tools_import_with_tools_limited_to_the_bundle',
    'tests/terminal/test_launch_dsv41.py::LaunchDeepSeek::test_a_small_computer_gets_the_warnings_not_a_refusal',
    'tests/terminal/test_launch_dsv41.py::LaunchDeepSeek::test_the_human_plan_shows_the_placement',
    'tests/terminal/test_launch_dsv41.py::LaunchDeepSeek::test_the_model_is_in_the_list',
    'tests/terminal/test_launch_dsv41.py::LaunchDeepSeek::test_the_plan_places_the_model_on_the_four_cards',
    'tests/terminal/test_launch_tui.py::FolderStep::test_default_folder_and_free_space_and_change',
    'tests/terminal/test_bench_standard.py::RunTests::test_a_round_total_is_null_when_a_counted_request_has_no_token_count'
)

$Results = New-Object System.Collections.ArrayList
$Progress = Join-Path $OutDir "$Stamp-progress.log"
function Note([string]$Line) { try { Add-Content -Path $Progress -Value ('{0}  {1}' -f (Get-Date -Format 'HH:mm:ss'), $Line) -Encoding UTF8 } catch { } }
function Say([string]$Text) { Write-Host "dev-check: $Text"; Note $Text }
function Add-Result([string]$Step, [string]$State, [string]$Text) {
    [void]$Results.Add([pscustomobject]@{ Step = $Step; State = $State; Text = $Text })
    Note ('{0,-5} {1}: {2}' -f $State, $Step, $Text)
    $color = switch ($State) { 'OK' { 'Green' } 'WARN' { 'Yellow' } 'SKIP' { 'DarkGray' } default { 'Red' } }
    Write-Host ("  {0,-5} {1}: {2}" -f $State, $Step, $Text) -ForegroundColor $color
}
function Run-Logged([string]$Name, [string]$File, [string[]]$Arguments, [string]$WorkDir) {
    $log = Join-Path $OutDir "$Stamp-$Name.log"
    Push-Location $WorkDir
    try {
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        # UTF-8: mit *> schriebe PowerShell 5.1 UTF-16, das andere Werkzeuge schlecht lesen.
        & $File @Arguments 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Encoding utf8
        $code = $LASTEXITCODE
        $ErrorActionPreference = $old
    } finally { Pop-Location }
    return @{ Code = $code; Log = $log; Text = (Get-Content -Raw -Encoding UTF8 $log) }
}

# Nur ein dev-check zur Zeit: zwei pytest-Laeufe stoeren sich (beide starten und beenden Strata-Prozesse).
$mutex = New-Object System.Threading.Mutex($false, 'Local\StrataDevCheck')
$owned = $false
try { $owned = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $owned = $true }
if (-not $owned) { Say 'Es laeuft schon ein dev-check. Warte, bis er fertig ist, und starte dann neu.'; exit 2 }
# Die Live-App muss zu sein: ein Teil der Python-Tests beendet Strata-Prozesse.
$liveUp = [bool](Get-NetTCPConnection -LocalPort 47831 -State Listen -ErrorAction SilentlyContinue)
$liveUp = $liveUp -or [bool](Get-Process -ErrorAction SilentlyContinue | Where-Object { try { $_.Path -like '*\Programs\Strata\*' } catch { $false } })
if ($liveUp -and -not $AllowLive) { Say 'Die Live-App (Strata) laeuft. Bitte ganz schliessen (auch im Tray "Beenden") und dev-check neu starten.'; exit 2 }
if ($liveUp -and -not $SkipPytest) { $SkipPytest = $true; Say 'Live laeuft (-AllowLive): pytest wird ausgelassen.' }
Say "Bericht und Logs: $OutDir"
Say "Fortschritt: $Progress"

if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Say 'Node.js fehlt (>= 24.12).'; exit 1 }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { Say 'npm fehlt (kommt mit Node.js).'; exit 1 }
if (-not (Test-Path (Join-Path $HostDir 'node_modules'))) { Say 'npm ci in apps/host ...'; & npm --prefix $HostDir ci --no-audit --no-fund; if ($LASTEXITCODE) { Say 'npm ci ist fehlgeschlagen.'; exit 1 } }

# --- 1. Validator --------------------------------------------------------------------------------------------------
Say 'Extensions, Hooks, Skills, Modelle und dev.config.json pruefen ...'
$env:STRATA_DEV_CONFIG = Join-Path $Repo 'dev.config.json'
$v = Run-Logged 'validate' 'node' @((Join-Path $HostDir 'scripts\dev-validate.mjs')) $Repo
$summary = ($v.Text -split "`r?`n" | Where-Object { $_ -like 'dev-validate:*' } | Select-Object -Last 1)
if ($v.Code -eq 0) { Add-Result 'Validator' 'OK' $summary } else { Add-Result 'Validator' 'FAIL' "$summary (Details: $($v.Log))" }
foreach ($line in ($v.Text -split "`r?`n" | Where-Object { $_ -match '^(FEHL|WARN) ' })) { Write-Host "        $line" -ForegroundColor DarkYellow }

# --- 2. Typecheck und npm test -------------------------------------------------------------------------------------
if ($SkipTests) { Add-Result 'Typecheck' 'SKIP' '-SkipTests'; Add-Result 'npm test' 'SKIP' '-SkipTests' }
else {
    Say 'Typecheck (apps/host) ...'
    $t = Run-Logged 'typecheck' 'npm' @('run', 'typecheck') $HostDir
    if ($t.Code -eq 0) { Add-Result 'Typecheck' 'OK' 'tsc --noEmit ohne Fehler' } else { Add-Result 'Typecheck' 'FAIL' "siehe $($t.Log)" }
    Say 'npm test (apps/host, etwa 1 min) ...'
    $n = Run-Logged 'npm-test' 'npm' @('test') $HostDir
    $pass = if ($n.Text -match 'pass (\d+)') { $Matches[1] } else { '?' }
    $fail = if ($n.Text -match 'fail (\d+)') { $Matches[1] } else { '?' }
    $skip = if ($n.Text -match 'skipped (\d+)') { $Matches[1] } else { '?' }
    if ($n.Code -eq 0) { Add-Result 'npm test' 'OK' "$pass bestanden, $fail fehlgeschlagen, $skip uebersprungen" }
    else { Add-Result 'npm test' 'FAIL' "$pass bestanden, $fail fehlgeschlagen (Details: $($n.Log))" }
}

# --- 3. pytest -----------------------------------------------------------------------------------------------------
function Test-PythonDeps([string]$Exe) {
    if (-not $Exe) { return $false }
    try { & $Exe -c 'import rich, textual, jinja2, psutil, yaml, pytest' 2>$null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}
if ($SetupVenv) {
    $sys = (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $sys) { Say 'Kein python im PATH; -SetupVenv braucht Python 3.11 oder neuer.'; exit 1 }
    Say "venv anlegen: $Venv (laedt Pakete aus PyPI) ..."
    & $sys.Source -m venv $Venv
    $vpy = Join-Path $Venv 'Scripts\python.exe'
    if (-not (Test-Path $vpy)) { Say "Die venv wurde nicht angelegt ($vpy fehlt)."; exit 1 }
    & $vpy -m pip install -q --disable-pip-version-check --upgrade pip
    & $vpy -m pip install -q --disable-pip-version-check -r (Join-Path $Repo 'requirements.txt') -r (Join-Path $Repo 'requirements-monitor.txt') pytest
    if ($LASTEXITCODE) { Say 'pip install ist fehlgeschlagen (Netz?). Das System-Python ist unveraendert.'; exit 1 }
    if (-not (Test-PythonDeps $vpy)) { Say 'Die venv ist da, aber rich/textual/jinja2/psutil/pyyaml/pytest lassen sich nicht laden.'; exit 1 }
    Say "venv bereit: $vpy"
}
if ($SkipTests -or $SkipPytest) { Add-Result 'pytest' 'SKIP' $(if ($SkipTests) { '-SkipTests' } else { '-SkipPytest' }) }
else {
    $py = $null
    foreach ($candidate in @($Python, (Join-Path $Venv 'Scripts\python.exe'), $env:STRATA_DEV_PYTHON, 'python')) {
        if ($candidate -and (Test-PythonDeps $candidate)) { $py = $candidate; break }
    }
    if (-not $py) { Add-Result 'pytest' 'WARN' "uebersprungen: kein Python mit rich, textual, jinja2, psutil, pyyaml und pytest. Einmal '.\dev-check.bat -SetupVenv' (eigene venv, System-Python bleibt) oder -Python <exe>." }
    else {
        Say "pytest tests/terminal mit $py (dauert etwa 35 min) ..."
        # Gleiche Umgebung, egal aus welchem Fenster dev-check startet: Die Shell-Tests brauchen eine echte bash (C:\Windows\System32\bash.exe
        # ist ohne WSL-Distribution nur ein Stub), die TUI-Tests eine moderne Konsole (TERM) und Farben (kein NO_COLOR).
        $saved = @{ PATH = $env:PATH; TERM = $env:TERM; NO_COLOR = $env:NO_COLOR }
        $git = Get-Command git -ErrorAction SilentlyContinue
        if ($git) { $usrBin = Join-Path (Split-Path (Split-Path $git.Source)) 'usr\bin'; if (Test-Path (Join-Path $usrBin 'bash.exe')) { $env:PATH = "$usrBin;$env:PATH" } }
        if (-not $env:TERM) { $env:TERM = 'xterm-256color' }
        Remove-Item Env:NO_COLOR -ErrorAction SilentlyContinue
        try { $p = Run-Logged 'pytest' $py @('-m', 'pytest', 'tests/terminal', '-q', '-p', 'no:cacheprovider') $Repo }
        finally { $env:PATH = $saved.PATH; $env:TERM = $saved.TERM; if ($null -ne $saved.NO_COLOR) { $env:NO_COLOR = $saved.NO_COLOR } }
        $failed = @($p.Text -split "`r?`n" | Where-Object { $_ -match '^FAILED ' } | ForEach-Object { ($_ -replace '^FAILED ', '') -replace ' - .*$', '' })
        $new = @($failed | Where-Object { $KnownFailures -notcontains $_ })
        $known = $failed.Count - $new.Count
        $line = ($p.Text -split "`r?`n" | Where-Object { $_ -match '(passed|failed).* in [\d.]+s' } | Select-Object -Last 1)
        if ($p.Text -match 'errors? during collection') { Add-Result 'pytest' 'FAIL' "Sammelfehler (fehlende Pakete?) - siehe $($p.Log)" }
        elseif ($new.Count -eq 0) { Add-Result 'pytest' 'OK' "$line; $known bekannte, hardwarebedingte Fehler" }
        else { Add-Result 'pytest' 'FAIL' "$line; NEU fehlgeschlagen: $($new -join ', ')" }
    }
}

# --- 4. Smoketest ----------------------------------------------------------------------------------------------------
function Get-Listener([int]$Port) {
    $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $c) { return $null }
    $proc = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
    $path = ''
    if ($proc) { try { $path = $proc.Path } catch { $path = '' } }
    return @{ Pid = $c.OwningProcess; Path = $path }
}
function Test-DevPath([string]$Path) { return ($Path -and ($Path -like '*Strata Dev*' -or $Path -like '*Strata-Dev-Build*' -or $Path -like '*StrataDev*')) }
function Get-Http([string]$Url) {
    try { $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 10; return @{ Status = [int]$r.StatusCode; Body = $r.Content } }
    catch { $s = 0; if ($_.Exception.Response) { $s = [int]$_.Exception.Response.StatusCode }; return @{ Status = $s; Body = $_.Exception.Message } }
}
if ($SkipSmoke) { Add-Result 'Smoketest' 'SKIP' '-SkipSmoke' }
else {
    Say 'Smoketest ...'
    $before = @{}; foreach ($p in $LivePorts) { $before[$p] = Get-Listener $p }
    $started = $null
    $mode = Get-Http "http://127.0.0.1:$($Ports.Window)/api/app/mode"
    if ($mode.Status -ne 200) {
        # Die neueste gebaute Dev-App: eine aeltere Installation kennt neuere Routen nicht.
        $exe = @((Join-Path $Local 'Programs\Strata Dev\strata-dev.exe'), (Join-Path $Repo 'apps\desktop\src-tauri\target\release\strata-dev.exe')) | Where-Object { Test-Path $_ } | Sort-Object { (Get-Item $_).LastWriteTime } -Descending | Select-Object -First 1
        if ($exe -and (Test-Path (Join-Path $Local 'Programs\Strata Dev\strata-dev.exe')) -and $exe -notlike '*Programs*') { Say 'Die installierte Dev-App ist aelter als der Build; dev-build.bat -Install aktualisiert sie.' }
        if (-not $exe) { Add-Result 'Smoketest' 'WARN' 'keine gebaute Dev-App gefunden (dev-build.bat -NoStart) und keine laufende auf 47832' }
        else {
            Say "starte $exe fuer den Smoketest (wird danach wieder beendet) ..."
            $started = Start-Process -FilePath $exe -PassThru
            for ($i = 0; $i -lt 90; $i++) { Start-Sleep -Seconds 1; $mode = Get-Http "http://127.0.0.1:$($Ports.Window)/api/app/mode"; if ($mode.Status -eq 200) { break } }
        }
    }
    if ($mode.Status -eq 200) {
        if ($mode.Body -match '"dev"\s*:') { Add-Result 'Dev-Host' 'OK' "antwortet auf $($Ports.Window) als Strata Dev" } else { Add-Result 'Dev-Host' 'FAIL' "auf $($Ports.Window) antwortet ein Host ohne Dev-Modus" }
        $ui = Get-Http "http://127.0.0.1:$($Ports.Window)/"
        if ($ui.Status -eq 200 -and $ui.Body -match 'id="root"') { Add-Result 'UI' 'OK' "laedt auf $($Ports.Window)" } else { Add-Result 'UI' 'FAIL' "GET / auf $($Ports.Window): $($ui.Status)" }
        $hostModels = Get-Http "http://127.0.0.1:$($Ports.Window)/v1/models"
        if ($hostModels.Status -eq 200) { Add-Result '/v1/models (Host)' 'OK' "Host-API auf $($Ports.Window) antwortet" } else { Add-Result '/v1/models (Host)' 'WARN' "Status $($hostModels.Status)" }
        $state = Get-Http "http://127.0.0.1:$($Ports.Window)/api/dev/state"
        if ($state.Status -eq 200) { Add-Result 'Dev-Routen' 'OK' '/api/dev/state antwortet' } else { Add-Result 'Dev-Routen' 'FAIL' "/api/dev/state: $($state.Status)" }
    } elseif ($started -or (Test-Path (Join-Path $Repo 'apps\desktop\src-tauri\target\release\strata-dev.exe'))) { Add-Result 'Dev-Host' 'FAIL' "keine Antwort auf $($Ports.Window) nach 90 s" }
    $engine = Get-Listener $Ports.Engine
    if ($engine) {
        $em = Get-Http "http://127.0.0.1:$($Ports.Engine)/v1/models"
        if ($em.Status -eq 200) { Add-Result '/v1/models (8081)' 'OK' 'die Dev-Engine antwortet' } else { Add-Result '/v1/models (8081)' 'FAIL' "Status $($em.Status)" }
    } else { Add-Result '/v1/models (8081)' 'SKIP' 'kein Modell geladen (8081 frei); geprueft wird nur, wenn ein Modell schon laeuft' }
    foreach ($p in $LivePorts) {
        $now = Get-Listener $p
        if ($now -and (Test-DevPath $now.Path)) { Add-Result "Live-Port $p" 'FAIL' "ein Dev-Prozess lauscht dort: $($now.Path)" }
        elseif ($now) { Add-Result "Live-Port $p" 'OK' "von Live/anderem Programm belegt ($([IO.Path]::GetFileName($now.Path))), nicht von Dev" }
        else { Add-Result "Live-Port $p" 'OK' 'frei' }
    }
    if ($started) {
        Stop-Process -Id $started.Id -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 4
        Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and (Test-DevPath $_.Path) -and $_.ProcessName -in @('node', 'strata-dev', 'python', 'strata-engine', 'strata-dsv41', 'llama-server') } | ForEach-Object { Stop-Process -Id $_.Id -ErrorAction SilentlyContinue }
        Say 'Dev-App fuer den Smoketest wieder beendet.'
    }
}

# --- Bericht -------------------------------------------------------------------------------------------------------
$fails = @($Results | Where-Object { $_.State -eq 'FAIL' }).Count
$warns = @($Results | Where-Object { $_.State -eq 'WARN' }).Count
$report = Join-Path $OutDir "report-$Stamp.txt"
$lines = @("Strata Dev - dev-check $Stamp", "Repo: $Repo ($(& git -C $Repo rev-parse --abbrev-ref HEAD) @ $(& git -C $Repo rev-parse --short HEAD))", '')
$lines += $Results | ForEach-Object { "{0,-5} {1}: {2}" -f $_.State, $_.Step, $_.Text }
$lines += @('', "Ergebnis: $fails Fehler, $warns Warnung(en). Logs: $OutDir")
Set-Content -Path $report -Value $lines -Encoding UTF8
Write-Host ''
Write-Host "dev-check: $fails Fehler, $warns Warnung(en). Bericht: $report" -ForegroundColor $(if ($fails) { 'Red' } elseif ($warns) { 'Yellow' } else { 'Green' })
Note "FERTIG: $fails Fehler, $warns Warnung(en). Bericht: $report"
if ($owned) { $mutex.ReleaseMutex() }
exit $(if ($fails) { 1 } else { 0 })
