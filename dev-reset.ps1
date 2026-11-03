# Strata Dev zuruecksetzen: loescht NUR die Daten von Strata Dev, nach Rueckfrage. Siehe docs/DEV_BUILD.md.
#
#   .\dev-reset.ps1          # zeigt, was geloescht wird, und fragt nach
#   .\dev-reset.ps1 -Force   # ohne Rueckfrage (fuer Skripte)
#
# Geloescht werden:
#   %LOCALAPPDATA%\StrataDev               Chats, Host-State, Skills, Profile, Presets, Tuning, Logs, Einstellungen von Dev
#   %LOCALAPPDATA%\ai.strata.desktop.dev   Browser-Speicher des Dev-Fensters (WebView2)
# Nie geloescht: Strata (Live), seine Daten, Modelle, Engine-Komponenten, die installierte Dev-App selbst.
[CmdletBinding()]
param([switch]$Force)
$ErrorActionPreference = 'Stop'

$Local = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $env:USERPROFILE 'AppData\Local' }
$targets = @((Join-Path $Local 'StrataDev'), (Join-Path $Local 'ai.strata.desktop.dev'))

# Nur diese zwei Namen, direkt unter LOCALAPPDATA: nie ein Live-Ordner, nie etwas anderes.
$allowed = @('StrataDev', 'ai.strata.desktop.dev')
foreach ($t in $targets) {
    $full = [IO.Path]::GetFullPath($t).TrimEnd('\')
    if ((Split-Path $full -Parent) -ine [IO.Path]::GetFullPath($Local).TrimEnd('\') -or $allowed -notcontains (Split-Path $full -Leaf)) {
        [Console]::Error.WriteLine("dev-reset: unerwarteter Pfad $full, Abbruch."); exit 1
    }
}

$running = @(Get-Process -Name 'strata-dev' -ErrorAction SilentlyContinue)
if ($running.Count) { [Console]::Error.WriteLine('dev-reset: Strata Dev laeuft noch. Bitte erst beenden.'); exit 1 }

$present = @($targets | Where-Object { Test-Path -LiteralPath $_ })
if (-not $present.Count) { Write-Host 'dev-reset: Es gibt keine Daten von Strata Dev. Nichts zu tun.'; exit 0 }

Write-Host 'dev-reset: Diese Ordner werden geloescht:'
$present | ForEach-Object { Write-Host "  $_" }
Write-Host 'Strata (Live), Modelle und Engine-Komponenten bleiben unberuehrt.'
if (-not $Force) {
    $answer = Read-Host 'Wirklich loeschen? (ja/nein)'
    if ($answer -notmatch '^(j|ja|y|yes)$') { Write-Host 'dev-reset: Abgebrochen, nichts geloescht.'; exit 0 }
}
foreach ($t in $present) { Remove-Item -LiteralPath $t -Recurse -Force }
Write-Host 'dev-reset: Fertig. Strata Dev startet beim naechsten Mal leer.'
