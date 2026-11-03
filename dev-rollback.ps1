# Strata Dev: eine Uebernahme (dev-promote.ps1) zuruecknehmen. Siehe docs/DEV_BUILD.md ("Zurueckrollen").
#
#   .\dev-rollback.ps1                                     # nur ansehen: Backup-Branches backup/live-* und Branches promote/*
#   .\dev-rollback.ps1 -DeletePromote mtp-fix -Yes         # den lokalen Branch promote/mtp-fix loeschen (nie gepusht? dann ist alles weg)
#   .\dev-rollback.ps1 -Backup backup/live-20261102 -Yes   # den lokalen main auf den Stand des Backups setzen
#   .\dev-rollback.ps1 -DeleteBackup backup/live-20261102 -Yes
#
# Ohne -Yes wird nur gezeigt, was passieren wuerde. Nichts wird gepusht; was schon auf GitHub ist, nimmst du selbst zurueck
# (die passenden Befehle werden ausgegeben). Ein ausgecheckter main wird nicht umgesetzt (dann erst auf dev-sandbox wechseln).
[CmdletBinding()]
param(
    [string]$Backup = '',
    [string]$DeletePromote = '',
    [string]$DeleteBackup = '',
    [switch]$Yes
)
$ErrorActionPreference = 'Stop'
$Repo = $PSScriptRoot
function Say([string]$Text) { Write-Host "dev-rollback: $Text" }
function Fail([string]$Text) { [Console]::Error.WriteLine("dev-rollback: $Text"); exit 1 }
function RepoGit { $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'; $out = & git.exe -C $Repo @args 2>&1; $code = $LASTEXITCODE; $ErrorActionPreference = $old; return @{ Code = $code; Out = @($out | ForEach-Object { "$_" }) } }
function Has([string]$Ref) { return (RepoGit rev-parse --verify --quiet "refs/heads/$Ref").Code -eq 0 }

$current = (RepoGit rev-parse --abbrev-ref HEAD).Out[0]
$backups = @((RepoGit for-each-ref --format='%(refname:short)  %(objectname:short)  %(committerdate:short)' 'refs/heads/backup/live-*').Out | Where-Object { $_ })
$promotes = @((RepoGit for-each-ref --format='%(refname:short)  %(objectname:short)  %(upstream:short)' 'refs/heads/promote/*').Out | Where-Object { $_ })

if (-not $Backup -and -not $DeletePromote -and -not $DeleteBackup) {
    Say "ausgecheckt: $current"
    Write-Host 'Backups (Stand von main vor einer Uebernahme):'; if ($backups.Count) { $backups | ForEach-Object { Write-Host "  $_" } } else { Write-Host '  keine' }
    Write-Host 'Uebernahme-Branches (mit Upstream = schon gepusht):'; if ($promotes.Count) { $promotes | ForEach-Object { Write-Host "  $_" } } else { Write-Host '  keine' }
    Write-Host ''
    Write-Host 'Zurueck: -DeletePromote <name> -Yes (Branch weg), -Backup <backup-branch> -Yes (lokaler main auf das Backup).'
    exit 0
}

if ($DeletePromote) {
    $branch = "promote/$($DeletePromote -replace '^promote/', '')"
    if (-not (Has $branch)) { Fail "$branch gibt es nicht." }
    if ($current -eq $branch) { Fail "$branch ist ausgecheckt; erst auf dev-sandbox wechseln." }
    $upstream = (RepoGit for-each-ref --format='%(upstream:short)' "refs/heads/$branch").Out[0]
    if (-not $Yes) { Say "wuerde $branch loeschen$(if ($upstream) { " (gepusht nach $upstream)" }). Mit -Yes ausfuehren."; exit 0 }
    $r = RepoGit branch -D $branch; if ($r.Code) { Fail ($r.Out -join ' ') }
    Say "$branch geloescht."
    if ($upstream) { Write-Host "Auf GitHub liegt er noch. Wenn du ihn dort auch entfernen willst: git push origin --delete $branch  (ein offener PR wird dabei geschlossen)" }
}

if ($Backup) {
    if (-not (Has $Backup)) { Fail "$Backup gibt es nicht (dev-rollback.ps1 ohne Argumente zeigt die Backups)." }
    if (-not (Has 'main')) {
        Say "Es gibt keinen lokalen main: lokal ist nichts zurueckzusetzen. Die Uebernahme lebt nur auf promote/*-Branches (-DeletePromote)."
        Write-Host "Falls sie schon auf GitHub in main gemergt ist, nimmst du sie dort zurueck, z. B. mit einem Revert-PR (git revert <merge-commit>) oder, nur wenn du sicher bist: git push --force-with-lease origin ${Backup}:main"
        exit 0
    }
    if ($current -eq 'main') { Fail 'main ist ausgecheckt; erst auf dev-sandbox wechseln (git switch dev-sandbox), dann noch einmal.' }
    $from = (RepoGit rev-parse --short main).Out[0]; $to = (RepoGit rev-parse --short $Backup).Out[0]
    if (-not $Yes) { Say "wuerde den lokalen main von $from auf $to ($Backup) setzen. Mit -Yes ausfuehren."; exit 0 }
    $r = RepoGit branch -f main $Backup; if ($r.Code) { Fail ($r.Out -join ' ') }
    Say "lokaler main: $from -> $to ($Backup). Der alte Stand bleibt erreichbar ueber: git branch rescued-main $from"
    Write-Host "Auf GitHub ist nichts geaendert. Nur wenn main dort schon den alten Stand hatte und du ihn ersetzen willst: git push --force-with-lease origin main"
}

if ($DeleteBackup) {
    if ($DeleteBackup -notlike 'backup/live-*') { Fail 'Nur Branches backup/live-* werden hier geloescht.' }
    if (-not (Has $DeleteBackup)) { Fail "$DeleteBackup gibt es nicht." }
    if (-not $Yes) { Say "wuerde $DeleteBackup loeschen. Mit -Yes ausfuehren."; exit 0 }
    $r = RepoGit branch -D $DeleteBackup; if ($r.Code) { Fail ($r.Out -join ' ') }
    Say "$DeleteBackup geloescht."
}
