# Strata Dev: Aenderungen aus dev-sandbox fuer Live vorbereiten. Siehe docs/DEV_BUILD.md ("Uebernehmen").
#
#   .\dev-promote.ps1                                   # nur ansehen: Commits und Dateien, getrennt nach Kern-Aenderungen und Dev-only
#   .\dev-promote.ps1 -Name mtp-fix -Commits e26737c8 -DryRun   # zeigt, was uebernommen wuerde, aendert nichts
#   .\dev-promote.ps1 -Name mtp-fix -Commits e26737c8 -Yes      # uebernimmt (nur mit -Yes, deiner ausdruecklichen Freigabe)
#   .\dev-promote.ps1 -Name x -Files tools\mtp_fetch.py,tools\test_mtp_fetch.py -Yes   # einzelne Dateien statt ganzer Commits
#   -IncludeDevOnly                                     # Dev-only-Dateien der gewaehlten Commits/Dateien mitnehmen (sonst nie)
#
# Was -Yes tut, und nur das:
#   1. lokaler Backup-Branch backup/live-<Datum> vom aktuellen main (kein Tag)
#   2. lokaler Branch promote/<Name> von main, in einem temporaeren Worktree (dein Checkout bleibt, wie er ist)
#   3. die gewaehlten Commits (cherry-pick -x) bzw. Dateien darauf, ohne Dev-only-Dateien, ausser -IncludeDevOnly
#   4. fertige Befehle fuer Push und Pull Request ausgeben
# Kein Push, kein Release, kein Tag, main wird nicht veraendert. Zurueck: dev-rollback.ps1.
[CmdletBinding()]
param(
    [string]$Name = '',
    [string[]]$Commits = @(),
    [string[]]$Files = @(),
    [switch]$IncludeDevOnly,
    [switch]$DryRun,
    [switch]$Yes,
    [string]$Base = '',
    [string]$Source = 'dev-sandbox',
    [string]$Title = ''
)
$ErrorActionPreference = 'Stop'
$Repo = $PSScriptRoot
$Local = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $env:USERPROFILE 'AppData\Local' }
$OutDir = Join-Path $Local 'StrataDev\promote'

function Say([string]$Text) { Write-Host "dev-promote: $Text" }
function Fail([string]$Text) { [Console]::Error.WriteLine("dev-promote: $Text"); exit 1 }
function RepoGit { $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'; $out = & git.exe -c core.longpaths=true -C $Repo @args 2>&1; $code = $LASTEXITCODE; $ErrorActionPreference = $old; return @{ Code = $code; Out = @($out | ForEach-Object { "$_" }) } }
function GitDo([string]$Dir) { $rest = $args; $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'; $out = & git.exe -c core.longpaths=true -C $Dir @rest 2>&1; $code = $LASTEXITCODE; $ErrorActionPreference = $old; return @{ Code = $code; Out = @($out | ForEach-Object { "$_" }) } }

# Dev-only: was nur Strata Dev braucht (Badge, Ports, Identifier, Tauri-Dev-Config, dev\, Skripte, Berichte, Dev-Module und ihre Tests).
$DevOnlyPatterns = @(
    '^dev/', '^dev\.config\.json$', '^dev-[a-z]+\.(ps1|bat)$', '^docs/DEV_BUILD\.md$', '^PHASE\d+_BERICHT\.md$', '^PROMPTS\.md$', '^ANALYSE\.md$',
    '^skill-struktur\.md$', '^\.claude/', '^apps/desktop/src-tauri/tauri\.dev-build\.conf\.json$', '^apps/desktop/src-tauri/src/devbuild\.rs$',
    '^apps/host/server/dev-[a-z-]+\.mjs$', '^apps/host/server/agent/plugins/strata-dev-hooks\.mjs$', '^apps/host/src/dev/',
    '^apps/host/src/components/DevBadge\.tsx$', '^apps/host/tests/dev-[a-z-]+\.test\.mjs$', '^apps/host/tests/fixtures/mock-llama-server\.mjs$',
    '^apps/host/scripts/dev-validate\.mjs$', '^strata_control/dev_models\.py$', '^tests/terminal/test_dev_models\.py$'
)
function Test-DevOnly([string]$Path) { foreach ($p in $DevOnlyPatterns) { if ($Path -match $p) { return $true } }; return $false }
# Hinweise auf Dev-Code in einer Kern-Datei (die Zeilen, die dev-sandbox dort hinzufuegt)
$DevMarks = 'STRATA_DEV|isDevMode|devHooks|DEV_SKILLS|DEV_SETTINGS|devLlama|dev-mode\.mjs|dev-[a-z]+\.mjs|/dev/|DevElement|DevSlot|useDevList|DevBadge|DevModels|DevSkills|devChatMode|devModeOverride|devSkillsOverride|devLoop|dev-build'

# --- Basis ---------------------------------------------------------------------------------------------------------
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail 'git fehlt.' }
if (-not $Base) { $Base = if ((RepoGit rev-parse --verify --quiet 'refs/heads/main').Code -eq 0) { 'main' } else { 'origin/main' } }
if ((RepoGit rev-parse --verify --quiet $Base).Code -ne 0) { Fail "Die Basis $Base gibt es nicht. 'git fetch origin main' oder -Base angeben." }
if ((RepoGit rev-parse --verify --quiet $Source).Code -ne 0) { Fail "Den Quell-Branch $Source gibt es nicht." }
$baseSha = (RepoGit rev-parse --short $Base).Out[0]
$mb = (RepoGit merge-base $Base $Source).Out[0]

$commitList = @((RepoGit log --reverse --format='%h%x09%s' "$Base..$Source").Out | Where-Object { $_ })
$changed = @((RepoGit diff --name-status "$mb" $Source).Out | Where-Object { $_ })
# One diff for all files: which core files gain lines that mention Dev code.
$hooked = @{}; $current = $null
foreach ($line in (RepoGit diff --unified=0 "$mb" $Source).Out) {
    if ($line -match '^\+\+\+ b/(.+)$') { $current = $Matches[1]; continue }
    if ($current -and $line -match '^\+' -and $line -match $DevMarks) { $hooked[$current] = $true }
}
# One log for all files: which commits touched each file.
$touched = @{}; $commit = $null
foreach ($line in (RepoGit log --name-only --format='@%h' "$Base..$Source").Out) {
    if ($line -match '^@([0-9a-f]+)$') { $commit = $Matches[1] } elseif ($line) { if (-not $touched[$line]) { $touched[$line] = @() }; $touched[$line] += $commit }
}
$fileRows = foreach ($row in $changed) {
    $parts = $row -split "`t"
    $path = $parts[-1]
    $dev = Test-DevOnly $path
    [pscustomobject]@{ Status = $parts[0]; Path = $path; DevOnly = $dev; DevHook = (-not $dev -and $hooked.ContainsKey($path)); Commits = (@($touched[$path]) -join ',') }
}

function Show-Overview {
    Say "Basis $Base ($baseSha), Quelle $Source, $($commitList.Count) Commit(s) voraus."
    Write-Host ''
    Write-Host 'Commits (Kern-Dateien / Dev-only-Dateien):'
    foreach ($c in $commitList) {
        $h, $s = $c -split "`t", 2
        $files = @($touched.Keys | Where-Object { @($touched[$_]) -contains $h })
        $core = @($files | Where-Object { -not (Test-DevOnly $_) }).Count
        Write-Host ("  {0}  K {1,3} / D {2,3}  {3}" -f $h, $core, ($files.Count - $core), $s)
    }
    Write-Host ''
    Write-Host 'Kern-Aenderungen (wuerden Live aendern):'
    foreach ($f in ($fileRows | Where-Object { -not $_.DevOnly })) {
        $mark = if ($f.DevHook) { '  [Dev-Haken: nur mit STRATA_DEV aktiv; braucht die Dev-only-Module]' } else { '' }
        Write-Host ("  {0} {1}  ({2}){3}" -f $f.Status, $f.Path, $f.Commits, $mark)
    }
    Write-Host ''
    Write-Host 'Dev-only (nie uebernommen, ausser -IncludeDevOnly):'
    foreach ($f in ($fileRows | Where-Object { $_.DevOnly })) { Write-Host ("  {0} {1}" -f $f.Status, $f.Path) }
    Write-Host ''
    Write-Host 'Auswaehlen: dev-promote.bat -Name <name> -Commits <hash>[,<hash>] -DryRun   (oder -Files <pfad>[,<pfad>])'
    Write-Host 'Dann mit -Yes statt -DryRun uebernehmen. Ein Kern-Fix ohne Dev-Haken ist z. B. e26737c8 (MTP-Check, tools\mtp_fetch.py).'
}

if (-not $Name -and -not $Commits.Count -and -not $Files.Count) { Show-Overview; exit 0 }
if (-not $Name) { Fail '-Name fehlt (der Branch heisst dann promote/<Name>).' }
if ($Name -notmatch '^[a-z0-9][a-z0-9._-]{0,60}$') { Fail '-Name: Kleinbuchstaben, Ziffern, Punkt, Minus, Unterstrich.' }
if (-not $Commits.Count -and -not $Files.Count) { Fail 'Waehle -Commits oder -Files.' }
$Commits = @($Commits | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$Files = @($Files | ForEach-Object { $_ -split ',' } | Where-Object { $_ } | ForEach-Object { $_ -replace '\\', '/' })
foreach ($c in $Commits) { if ((RepoGit merge-base --is-ancestor $c $Source).Code -ne 0) { Fail "$c ist kein Commit von $Source." } }
foreach ($f in $Files) { if (-not ($fileRows | Where-Object { $_.Path -eq $f })) { Fail "$f ist zwischen $Base und $Source nicht geaendert." } }

# Was uebernommen wird
$picked = if ($Commits.Count) { @($Commits | ForEach-Object { (RepoGit show --name-only --format= $_).Out } | Where-Object { $_ } | Sort-Object -Unique) } else { $Files }
$keep = @($picked | Where-Object { $IncludeDevOnly -or -not (Test-DevOnly $_) })
$dropped = @($picked | Where-Object { -not $IncludeDevOnly -and (Test-DevOnly $_) })
if (-not $keep.Count) { Fail 'Nach dem Weglassen der Dev-only-Dateien bleibt nichts uebrig (-IncludeDevOnly nimmt sie mit).' }

# Verweise einer Kern-Datei auf Dateien, die nicht mitkommen (sonst bricht der Build auf main)
$missing = New-Object System.Collections.ArrayList
foreach ($f in ($keep | Where-Object { $_ -match '\.(mjs|js|ts|tsx|py)$' })) {
    $text = ((RepoGit show "${Source}:$f").Out) -join "`n"
    foreach ($m in [regex]::Matches($text, "(?:from|import)\s*\(?\s*['""](\.{1,2}/[^'""]+)['""]")) {
        $target = [IO.Path]::GetFullPath((Join-Path (Join-Path $Repo (Split-Path $f)) $m.Groups[1].Value)).Substring($Repo.Length + 1) -replace '\\', '/'
        $candidates = @($target, "$target.ts", "$target.tsx", "$target.mjs", "$target.js")
        $inBase = $candidates | Where-Object { (RepoGit cat-file -e "${Base}:$_").Code -eq 0 } | Select-Object -First 1
        $inKeep = $candidates | Where-Object { $keep -contains $_ } | Select-Object -First 1
        if (-not $inBase -and -not $inKeep) { [void]$missing.Add("$f -> $target") }
    }
    if ($f -match '\.py$') { foreach ($m in [regex]::Matches($text, 'from strata_control import ([a-z_]+)')) { $p = "strata_control/$($m.Groups[1].Value).py"; if ((RepoGit cat-file -e "${Base}:$p").Code -ne 0 -and $keep -notcontains $p) { [void]$missing.Add("$f -> $p") } } }
}

Say "promote/$Name von $Base ($baseSha)"
Write-Host "  uebernimmt: $($keep.Count) Datei(en)"; $keep | ForEach-Object { Write-Host "    $_" }
if ($dropped.Count) { Write-Host "  laesst weg (Dev-only): $($dropped.Count)"; $dropped | ForEach-Object { Write-Host "    $_" } }
if ($missing.Count) { Write-Host '  WARNUNG: diese Verweise zeigen auf Dateien, die auf main fehlen und nicht mitkommen (der Build wuerde brechen):' -ForegroundColor Yellow; $missing | Sort-Object -Unique | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow } }
if ($DryRun -or -not $Yes) {
    if (-not $DryRun) { Say 'Nichts geaendert: zum Uebernehmen -Yes anhaengen (deine ausdrueckliche Freigabe).' } else { Say 'Probelauf: nichts geaendert.' }
    exit 0
}

# --- Uebernehmen (-Yes) ----------------------------------------------------------------------------------------------
if ((RepoGit rev-parse --verify --quiet "refs/heads/promote/$Name").Code -eq 0) { Fail "promote/$Name gibt es schon. Anderen -Name waehlen oder mit dev-rollback.ps1 -DeletePromote $Name entfernen." }
$backup = "backup/live-$(Get-Date -Format 'yyyyMMdd')"
if ((RepoGit rev-parse --verify --quiet "refs/heads/$backup").Code -eq 0) { $backup = "$backup-$(Get-Date -Format 'HHmmss')" }
$r = RepoGit branch $backup $Base; if ($r.Code) { Fail "Backup-Branch: $($r.Out -join ' ')" }
Say "Backup-Branch $backup -> $Base ($baseSha)"

# Ein kurzer Ordner: das Repo hat lange Pfade (bench\results), und der Worktree haelt kurz einen ganzen Checkout von main.
$tree = Join-Path $Local "StrataDev\pw\$(Get-Random)"
$r = RepoGit worktree add -b "promote/$Name" $tree $Base; if ($r.Code) { Fail "Worktree: $($r.Out -join ' ')" }
$ok = $true
try {
    if ($Commits.Count) {
        foreach ($c in $Commits) {
            $subject = (RepoGit log -1 --format=%s $c).Out[0]
            $cp = GitDo $tree cherry-pick --no-commit $c
            if ($cp.Code) {
                # Konflikte nur in Dev-only-Dateien (die ohnehin wegbleiben) loest das Skript selbst; einer in einer Kern-Datei bricht ab.
                $conflicts = @((GitDo $tree diff --name-only --diff-filter=U).Out | Where-Object { $_ })
                $coreConflicts = @($conflicts | Where-Object { $IncludeDevOnly -or -not (Test-DevOnly $_) })
                if ($coreConflicts.Count -or -not $conflicts.Count) {
                    GitDo $tree cherry-pick --abort | Out-Null; GitDo $tree reset -q --hard | Out-Null
                    $earlier = @($coreConflicts | ForEach-Object { @($touched[$_]) } | Where-Object { $_ -and $_ -ne $c -and $Commits -notcontains $_ } | Sort-Object -Unique)
                    Say "Konflikt bei $c ($subject) in: $($coreConflicts -join ', ')"
                    if ($earlier.Count) { Say "Diese Dateien bauen auf frueheren Commits von $Source auf. Waehle sie mit, in dieser Reihenfolge vorne: $($earlier -join ', ')" }
                    $ok = $false; break
                }
                foreach ($f in $conflicts) { if ((GitDo $tree cat-file -e "HEAD:$f").Code -eq 0) { GitDo $tree checkout HEAD -- $f | Out-Null } else { GitDo $tree rm -q -f -- $f | Out-Null } }
            }
            $staged = @((GitDo $tree diff --cached --name-only).Out | Where-Object { $_ })
            foreach ($f in ($staged | Where-Object { -not $IncludeDevOnly -and (Test-DevOnly $_) })) {
                if ((GitDo $tree cat-file -e "HEAD:$f").Code -eq 0) { GitDo $tree restore --source=HEAD --staged --worktree -- $f | Out-Null } else { GitDo $tree rm -q --cached -f -- $f | Out-Null; Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $tree $f) }
            }
            if (-not ((GitDo $tree diff --cached --name-only).Out | Where-Object { $_ })) { Say "$c bringt ohne Dev-only-Dateien nichts mit; uebersprungen."; continue }
            $msg = "$subject`n`n(cherry picked from commit $((RepoGit rev-parse $c).Out[0]) on $Source$(if (-not $IncludeDevOnly -and $dropped.Count) { ', without the Dev-only files' }))"
            $cm = GitDo $tree commit -q -m $msg; if ($cm.Code) { Say "Commit: $($cm.Out -join ' ')"; $ok = $false; break }
        }
    } else {
        $co = GitDo $tree checkout $Source -- @($keep); if ($co.Code) { Say "Dateien holen: $($co.Out -join ' ')"; $ok = $false }
        else { $cm = GitDo $tree commit -q -m "Promote from $Source`: $($keep -join ', ')"; if ($cm.Code) { Say "Commit: $($cm.Out -join ' ')"; $ok = $false } }
    }
    $made = (GitDo $tree log --oneline "$Base..HEAD").Out
} finally {
    GitDo $Repo worktree remove --force $tree | Out-Null
    $pw = Split-Path $tree
    if ((Test-Path $pw) -and -not (Get-ChildItem $pw -Force)) { Remove-Item $pw -Force -ErrorAction SilentlyContinue }
}
if (-not $ok) { RepoGit branch -D "promote/$Name" | Out-Null; RepoGit branch -D $backup | Out-Null; Fail "Abgebrochen; nichts geaendert (promote/$Name und das Backup $backup dieses Laufs wurden wieder entfernt)." }

# --- Befehle fuer Push und PR (nur ausgeben) ------------------------------------------------------------------------
$areas = @($keep | ForEach-Object { if ($_ -match '^(apps/host|apps/desktop)/') { 'GUI' } elseif ($_ -match '^(strata_cli|strata_control|strata\.py)') { 'TUI' } elseif ($_ -match '^(src|include|serve|sycl|metal)/') { 'Engine' } elseif ($_ -match '^(docs/|README)') { 'Docs' } elseif ($_ -match '^(bench|tools/bench)') { 'Benchmark' } else { 'Other' } } | Sort-Object -Unique)
$area = if ($areas.Count -eq 1) { $areas[0] } elseif ($areas -contains 'GUI') { 'GUI' } else { 'Other' }
if (-not $Title) { $Title = (RepoGit log -1 --format=%s $(if ($Commits.Count) { $Commits[-1] } else { $Source })).Out[0] -replace '^Dev build: ', '' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$body = Join-Path $OutDir "$Name-pr.md"
Set-Content -Encoding UTF8 -Path $body -Value @(
    '## Problem and result', '', '<what was wrong, what it does now>', '',
    '## Approach', '', "Taken from the local dev-sandbox branch: $(if ($Commits.Count) { $Commits -join ', ' } else { 'files ' + ($keep -join ', ') }).", '',
    '## Validation', '', '- Passed: <npm test in apps/host, python -m pytest tests/terminal, ...>', '- Failed: <none / which>', '- Not run: <which, why>', '',
    '## Limits', '', '<what was not tested>', '',
    'The default path stays byte-identical to the last release: <how checked>.')
Write-Host ''
Say "fertig: promote/$Name"; $made | ForEach-Object { Write-Host "    $_" }
Write-Host ''
Write-Host 'Nichts wurde gepusht. Wenn du es willst, sind das die Befehle (erst pruefen, z. B. git log main..promote/<Name>):'
Write-Host "  git push -u origin promote/$Name"
Write-Host "  gh pr create --base main --head promote/$Name --title `"[$area] $Title`" --body-file `"$body`""
Write-Host "Titel und Text nach CONTRIBUTING.md: Bereich vorne ([Engine], [GUI], [TUI], [Benchmark], [Docs], [Other]); den Text in $body ausfuellen."
Write-Host "Zurueck: dev-rollback.bat -DeletePromote $Name -Yes  (Backup: $backup)"
