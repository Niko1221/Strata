param([string]$Root = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath($Root)
$appPath = Join-Path $projectRoot 'dist\Strata\Strata.exe'
if (-not (Test-Path -LiteralPath $appPath -PathType Leaf)) { throw 'Build the desktop app first.' }
$backupDirectory = Join-Path $projectRoot '.local\desktop-shortcut-backups'
New-Item -ItemType Directory -Path $backupDirectory -Force | Out-Null
$desktopDirectory = [Environment]::GetFolderPath('Desktop')
$programsDirectory = Join-Path ([Environment]::GetFolderPath('Programs')) 'Strata'
New-Item -ItemType Directory -Path $programsDirectory -Force | Out-Null
$shortcutPaths = @((Join-Path $desktopDirectory 'Strata.lnk'), (Join-Path $programsDirectory 'Strata.lnk'))
$shortcutWriter = New-Object -ComObject WScript.Shell
foreach ($shortcutPath in $shortcutPaths) {
    if (Test-Path -LiteralPath $shortcutPath) {
        $backupName = ([IO.Path]::GetFileName([IO.Path]::GetDirectoryName($shortcutPath))) + '-' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fffffff') + '.lnk'
        Copy-Item -LiteralPath $shortcutPath -Destination (Join-Path $backupDirectory $backupName)
    }
    $shortcut = $shortcutWriter.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $appPath
    $shortcut.Arguments = '--root "' + $projectRoot + '"'
    $shortcut.WorkingDirectory = $projectRoot
    $shortcut.IconLocation = (Join-Path $projectRoot 'dist\Strata\strata.ico') + ',0'
    $shortcut.Description = 'Strata - Local models and chat history'
    $shortcut.Save()
    $verified = $shortcutWriter.CreateShortcut($shortcutPath)
    if ($verified.TargetPath -ne $appPath -or $verified.WorkingDirectory -ne $projectRoot) { throw 'Shortcut verification failed.' }
}
@{ executable = $appPath; shortcuts = $shortcutPaths; verified = $true } | ConvertTo-Json
