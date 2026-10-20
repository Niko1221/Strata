# Strata Terminal: one-line installer for Windows (PowerShell 5.1 or newer).
#
#     irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
#
# What it does, in order (and nothing else):
#   1. finds out which computer this is (Windows, processor, graphics card);
#   2. downloads the Terminal archive named in the release folder's install-manifest.txt and checks its SHA-256 against it;
#   3. unpacks it into  %LOCALAPPDATA%\Strata\terminal\<version>  (no administrator rights);
#   4. checks the release's signed feed with the Terminal it just unpacked (`strata update --check`); if that fails it removes what it
#      unpacked and stops;
#   5. adds  %LOCALAPPDATA%\Strata\terminal\current\bin  to your user PATH and prints:  Installed. Run: strata launch
#
# The engine and the model are not part of this: `strata launch` fetches the engine for your graphics card the first time you choose
# a model, from the same release folder, signed.
#
# <feed> is where the release folder is hosted: GitHub Releases of Niko1221/Strata (the "latest" release's files), built in as the default.
# Set $env:STRATA_FEED to another folder (a URL) to use that instead, for example a local test feed:  $env:STRATA_FEED='http://127.0.0.1:8099'; irm http://127.0.0.1:8099/install.ps1 | iex
# Optional: $env:STRATA_DATA_DIR (default %LOCALAPPDATA%\Strata), $env:STRATA_INSTALL_TEST_TRUST='1' (a test feed: trust only its
# test-trust.json for step 4), $env:STRATA_NO_PATH='1' (do not change the user PATH).
& {   # a child scope: `irm | iex` runs in the caller's scope, and this script's variables must not overwrite the caller's
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Die([string]$msg) { [Console]::Error.WriteLine("strata installer: $msg"); throw "strata installer stopped" }

$feed = if ($env:STRATA_FEED) { $env:STRATA_FEED } else { 'https://github.com/Niko1221/Strata/releases/latest/download' }
$feed = $feed.TrimEnd('/')
if (-not $feed) {
    Die 'this installer does not know where the Strata release folder is. Set $env:STRATA_FEED to its address and run it again, for example: $env:STRATA_FEED=''https://example.org/strata'''
}
if ([Environment]::OSVersion.Platform -ne 'Win32NT') { Die 'this is the Windows installer; on Linux or macOS use install.sh.' }
$arch = $env:PROCESSOR_ARCHITEW6432; if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
if ($arch -ne 'AMD64') { Die "this computer ($arch) has no Strata Terminal yet. Supported: Windows on x64." }
$platform = 'windows-x86_64'
$gpu = ''
# a discrete card is named before a built-in one (a Ryzen's "AMD Radeon(TM) Graphics" is listed first on many PCs); this line is only for the person to read
try { $cards = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch 'Basic|Remote' }); $gpu = ($cards | Sort-Object @{ Expression = { if ($_.Name -match 'NVIDIA|GeForce|RTX|GTX|Quadro|Radeon (RX|PRO|AI)|Arc|Instinct') { 0 } else { 1 } } } | Select-Object -First 1 -ExpandProperty Name) } catch { }
Write-Host 'Strata Terminal installer'
Write-Host "  This computer: Windows $arch ($platform)$(if ($gpu) { ", graphics: $gpu" })"
Write-Host "  Release folder: $feed"

$data = if ($env:STRATA_DATA_DIR) { $env:STRATA_DATA_DIR } else { Join-Path $env:LOCALAPPDATA 'Strata' }
$work = Join-Path ([IO.Path]::GetTempPath()) ("strata-install-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $work | Out-Null
try {
    try { Invoke-WebRequest -UseBasicParsing -Uri "$feed/install-manifest.txt" -OutFile (Join-Path $work 'manifest.txt') }
    catch { Die "could not read $feed/install-manifest.txt. Check the address and your internet connection." }
    $line = Get-Content (Join-Path $work 'manifest.txt') | Where-Object { $_ -match "^$platform\s" } | Select-Object -First 1
    if (-not $line) { Die "this release has no Terminal for $platform." }
    $f = -split $line
    $version, $rel, $want, $size = $f[1], $f[2], $f[3].ToLower(), $f[4]
    Write-Host "  Terminal $version ($([math]::Round([double]$size / 1MB)) MB)"
    Write-Host 'Downloading...'
    $zip = Join-Path $work 'terminal.zip'
    try { Invoke-WebRequest -UseBasicParsing -Uri "$feed/$rel" -OutFile $zip }
    catch { Die 'the download failed. Check your internet connection and run this again; nothing was installed.' }
    $got = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
    if ($got -ne $want) { Die "the download does not match the release's checksum (got $got, wanted $want). It was not installed; try again." }
    Write-Host '  Checksum ok.'

    $dest = Join-Path $data "terminal\$version"
    $incoming = "$dest.incoming"
    New-Item -ItemType Directory -Force -Path (Join-Path $data 'terminal') | Out-Null
    if (Test-Path $incoming) { Remove-Item -Recurse -Force $incoming }
    try { Expand-Archive -LiteralPath $zip -DestinationPath $incoming -Force }
    catch { Die 'the archive could not be unpacked (is the disk full?). Nothing was installed.' }
    # the archive of a release folder is rooted at the bundle itself (control package) or in one folder (terminal archive)
    if (-not (Test-Path (Join-Path $incoming 'bin\strata.cmd'))) {
        $inner = Get-ChildItem $incoming -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'bin\strata.cmd') } | Select-Object -First 1
        if (-not $inner) { Remove-Item -Recurse -Force $incoming; Die 'the archive does not look like a Strata Terminal.' }
        $moved = "$dest.moved"; Move-Item $inner.FullName $moved; Remove-Item -Recurse -Force $incoming; Move-Item $moved $incoming
    }

    if ($env:STRATA_INSTALL_TEST_TRUST -eq '1') {
        $trust = Join-Path $work 'test-trust.json'
        try { Invoke-WebRequest -UseBasicParsing -Uri "$feed/test-trust.json" -OutFile $trust } catch { Remove-Item -Recurse -Force $incoming; Die 'the test feed has no test-trust.json' }
        $env:STRATA_UPDATE_TRUSTED_KEYS = $trust
    }
    Write-Host "Checking the release's signature..."
    $strata = Join-Path $incoming 'bin\strata.cmd'
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'      # a native program's stderr note must not stop the script
    $out = & $strata update --check --no-engine --feed $feed 2>&1 | Out-String
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($code -ne 0) {
        ($out -split "`n" | Select-Object -First 6 | ForEach-Object { "    $_" }) | ForEach-Object { [Console]::Error.WriteLine($_) }
        Remove-Item -Recurse -Force $incoming
        Die "the release folder's signed feed could not be verified, so nothing was installed. If you trust this source, ask its owner to publish a signed feed."
    }
    Write-Host '  Signature ok.'

    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    Move-Item $incoming $dest
    $current = Join-Path $data 'terminal\current'
    if (Test-Path $current) { cmd /c rmdir "$current" | Out-Null }
    cmd /c mklink /J "$current" "$dest" | Out-Null
    $bin = Join-Path $current 'bin'
    if ($env:STRATA_NO_PATH -ne '1') {
        $user = [Environment]::GetEnvironmentVariable('Path', 'User')
        if (-not $user) { $user = '' }
        if (($user -split ';') -notcontains $bin) {
            [Environment]::SetEnvironmentVariable('Path', ($user.TrimEnd(';') + ';' + $bin).TrimStart(';'), 'User')
        }
        $env:Path = "$env:Path;$bin"
    }
    Write-Host 'Installed. Run: strata launch'
    if ($env:STRATA_NO_PATH -eq '1') { Write-Host "  (the program is $bin\strata.cmd)" }
    else { Write-Host '  (open a new terminal window if the command is not found yet)' }
}
finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
}
