param([string]$OutputDirectory = (Join-Path $PSScriptRoot '..\..\dist\windows-tray'), [switch]$Test)
$ErrorActionPreference = 'Stop'
$out = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $out -Force | Out-Null
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (!(Test-Path $compiler)) { throw 'Windows with .NET Framework 4.8 is required.' }
$name = if ($Test) { 'TrayTests.exe' } else { 'StrataTray.exe' }
$arguments = @('/nologo','/platform:x64',('/out:' + (Join-Path $out $name)),
    ('/win32manifest:' + (Join-Path $PSScriptRoot 'app.manifest')),
    ('/win32icon:' + (Join-Path $PSScriptRoot 'Strata.ico')),
    ('/resource:' + (Join-Path $PSScriptRoot 'Strata.ico') + ',Strata.ico'),
    '/reference:System.dll,System.Core.dll,System.Drawing.dll,System.Windows.Forms.dll,System.Web.Extensions.dll,System.Net.Http.dll',
    (Join-Path $PSScriptRoot 'Program.cs'),(Join-Path $PSScriptRoot 'AssemblyInfo.cs'))
if ($Test) { $arguments = @('/target:exe','/main:TrayTests') + $arguments + @((Join-Path $PSScriptRoot 'TrayTests.cs')) }
else { $arguments = @('/target:winexe') + $arguments }
& $compiler @arguments
if ($LASTEXITCODE -ne 0) { throw 'Tray compilation failed.' }
Copy-Item (Join-Path $PSScriptRoot 'App.config') (Join-Path $out ($name + '.config'))
Write-Output (Join-Path $out $name)
