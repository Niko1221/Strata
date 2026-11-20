# Compile and run the actual installer cleanup macros in fresh, isolated Windows fixtures.
# No GPU, model download or real app installation is used. Keep OutputDir for audit.
param(
 [Parameter(Mandatory=$true)][string]$OutputDir,
 [string]$NsisCompiler=(Join-Path $env:LOCALAPPDATA 'tauri\NSIS\Bin\makensis.exe')
)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath($OutputDir)
if(Test-Path -LiteralPath $root){throw 'Fresh immutable fixture directory required'}
if(-not (Test-Path -LiteralPath $NsisCompiler -PathType Leaf)){throw 'Pass -NsisCompiler with the installed makensis.exe path'}
New-Item -ItemType Directory -Path $root|Out-Null
$source=Join-Path (Split-Path -Parent $PSScriptRoot) 'app\desktop\src-tauri\nsis\installer.nsi'
$sourceHash=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath (Join-Path $root 'source-sha256.txt') -Value $sourceHash
$text=Get-Content -LiteralPath $source -Raw
$match=[regex]::Match($text,'(?s); STRATA-CLEAN-BEGIN(.*?); STRATA-CLEAN-END')
if(-not $match.Success){throw 'Actual production cleanup macros not found'}
$macros=$match.Groups[1].Value
$names=@('control','host','node','harness','engine-bundle','local-backends','notices')
$results=@();$oldTemp=$env:TEMP;$oldTmp=$env:TMP
try{
 foreach($scenario in @('upgrade','fresh','empty','locked','junction','root-junction')){
  $case=Join-Path $root $scenario;$app=Join-Path $case 'app';$data=Join-Path $case 'user-data';$models=Join-Path $case 'models';$temp=Join-Path $case 'temp'
  foreach($p in @($case,$app,$data,$models,$temp)){New-Item -ItemType Directory -Path $p|Out-Null}
  foreach($p in @($app,$data,$models)){if(-not ([IO.Path]::GetFullPath($p).StartsWith([IO.Path]::GetFullPath($root)+'\',[StringComparison]::OrdinalIgnoreCase))){throw 'Fixture path escaped its owned root'}}
  if($scenario -eq 'root-junction'){[IO.Directory]::Delete($app);New-Item -ItemType Junction -Path $app -Target $data|Out-Null}
  [IO.File]::WriteAllText((Join-Path $data 'memory.db'),'private-memory-sentinel')
  [IO.File]::WriteAllText((Join-Path $models 'model.gguf'),'external-model-sentinel')
  [IO.File]::WriteAllText((Join-Path $app 'personal-note.txt'),'unrelated-root-file')
  $newEngine=Join-Path $case 'new-engine.txt';$newGUI=Join-Path $case 'new-gui.txt'
  [IO.File]::WriteAllText($newEngine,'fresh-runtime-0.4.0');[IO.File]::WriteAllText($newGUI,'fresh-gui-0.4.0')
  if($scenario -eq 'empty'){foreach($name in $names){New-Item -ItemType Directory -Path (Join-Path $app $name)|Out-Null}}
  if($scenario -notin @('fresh','empty')){
   foreach($name in $names){$p=Join-Path $app $name;New-Item -ItemType Directory -Path $p|Out-Null;[IO.File]::WriteAllText((Join-Path $p 'legacy.txt'),'obsolete-managed-file')}
   $legacy=Join-Path $app 'engine-bundle\0.1.47';New-Item -ItemType Directory -Path $legacy|Out-Null;[IO.File]::WriteAllText((Join-Path $legacy 'strata.exe'),'old-runtime')
  }
  if($scenario -eq 'junction'){New-Item -ItemType Junction -Path (Join-Path $app 'engine-bundle\old-link') -Target $data|Out-Null}
  $exe=Join-Path $case 'fixture-setup.exe';$nsi=Join-Path $case 'fixture.nsi'
  $nsiExe=$exe.Replace('$','$$');$nsiApp=$app.Replace('$','$$');$nsiEngine=$newEngine.Replace('$','$$');$nsiGUI=$newGUI.Replace('$','$$')
  @"
Unicode true
Name "Strata managed-runtime test fixture"
OutFile "$nsiExe"
InstallDir "$nsiApp"
RequestExecutionLevel user
SilentInstall silent
AutoCloseWindow true
!include "LogicLib.nsh"
!include "FileFunc.nsh"
$macros
Section
  !insertmacro StrataRemoveAppFolders "`$INSTDIR" ""
  SetOutPath "`$INSTDIR\engine-bundle\0.4.0"
  File /oname=strata.exe "$nsiEngine"
  SetOutPath "`$INSTDIR\host\dist"
  File /oname=index.html "$nsiGUI"
  WriteUninstaller "`$INSTDIR\fixture-uninstall.exe"
SectionEnd
Section Uninstall
  !insertmacro StrataRemoveAppFolders "`$INSTDIR" "un."
SectionEnd
"@|Set-Content -LiteralPath $nsi -Encoding utf8
  $env:TEMP=$temp;$env:TMP=$temp
  $compiler=Start-Process -FilePath $NsisCompiler -ArgumentList @('/V2',"`"$nsi`"") -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput (Join-Path $case 'compile.log') -RedirectStandardError (Join-Path $case 'compile-error.log')
  if($compiler.ExitCode -ne 0){throw "Fixture compilation failed: $scenario (exit $($compiler.ExitCode)); see compile.log and compile-error.log"}
  $lock=$null
  try{
   if($scenario -eq 'locked'){$lock=[IO.File]::Open((Join-Path $app 'engine-bundle\0.1.47\strata.exe'),[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::None)}
   $process=Start-Process -FilePath $exe -ArgumentList '/S' -WindowStyle Hidden -Wait -PassThru
  }finally{if($lock){$lock.Dispose()}}
  $oldExists=Test-Path -LiteralPath (Join-Path $app 'engine-bundle\0.1.47\strata.exe')
  $newExists=Test-Path -LiteralPath (Join-Path $app 'engine-bundle\0.4.0\strata.exe')
  $sentinels=(Test-Path -LiteralPath (Join-Path $data 'memory.db')) -and (Test-Path -LiteralPath (Join-Path $models 'model.gguf')) -and (Test-Path -LiteralPath (Join-Path $app 'personal-note.txt'))
  if($sentinels){$sentinels=([IO.File]::ReadAllText((Join-Path $data 'memory.db')) -eq 'private-memory-sentinel') -and ([IO.File]::ReadAllText((Join-Path $models 'model.gguf')) -eq 'external-model-sentinel') -and ([IO.File]::ReadAllText((Join-Path $app 'personal-note.txt')) -eq 'unrelated-root-file')}
  $legacyFiles=@($names|Where-Object {Test-Path -LiteralPath (Join-Path $app ($_+'\legacy.txt'))})
  if($scenario -in @('locked','junction','root-junction')){$ok=$process.ExitCode -ne 0 -and $oldExists -and -not $newExists -and $sentinels}else{$ok=$process.ExitCode -eq 0 -and -not $oldExists -and $newExists -and $sentinels -and $legacyFiles.Count -eq 0}
  $uninstallCode=$null;$uninstallPreserved=$null
  if($ok -and $scenario -notin @('locked','junction','root-junction')){
   $un=Start-Process -FilePath (Join-Path $app 'fixture-uninstall.exe') -ArgumentList @('/S',"_?=$app") -WindowStyle Hidden -Wait -PassThru
   $uninstallCode=$un.ExitCode
   $uninstallPreserved=(Test-Path -LiteralPath (Join-Path $data 'memory.db')) -and (Test-Path -LiteralPath (Join-Path $models 'model.gguf')) -and (Test-Path -LiteralPath (Join-Path $app 'personal-note.txt'))
   if($uninstallPreserved){$uninstallPreserved=([IO.File]::ReadAllText((Join-Path $data 'memory.db')) -eq 'private-memory-sentinel') -and ([IO.File]::ReadAllText((Join-Path $models 'model.gguf')) -eq 'external-model-sentinel') -and ([IO.File]::ReadAllText((Join-Path $app 'personal-note.txt')) -eq 'unrelated-root-file')}
   $remaining=@($names|Where-Object {Test-Path -LiteralPath (Join-Path $app $_)})
   $ok=$ok -and $uninstallCode -eq 0 -and $uninstallPreserved -and $remaining.Count -eq 0
  }
  $results+=@{scenario=$scenario;ok=$ok;exit=$process.ExitCode;oldEngineRemains=$oldExists;newEngineCopied=$newExists;userSentinelsPreserved=$sentinels;obsoleteManagedFolders=$legacyFiles;uninstallExit=$uninstallCode;uninstallPreservedUserData=$uninstallPreserved}
 }
}finally{$env:TEMP=$oldTemp;$env:TMP=$oldTmp;$results|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $root 'results.json') -Encoding utf8}
$results|ConvertTo-Json -Depth 8
if((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceHash){throw 'Production installer source changed during the test'}
if(@($results|Where-Object {-not $_.ok}).Count){exit 1}
