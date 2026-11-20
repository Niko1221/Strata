param([string]$AuditDirectory='build/gdn-resource-qualified', [switch]$BuildOnly)
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$dir=(Resolve-Path $AuditDirectory).Path
$nvcc=(Get-Command nvcc.exe).Source
$lib='H:\Strata\releases\hakua040-native-upstream-r1\Release\strata_kernels.lib'
if (-not (Test-Path $lib)) { throw 'Recorded reference kernel library is unavailable.' }
if (Test-Path (Join-Path $dir 'runtime-binding.json')) { throw 'Runtime evidence already exists; choose a fresh audit directory.' }
$vswhere=Join-Path ([Environment]::GetEnvironmentVariable('ProgramFiles(x86)')) 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs=& $vswhere -latest -version '[17,18)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) {throw 'Visual Studio 2022 C++ tools required'}
$vcvars=Join-Path $vs 'VC/Auxiliary/Build/vcvars64.bat'
$utf8=[Text.UTF8Encoding]::new($false)
$variants=@('verify_kernels','verify_kernels_ungrouped_q')
$lines=@('@echo off',('call "'+$vcvars+'" >nul'),'if errorlevel 1 exit /b %errorlevel%')
foreach ($test in @(@{name='parity';source='src/kernels/gdn_parity.cpp'},@{name='bench';source='bench/gdn_multi_resource_bench.cpp'})) {
    $obj=Join-Path $dir ($test.name+'.obj')
    $lines+=('"'+$nvcc+'" -std=c++20 -O2 -arch=sm_120 -DNDEBUG -DSTRATA_VERIFY_MAX_T=8 -Xcompiler /EHsc,/utf-8,/MT,/Ob2 -I "'+(Join-Path $repo 'include')+'" -c "'+(Join-Path $repo $test.source)+'" -o "'+$obj+'"')
    $lines+='if errorlevel 1 exit /b %errorlevel%'
    foreach ($variant in $variants) {
        if (-not (Test-Path (Join-Path $dir ($variant+'.obj')))) {throw 'Run the compiler audit with -CompareUngroupedQuantized first.'}
        $lines+=('"'+$nvcc+'" "'+$obj+'" "'+(Join-Path $dir ($variant+'.obj'))+'" "'+(Join-Path $dir 'fused_gdn.obj')+'" "'+$lib+'" -o "'+(Join-Path $dir ($variant+'_'+$test.name+'.exe'))+'"')
        $lines+='if errorlevel 1 exit /b %errorlevel%'
    }
}
$cmdfile=Join-Path $dir 'build-runtime.cmd'
[IO.File]::WriteAllLines($cmdfile,$lines,$utf8)
$inputs=@($lib,(Join-Path $repo 'src/kernels/gdn_parity.cpp'),(Join-Path $repo 'bench/gdn_multi_resource_bench.cpp'),$PSCommandPath) | ForEach-Object {@{path=$_;sha256=(Get-FileHash -LiteralPath $_).Hash}}
$binding=@{utc=[DateTime]::UtcNow.ToString('o'); inputs=@($inputs); scope='Current GDN parity/benchmark and verify/fused objects; other kernels resolved from recorded existing library. Synthetic; not an engine build or E2E qualification.'; environment='All inherited STRATA_* variables removed in children; no settings or external processes changed'; performance_qualified=$false; reason='Shared GPU, unfixed clocks, synthetic inputs and cache reuse; no E2E or DRAM counters'}
[IO.File]::WriteAllText((Join-Path $dir 'runtime-binding.json'),($binding|ConvertTo-Json -Depth 6),$utf8)
& $env:ComSpec /d /c $cmdfile 2>&1 | Out-File (Join-Path $dir 'build-runtime.log') -Encoding utf8
if ($LASTEXITCODE -ne 0) {throw 'Runtime build failed; inspect build-runtime.log'}
function Run-Owned([string]$Exe,[string[]]$Arguments,[string]$Log) {
    $psi=[Diagnostics.ProcessStartInfo]::new($Exe)
    $psi.UseShellExecute=$false; $psi.CreateNoWindow=$true
    $psi.RedirectStandardOutput=$true; $psi.RedirectStandardError=$true
    foreach($key in @($psi.Environment.Keys)) {if($key -like 'STRATA_*') {$psi.Environment.Remove($key)|Out-Null}}
    foreach($arg in $Arguments) {$psi.ArgumentList.Add($arg)}
    $process=[Diagnostics.Process]::new(); $process.StartInfo=$psi
    if(-not $process.Start()) {throw 'Cannot start owned test'}
    $stdout=$process.StandardOutput.ReadToEndAsync(); $stderr=$process.StandardError.ReadToEndAsync()
    if(-not $process.WaitForExit(120000)) {$process.Kill($true); $process.WaitForExit(); throw 'Owned test exceeded 120 seconds'}
    [IO.File]::WriteAllText($Log,$stdout.GetAwaiter().GetResult(),$utf8)
    [IO.File]::WriteAllText(($Log+'.stderr'),$stderr.GetAwaiter().GetResult(),$utf8)
    if($process.ExitCode -ne 0) {throw ('Owned test failed: '+$Log+' exit '+$process.ExitCode)}
    $process.Dispose()
}
if(-not $BuildOnly) {
    foreach($variant in $variants) {
        Run-Owned (Join-Path $dir ($variant+'_parity.exe')) @('--selftest') (Join-Path $dir ($variant+'.parity.log'))
        Write-Output ($variant+' parity passed')
    }
    for($trial=0;$trial -lt 4;++$trial) {
        $order=if($trial%2 -eq 0){$variants}else{@($variants[1],$variants[0])}
        foreach($variant in $order) {
            $name=$variant+'-r'+$trial
            & nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu,clocks.sm,clocks.mem --format=csv | Out-File (Join-Path $dir ($name+'.gpu-before.csv')) -Encoding utf8
            Run-Owned (Join-Path $dir ($variant+'_bench.exe')) @() (Join-Path $dir ($name+'.jsonl'))
            & nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu,clocks.sm,clocks.mem --format=csv | Out-File (Join-Path $dir ($name+'.gpu-after.csv')) -Encoding utf8
        }
        Write-Output ('Timing pair '+$trial+' completed (not performance-qualified)')
    }
}