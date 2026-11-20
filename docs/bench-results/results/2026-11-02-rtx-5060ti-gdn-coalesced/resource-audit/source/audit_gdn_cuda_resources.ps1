param(
    [string]$OutputDirectory = '',
    [switch]$IncludeNativeMmvq,
    [switch]$CompareTwoBlockBound,
    [switch]$CompareUngroupedQuantized
)
# CPU-only compiler/resource inspection. Does not load a model or change production sources.
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repo 'build/gdn-resource-audit' }
$out = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path (Join-Path $out 'binding.json')) { throw 'Choose a new output directory; an audit already exists here.' }
New-Item -ItemType Directory -Path $out -Force | Out-Null
$nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
$cuobjdump = Join-Path (Split-Path $nvcc) 'cuobjdump.exe'
$vswhere = Join-Path ([Environment]::GetEnvironmentVariable('ProgramFiles(x86)')) 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -version '[17,18)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio 2022 C++ tools required' }
$vcvars = Join-Path $vs 'VC/Auxiliary/Build/vcvars64.bat'
$utf8 = [Text.UTF8Encoding]::new($false)
$units = @(
    @{name='fused_gdn'; source=(Join-Path $repo 'src/kernels/cuda/fused_gdn.cu'); fast=$false},
    @{name='verify_kernels'; source=(Join-Path $repo 'src/kernels/cuda/verify_kernels.cu'); fast=$false}
)
if ($IncludeNativeMmvq) {
    $units += @{name='native_mmvq'; source=(Join-Path $repo 'src/kernels/cuda/native_mmvq.cu'); fast=$true}
}
if ($CompareTwoBlockBound) {
    $original = [IO.File]::ReadAllText($units[1].source)
    $needle = '__launch_bounds__(S * RG) gdn_step_norm_multi_kernel'
    if (($original.Split(@($needle), [StringSplitOptions]::None)).Count -ne 2) {
        throw 'Expected exactly one multi-token GDN launch bound; source changed.'
    }
    $scratch = Join-Path $out 'verify_kernels_two_blocks.cu'
    [IO.File]::WriteAllText($scratch, $original.Replace($needle, '__launch_bounds__(S * RG, 2) gdn_step_norm_multi_kernel'), $utf8)
    $units += @{name='verify_kernels_two_blocks'; source=$scratch; fast=$false}
}
if ($CompareUngroupedQuantized) {
    $original = [IO.File]::ReadAllText($units[1].source)
    $from = @('template <bool ALL_OUT, bool Q>', 'if (grp.n > 0) {   // group blockIdx.y', 'else gdn_step_norm_multi_kernel<false, true><<<g1, b1, 0, (cudaStream_t) stream>>>(')
    $to = @('template <bool ALL_OUT, bool Q, bool UNGROUPED = false>', 'if (!UNGROUPED && grp.n > 0) {   // group blockIdx.y', 'else gdn_step_norm_multi_kernel<false, true, true><<<g1, b1, 0, (cudaStream_t) stream>>>(')
    for ($i=0; $i -lt $from.Count; ++$i) {
        if (($original.Split(@($from[$i]), [StringSplitOptions]::None)).Count -ne 2) { throw 'Ungrouped specialization source changed.' }
        $original = $original.Replace($from[$i], $to[$i])
    }
    $scratch = Join-Path $out 'verify_kernels_ungrouped_q.cu'
    [IO.File]::WriteAllText($scratch, $original, $utf8)
    $units += @{name='verify_kernels_ungrouped_q'; source=$scratch; fast=$false}
}
# Same device target, C++ standard and per-file fast-math split as Release CMake.
# -Xptxas=-v adds diagnostics; no -maxrregcount and no global --use_fast_math.
$common = '"'+$nvcc+'" -std=c++20 -O2 -arch=sm_120 -Xptxas=-v -DNDEBUG -D_WINDOWS -DSTRATA_VERIFY_MAX_T=8 -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_HAVE_DFLASH2=1 -Xcompiler /EHsc,/utf-8,/MT,/Ob2 -I "'+(Join-Path $repo 'include')+'" -I "'+(Join-Path $repo 'third_party/ggml')+'" -I "'+(Join-Path $repo 'src/kernels/cuda')+'"'
$inputs = @(Get-ChildItem (Join-Path $repo 'include'), (Join-Path $repo 'src/kernels/cuda'), (Join-Path $repo 'third_party/ggml') -File -Recurse | Where-Object { $_.Extension -in '.hpp','.h','.cuh','.cu' } | Sort-Object FullName | ForEach-Object {
    @{path=$_.FullName.Substring($repo.Length+1); sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$binding = @{schema=1; utc=[DateTime]::UtcNow.ToString('o'); source_root=$repo; head=(& git -C $repo rev-parse HEAD); nvcc=$nvcc; nvcc_version=(& $nvcc --version | Out-String); vcvars=$vcvars; common_flags=$common; units=$units; inputs=$inputs; script_sha256=(Get-FileHash -LiteralPath $PSCommandPath).Hash; scope='Standalone sm_120 compiler audit; not a full engine build or runtime benchmark'}
[IO.File]::WriteAllText((Join-Path $out 'binding.json'), ($binding | ConvertTo-Json -Depth 8), $utf8)
foreach ($unit in $units) {
    $obj = Join-Path $out ($unit.name+'.obj')
    $flags = if ($unit.fast) { ' --use_fast_math' } else { '' }
    $command = $common+$flags+' -c "'+$unit.source+'" -o "'+$obj+'"'
    $cmdfile = Join-Path $out ($unit.name+'.cmd')
    [IO.File]::WriteAllLines($cmdfile, @('@echo off', ('call "'+$vcvars+'" >nul'), 'if errorlevel 1 exit /b %errorlevel%', $command, 'exit /b %errorlevel%'), $utf8)
    & $env:ComSpec /d /c $cmdfile 2>&1 | Out-File -LiteralPath (Join-Path $out ($unit.name+'.ptxas.log')) -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw ('Compilation failed for '+$unit.name+'; inspect its ptxas log.') }
    & $cuobjdump --dump-resource-usage $obj | Out-File -LiteralPath (Join-Path $out ($unit.name+'.resources.txt')) -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw ('cuobjdump failed for '+$unit.name) }
    & uv run --no-project --python 3.12 python (Join-Path $repo 'tools/check_cuda_spills.py') --list --cuobjdump $cuobjdump $obj | Out-File -LiteralPath (Join-Path $out ($unit.name+'.resources.csv')) -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw ('Resource table failed for '+$unit.name) }
    Write-Output ('Compiled and inspected '+$unit.name)
}
$receipt = @{completed_utc=[DateTime]::UtcNow.ToString('o'); artifacts=@(Get-ChildItem -LiteralPath $out -File | Sort-Object Name | ForEach-Object { @{file=$_.Name; bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName).Hash.ToLowerInvariant()} })}
[IO.File]::WriteAllText((Join-Path $out 'receipt.json'), ($receipt | ConvertTo-Json -Depth 6), $utf8)