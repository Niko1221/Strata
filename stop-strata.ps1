<#
  stop-strata.ps1 - stop Strata, and only Strata.

  The launcher this replaces matched python.exe by name:

      for /f "tokens=2" %%a in ('tasklist ^| findstr "python.exe"') do taskkill /f /pid %%a

  which ends every Python process on the PC - a Jupyter kernel, a build script, another model server, anything.  This
  script matches on what a process *is* instead of what it is called:

    1. a launcher   - a python.exe whose command line runs serve\server.py (the run-<model>.bat setup writes)
    2. an engine    - strata.exe or strata-vision.exe whose executable sits inside this install
    3. nothing else - a process is touched only when one of those two rules names it

  The install is the folder this script is in (STOP.bat sits beside it); -Root points it elsewhere.  A launcher is
  stopped first, so a supervisor cannot restart the engine behind us.  Each process gets a plain taskkill (WM_CLOSE)
  first and only then /F, so the engine's own shutdown runs when it can.

    STOP.bat                 stop it
    STOP.bat -WhatIf         list what would be stopped, stop nothing
    STOP.bat -Root D:\Strata stop the install somewhere else
    STOP.bat -TimeoutSeconds 30   wait longer for a graceful exit (default 10)

  Exit code: 0 when nothing was running or everything stopped, 1 when a process survived.
#>
[CmdletBinding()]
param(
    [string]$Root,
    [switch]$WhatIf,
    [int]$TimeoutSeconds = 10
)

$ErrorActionPreference = 'Stop'

function Say($text, $color = 'Gray') { Write-Host $text -ForegroundColor $color }

if (-not $Root) { $Root = $PSScriptRoot }
if (-not $Root) {
    Say '[错误] 无法确定 Strata 安装目录，请用 -Root 指定。' 'Red'
    exit 1
}
if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
    Say ('[错误] 目录不存在: ' + $Root) 'Red'
    exit 1
}
$rootFull = (Resolve-Path -LiteralPath $Root).Path.TrimEnd('\') + '\'
$rootLow = $rootFull.ToLowerInvariant()
# the server script this install runs, in the two forms a command line can spell it
$scriptAbs = @(($rootLow + 'serve\server.py'), ($rootLow + 'serve/server.py'))

Say ''
Say '==========================================================' 'Cyan'
Say '  Strata 停止工具（只终止 Strata 自己的进程）' 'Cyan'
Say ('  安装目录: ' + $rootFull) 'Cyan'
Say '==========================================================' 'Cyan'
Say ''

# ---- 找出属于 Strata 的进程 ------------------------------------------------------------------------------------
$targets = New-Object System.Collections.ArrayList
$skipped = New-Object System.Collections.ArrayList

$procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
         Select-Object ProcessId, ParentProcessId, Name, ExecutablePath, CommandLine

# this script's own process and the shells that started it: never a target, whatever their command line says
$selfIds = New-Object System.Collections.Generic.HashSet[int]
$byId = @{}
foreach ($p in $procs) { $byId[[int]$p.ProcessId] = $p }
$cur = $PID
for ($i = 0; $i -lt 8 -and $byId.ContainsKey($cur); $i++) {
    [void]$selfIds.Add($cur)
    $cur = [int]$byId[$cur].ParentProcessId
    if ($cur -le 0) { break }
}

foreach ($p in $procs) {
    $pid_ = [int]$p.ProcessId
    if ($selfIds.Contains($pid_)) { continue }
    $name = if ($p.Name) { $p.Name.ToLowerInvariant() } else { '' }
    $exe  = $p.ExecutablePath
    $cmdLow = if ($p.CommandLine) { $p.CommandLine.ToLowerInvariant() } else { '' }
    $exeUnderRoot = ($exe -and $exe.ToLowerInvariant().StartsWith($rootLow))

    # 规则 1: 启动器 —— 一个跑 serve\server.py 的 python，且那个脚本就是本安装的脚本。
    # 两个条件必须同时成立：命令行里出现 serve\server.py，并且它要么由本安装里的 python 跑，要么写的就是本安装的
    # 绝对路径。少了后半句，任何命令行里恰好提到这个路径的 python 都会被当成启动器。
    $isLauncher = $false
    if ($name -match '^python(w)?\.exe$' -and $cmdLow -match 'serve[\\/]server\.py') {
        $absScript = ($cmdLow.Contains($scriptAbs[0]) -or $cmdLow.Contains($scriptAbs[1]))
        if ($exeUnderRoot -or $absScript) { $isLauncher = $true }
    }

    # 规则 2: 引擎 —— 名字是 strata.exe / strata-vision.exe，且可执行文件就在本安装目录里
    $isEngine = $false
    if ($name -eq 'strata.exe' -or $name -eq 'strata-vision.exe') {
        if (-not $exe) {
            [void]$skipped.Add([pscustomobject]@{ Pid = $pid_; Name = $p.Name; Why = '读不到可执行文件路径（权限不足），不猜' })
        } elseif ($exeUnderRoot) {
            $isEngine = $true
        }
    }

    if ($isLauncher -or $isEngine) {
        [void]$targets.Add([pscustomobject]@{
            Pid     = $pid_
            Name    = $p.Name
            Exe     = $exe
            Role    = if ($isLauncher) { '启动器' } else { '引擎' }
            Order   = if ($isLauncher) { 0 } else { 1 }      # 先停启动器，免得它把引擎再拉起来
        })
    }
}

# ---- 没有目标：友好退出，不是错误 ------------------------------------------------------------------------------
if ($targets.Count -eq 0) {
    Say '没有找到正在运行的 Strata（可能已经停止）。' 'Green'
    if ($skipped.Count -gt 0) {
        Say ''
        Say '以下进程被跳过:' 'Yellow'
        foreach ($s in $skipped) { Say ('  pid=' + $s.Pid + '  ' + $s.Name + '  - ' + $s.Why) 'Yellow' }
    }
    Say ''
    Say '其他程序未受影响。' 'Green'
    exit 0
}

# ---- 先列出来 --------------------------------------------------------------------------------------------------
$targets = $targets | Sort-Object Order, Pid
Say '找到以下 Strata 进程:' 'Yellow'
foreach ($t in $targets) {
    Say ('  [' + $t.Role + '] pid=' + $t.Pid + '  ' + $t.Name)
    if ($t.Exe) { Say ('           ' + $t.Exe) 'DarkGray' }
}
Say ''

if ($skipped.Count -gt 0) {
    Say '以下进程被跳过（不属于本安装，或读不到路径）:' 'Yellow'
    foreach ($s in $skipped) { Say ('  pid=' + $s.Pid + '  ' + $s.Name + '  - ' + $s.Why) 'Yellow' }
    Say ''
}

if ($WhatIf) {
    Say '-WhatIf: 只列出，没有终止任何进程。' 'Cyan'
    exit 0
}

# ---- 终止 ------------------------------------------------------------------------------------------------------
$failed = New-Object System.Collections.ArrayList

foreach ($t in $targets) {
    $alive = Get-Process -Id $t.Pid -ErrorAction SilentlyContinue
    if (-not $alive) {
        Say ('  pid=' + $t.Pid + ' 已经退出') 'DarkGray'
        continue
    }

    Say ('正在停止 ' + $t.Name + ' (pid=' + $t.Pid + ')...') 'Yellow'
    & taskkill.exe /PID $t.Pid 2>&1 | Out-Null        # 先礼后兵：WM_CLOSE，引擎能自己收尾
    $waited = 0
    while ($waited -lt $TimeoutSeconds -and (Get-Process -Id $t.Pid -ErrorAction SilentlyContinue)) {
        Start-Sleep -Milliseconds 500
        $waited += 0.5
    }
    if (Get-Process -Id $t.Pid -ErrorAction SilentlyContinue) {
        Say ('  未在 ' + $TimeoutSeconds + ' 秒内退出，强制终止') 'Yellow'
        & taskkill.exe /PID $t.Pid /F /T 2>&1 | Out-Null
        Start-Sleep -Milliseconds 500
    }
    if (Get-Process -Id $t.Pid -ErrorAction SilentlyContinue) {
        Say ('  [失败] pid=' + $t.Pid + ' 仍在运行（可能权限不足，请用管理员身份运行）') 'Red'
        [void]$failed.Add($t)
    } else {
        Say ('  [完成] pid=' + $t.Pid) 'Green'
    }
}

Say ''
if ($failed.Count -gt 0) {
    Say ('有 ' + $failed.Count + ' 个进程未能停止。') 'Red'
    exit 1
}
Say 'Strata 已停止。其他程序未受影响。' 'Green'
exit 0
