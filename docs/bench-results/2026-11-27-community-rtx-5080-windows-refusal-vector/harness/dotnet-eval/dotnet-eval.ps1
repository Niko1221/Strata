# dotnet-eval.ps1 - .NET eval pre lokalny model (Swift 1.5 vs base Qwen3.8-Flash-Next).
#
# Preco: verejne benchmarky nemaju C#/.NET. Toto meria presne to, co robis:
# model dostane spec + sucasne subory, vrati cele subory, harness ich zapise,
# spusti `dotnet build` a skryte testy. Skore = zeleny build + vsetky testy.
#
# Bez NuGet: vsetky projekty stoja len na shared frameworku (SDK), takze eval
# funguje offline.
#
# Pouzitie:
#   .\dotnet-eval.ps1 -SelfTest              # overi, ze referencia prejde a sablona nie (bez engine)
#   .\dotnet-eval.ps1 -NoRun                 # vypise ulohy
#   .\dotnet-eval.ps1 -RunId base -Reps 1    # zmeria bezacy model
#
# Vystup: dotnet-eval-results\<RunId>.json
param(
    [string]$RunId = 'base',
    [string]$Model = '',
    [int]$Reps = 1,
    [string]$Only = '',
    [switch]$SelfTest,
    [switch]$NoRun
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:DOTNET_CLI_TELEMETRY_OPTOUT = '1'
$env:DOTNET_NOLOGO = '1'
$env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = '1'
$env:PYTHONUTF8 = '1'

$baseUri = 'http://127.0.0.1:8080'
$uriChat = "$baseUri/v1/chat/completions"
# jeden zdroj pravdy pre parsovanie engine logu; overuje ho -SelfTest
$script:logPromptPattern = 'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)'
$tasksDir = Join-Path $PSScriptRoot 'tasks'
$outDir = Join-Path $PSScriptRoot 'dotnet-eval-results'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$workRoot = Join-Path $env:TEMP ('dotnet-eval-' + $RunId)

# ------------------------------------------------------------------ HTTP
function Post-Json([string]$url, [string]$jsonBody, [int]$timeoutSec = 900) {
    $req = [System.Net.HttpWebRequest]::Create($url)
    $req.Method = 'POST'
    $req.ContentType = 'application/json; charset=utf-8'
    $req.Timeout = $timeoutSec * 1000
    $req.ReadWriteTimeout = $timeoutSec * 1000
    $bytes = [Text.Encoding]::UTF8.GetBytes($jsonBody)
    $req.ContentLength = $bytes.Length
    $rs = $req.GetRequestStream(); $rs.Write($bytes, 0, $bytes.Length); $rs.Close()
    $resp = $req.GetResponse()
    $ms = New-Object System.IO.MemoryStream
    $src = $resp.GetResponseStream(); $src.CopyTo($ms); $src.Close(); $resp.Close()
    return ([Text.Encoding]::UTF8.GetString($ms.ToArray()) | ConvertFrom-Json)
}

# ------------------------------------------------------------------ procesy
function Invoke-Proc([string]$exe, [string[]]$argList, [string]$workDir, [int]$timeoutSec) {
    $psi = New-Object Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = ($argList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    $psi.WorkingDirectory = $workDir
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.StandardOutputEncoding = [Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [Text.Encoding]::UTF8
    $p = [Diagnostics.Process]::Start($psi)
    $outTask = $p.StandardOutput.ReadToEndAsync()
    $errTask = $p.StandardError.ReadToEndAsync()
    $timedOut = $false
    if (-not $p.WaitForExit($timeoutSec * 1000)) {
        $timedOut = $true
        try { $p.Kill($true) } catch { }
        $p.WaitForExit(5000) | Out-Null
    }
    $out = ''
    $err = ''
    try { $out = $outTask.Result } catch { }
    try { $err = $errTask.Result } catch { }
    $code = $null
    if (-not $timedOut) { $code = $p.ExitCode }
    return [pscustomobject]@{ code = $code; out = $out; err = $err; timedOut = $timedOut }
}

# ------------------------------------------------------------------ generovanie projektov
function Write-ProjectFiles([string]$workDir, [switch]$Exe) {
    $props = @'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <LangVersion>latest</LangVersion>
    <TreatWarningsAsErrors>false</TreatWarningsAsErrors>
  </PropertyGroup>
  <ItemGroup>
    <FrameworkReference Include="Microsoft.AspNetCore.App" />
  </ItemGroup>
</Project>
'@
    if (-not $Exe) { $props = $props -replace '(?m)^\s*<OutputType>Exe</OutputType>\r?\n', '' }
    $appDir = Join-Path $workDir 'app'
    $testDir = Join-Path $workDir 'tests'
    New-Item -ItemType Directory -Force -Path $appDir | Out-Null
    New-Item -ItemType Directory -Force -Path $testDir | Out-Null
    Set-Content -Path (Join-Path $appDir 'app.csproj') -Value $props -Encoding UTF8
    $testProps = @'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <LangVersion>latest</LangVersion>
  </PropertyGroup>
  <ItemGroup>
    <FrameworkReference Include="Microsoft.AspNetCore.App" />
    <ProjectReference Include="..\app\app.csproj" />
  </ItemGroup>
</Project>
'@
    Set-Content -Path (Join-Path $testDir 'tests.csproj') -Value $testProps -Encoding UTF8
}

# ------------------------------------------------------------------ parsovanie odpovede
function Get-FileBlocks([string]$text) {
    # Format: "=== FILE: app/X.cs ===" na vlastnom riadku, potom obsah.
    $res = @()
    $pattern = '(?m)^===\s*FILE:\s*(.+?)\s*===\s*$'
    $m = [regex]::Matches($text, $pattern)
    for ($i = 0; $i -lt $m.Count; $i++) {
        $start = $m[$i].Index + $m[$i].Length
        $end = if ($i + 1 -lt $m.Count) { $m[$i + 1].Index } else { $text.Length }
        $body = $text.Substring($start, $end - $start)
        $body = $body -replace '^\r?\n', ''
        $body = $body -replace '```[a-zA-Z]*\s*$', ''
        $res += [pscustomobject]@{ path = $m[$i].Groups[1].Value.Trim(); body = $body.TrimEnd() }
    }
    return $res
}
function Test-SafePath([string]$p) {
    if ($p -match '^[a-zA-Z]:' -or $p.StartsWith('/') -or $p -match '\.\.') { return $false }
    return $true
}

# ------------------------------------------------------------------ skore (build + testy)
function Invoke-Score($task, [string]$workDir) {
    $out = @{ build_ok = $false; passed = 0; total = 0; green = $false; build_error = ''; test_output = ''; timed_out = $false }
    $buildArgs = @('build', 'app\app.csproj', '-nologo', '-v', 'q')
    if ($task.buildArgs) { $buildArgs += @($task.buildArgs) }
    $b = Invoke-Proc 'dotnet' $buildArgs $workDir 180
    $out.build_ok = ($b.code -eq 0)
    if (-not $out.build_ok) {
        $out.build_error = ((($b.out + "`n" + $b.err) -split "`r?`n" | Where-Object { $_ -match 'error' } | Select-Object -First 3) -join ' | ')
    }
    if ($out.build_ok) {
        $runArgs = @('run', '--project', 'tests\tests.csproj')
        if ($task.exe) { $runArgs += @('--', (Join-Path $workDir 'app')) }
        $t = Invoke-Proc 'dotnet' $runArgs $workDir ([int]$task.timeoutSec)
        $outText = $t.out + "`n" + $t.err
        if ($outText -match 'RESULT:\s*(\d+)\s*/\s*(\d+)') {
            $out.passed = [int]$Matches[1]
            $out.total = [int]$Matches[2]
        }
        $out.timed_out = [bool]$t.timedOut
        $out.test_output = (($outText -split "`r?`n" | Where-Object { $_ -match 'FAIL|RESULT|Exception' } | Select-Object -First 6) -join ' | ')
    }
    $out.green = ($out.build_ok -and $out.total -gt 0 -and $out.passed -eq $out.total)
    return $out
}

# ------------------------------------------------------------------ jedna uloha
function Invoke-Task($task, [string]$variant) {
    # variant: 'template' | 'reference' | 'model'
    $id = $task.id
    $workDir = Join-Path $workRoot $id
    if (Test-Path $workDir) { Remove-Item -Recurse -Force $workDir }
    $appDir = Join-Path $workDir 'app'
    $testDir = Join-Path $workDir 'tests'
    New-Item -ItemType Directory -Force -Path $appDir | Out-Null
    New-Item -ItemType Directory -Force -Path $testDir | Out-Null
    Write-ProjectFiles -workDir $workDir -Exe:([bool]$task.exe)

    # skryte testy vzdy z repa
    Copy-Item (Join-Path $tasksDir "$id\tests\Program.cs") (Join-Path $testDir 'Program.cs') -Force

    # startovaci stav = sablona (rovnaka pre model aj pre referenciu)
    foreach ($f in @($task.files)) {
        $src = Join-Path $tasksDir "$id\template\$f"
        if (-not (Test-Path $src)) { continue }
        $target = Join-Path $workDir $f
        $parent = Split-Path $target -Parent
        if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
        Copy-Item $src $target -Force
    }

    $res = [ordered]@{ task = $id; variant = $variant; build_ok = $false; passed = 0; total = 0; green = $false }

    if ($variant -eq 'model') {
        $files = @($task.files)
        $current = @()
        foreach ($f in $files) {
            $p = Join-Path $tasksDir "$id\template\$f"
            if (Test-Path $p) { $current += ("=== FILE: $f ===`n" + (Get-Content $p -Raw -Encoding UTF8).TrimEnd()) }
        }
        $prompt = $task.prompt + "`n`n=== SUCASNE SUBORY ===`n" + ($current -join "`n`n") +
            "`n`nOdpovedz LEN blokmi v tvare:`n=== FILE: app/Subor.cs ===`n<kompletny obsah suboru>`n" +
            "Prepis cely obsah kazdeho suboru, ktory menis. Ziadne vysvetlenie."
        $body = @{
            model = $script:model
            messages = @(@{ role = 'user'; content = $prompt })
            max_tokens = 4000
            temperature = 0
            stream = $false
            chat_template_kwargs = @{ enable_thinking = $false }
        } | ConvertTo-Json -Depth 8
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $r = Post-Json $uriChat $body
        $sw.Stop()
        Start-Sleep -Milliseconds 500
        $log = Read-LogDelta
        $content = ''
        if ($r.choices -and $r.choices.Count -gt 0 -and $null -ne $r.choices[0].message.content) {
            $content = [string]$r.choices[0].message.content
        }
        $blocks = @(Get-FileBlocks $content)
        $written = @()
        foreach ($b in $blocks) {
            if (-not (Test-SafePath $b.path)) { continue }
            $target = Join-Path $workDir $b.path
            $parent = Split-Path $target -Parent
            if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
            Set-Content -Path $target -Value $b.body -Encoding UTF8
            $written += $b.path
        }
        $res['req_s'] = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        $res['gen_tok'] = $log.gen_tok
        $res['decode_tps'] = $log.decode_tps
        $res['prompt_tok'] = $log.prompt_tok
        $res['files_written'] = $written
        $res['answer_chars'] = $content.Length
    } else {
        # 'template' uz je nakopirovana; 'reference' ju prepise spravnym riesenim
        if ($variant -eq 'reference') {
            foreach ($f in @($task.files)) {
                $src = Join-Path $tasksDir "$id\reference\$f"
                if (Test-Path $src) { Copy-Item $src (Join-Path $workDir $f) -Force }
            }
        }
    }

    # Copy-Item zachovava mtime zdroja; bez tohto MSBuild vyhodnoti build ako
    # aktualny a spusti stary bin (stale build) - overene na T4.
    Get-ChildItem -Path (Join-Path $workDir 'app') -Recurse -Filter *.cs -ErrorAction SilentlyContinue |
        ForEach-Object { $_.LastWriteTime = (Get-Date) }

    # build + testy
    $score = Invoke-Score $task $workDir
    $res['passed_first'] = $score.passed
    $res['total_first'] = $score.total
    $res['green_first'] = $score.green
    $res['attempts'] = 1

    # opravne kolo: model dostane chybovy vystup a jednu moznost opravit
    $repairRounds = 0
    if ($task.repair) { $repairRounds = [int]$task.repair }
    for ($round = 1; $round -le $repairRounds -and $variant -eq 'model' -and -not $score.green; $round++) {
        $feedback = if (-not $score.build_ok) { "BUILD FAILED:`n" + $score.build_error } else { "TESTS FAILED:`n" + $score.test_output }
        $now = @()
        foreach ($f in @($task.files)) {
            $p = Join-Path $workDir $f
            if (Test-Path $p) { $now += ("=== FILE: $f ===`n" + (Get-Content $p -Raw -Encoding UTF8).TrimEnd()) }
        }
        $prompt = $task.prompt + "`n`n=== SUCASNE SUBORY ===`n" + ($now -join "`n`n") +
            "`n`n=== VYSTUP Z BUILDU/TESTOV ===`n" + $feedback +
            "`n`nOprav chyby. Odpovedz LEN blokmi v tvare:`n=== FILE: app/Subor.cs ===`n<kompletny obsah suboru>`n" +
            "Prepis cely obsah kazdeho suboru, ktory menis. Ziadne vysvetlenie."
        $body = @{
            model = $script:model
            messages = @(@{ role = 'user'; content = $prompt })
            max_tokens = 4000
            temperature = 0
            stream = $false
            chat_template_kwargs = @{ enable_thinking = $false }
        } | ConvertTo-Json -Depth 8
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $r = Post-Json $uriChat $body
        $sw.Stop()
        Start-Sleep -Milliseconds 500
        $log = Read-LogDelta
        $content = ''
        if ($r.choices -and $r.choices.Count -gt 0 -and $null -ne $r.choices[0].message.content) { $content = [string]$r.choices[0].message.content }
        $blocks = @(Get-FileBlocks $content)
        foreach ($bk in $blocks) {
            if (-not (Test-SafePath $bk.path)) { continue }
            $target = Join-Path $workDir $bk.path
            $parent = Split-Path $target -Parent
            if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
            Set-Content -Path $target -Value $bk.body -Encoding UTF8
        }
        $res['repair_s'] = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        $res['repair_gen_tok'] = $log.gen_tok
        $res['repair_blocks'] = @($blocks | ForEach-Object { $_.path })
        $score = Invoke-Score $task $workDir
        $res['attempts'] = 1 + $round
    }

    $res['build_ok'] = $score.build_ok
    $res['passed'] = $score.passed
    $res['total'] = $score.total
    $res['green'] = $score.green
    if ($score.build_error) { $res['build_error'] = $score.build_error }
    if ($score.test_output) { $res['test_output'] = $score.test_output }
    $res['timed_out'] = $score.timed_out
    return [pscustomobject]$res
}

# ------------------------------------------------------------------ engine log
$script:model = $null
$script:log = $null
$script:logPos = 0
function Init-Log {
    $script:log = (Get-ChildItem (Join-Path $env:LOCALAPPDATA 'Strata\launch') -Filter '*.engine.log' |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    $script:logPos = @(Get-Content $script:log).Count
}
function Read-LogDelta {
    if (-not $script:log) { return [pscustomobject]@{ prompt_tok = $null; gen_tok = $null; decode_tps = $null } }
    $all = @(Get-Content $script:log)
    if ($all.Count -gt $script:logPos) { $new = $all[$script:logPos..($all.Count - 1)] } else { $new = @() }
    $script:logPos = $all.Count
    $row = [pscustomobject]@{ prompt_tok = $null; gen_tok = $null; decode_tps = $null }
    $pl = ($new | Select-String 'strata serve: prompt ' | Select-Object -Last 1).Line
    if ($pl -match $script:logPromptPattern) {
        $row.prompt_tok = [int]$Matches[1]; $row.gen_tok = [int]$Matches[6]; $row.decode_tps = [double]$Matches[8]
    }
    return $row
}

# ------------------------------------------------------------------ naci tasky
$tasks = @()
foreach ($d in Get-ChildItem $tasksDir -Directory | Sort-Object Name) {
    $jf = Join-Path $d.FullName 'task.json'
    if (-not (Test-Path $jf)) { continue }
    $t = Get-Content $jf -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $t.id) { $t | Add-Member -NotePropertyName id -NotePropertyValue $d.Name -Force }
    if ($Only -and $t.id -notlike "*$Only*") { continue }
    $tasks += $t
}

# ------------------------------------------------------------------ selftest
if ($SelfTest) {
    '=== SELFTEST: referencia musi prejst, sablona musi zlyhat (bez engine) ==='
    $bad = 0

    # parser odpovede modelu
    $sample = "=== FILE: app/Fetcher.cs ===`nnamespace Eval;`n`n=== FILE: app/Sub/X.cs ===`nclass X {}"
    $blocks = @(Get-FileBlocks $sample)
    if ($blocks.Count -eq 2 -and $blocks[0].path -eq 'app/Fetcher.cs' -and $blocks[0].body -eq 'namespace Eval;' -and $blocks[1].path -eq 'app/Sub/X.cs') {
        '  parser                 PASS  2 bloky, spravne cesty a telo'
    } else { '  parser                 FAIL  ' + (($blocks | ForEach-Object { $_.path }) -join ','); $bad++ }

    if ((Test-SafePath 'app/X.cs') -and -not (Test-SafePath '../evil.cs') -and -not (Test-SafePath 'C:\evil.cs') -and -not (Test-SafePath '/etc/x')) {
        '  path-safety            PASS  relativne ano, absolutne a .. nie'
    } else { '  path-safety            FAIL'; $bad++ }

    # regex na engine log musi sedet na realny riadok
    $sampleLine = 'strata serve: prompt 1234 tokens = 100 reused + 1134 read in 400 ms (3085.0 tok/s), 42 generated in 500 ms (84.0 tok/s), drafts accepted 30 of 40'
    if ($sampleLine -match $script:logPromptPattern) {
        if ([int]$Matches[1] -eq 1234 -and [int]$Matches[6] -eq 42 -and [double]$Matches[8] -eq 84.0) {
            '  log-regex              PASS  prompt=1234 gen=42 decode=84'
        } else { '  log-regex              FAIL  spatne skupiny'; $bad++ }
    } else { '  log-regex              FAIL  nesedi na vzorku'; $bad++ }

    foreach ($t in $tasks) {
        $ref = Invoke-Task $t 'reference'
        $tag = if ($ref.green) { 'PASS' } else { 'FAIL' }
        if (-not $ref.green) { $bad++ }
        "  {0,-22} referencia  {1}  build={2} testy={3}/{4} {5}" -f $t.id, $tag, $ref.build_ok, $ref.passed, $ref.total, $ref.build_error
        if (-not $ref.green) { "     detail: " + $ref.test_output }
        $tmpTpl = Invoke-Task $t 'template'
        $neg = (-not $tmpTpl.green)
        $tag2 = if ($neg) { 'PASS' } else { 'FAIL' }
        if (-not $neg) { $bad++ }
        "  {0,-22} sablona     {1}  (ocakavane zlyhanie) build={2} testy={3}/{4}" -f $t.id, $tag2, $tmpTpl.build_ok, $tmpTpl.passed, $tmpTpl.total
    }
    '  ----'
    if ($bad -eq 0) { 'SELFTEST: OK' } else { "SELFTEST: $bad zlyhani"; exit 1 }
    return
}

if ($NoRun) {
    'ulohy:'
    foreach ($t in $tasks) { "  {0,-22} {1}" -f $t.id, $t.title }
    'SelfTest:  .\dotnet-eval.ps1 -SelfTest'
    'Meranie:   .\dotnet-eval.ps1 -RunId base'
    return
}

# ------------------------------------------------------------------ beh
$models = Invoke-RestMethod "$baseUri/v1/models" -TimeoutSec 10
$script:model = if ($Model) { $Model } else { [string]$models.data[0].id }
"model: $($script:model)"
Init-Log

$all = @()
foreach ($rep in 1..$Reps) {
    foreach ($t in $tasks) {
        $r = Invoke-Task $t 'model'
        $r | Add-Member -NotePropertyName rep -NotePropertyValue $rep -Force
        $all += $r
        "  {0,-22} rep{1} green={2} build={3} testy={4}/{5} gen={6} tok  {7}s" -f `
            $t.id, $rep, $r.green, $r.build_ok, $r.passed, $r.total, $r.gen_tok, $r.req_s
    }
}

$taskIds = @($tasks | ForEach-Object { $_.id })
$greenTasks = @()
foreach ($id in $taskIds) {
    $rows = @($all | Where-Object { $_.task -eq $id })
    $badRows = @($rows | Where-Object { -not $_.green })
    if ($rows.Count -gt 0 -and $badRows.Count -eq 0) { $greenTasks += $id }
}

$summary = [ordered]@{
    run_id = $RunId
    model = $script:model
    engine_line = ((Get-Content $script:log | Select-String 'engine started:' | Select-Object -Last 1).Line)
    started_utc = (Get-Date).ToUniversalTime().ToString('s') + 'Z'
    green_results = @($all | Where-Object { $_.green }).Count
    total_results = @($all).Count
    tasks = $taskIds.Count
    tasks_all_green = $greenTasks.Count
    reps = $Reps
    total_wall_s = [math]::Round((@($all | ForEach-Object { [double]$_.req_s } | Measure-Object -Sum).Sum), 1)
    gen_tok_total = (@($all | ForEach-Object { [int]$_.gen_tok } | Measure-Object -Sum).Sum)
    results = $all
}
$file = Join-Path $outDir ("dotnet-eval-$RunId.json")
($summary | ConvertTo-Json -Depth 8) | Set-Content -Path $file -Encoding UTF8
''
"green behy:  $($summary.green_results)/$($summary.total_results)"
"ulohy plne zelene: $($summary.tasks_all_green)/$($summary.tasks)$(if ($greenTasks.Count -gt 0) { '  (' + ($greenTasks -join ', ') + ')' })"
'--- po ulohach ---'
foreach ($id in $taskIds) {
    $rows = @($all | Where-Object { $_.task -eq $id })
    $g = @($rows | Where-Object { $_.green }).Count
    $tests = ($rows | ForEach-Object { "$($_.passed)/$($_.total)" }) -join ' '
    $builds = @($rows | Where-Object { -not $_.build_ok }).Count
    "  {0,-22} green {1}/{2}   testy: {3}   build zlyhal: {4}x" -f $id, $g, $rows.Count, $tests, $builds
}
"gen_tok spolu: $($summary.gen_tok_total), wall: $($summary.total_wall_s) s"
"zapisane: $file"
