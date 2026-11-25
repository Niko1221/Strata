#requires -Version 5.1
<#
.SYNOPSIS
The MoE project's speed tests (16K decode benchmark and context ladder), run against Strata.
.DESCRIPTION
Ported from ..\..\..\..\MoE\run_model_benchmarks.ps1 (mode 'short') and run_context_ladder.ps1
(mode 'ladder'). The prompts, the recall needle, the request body (temperature 0, seed 1234,
streamed, cache_prompt false) and the report columns are the same, so the numbers compare with the
llama-server results in MoE\benchmark_runs and UsingTheBest.md.

What differs, because Strata is not llama-server:
- The server is started with `strata launch ... --restart --yes --no-tui` instead of llama-server.
  Each config is a list of extra `strata launch` flags (for example --drafting mtp, --kv q4_0,
  --set STRATA_SM75_PERF=1). -NoLaunch measures whatever is serving at -BaseUrl (Basalt, a
  llama-server you started yourself, ...), so the same prompts can be used on other servers.
- Strata has no /tokenize. The ladder sizes its fill with /v1/messages/count_tokens, which also
  counts the chat template; that makes the fill a few dozen tokens smaller, never larger.
- After each load one short request (64 tokens) is sent and not scored, because Strata's first
  request pays one-off costs (graph capture, the SSD table). The MoE ladder had no warmup.
- The free-RAM floor defaults to 1 GiB, not 4: Strata pins the experts it keeps in RAM (about
  40 GiB for UD-IQ2_XS), so free physical RAM on a 64 GiB PC is about 2 GiB while it serves.

Mode short: per config, a 16K context, a 1,024-token warmup prompt, then the measured prompt.
Mode ladder: per config and context level, a fresh start at that context, one prompt that fills
it, 256 generated tokens, and a recall check.

Results go to -OutputDirectory (results.json checkpoint, results.csv, results.md). Pass the same
folder again to resume; finished rows are skipped.
.EXAMPLE
.\run_strata_speed.ps1 -Mode ladder -ConfigFile .\configs\ladder.psd1
.EXAMPLE
.\run_strata_speed.ps1 -Mode short -NoLaunch -BaseUrl http://127.0.0.1:8081 -Label basalt
#>
[CmdletBinding()]
param(
    [ValidateSet('short', 'ladder')][string]$Mode = 'short',
    # A .psd1 with @{ Configs = @( @{ Label = '...'; Args = @('--drafting','mtp') }, ... ) }.
    [string]$ConfigFile = '',
    [string]$OutputDirectory = '',
    [string]$BaseUrl = 'http://127.0.0.1:8080',
    [string]$StrataCmd = "$env:LOCALAPPDATA\Strata\terminal\current\bin\strata.cmd",
    # The model every config launches. Configs can add flags, not change these.
    [string[]]$ModelArgs = @('--family', 'unsloth', '--model', 'UD-IQ2_XS'),
    [int[]]$Contexts = @(32768, 65536, 131072, 196608, 262144),
    # Mode short: the context it allocates (MoE used 16000).
    [int]$ShortContext = 16384,
    [int]$MaxTokens = 0,
    [double]$MinFreeRamGiB = 1,
    [double]$MinDecodeTokS = 10,
    [int]$RequestTimeoutMin = 120,
    [int]$LoadTimeoutSec = 1800,
    # Measure the server already at -BaseUrl; start nothing.
    [switch]$NoLaunch,
    [string]$Label = 'external',
    # Stop serving when the run ends (default: leave the last config serving).
    [switch]$StopAtEnd
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Net.Http
if ($MaxTokens -le 0) { $MaxTokens = if ($Mode -eq 'short') { 1024 } else { 256 } }
$BaseUrl = $BaseUrl.TrimEnd('/')
$minFreeRamMiB = [int]($MinFreeRamGiB * 1024)

if ($NoLaunch) {
    $configs = @(@{ Label = $Label; Args = @() })
} elseif ($ConfigFile) {
    $configs = @((Import-PowerShellDataFile -LiteralPath $ConfigFile).Configs)
} else {
    $configs = @(@{ Label = 'default'; Args = @() })
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $PSScriptRoot ('results\' + (Get-Date -Format 'yyyyMMdd-HHmmss') + "-$Mode")
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$checkpointPath = Join-Path $OutputDirectory 'results.json'
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromMinutes($RequestTimeoutMin)
$hasNvidiaSmi = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)

# ---- prompts, copied unchanged from the MoE scripts ----
$warmupPrompt = @'
Describe how to build a reliable local document search service for a small team.
Write a detailed technical guide of at least 1500 words, using numbered sections.
Cover document ingestion, text extraction, chunking, metadata, embeddings, indexing,
retrieval, ranking, permissions, caching, observability, backup, testing, and deployment.
Include concrete examples, pseudocode, failure scenarios, and recovery procedures.
Do not conclude early; continue developing each section until the response limit.
'@
$measuredPrompt = @'
Explain how to design and operate a reliable local mixture-of-experts inference service.
Write a detailed technical guide of at least 1500 words, using numbered sections.
Cover token routing, active versus total parameters, expert placement, GPU memory,
CPU offloading, KV caches, batching, prompt processing, decoding, scheduling, and monitoring.
Include concrete examples, pseudocode, failure scenarios, and recovery procedures.
Do not conclude early; continue developing each section until the response limit.
'@
$services = @('auth', 'billing', 'search', 'storage', 'gateway', 'metrics', 'scheduler', 'mailer')
$regions = @('us-east', 'us-west', 'eu-central', 'ap-south')
$statuses = @('OK', 'OK', 'OK', 'DEGRADED', 'RETRYING', 'TIMEOUT')
$question = "`n`nThe text above is a service log. Answer two questions. First line: the vault access code given in the NOTICE entry, exactly as written. Second line: which service appears most often in the log."

function New-FillText([int]$Records, [string]$Code) {
    $rng = New-Object Random 20261027
    $sb = New-Object Text.StringBuilder
    $needleAt = [math]::Max(1, [int]($Records / 2))
    for ($i = 1; $i -le $Records; $i++) {
        if ($i -eq $needleAt) {
            [void]$sb.AppendLine(('Log entry {0:D6}: NOTICE the vault access code is {1}. Keep it for the audit.' -f $i, $Code))
            continue
        }
        [void]$sb.AppendLine(('Log entry {0:D6}: service {1} on node n-{2} in {3} reported latency {4} ms, status {5}.' -f
            $i, $services[$rng.Next($services.Count)], $rng.Next(100, 999), $regions[$rng.Next($regions.Count)],
            $rng.Next(3, 900), $statuses[$rng.Next($statuses.Count)]))
    }
    $sb.ToString()
}

function Get-TokenCount([string]$Text) {
    # Strata has no /tokenize; count_tokens includes the chat template, so this over-counts slightly.
    $body = @{ model = 'benchmark'; max_tokens = 1; messages = @(@{ role = 'user'; content = $Text }) } | ConvertTo-Json -Compress -Depth 5
    try {
        $r = Invoke-RestMethod -Uri "$BaseUrl/v1/messages/count_tokens" -Method Post -ContentType 'application/json; charset=utf-8' `
            -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 600
        return [int]$r.input_tokens
    } catch {
        # llama-server based servers (Basalt, llama.cpp): /tokenize.
        $body = @{ content = $Text } | ConvertTo-Json -Compress
        $r = Invoke-RestMethod -Uri "$BaseUrl/tokenize" -Method Post -ContentType 'application/json; charset=utf-8' `
            -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 600
        return @($r.tokens).Count
    }
}

function New-FillPrompt([int]$Context, [string]$Code) {
    $target = $Context - $MaxTokens - 512
    $records = [int]($target / 30)
    $text = $null; $count = 0
    for ($i = 0; $i -lt 4; $i++) {
        $text = (New-FillText $records $Code) + $question
        $count = Get-TokenCount $text
        if ($count -le $target -and $count -ge $target * 0.97) { break }
        $records = [int][math]::Floor($records * ($target * 0.985) / $count)
    }
    while ($count -gt $target) {
        $records = [int][math]::Floor($records * 0.98)
        $text = (New-FillText $records $Code) + $question
        $count = Get-TokenCount $text
    }
    [pscustomobject]@{ Text = $text; Tokens = $count }
}

# ---- monitoring ----
function Get-FreeRamMiB { [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB) }
function Get-GpuUsedMiB {
    if (-not $hasNvidiaSmi) { return $null }
    $lines = @(& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0) { return $null }
    ($lines | ForEach-Object { [int]$_.Trim() } | Measure-Object -Sum).Sum
}
function Get-ServerProcess { Get-Process -Name strata, llama-server -ErrorAction SilentlyContinue | Sort-Object WorkingSet64 -Descending | Select-Object -First 1 }
$script:lastPoll = [datetime]::MinValue
function Update-Peaks($Row) {
    if (((Get-Date) - $script:lastPoll).TotalMilliseconds -lt 1000) { return }
    $script:lastPoll = Get-Date
    $free = Get-FreeRamMiB
    if ($null -eq $Row.MinFreeRamMiB -or $free -lt $Row.MinFreeRamMiB) { $Row.MinFreeRamMiB = $free }
    $gpu = Get-GpuUsedMiB
    if ($null -ne $gpu -and ($null -eq $Row.PeakGpuMiB -or $gpu -gt $Row.PeakGpuMiB)) { $Row.PeakGpuMiB = $gpu }
    $p = Get-ServerProcess
    if ($p) {
        $ws = [int]($p.WorkingSet64 / 1MB)
        if ($null -eq $Row.PeakServerRamMiB -or $ws -gt $Row.PeakServerRamMiB) { $Row.PeakServerRamMiB = $ws }
    }
    if ($free -lt $minFreeRamMiB) { throw "RAM limit: free RAM fell to $free MiB (floor $minFreeRamMiB MiB)." }
}

function Invoke-Streamed($Row, [string]$Prompt, [int]$Tokens, [string]$ResponsePath) {
    $body = @{
        model = 'benchmark'
        messages = @(@{ role = 'user'; content = $Prompt })
        temperature = 0; seed = 1234; max_tokens = $Tokens; stream = $true; cache_prompt = $false
    } | ConvertTo-Json -Depth 6
    $request = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, "$BaseUrl/v1/chat/completions")
    $request.Content = [System.Net.Http.StringContent]::new($body, [Text.Encoding]::UTF8, 'application/json')
    $response = $null; $reader = $null
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $text = [Text.StringBuilder]::new()
    $ttft = $null; $timings = $null; $done = $false
    try {
        $task = $client.SendAsync($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead)
        while (-not $task.IsCompleted) { if ($Row) { Update-Peaks $Row }; $null = $task.Wait(250) }
        $response = $task.GetAwaiter().GetResult()
        if (-not $response.IsSuccessStatusCode) {
            throw "HTTP $([int]$response.StatusCode): $($response.Content.ReadAsStringAsync().GetAwaiter().GetResult())"
        }
        $reader = [IO.StreamReader]::new($response.Content.ReadAsStreamAsync().GetAwaiter().GetResult())
        while ($true) {
            $readTask = $reader.ReadLineAsync()
            while (-not $readTask.IsCompleted) { if ($Row) { Update-Peaks $Row }; $null = $readTask.Wait(250) }
            $line = $readTask.GetAwaiter().GetResult()
            if ($null -eq $line) { break }
            if (-not $line.StartsWith('data:')) { continue }
            $data = $line.Substring(5).Trim()
            if ($data -eq '[DONE]') { $done = $true; break }
            $chunk = $data | ConvertFrom-Json
            if ($chunk.error) { throw ($chunk.error | ConvertTo-Json -Compress) }
            if ($chunk.timings) { $timings = $chunk.timings }
            if ($chunk.choices) {
                foreach ($piece in @($chunk.choices[0].delta.reasoning_content, $chunk.choices[0].delta.content)) {
                    if ($piece) {
                        if ($null -eq $ttft) { $ttft = [math]::Round($watch.Elapsed.TotalMilliseconds) }
                        [void]$text.Append($piece)
                    }
                }
            }
        }
        if (-not $done -or -not $timings) { throw 'Incomplete stream or missing server timings.' }
        [pscustomobject]@{
            TTFT_ms = $ttft
            PromptTokens = $timings.prompt_n
            PromptTok_s = [math]::Round([double]$timings.prompt_per_second, 2)
            GeneratedTokens = $timings.predicted_n
            DecodeTok_s = [math]::Round([double]$timings.predicted_per_second, 2)
            DraftN = $timings.draft_n
            DraftAccepted = $timings.draft_n_accepted
            RequestSec = [math]::Round($watch.Elapsed.TotalSeconds, 1)
            Text = $text.ToString()
        }
    } finally {
        if ($ResponsePath) { $text.ToString() | Set-Content -LiteralPath $ResponsePath -Encoding UTF8 }
        if ($reader) { $reader.Dispose() }
        if ($response) { $response.Dispose() }
        $request.Dispose()
    }
}

# ---- launching ----
function Start-Config([hashtable]$Config, [int]$Context, $Row) {
    $watch = [Diagnostics.Stopwatch]::StartNew()
    if (-not $NoLaunch) {
        $ctxArg = if ($Context % 1024 -eq 0) { "$($Context / 1024)k" } else { "$Context" }
        $launch = @('launch') + $ModelArgs + @('--context', $ctxArg) + @($Config.Args) + @('--restart', '--yes', '--no-tui')
        $Row.LaunchArgs = ($launch -join ' ')
        Write-Output "LAUNCH strata $($Row.LaunchArgs)"
        $log = Join-Path $OutputDirectory ("$($Row.Safe).launch.log")
        & $StrataCmd @launch *> $log
        if ($LASTEXITCODE -ne 0) { throw "strata launch exited $LASTEXITCODE; see $log" }
    }
    $deadline = (Get-Date).AddSeconds($LoadTimeoutSec)
    while ($true) {
        try {
            $h = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 5
            if ($h.status -eq 'ok' -and $h.loaded -ne $false) { break }
        } catch { }
        if ((Get-Date) -gt $deadline) { throw "Server at $BaseUrl not ready after $LoadTimeoutSec s." }
        Start-Sleep -Seconds 3
    }
    $Row.LoadSec = [math]::Round($watch.Elapsed.TotalSeconds, 1)
    try { $Row.ActualContext = (Invoke-RestMethod -Uri "$BaseUrl/props" -TimeoutSec 10).default_generation_settings.n_ctx } catch { $Row.ActualContext = $Context }
    try { $Row.Engine = (Invoke-RestMethod -Uri "$BaseUrl/v1/status" -TimeoutSec 10).engine } catch { }
    # One unscored request: Strata's first request pays one-off costs.
    $null = Invoke-Streamed $null 'Reply with the word ready.' 64 $null
    $Row.ReadyGpuMiB = Get-GpuUsedMiB
}

# ---- checkpoint and report ----
$columns = 'Label', 'Context', 'ActualContext', 'Status', 'PromptTokens', 'PromptTok_s', 'DecodeTok_s', 'GeneratedTokens',
    'DraftN', 'DraftAccepted', 'TTFT_ms', 'RequestSec', 'WarmupDecodeTok_s', 'LoadSec', 'ReadyGpuMiB', 'PeakGpuMiB',
    'PeakServerRamMiB', 'MinFreeRamMiB', 'RecallCorrect', 'Engine', 'Error', 'LaunchArgs'
function Save-Progress {
    $state.UpdatedAt = Get-Date -Format o
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$checkpointPath.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$checkpointPath.tmp" -Destination $checkpointPath -Force
    $state.Rows | Select-Object $columns | Export-Csv -LiteralPath (Join-Path $OutputDirectory 'results.csv') -NoTypeInformation -Encoding UTF8
    $gpu = if ($hasNvidiaSmi) { (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) -join '; ' } else { '' }
    $report = @(
        "# Strata $($state.Mode) run", ''
        "Updated: $($state.UpdatedAt). GPU: $gpu. Server: $BaseUrl."
        ''
    )
    if ($state.Mode -eq 'short') {
        $report += "Context $ShortContext. Each config gets a $MaxTokens-token-cap warmup (MoE warmup prompt), then the MoE measured prompt with the same cap."
    } else {
        $report += "Each level starts Strata with --context <level>, fills it with one prompt (service log, recall code in the middle) and generates up to $MaxTokens tokens."
    }
    $report += 'Temperature 0, seed 1234, cache_prompt false. Rates are the server''s own timings. Draft = drafted / accepted tokens. One sample per row.'
    $report += "RAM floor $MinFreeRamGiB GiB free; speed floor $MinDecodeTokS tok/s (higher levels are skipped below it)."
    $report += ''
    foreach ($label in @($state.Rows | Select-Object -ExpandProperty Label -Unique)) {
        $rows = @($state.Rows | Where-Object Label -eq $label)
        $report += "## $label", ''
        $report += "``strata $($rows[0].LaunchArgs)``", ''
        $report += '| Context | Status | Prompt tokens | Prompt tok/s | Decode tok/s | Draft acc. | TTFT s | Load s | Peak GPU MiB | Peak server RAM MiB | Min free RAM MiB | Recall |'
        $report += '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|'
        foreach ($r in $rows) {
            $ttft = if ($r.TTFT_ms) { [math]::Round($r.TTFT_ms / 1000, 1) } else { '' }
            $acc = if ($r.DraftN) { '{0}/{1}' -f $r.DraftAccepted, $r.DraftN } else { '' }
            $report += "| $($r.ActualContext) | $($r.Status) | $($r.PromptTokens) | $($r.PromptTok_s) | $($r.DecodeTok_s) | $acc | $ttft | $($r.LoadSec) | $($r.PeakGpuMiB) | $($r.PeakServerRamMiB) | $($r.MinFreeRamMiB) | $($r.RecallCorrect) |"
        }
        foreach ($r in $rows | Where-Object Error) { $report += ''; $report += "- $($r.Context): $($r.Error)" }
        $report += ''
    }
    $report | Set-Content -LiteralPath (Join-Path $OutputDirectory 'results.md') -Encoding UTF8
}

if (Test-Path -LiteralPath $checkpointPath) {
    $state = Get-Content -LiteralPath $checkpointPath -Raw | ConvertFrom-Json
    $state.Rows = @($state.Rows)
    foreach ($r in $state.Rows) { if ($r.Status -notin @('OK', 'Skipped', 'Failed', 'TooSlow')) { $r.Status = 'Pending' } }
} else {
    $rows = foreach ($c in $configs) {
        $levels = if ($Mode -eq 'short') { @($ShortContext) } else { $Contexts }
        foreach ($ctx in $levels) {
            $safe = ('{0}-{1}' -f $c.Label, $ctx) -replace '[^\w.-]', '_'
            [pscustomobject]@{
                Label = $c.Label; Context = $ctx; Safe = $safe; Status = 'Pending'; ActualContext = $null
                PromptTokens = $null; PromptTok_s = $null; DecodeTok_s = $null; GeneratedTokens = $null
                DraftN = $null; DraftAccepted = $null; TTFT_ms = $null; RequestSec = $null; WarmupDecodeTok_s = $null
                LoadSec = $null; ReadyGpuMiB = $null; PeakGpuMiB = $null; PeakServerRamMiB = $null; MinFreeRamMiB = $null
                RecallCorrect = $null; Engine = $null; Error = $null; LaunchArgs = ''
            }
        }
    }
    $state = [pscustomobject]@{ Mode = $Mode; UpdatedAt = $null; Rows = @($rows) }
}
Save-Progress

foreach ($c in $configs) {
    $skipRest = $false
    foreach ($row in @($state.Rows | Where-Object Label -eq $c.Label | Sort-Object Context)) {
        if ($row.Status -ne 'Pending') { if ($row.Status -in 'Failed', 'TooSlow') { $skipRest = $true }; continue }
        if ($skipRest) { $row.Status = 'Skipped'; Save-Progress; continue }
        try {
            $row.Status = 'Loading'; Save-Progress
            Start-Config $c $row.Context $row
            $row.Status = 'Running'; Save-Progress
            if ($Mode -eq 'short') {
                $w = Invoke-Streamed $row $warmupPrompt $MaxTokens (Join-Path $OutputDirectory "$($row.Safe).warmup.txt")
                $row.WarmupDecodeTok_s = $w.DecodeTok_s
                $m = Invoke-Streamed $row $measuredPrompt $MaxTokens (Join-Path $OutputDirectory "$($row.Safe).measured.txt")
            } else {
                $code = 'VAULT-{0:X6}' -f ((New-Object Random $row.Context).Next(0x100000, 0xFFFFFF))
                $fill = New-FillPrompt $row.ActualContext $code
                Write-Output "RUN $($row.Label) @ $($row.Context): $($fill.Tokens) prompt tokens"
                $m = Invoke-Streamed $row $fill.Text $MaxTokens (Join-Path $OutputDirectory "$($row.Safe).response.txt")
                $row.RecallCorrect = $m.Text.Contains($code)
            }
            foreach ($k in 'TTFT_ms', 'PromptTokens', 'PromptTok_s', 'GeneratedTokens', 'DecodeTok_s', 'DraftN', 'DraftAccepted', 'RequestSec') { $row.$k = $m.$k }
            $row.Status = 'OK'
            if ($MinDecodeTokS -gt 0 -and $row.DecodeTok_s -lt $MinDecodeTokS) { $row.Status = 'TooSlow'; $skipRest = $true }
            Write-Output "OK $($row.Label) @ $($row.Context): prompt $($row.PromptTok_s) tok/s, decode $($row.DecodeTok_s) tok/s, draft $($row.DraftAccepted)/$($row.DraftN), recall $($row.RecallCorrect)"
        } catch {
            $row.Status = 'Failed'; $row.Error = $_.Exception.Message; $skipRest = $true
            Write-Warning "FAILED $($row.Label) @ $($row.Context): $($row.Error)"
        }
        Save-Progress
    }
}
if ($StopAtEnd -and -not $NoLaunch) { & $StrataCmd launch --stop *> $null }
Write-Output "Done. Report: $(Join-Path $OutputDirectory 'results.md')"
