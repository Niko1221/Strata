param([string]$Out = "telemetry.jsonl")
$sw = [System.Diagnostics.Stopwatch]::StartNew()
while ($true) {
    $os = Get-CimInstance Win32_OperatingSystem
    $ramUsedGB = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 2)
    $gpuUtil = $null; $vramMB = $null
    try {
        $u = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction Stop).CounterSamples |
             Where-Object { $_.InstanceName -match 'engtype_(3D|Compute)' -and $_.CookedValue -gt 0 } |
             Measure-Object CookedValue -Sum
        if ($u.Count -gt 0) { $gpuUtil = [math]::Round($u.Sum, 1) }
        $v = (Get-Counter '\GPU Process Memory(*)\Local Usage' -ErrorAction Stop).CounterSamples |
             Measure-Object CookedValue -Sum
        $vramMB = [math]::Round($v.Sum / 1MB, 0)
    } catch {}
    $row = [ordered]@{
        t_s = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        ram_used_gb = $ramUsedGB
        gpu_util_pct = $gpuUtil
        vram_dedicated_mb = $vramMB
    }
    ($row | ConvertTo-Json -Compress) | Out-File -Append -Encoding utf8 $Out
    Start-Sleep -Seconds 1
}