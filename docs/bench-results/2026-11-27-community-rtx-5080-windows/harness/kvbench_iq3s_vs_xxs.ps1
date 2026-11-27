# kvbench_iq3s_vs_xxs.ps1 — benchmark "je IQ3_S skutocne lepsia ako IQ3_XXS?"
# Metodika: IQ3S_VS_XXS_METHOD_2026-11-27.md
#
# Vrstvy:
#   A = perplexita / NLL (len ak engine podpori token logprobs)
#   B = behavioralne ulohy so spojitym skore (B1 cisla, B2 recitacia, B3 tabulka, B4 kod)
#   C = rychlost a pamat (kontrola platnosti behu)
#
# Priklady:
#   powershell -File .\kvbench_iq3s_vs_xxs.ps1 -SelfTest
#   powershell -File .\kvbench_iq3s_vs_xxs.ps1 -RunId A1 -Round A1 -Layer all
#   powershell -File .\kvbench_iq3s_vs_xxs.ps1 -RunId B1 -Round B1 -Layer B -Reps 5
#
# Runner NIC neinicializuje ani nerestartuje. Oczekava, ze engine uz servíruje
# spravny pack; pri nesulade modelu sa OKAMZITE zastavi.
param(
    [string]$RunId = 'A1',
    [string]$Round = '',
    [ValidateSet('A', 'B', 'C', 'all')][string]$Layer = 'all',
    [string]$TaskFile = (Join-Path $PSScriptRoot 'iq3s_vs_xxs_tasks.json'),
    [string]$ExpectModel = '',
    [int]$Reps = 0,
    [switch]$WithCode,
    [switch]$SelfTest,
    [switch]$NoRun,
    [int]$LeadInSec = 25
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$base = 'http://127.0.0.1:8080'
$uriChat = "$base/v1/chat/completions"
$uriModels = "$base/v1/models"
$doc = Get-Content $TaskFile -Raw -Encoding UTF8 | ConvertFrom-Json
$pools = $doc.pools
$tmp = Join-Path $env:TEMP ('kv-iq3s-' + $RunId)
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONDONTWRITEBYTECODE = '1'

# ------------------------------------------------------------------ HTTP
function Post-Json([string]$url, [string]$jsonBody, [int]$timeoutSec = 1800) {
    # PS 5.1 dekoduje JSON bez charsetu ako Latin-1; preto raw bytes + rucne UTF-8.
    $req = [System.Net.HttpWebRequest]::Create($url)
    $req.Method = 'POST'
    $req.ContentType = 'application/json; charset=utf-8'
    $req.Timeout = $timeoutSec * 1000
    $req.ReadWriteTimeout = $timeoutSec * 1000
    $bytes = [Text.Encoding]::UTF8.GetBytes($jsonBody)
    $req.ContentLength = $bytes.Length
    $rs = $req.GetRequestStream()
    $rs.Write($bytes, 0, $bytes.Length)
    $rs.Close()
    $resp = $req.GetResponse()
    $ms = New-Object System.IO.MemoryStream
    $src = $resp.GetResponseStream()
    $src.CopyTo($ms)
    $src.Close(); $resp.Close()
    return ([Text.Encoding]::UTF8.GetString($ms.ToArray()))
}
function Post-JsonObj([string]$url, [string]$jsonBody, [int]$timeoutSec = 1800) {
    return ((Post-Json $url $jsonBody $timeoutSec) | ConvertFrom-Json)
}
function FreeRam { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2) }
function VramNow {
    try {
        $v = (& nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader) -split ','
        [pscustomobject]@{ used_mib = [int]($v[0] -replace '\D', ''); free_mib = [int]($v[1] -replace '\D', '') }
    } catch { [pscustomobject]@{ used_mib = -1; free_mib = -1 } }
}

# ------------------------------------------------------------------ deterministicky RNG
# Jeden generator na tok; rovnaky seed = rovnake polozky v kazdom kole a pre oba modely.
function New-Rng([long]$seed) { return @{ s = ($seed % 2147483647) } }
function Next-Int($rng, [int]$max) {
    $rng.s = ([long]$rng.s * 1103515245 + 12345) % 2147483648
    $x = [double]$rng.s / 2147483648.0
    return [int][math]::Floor($x * $max)
}
function Next-Range($rng, [int]$min, [int]$maxIncl) { return $min + (Next-Int $rng ($maxIncl - $min + 1)) }

# ------------------------------------------------------------------ B1: ciselna presnost
function New-NumericTask([int]$rep = 0) {
    $cfg = $doc.numeric
    $rng = New-Rng ([long]$cfg.seed + 101 * $rep)
    $items = @()
    for ($i = 1; $i -le $cfg.count; $i++) {
        $t = $i % 4
        switch ($t) {
            0 {
                $a = Next-Range $rng 11 99; $b = Next-Range $rng 11 99
                $c = Next-Range $rng 100 999; $d = Next-Range $rng 100 999
                $text = "($a krat $b) plus $c minus $d"
                $ans = $a * $b + $c - $d
            }
            1 {
                $x = @(5, 10, 15, 20, 25, 40, 60, 75)[(Next-Int $rng 8)]
                $y = 20 * (Next-Range $rng 5 495)
                $text = "kolko je $x percent z cisla $y"
                $ans = $x * $y / 100
            }
            2 {
                $h = Next-Range $rng 1 23; $m = Next-Range $rng 0 59
                $text = "kolko sekund je $h hodin a $m minut"
                $ans = $h * 3600 + $m * 60
            }
            default {
                $a = Next-Range $rng 101 999; $b = Next-Range $rng 11 99
                $text = "$a krat $b"
                $ans = $a * $b
            }
        }
        $items += [pscustomobject]@{ n = $i; text = $text; ans = [int]$ans }
    }
    $lines = @()
    $lines += 'Vypocitaj tychto ' + $cfg.count + ' prikladov. Pocitaj presne, bez zaokruhlovania.'
    $lines += 'Odpovedz presne ' + $cfg.count + ' riadkov, kazdy v tvare N=hodnota (napr. 1=3900).'
    $lines += 'Nic ine nepis, ziadne vysvetlenie, ziadne odrazky.'
    $lines += ''
    foreach ($it in $items) { $lines += ('{0}) {1}' -f $it.n, $it.text) }
    return [pscustomobject]@{
        id = "$($cfg.id)_r$rep"; kind = 'B1'; prompt = ($lines -join "`n")
        max_tokens = [int]$cfg.max_tokens; items = $items
    }
}
function Score-Numeric($task, [string]$completion) {
    $map = @{}
    foreach ($line in ($completion -split "`r?`n")) {
        if ($line -match '^\s*(\d{1,2})\s*(?:[\)\.\:\-]\s*)?=?\s*(-?\d+)\s*$') {
            $map[[int]$Matches[1]] = [long]$Matches[2]
        }
    }
    $flags = @(); $ok = 0
    foreach ($it in $task.items) {
        $got = $null
        if ($map.ContainsKey($it.n)) { $got = $map[$it.n] }
        $good = ($null -ne $got -and [long]$got -eq [long]$it.ans)
        if ($good) { $ok++ }
        $flags += [pscustomobject]@{ n = $it.n; expected = $it.ans; got = $got; ok = $good }
    }
    return [pscustomobject]@{
        score = $ok; total = @($task.items).Count; unit = 'problems'
        items = $flags; note = "parsed=$($map.Count)/$(@($task.items).Count)"
    }
}

# ------------------------------------------------------------------ B2: recitacia s near-miss
function New-RecallTask([int]$rep) {
    $cfg = $doc.recall
    $needleSet = @($doc.recall_needles)[$rep % @($doc.recall_needles).Count]
    $rng = New-Rng ([long]$cfg.seed + 1000 * $rep)
    $truePos = 0.30 + 0.10 * ($rep % 4)
    $slots = @(0.10, 0.18, 0.24, 0.38, 0.52, 0.66, 0.80, 0.90)
    $texts = @()
    $places = @()
    $k = 0
    foreach ($d in @($needleSet.decoys)) {
        $texts += [string]$d
        $places += [double]$slots[$k]; $k++
    }
    $texts += [string]$needleSet.true
    $places += [double]$truePos

    $order = @()
    for ($i = 0; $i -lt $texts.Count; $i++) { $order += $i }
    # deterministic shuffle
    for ($i = $order.Count - 1; $i -gt 0; $i--) {
        $j = Next-Int $rng ($i + 1)
        $t = $order[$i]; $order[$i] = $order[$j]; $order[$j] = $t
    }
    $ins = @()
    for ($i = 0; $i -lt $texts.Count; $i++) {
        $ins += [pscustomobject]@{ pos = [double]$places[$i]; text = $texts[$order[$i]] }
    }

    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append("[pbench $($cfg.id)-r$rep] ")
    $sorted = @($ins | Sort-Object { [double]$_.pos })
    $fi = 0; $n = 0; $si = 0
    $pool = @($pools.sk)
    while ($n -lt [int]$cfg.doc_words) {
        while ($fi -lt $sorted.Count -and $n -ge [int]([double]$sorted[$fi].pos * [int]$cfg.doc_words)) {
            [void]$sb.Append($sorted[$fi].text).Append(' ')
            $n += @($sorted[$fi].text -split '\s+').Count; $fi++
        }
        $s = $pool[$si % $pool.Count] + " [$($cfg.id)-r$rep-$si]"
        [void]$sb.Append($s).Append(' '); $n += @($s -split '\s+').Count; $si++
    }
    while ($fi -lt $sorted.Count) { [void]$sb.Append($sorted[$fi].text).Append(' '); $fi++ }

    return [pscustomobject]@{
        id = "$($cfg.id)_r$rep"; kind = 'B2'
        prompt = ($sb.ToString().Trim() + "`n`n" + [string]$needleSet.question)
        max_tokens = [int]$cfg.max_tokens
        target = [string]$needleSet.true; rep = $rep
    }
}
function Get-WordList([string]$s) {
    $t = $s.ToLowerInvariant()
    $t = [regex]::Replace($t, '[^\p{L}\p{Nd}]+', ' ')
    return @($t.Trim() -split '\s+' | Where-Object { $_ -ne '' })
}
function Compare-Words($a, $b) {
    $n = $a.Count; $m = $b.Count
    $d = New-Object 'int[,]' -ArgumentList ($n + 1), ($m + 1)
    for ($i = 0; $i -le $n; $i++) { $d.SetValue($i, $i, 0) }
    for ($j = 0; $j -le $m; $j++) { $d.SetValue($j, 0, $j) }
    for ($i = 1; $i -le $n; $i++) {
        for ($j = 1; $j -le $m; $j++) {
            $cost = 0; if ($a[$i - 1] -ne $b[$j - 1]) { $cost = 1 }
            $up = [int]$d.GetValue($i - 1, $j) + 1
            $left = [int]$d.GetValue($i, $j - 1) + 1
            $diag = [int]$d.GetValue($i - 1, $j - 1) + $cost
            $d.SetValue([math]::Min([math]::Min($up, $left), $diag), $i, $j)
        }
    }
    # spatna stopa: ktore slova ciela boli trafene
    $flags = New-Object 'bool[]' $m
    $i = $n; $j = $m
    while ($i -gt 0 -and $j -gt 0) {
        if ($a[$i - 1] -eq $b[$j - 1]) { $flags[$j - 1] = $true; $i--; $j-- }
        elseif ([int]$d.GetValue($i - 1, $j) -le [int]$d.GetValue($i, $j - 1)) { $i-- }
        else { $j-- }
    }
    return [pscustomobject]@{ dist = [int]$d.GetValue($n, $m); n = $n; m = $m; flags = $flags }
}
function Score-Recall($task, [string]$completion) {
    $want = Get-WordList $task.target
    $got = Get-WordList $completion
    $cmp = Compare-Words $want $got
    $wer = if ($cmp.n -gt 0) { [double]$cmp.dist / $cmp.n } else { 1.0 }
    $normC = (($completion.ToLowerInvariant()) -replace '\s+', ' ')
    $checks = @(
        [pscustomobject]@{ name = 'protocol_id'; ok = ($normC -match 'sp-2771-ng' -or $normC -match 'au-5140-xk') },
        [pscustomobject]@{ name = 'part_or_torque'; ok = ($normC -match '88-401-b' -or $normC -match '84,5') },
        [pscustomobject]@{ name = 'date_or_screw'; ok = ($normC -match '14\. okt' -or $normC -match 'šestnást') },
        [pscustomobject]@{ name = 'no_decoy_digits'; ok = -not ($normC -match 'sp-2771-nq|sp-2777-ng|sp-2711-ng|88-410-b|au-5140-xq|au-5104-xk|48,5') },
        [pscustomobject]@{ name = 'word_overlap_ge_0.90'; ok = ((1.0 - $wer) -ge 0.90) }
    )
    return [pscustomobject]@{
        score = 1.0 - $wer; total = 1.0; unit = 'wer_score'
        wer = [math]::Round($wer, 4); items = $checks
        note = "wer=$([math]::Round($wer,4))"
    }
}

# ------------------------------------------------------------------ B3: transformacia tabulky
function New-TransformTask([int]$rep) {
    $cfg = $doc.transform
    $rng = New-Rng ([long]$cfg.seed + 100 * $rep)
    $letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    $rows = @()
    for ($i = 1; $i -le $cfg.rows; $i++) {
        $k1 = $letters[(Next-Int $rng 26)]
        $k2 = $letters[(Next-Int $rng 26)]
        $mut = Next-Int $rng 2
        if ($mut -eq 1) { $k2 = $k1 }
        $kod = '{0}{1}-{2:0000}' -f $k1, $k2, (Next-Range $rng 1000 9999)
        $mnoz = Next-Range $rng 1 999
        $cena = (Next-Range $rng 100 99999) / 100.0
        $d = (Get-Date '2026-01-01').AddDays((Next-Range $rng 0 364)).ToString('yyyy-MM-dd')
        $rows += [pscustomobject]@{ id = $i; kod = $kod; mnozstvo = $mnoz; cena = $cena; datum = $d }
    }
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine('Preved nasledujucu tabulku na JSON pole objektov.')
    [void]$sb.AppendLine('Kluce: id (cislo), kod (text), mnozstvo (cislo), cena (cislo s dvoma desatinnymi miestami), datum (text YYYY-MM-DD).')
    [void]$sb.AppendLine('Zachovaj poradie riadkov a vsetky hodnoty presne. Kazdy objekt napis na jeden riadok, kompaktne, bez odsadzovania. Odpovedz LEN JSON polom, bez textu a bez komentara.')
    [void]$sb.AppendLine('')
    [void]$sb.AppendLine('id;kod;mnozstvo;cena;datum')
    foreach ($r in $rows) {
        [void]$sb.AppendLine(('{0};{1};{2};{3};{4}' -f $r.id, $r.kod, $r.mnozstvo, $r.cena.ToString('0.00', [Globalization.CultureInfo]::InvariantCulture), $r.datum))
    }
    return [pscustomobject]@{
        id = "$($cfg.id)_r$rep"; kind = 'B3'; prompt = $sb.ToString().Trim()
        max_tokens = [int]$cfg.max_tokens; rows = $rows; rep = $rep
    }
}
function Get-JsonArray([string]$text) {
    $t = $text.Trim()
    if ($t -match '```(?:json)?\s*([\s\S]*?)```') { $t = $Matches[1].Trim() }
    $i = $t.IndexOf('[')
    if ($i -lt 0) { return $null }
    $j = $t.LastIndexOf(']')
    if ($j -gt $i) {
        $body = $t.Substring($i, $j - $i + 1)
    } else {
        # odrezana odpoved (finish=length): zober vsetky kompletne objekty
        $k = $t.LastIndexOf('}')
        if ($k -le $i) { return $null }
        $body = $t.Substring($i, $k - $i + 1) + ']'
    }
    try { $parsed = $body | ConvertFrom-Json } catch { return $null }
    if ($null -eq $parsed) { return $null }
    return @($parsed)
}
function Score-Transform($task, [string]$completion) {
    $gotRaw = Get-JsonArray $completion
    if ($null -eq $gotRaw) {
        $want = @($task.rows).Count * 5
        return [pscustomobject]@{
            score = 0; total = $want; unit = 'fields'
            items = @([pscustomobject]@{ field = 'json_parse'; ok = $false })
            note = 'odpoved sa neda parsovat ako JSON pole'
        }
    }
    $rowsGot = @($gotRaw)
    if ($rowsGot.Count -eq 1 -and $rowsGot[0] -is [System.Array]) { $rowsGot = @($rowsGot[0]) }
    $items = @(); $ok = 0
    $wantRows = @($task.rows)
    for ($i = 0; $i -lt $wantRows.Count; $i++) {
        $w = $wantRows[$i]
        $g = $null
        if ($i -lt $rowsGot.Count) { $g = $rowsGot[$i] }
        $pairs = @(
            @{ f = 'id'; ok = ($null -ne $g -and [string]$g.id -eq [string]$w.id) },
            @{ f = 'kod'; ok = ($null -ne $g -and ([string]$g.kod).Trim().ToUpperInvariant() -eq $w.kod) },
            @{ f = 'mnozstvo'; ok = ($null -ne $g -and ("$($g.mnozstvo)".Trim() -match '^\d+$') -and [int]("$($g.mnozstvo)".Trim()) -eq [int]$w.mnozstvo) },
            @{ f = 'cena'; ok = ($null -ne $g -and [math]::Abs([double]("$($g.cena)".Trim().Replace(',', '.')) - [double]$w.cena) -le 0.005) },
            @{ f = 'datum'; ok = ($null -ne $g -and ([string]$g.datum).Trim() -eq $w.datum) }
        )
        foreach ($p in $pairs) {
            if ($p.ok) { $ok++ }
            $items += [pscustomobject]@{ row = $w.id; field = $p.f; ok = [bool]$p.ok }
        }
    }
    # chybajuce/extra riadky
    $extra = $rowsGot.Count - $wantRows.Count
    return [pscustomobject]@{
        score = $ok; total = ($wantRows.Count * 5); unit = 'fields'
        items = $items; note = "rows_got=$($rowsGot.Count) rows_want=$($wantRows.Count) extra=$extra"
    }
}

# ------------------------------------------------------------------ B4: dlhy kod so skrytou chybou
function New-CodeTask([int]$rep) {
    $cfg = $doc.code
    $rng = New-Rng ([long]$cfg.seed + 10 * $rep)
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine('"""Modul pro zpracovani meranych hodnot."""')
    [void]$sb.AppendLine('')
    $n = 0
    while ($n -lt 60) {
        $k = Next-Range $rng 1 97
        [void]$sb.AppendLine("def pomocna_$('{0:000}' -f $n)(x):")
        [void]$sb.AppendLine("    return x * $k + $n")
        [void]$sb.AppendLine('')
        $n++
    }
    if ($rep % 2 -eq 0) {
        [void]$sb.AppendLine('def sliding_max(vals, k):')
        [void]$sb.AppendLine('    """Maximum v kazdom okne delky k."""')
        [void]$sb.AppendLine('    out = []')
        [void]$sb.AppendLine('    for i in range(len(vals) - k):')
        [void]$sb.AppendLine('        out.append(max(vals[i:i + k]))')
        [void]$sb.AppendLine('    return out')
    } else {
        [void]$sb.AppendLine('def sliding_max(vals, k):')
        [void]$sb.AppendLine('    """Maximum v kazdom okne delky k."""')
        [void]$sb.AppendLine('    out = []')
        [void]$sb.AppendLine('    for i in range(len(vals) - k + 1):')
        [void]$sb.AppendLine('        out.append(max(vals[i:i + k - 1]))')
        [void]$sb.AppendLine('    return out')
    }
    [void]$sb.AppendLine('')
    while ($n -lt 75) {
        $k = Next-Range $rng 1 97
        [void]$sb.AppendLine("def doplnkova_$('{0:000}' -f $n)(y):")
        [void]$sb.AppendLine("    return y - $k")
        [void]$sb.AppendLine('')
        $n++
    }
    $prompt = "V tomto súbore je v jednej funkcii chyba. Nájdi ju a vráť LEN opravenú funkciu sliding_max v jednom python bloku, bez testov a bez komentárov.`n`n" + $sb.ToString().Trim()
    return [pscustomobject]@{
        id = "$($cfg.id)_r$rep"; kind = 'B4'; prompt = $prompt
        max_tokens = [int]$cfg.max_tokens; rep = $rep
    }
}
function Score-Code($task, [string]$completion, [string]$dir) {
    $code = $completion
    if ($code -match '```(?:python)?\s*([\s\S]*?)```') { $code = $Matches[1] }
    $code = $code.Trim()
    $tests = @'
import json, sys
from m import sliding_max
cases = [
    (([1,3,2,5,4], 2), [3,3,5,5]),
    (([7,1,9,2], 3), [9,9]),
    (([4,4,4,4], 4), [4]),
    (([5], 1), [5]),
    (([2,1,3,1,4,1,5], 3), [3,3,4,4,5]),
]
res = []
for (args, want) in cases:
    try:
        got = sliding_max(*args)
        res.append({"ok": got == want, "got": got, "want": want})
    except Exception as e:
        res.append({"ok": False, "got": "EXC: " + repr(e), "want": want})
print(json.dumps(res))
'@
    $modPath = Join-Path $dir 'm.py'
    $testPath = Join-Path $dir 't.py'
    Set-Content -Path $modPath -Value $code -Encoding UTF8
    Set-Content -Path $testPath -Value $tests -Encoding UTF8
    $out = & python $testPath 2>&1
    $joined = ($out -join "`n")
    $res = @()
    $parsed = $null
    try { $parsed = $joined | ConvertFrom-Json } catch { $parsed = $null }
    if ($null -ne $parsed) { foreach ($x in $parsed) { $res += $x } }
    if ($res.Count -eq 0) {
        return [pscustomobject]@{
            score = 0; total = 5; unit = 'hidden_tests'
            items = @([pscustomobject]@{ test = 0; ok = $false; detail = ($joined -split "`n" | Select-Object -First 3) -join ' | ' })
            note = 'python nevyprodukoval JSON'
        }
    }
    $items = @(); $ok = 0
    for ($i = 0; $i -lt $res.Count; $i++) {
        if ($res[$i].ok) { $ok++ }
        $items += [pscustomobject]@{ test = $i; ok = [bool]$res[$i].ok; got = $res[$i].got }
    }
    return [pscustomobject]@{ score = $ok; total = 5; unit = 'hidden_tests'; items = $items; note = '' }
}

# ------------------------------------------------------------------ korpus pre vrstvu A
function New-PplChunks {
    $cfg = $doc.corpus
    $out = @()
    foreach ($dom in @('sk', 'py', 'data')) {
        $pool = @($pools.$dom)
        $perDom = [int][math]::Ceiling([int]$cfg.chunks / 3)
        for ($i = 0; $i -lt $perDom; $i++) {
            $sb = New-Object System.Text.StringBuilder
            [void]$sb.Append("[pbench $dom-$i] ")
            $j = 0
            while ($sb.Length -lt [int]$cfg.chars_per_chunk -and $j -lt ($pool.Count * 3)) {
                [void]$sb.Append($pool[($i * 3 + $j) % $pool.Count]).Append("`n")
                $j++
            }
            $out += [pscustomobject]@{
                id = "ppl_${dom}_$('{0:00}' -f $i)"; domain = $dom
                text = $sb.ToString().Trim()
            }
        }
    }
    return $out
}

# ------------------------------------------------------------------ logprobs (Gate 0 + vrstva A)
function Get-ChunkNll([string]$text) {
    # 1) OpenAI-kompatibilny /v1/completions s echo+logprobs
    try {
        $body = @{
            model = $script:model; prompt = $text; max_tokens = 1
            temperature = 0; logprobs = 1; echo = $true; stream = $false
        } | ConvertTo-Json -Depth 5
        $r = Post-JsonObj "$base/v1/completions" $body 900
        $lp = $null
        if ($r.choices -and $r.choices.Count -gt 0) { $lp = $r.choices[0].logprobs }
        if ($lp -and $lp.token_logprobs -and @($lp.token_logprobs).Count -gt 1) {
            $vals = @(@($lp.token_logprobs) | Where-Object { $null -ne $_ })
            if ($vals.Count -gt 1) {
                $sum = 0.0
                foreach ($v in $vals) { $sum += [double]$v }
                return [pscustomobject]@{ ok = $true; nll = (-$sum / $vals.Count); n = $vals.Count; method = 'openai:echo' }
            }
        }
    } catch { }
    # 2) llama.cpp native /completion s n_probs
    try {
        $body = @{ prompt = $text; n_predict = 0; n_probs = 1; temperature = 0; cache_prompt = $false } | ConvertTo-Json -Depth 5
        $r = Post-JsonObj "$base/completion" $body 900
        $cps = @($r.completion_probabilities)
        if ($cps.Count -gt 1) {
            $sum = 0.0; $n = 0
            foreach ($cp in $cps) {
                $p = $null
                foreach ($pr in @($cp.probs)) {
                    if ([string]$pr.tok_str -eq [string]$cp.content) { $p = [double]$pr.prob; break }
                }
                if ($null -eq $p -and @($cp.probs).Count -gt 0) { $p = [double](@($cp.probs)[0].prob) }
                if ($null -ne $p -and $p -gt 0) { $sum += [math]::Log($p); $n++ }
            }
            if ($n -gt 1) { return [pscustomobject]@{ ok = $true; nll = (-$sum / $n); n = $n; method = 'native:n_probs' } }
        }
    } catch { }
    return [pscustomobject]@{ ok = $false; nll = $null; n = 0; method = 'none' }
}
function Test-Gate0 {
    $probe = [string]($pools.py[0])
    $res = Get-ChunkNll $probe
    return $res
}

# ------------------------------------------------------------------ doc builder pre vrstvu C
function New-PerfDoc([int]$words, [string]$tag) {
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append("[pbench $tag seed=0x$(([long]$words * 2654435761) % 4294967296 | ForEach-Object { $_.ToString('x8') }))] ")
    $pool = @($pools.sk)
    $n = 0; $i = 0
    while ($n -lt $words) {
        # salt: kazdy odsek je unikatny, aby conversation cache nemohla
        # vratit KV z ineho (kratsieho) dokumentu a "prefill" nebol 7 tokenov
        $s = $pool[$i % $pool.Count] + " [$tag-$i]"
        [void]$sb.Append($s).Append(' '); $n += @($s -split '\s+').Count; $i++
    }
    return $sb.ToString().Trim()
}

# ------------------------------------------------------------------ meranie requestu z logu
$script:model = $null
$script:log = $null
$script:logPos = 0
function Init-Log {
    $script:log = (Get-ChildItem (Join-Path $env:LOCALAPPDATA 'Strata\launch') -Filter '*.engine.log' |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    $script:logPos = (Get-Content $script:log).Count
}
function Read-LogDelta {
    $all = @(Get-Content $script:log)
    if ($all.Count -gt $script:logPos) { $new = $all[$script:logPos..($all.Count - 1)] } else { $new = @() }
    $script:logPos = $all.Count
    $row = [ordered]@{
        prompt_tok = $null; reused_tok = $null; prefill_tps = $null
        gen_tok = $null; decode_tps = $null; drafts = ''
        skip_parking = @($new | Select-String 'skip parking').Count
        park_note = $null
    }
    $pl = ($new | Select-String 'strata serve: prompt ' | Select-Object -Last 1).Line
    if ($pl -match 'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)') {
        $row.prompt_tok = [int]$Matches[1]; $row.reused_tok = [int]$Matches[2]
        $row.prefill_tps = [double]$Matches[5]; $row.gen_tok = [int]$Matches[6]
        $row.decode_tps = [double]$Matches[8]; $row.drafts = "$($Matches[9])/$($Matches[10])"
    }
    $cl = ($new | Select-String 'conversation cache: (parked|skip parking)' | Select-Object -Last 1).Line
    if ($cl -match 'conversation cache: (.+)$') { $row.park_note = $Matches[1] }
    return [pscustomobject]$row
}

function Send-One([string]$text, [int]$maxTokens, [bool]$thinking = $false) {
    $body = @{
        model = $script:model
        messages = @(@{ role = 'user'; content = $text })
        max_tokens = $maxTokens
        temperature = 0
        stream = $false
        chat_template_kwargs = @{ enable_thinking = $thinking }
    } | ConvertTo-Json -Depth 8
    $free0 = FreeRam
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r = Post-JsonObj $uriChat $body 1800
    $sw.Stop()
    Start-Sleep -Milliseconds 600
    $m = Read-LogDelta
    $content = ''; $reasoning = ''; $finish = ''
    if ($r.choices -and $r.choices.Count -gt 0) {
        $finish = [string]$r.choices[0].finish_reason
        if ($null -ne $r.choices[0].message.content) { $content = [string]$r.choices[0].message.content }
        if ($null -ne $r.choices[0].message.reasoning_content) { $reasoning = [string]$r.choices[0].message.reasoning_content }
    }
    $v = VramNow
    return [pscustomobject]@{
        content = $content; reasoning_chars = $reasoning.Length; finish = $finish
        req_s = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        prompt_tok = $m.prompt_tok; reused_tok = $m.reused_tok; prefill_tps = $m.prefill_tps
        gen_tok = $m.gen_tok; decode_tps = $m.decode_tps; drafts = $m.drafts
        skip_parking = $m.skip_parking; park_note = $m.park_note
        free_before = $free0; free_after = (FreeRam)
        vram_used_mib = $v.used_mib; vram_free_mib = $v.free_mib
    }
}

# ------------------------------------------------------------------ konfiguracia a model
function Get-ConfigSnapshot {
    # Zdroj pravdy = skutocny command line strata.exe. Launch JSONy sa medzi
    # behmi lisia strukturou aj volatilnymi poliami (engine_ready_s), takze
    # hash z nich by dal falesny CONFOUND.
    # POZOR: launcher meni PORADIE flagov podla modelu (--vision pred/za
    # --kv-resident), preto sa hash pocita z kanonickej formy: flagy sa
    # rozdelia a ZORADIA. Inak by rovnaka konfiguracia dala iny hash.
    $cmd = ''
    try {
        $p = Get-CimInstance Win32_Process -Filter "Name='strata.exe'" | Select-Object -First 1
        if ($p) { $cmd = [string]$p.CommandLine }
    } catch { }
    if (-not $cmd) { $cmd = '(strata.exe nenajdeny)' }
    $norm = $cmd -replace '(?i)iq3_s|iq3_xxs', 'PACK'
    $norm = $norm -replace '(?i)C:\\STRATA-Model\\(models|packs)\\\S+', 'PACK'
    $norm = ($norm -replace '\s+', ' ').Trim()

    $tokens = @($norm -split ' --')
    $canon = @($tokens[0].Trim())
    for ($i = 1; $i -lt $tokens.Count; $i++) { $canon += ('--' + $tokens[$i].Trim()) }
    $sorted = @($canon[1..($canon.Count - 1)] | Sort-Object)
    $canonical = $canon[0] + "`n" + ($sorted -join "`n")

    $sha = [BitConverter]::ToString(
        [Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical))
    ).Replace('-', '').ToLowerInvariant()

    $jf = Get-ChildItem (Join-Path $env:LOCALAPPDATA 'Strata\launch') -Filter '*.json' |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $eng = ''
    if ($script:log) {
        $eng = ((Get-Content $script:log | Select-String 'engine started:' | Select-Object -Last 1).Line)
    }
    return [pscustomobject]@{
        hash = $sha
        launch_file = $(if ($jf) { $jf.Name } else { '' })
        engine_line = $eng
        launch_pairs = $canonical
    }
}
function Resolve-Expected {
    if ($ExpectModel) { return $ExpectModel }
    if ($RunId -match '(?i)xxs') { return 'iq3_xxs' }
    return 'iq3_s'
}

# ------------------------------------------------------------------ SELFTEST (bez engine)
if ($SelfTest) {
    '=== SELFTEST (ziadne volanie na engine) ==='
    $bad = 0
    function Check([string]$name, [bool]$cond, [string]$detail = '') {
        if ($cond) { '  ' + $name.PadRight(26) + ' PASS ' + $detail }
        else { '  ' + $name.PadRight(26) + ' FAIL ' + $detail; $script:bad++ }
    }

    $t1 = New-NumericTask
    $t1b = New-NumericTask
    Check 'numeric_deterministic' (($t1.prompt -eq $t1b.prompt) -and (@($t1.items).Count -eq 40)) "items=$(@($t1.items).Count)"
    $perfect = (@($t1.items) | ForEach-Object { "$($_.n)=$($_.ans)" }) -join "`n"
    $s1 = Score-Numeric $t1 $perfect
    Check 'numeric_perfect=40/40' ($s1.score -eq 40 -and $s1.total -eq 40) "$($s1.score)/$($s1.total)"
    $shifted = (@($t1.items) | ForEach-Object { "$($_.n)=$([long]$_.ans + 1)" }) -join "`n"
    $s1b = Score-Numeric $t1 $shifted
    Check 'numeric_shifted=0/40' ($s1b.score -eq 0) "$($s1b.score)/$($s1b.total)"
    $partial = (@($t1.items[0..19]) | ForEach-Object { "$($_.n)=$($_.ans)" }) -join "`n"
    $s1c = Score-Numeric $t1 $partial
    Check 'numeric_partial=20/40' ($s1c.score -eq 20) "$($s1c.score)/$($s1c.total)"

    $t2 = New-RecallTask 0
    $t2b = New-RecallTask 0
    Check 'recall_deterministic' ($t2.prompt -eq $t2b.prompt) "chars=$($t2.prompt.Length)"
    $s2 = Score-Recall $t2 $t2.target
    Check 'recall_exact_wer=0' ($s2.wer -le 0.02) "wer=$($s2.wer)"
    $s2b = Score-Recall $t2 'RECORD DELTA: neviem, v texte je vela podobnych zaznamov.'
    Check 'recall_wrong_detected' ($s2b.wer -gt 0.5) "wer=$($s2b.wer)"
    $s2c = Score-Recall $t2 ($t2.target -replace '88-401-B', '88-410-B')
    $negOk = @($s2c.items | Where-Object { $_.name -eq 'no_decoy_digits' -and -not $_.ok }).Count -eq 1
    Check 'recall_decoy_digits_flagged' $negOk "wer=$($s2c.wer)"
    $allOk = @($s2.items | Where-Object { -not $_.ok }).Count -eq 0
    Check 'recall_all_items_true' $allOk (($s2.items | ForEach-Object { "$($_.name)=$($_.ok)" }) -join ' ')
    $t2n = New-RecallTask 1
    $s2n = Score-Recall $t2n $t2n.target
    $allOk2 = @($s2n.items | Where-Object { -not $_.ok }).Count -eq 0
    Check 'recall_needle2_items' $allOk2 (($s2n.items | ForEach-Object { "$($_.name)=$($_.ok)" }) -join ' ')

    $t3 = New-TransformTask 0
    $t3b = New-TransformTask 0
    Check 'transform_deterministic' ($t3.prompt -eq $t3b.prompt) "rows=$(@($t3.rows).Count)"
    $goodJson = (@($t3.rows) | ForEach-Object { [pscustomobject]@{ id = $_.id; kod = $_.kod; mnozstvo = $_.mnozstvo; cena = $_.cena; datum = $_.datum } }) | ConvertTo-Json -Depth 4
    $s3 = Score-Transform $t3 $goodJson
    Check 'transform_perfect=300/300' ($s3.score -eq 300) "$($s3.score)/$($s3.total)"
    $badRows = @($t3.rows) | ForEach-Object { [pscustomobject]@{ id = $_.id; kod = 'XX-0000'; mnozstvo = $_.mnozstvo; cena = $_.cena; datum = $_.datum } }
    $s3b = Score-Transform $t3 ($badRows | ConvertTo-Json -Depth 4)
    Check 'transform_bad_kod=240/300' ($s3b.score -eq 240) "$($s3b.score)/$($s3b.total)"
    $s3c = Score-Transform $t3 'neviem'
    Check 'transform_garbage=0' ($s3c.score -eq 0) "$($s3c.score)/$($s3c.total)"
    $trunc = '[{"id":1,"kod":"AB-1042","mnozstvo":14,"cena":129.5,"datum":"2026-01-04"},{"id":2,"kod":"CD-22'
    $s3d = Score-Transform $t3 $trunc
    Check 'transform_truncated_partial' ($s3d.note -like '*rows_got=1*') "$($s3d.score)/$($s3d.total) $($s3d.note)"

    $t4 = New-CodeTask 0
    $good = "def sliding_max(vals, k):`n    return [max(vals[i:i+k]) for i in range(len(vals)-k+1)]"
    $d4 = Join-Path $tmp 'selftest_code'
    New-Item -ItemType Directory -Force -Path $d4 | Out-Null
    $s4 = Score-Code $t4 ('```python' + "`n" + $good + "`n" + '```') $d4
    Check 'code_good=5/5' ($s4.score -eq 5) "$($s4.score)/$($s4.total) $($s4.note)"
    $buggy = "def sliding_max(vals, k):`n    out = []`n    for i in range(len(vals) - k):`n        out.append(max(vals[i:i+k]))`n    return out"
    $s4b = Score-Code $t4 $buggy $d4
    Check 'code_buggy_lt_5' ($s4b.score -lt 5) "$($s4b.score)/$($s4b.total)"

    $c1 = New-PplChunks; $c2 = New-PplChunks
    $same = ($c1.Count -eq $c2.Count) -and (@(Compare-Object @($c1 | ForEach-Object { $_.id + '|' + $_.text }) @($c2 | ForEach-Object { $_.id + '|' + $_.text })).Count -eq 0)
    Check 'chunks_deterministic' $same "chunks=$($c1.Count) chars=$((@($c1 | ForEach-Object { $_.text.Length }) | Measure-Object -Sum).Sum)"
    $domOk = (@($c1 | Group-Object domain | ForEach-Object { $_.Count }) -join ',') -eq '8,8,8'
    Check 'chunks_3x8' $domOk ((@($c1 | Group-Object domain | ForEach-Object { "$($_.Name)=$($_.Count)" })) -join ' ')

    $d1 = New-PerfDoc 500 'C-x'; $d2 = New-PerfDoc 500 'C-x'
    Check 'perfdoc_deterministic' ($d1 -eq $d2) "chars=$($d1.Length)"

    # ordered dictionary zapis (takto runner uklada vysledky)
    $od = [ordered]@{ results = @(); gate0 = $null }
    $od.results += [pscustomobject]@{ a = 1 }
    $od.results += [pscustomobject]@{ a = 2 }
    $od.gate0 = [pscustomobject]@{ ok = $true }
    Check 'ordered_dict_append' (@($od.results).Count -eq 2) "count=$(@($od.results).Count)"
    Check 'ordered_dict_set' ($od.gate0.ok -eq $true) ''

    # parsovanie engine logu na syntetickych riadkoch (bez engine)
    $fakeLog = Join-Path $tmp 'fake.engine.log'
    $sample = @(
        '2026-11-27T10:00:00Z strata serve: prompt 1234 tokens = 100 reused + 1134 read in 400 ms (3085.0 tok/s), 42 generated in 500 ms (84.0 tok/s), drafts accepted 30 of 40',
        '2026-11-27T10:00:01Z conversation cache: parked 1000 tokens snapshot_bytes=12345'
    )
    Set-Content -Path $fakeLog -Value $sample -Encoding UTF8
    $script:log = $fakeLog; $script:logPos = 0
    $m = Read-LogDelta
    Check 'log_prompt_tok' ($m.prompt_tok -eq 1234) "prompt=$($m.prompt_tok)"
    Check 'log_reused_tok' ($m.reused_tok -eq 100) "reused=$($m.reused_tok)"
    Check 'log_prefill_tps' ([double]$m.prefill_tps -eq 3085.0) "prefill=$($m.prefill_tps)"
    Check 'log_decode_tps' ([double]$m.decode_tps -eq 84.0) "decode=$($m.decode_tps)"
    Check 'log_drafts' ($m.drafts -eq '30/40') "drafts=$($m.drafts)"
    Check 'log_park_note' ($m.park_note -like '*parked 1000 tokens*') "park=$($m.park_note)"
    Check 'log_skip_parking' ($m.skip_parking -eq 0) "skip=$($m.skip_parking)"
    $script:log = $null

    '  ----'
    if ($bad -eq 0) { 'SELFTEST: OK (0 zlyhani)' } else { "SELFTEST: $bad zlyhani"; exit 1 }
    return
}

# ------------------------------------------------------------------ dry run (nic neposle)
if ($NoRun) {
    'DRY RUN: vsetko sa postavi, nic sa neposle na engine'
    $t = New-NumericTask
    '  B1: items={0} chars={1} ~tok={2}' -f @($t.items).Count, $t.prompt.Length, [math]::Round($t.prompt.Length / 3.4)
    for ($r = 0; $r -lt [int]$doc.recall.reps; $r++) {
        $t = New-RecallTask $r
        '  B2 r{0}: chars={1} ~tok={2} (doc_words={3})' -f ($r + 1), $t.prompt.Length, [math]::Round($t.prompt.Length / 3.4), $doc.recall.doc_words
    }
    for ($r = 0; $r -lt [int]$doc.transform.reps; $r++) {
        $t = New-TransformTask $r
        '  B3 r{0}: rows={1} chars={2} ~tok={3}' -f ($r + 1), @($t.rows).Count, $t.prompt.Length, [math]::Round($t.prompt.Length / 4.0)
    }
    if ($WithCode) {
        for ($r = 0; $r -lt [int]$doc.code.reps; $r++) {
            $t = New-CodeTask $r
            '  B4 r{0}: chars={1} ~tok={2}' -f ($r + 1), $t.prompt.Length, [math]::Round($t.prompt.Length / 4.0)
        }
    }
    $chunks = New-PplChunks
    $tc = (@($chunks | ForEach-Object { $_.text.Length }) | Measure-Object -Sum).Sum
    '  A: chunks={0} total_chars={1} ~tok={2}' -f $chunks.Count, $tc, [math]::Round($tc / 3.2)
    foreach ($spec in @(@{ id = 'C5k'; words = 2400 }, @{ id = 'C32k'; words = 15000 }, @{ id = 'C128k'; words = 60000 }, @{ id = 'C240k'; words = 113000 })) {
        $d = New-PerfDoc ([int]$spec.words) $spec.id
        '  {0}: words={1} chars={2} ~tok={3}' -f $spec.id, $spec.words, $d.Length, [math]::Round($d.Length / 3.2)
    }
    'DRY RUN OK'
    return
}

# ------------------------------------------------------------------ hlavny beh
if ($LeadInSec -gt 0) {
    "cakam $LeadInSec s na docitanie vlastnej generacie..."
    Start-Sleep -Seconds $LeadInSec
}

$expect = Resolve-Expected
$models = Invoke-RestMethod -Uri $uriModels -TimeoutSec 10
$script:model = [string]$models.data[0].id
"{0}" -f "engine servíruje: $($script:model) (ocakavane: *$expect*)"
if ($script:model -notlike "*$expect*") {
    "STOP: engine neservíruje ocakavany model. Restartuj preset na pack '$expect' a spusti znova."
    exit 2
}
Init-Log
$cfgSnap = Get-ConfigSnapshot
"cmdline_hash: $($cfgSnap.hash)"

$res = [ordered]@{
    run_id = $RunId; round = $(if ($Round) { $Round } else { $RunId }); layer = $Layer
    model_id = $script:model; expected_model = $expect
    engine_line = $cfgSnap.engine_line; cmdline_hash = $cfgSnap.hash
    config_canonical = $cfgSnap.launch_pairs
    launch_file = $cfgSnap.launch_file; started_utc = (Get-Date).ToUniversalTime().ToString('s') + 'Z'
    gate0 = $null; results = @()
}

# fixny warmup: 3 male requesty, do skore nevstupuju (naplni expert cache rovnako)
'warmup: 3 requesty'
for ($i = 1; $i -le 3; $i++) {
    $w = Send-One 'Odpovedz jednym slovom: ano' 8
    "  warmup $i/3 decode=$($w.decode_tps) tok/s"
}

if ($Layer -in @('A', 'all')) {
    $g = Test-Gate0
    $res.gate0 = [pscustomobject]@{ ok = $g.ok; method = $g.method; n = $g.n; nll = $g.nll }
    if (-not $g.ok) {
        "Gate 0: engine nevratil token logprobs -> vrstva A sa preskakuje."
    } else {
        "Gate 0 OK ($($g.method))."
        $chunks = New-PplChunks
        $i = 0
        foreach ($c in $chunks) {
            $i++
            Write-Progress -Activity 'Vrstva A: perplexita' -Status "$i/$($chunks.Count) $($c.id)" -PercentComplete (100 * $i / $chunks.Count)
            $nll = Get-ChunkNll $c.text
            $ppl = $null
            if ($nll.ok) { $ppl = [math]::Exp($nll.nll) }
            $res.results += [pscustomobject]@{
                task = 'A_ppl'; id = $c.id; domain = $c.domain; rep = 0
                ok = [bool]$nll.ok; nll = $nll.nll; ppl = $ppl; tokens = $nll.n
                method = $nll.method; chars = $c.text.Length
            }
        }
        Write-Progress -Activity 'Vrstva A' -Completed
    }
}

if ($Layer -in @('B', 'all')) {
    $repsNum = if ($Reps -gt 0) { $Reps } else { [int]$doc.numeric.reps }
    $repsRec = if ($Reps -gt 0) { $Reps } else { [int]$doc.recall.reps }
    $repsTr = if ($Reps -gt 0) { $Reps } else { [int]$doc.transform.reps }
    $repsCode = if ($Reps -gt 0) { $Reps } else { [int]$doc.code.reps }

    "B1 cisla: $repsNum x 40 poloziek"
    for ($r = 1; $r -le $repsNum; $r++) {
        $t = New-NumericTask ($r - 1)
        $a = Send-One $t.prompt $t.max_tokens
        $s = Score-Numeric $t $a.content
        $res.results += [pscustomobject]@{
            task = 'B1'; id = $t.id; rep = $r
            score = $s.score; total = $s.total; unit = $s.unit; note = $s.note
            items = @($s.items | ForEach-Object { [pscustomobject]@{ n = $_.n; ok = $_.ok; expected = $_.expected; got = $_.got } })
            prompt_tok = $a.prompt_tok; decode_tps = $a.decode_tps; prefill_tps = $a.prefill_tps
            req_s = $a.req_s; gen_tok = $a.gen_tok; finish = $a.finish
            skip_parking = $a.skip_parking; park_note = $a.park_note
            vram_used_mib = $a.vram_used_mib
        }
        "  B1 r$r -> $($s.score)/$($s.total)  decode=$($a.decode_tps) tok/s  $($s.note)"
    }

    "B2 recitacia s near-miss: $repsRec x"
    for ($r = 0; $r -lt $repsRec; $r++) {
        $t = New-RecallTask $r
        $a = Send-One $t.prompt $t.max_tokens
        $s = Score-Recall $t $a.content
        $res.results += [pscustomobject]@{
            task = 'B2'; id = $t.id; rep = ($r + 1)
            score = $s.score; total = $s.total; unit = $s.unit; wer = $s.wer; note = $s.note
            items = @($s.items | ForEach-Object { [pscustomobject]@{ name = $_.name; ok = $_.ok } })
            answer = $a.content.Trim()
            prompt_tok = $a.prompt_tok; decode_tps = $a.decode_tps; prefill_tps = $a.prefill_tps
            req_s = $a.req_s; gen_tok = $a.gen_tok; finish = $a.finish
            skip_parking = $a.skip_parking; park_note = $a.park_note
            vram_used_mib = $a.vram_used_mib
        }
        "  B2 r$($r+1) -> wer=$($s.wer)  decode=$($a.decode_tps) tok/s"
    }

    "B3 tabulka->JSON: $repsTr x"
    for ($r = 0; $r -lt $repsTr; $r++) {
        $t = New-TransformTask $r
        $a = Send-One $t.prompt $t.max_tokens
        $s = Score-Transform $t $a.content
        $res.results += [pscustomobject]@{
            task = 'B3'; id = $t.id; rep = ($r + 1)
            score = $s.score; total = $s.total; unit = $s.unit; note = $s.note
            items = @($s.items | ForEach-Object { [pscustomobject]@{ row = $_.row; field = $_.field; ok = $_.ok } })
            prompt_tok = $a.prompt_tok; decode_tps = $a.decode_tps; prefill_tps = $a.prefill_tps
            req_s = $a.req_s; gen_tok = $a.gen_tok; finish = $a.finish
            skip_parking = $a.skip_parking; park_note = $a.park_note
            vram_used_mib = $a.vram_used_mib
        }
        "  B3 r$($r+1) -> $($s.score)/$($s.total) fieldov  $($s.note)"
        if ($a.finish -eq 'length') { "  POZOR: odpoved odrezana (max_tokens=$($t.max_tokens))" }
    }

    if ($WithCode) {
        "B4 dlhy kod: $repsCode x"
        for ($r = 0; $r -lt $repsCode; $r++) {
            $t = New-CodeTask $r
            $a = Send-One $t.prompt $t.max_tokens
            $d = Join-Path $tmp ("b4_r$r")
            New-Item -ItemType Directory -Force -Path $d | Out-Null
            $s = Score-Code $t $a.content $d
            $res.results += [pscustomobject]@{
                task = 'B4'; id = $t.id; rep = ($r + 1)
                score = $s.score; total = $s.total; unit = $s.unit; note = $s.note
                items = @($s.items | ForEach-Object { [pscustomobject]@{ test = $_.test; ok = $_.ok } })
                prompt_tok = $a.prompt_tok; decode_tps = $a.decode_tps; prefill_tps = $a.prefill_tps
                req_s = $a.req_s; gen_tok = $a.gen_tok; finish = $a.finish
                skip_parking = $a.skip_parking; park_note = $a.park_note
                vram_used_mib = $a.vram_used_mib
            }
            "  B4 r$($r+1) -> $($s.score)/$($s.total) testov  $($s.note)"
        }
    }
}

if ($Layer -in @('C', 'all')) {
    "C perf: 4 velkosti kontextu"
    foreach ($spec in @(
            @{ id = 'C5k'; words = 2400 }, @{ id = 'C32k'; words = 15000 },
            @{ id = 'C128k'; words = 60000 }, @{ id = 'C240k'; words = 113000 })) {
        $docText = New-PerfDoc ([int]$spec.words) $spec.id
        $q = "`n`nOtazka: ake je hlavne mesto Slovenska? Odpovedz jednym slovom."
        $a = Send-One ($docText + $q) 32
        $res.results += [pscustomobject]@{
            task = 'C'; id = $spec.id; rep = 1
            score = $null; total = $null; unit = 'perf'
            prompt_tok = $a.prompt_tok; reused_tok = $a.reused_tok; prefill_tps = $a.prefill_tps
            decode_tps = $a.decode_tps; gen_tok = $a.gen_tok; req_s = $a.req_s
            skip_parking = $a.skip_parking; park_note = $a.park_note
            vram_used_mib = $a.vram_used_mib; vram_free_mib = $a.vram_free_mib
            free_before = $a.free_before; free_after = $a.free_after; finish = $a.finish
        }
        $flag = ''
        if ($null -ne $a.reused_tok -and [int]$a.reused_tok -gt 0) { $flag = "  [REUSED $($a.reused_tok) tok -> prefill NEPLATNY]" }
        "  $($spec.id): prompt=$($a.prompt_tok) tok, prefill=$($a.prefill_tps) tok/s, decode=$($a.decode_tps) tok/s, $($a.req_s) s, skip_parking=$($a.skip_parking), vram=$($a.vram_used_mib) MiB$flag"
    }
}

# ------------------------------------------------------------------ zapis
$validity = [ordered]@{
    gate0_ok = $(if ($res.gate0) { $res.gate0.ok } else { $null })
    skip_parking_total = (@($res.results | ForEach-Object { [int]$_.skip_parking } | Measure-Object -Sum).Sum)
    decoded = (@($res.results | Where-Object { $_.decode_tps } | ForEach-Object { [double]$_.decode_tps }) | Measure-Object -Average -Minimum -Maximum)
}
$res['validity'] = $validity

$outDir = Join-Path $PSScriptRoot 'kvbench-results'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outFile = Join-Path $outDir ("iq3s-vs-xxs-$RunId-$Layer.json")
($res | ConvertTo-Json -Depth 12) | Set-Content -Path $outFile -Encoding UTF8
""
"zapisane: $outFile"
""
if ($validity.skip_parking_total -gt 0) {
    "POZOR: skip_parking=$($validity.skip_parking_total) — beh je podla metodiky neplatny pre rychlostnu cast."
}
