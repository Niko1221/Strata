# kvbench_iq3s_vs_xxs_compare.ps1 — vyhodnotenie A1 vs B1 vs A2 podla metodiky.
#
#   powershell -File .\kvbench_iq3s_vs_xxs_compare.ps1 -A1 A1 -B1 B1 -A2 A2
#
# princip:
#   - A = IQ3_S (kola RoundA1 a RoundA2 spojene), B = IQ3_XXS (kolo RoundB1)
#   - rozdiel A1 vs A2 = sumova podlaha; Δ mensi ako podlaha nie je dokaz
#   - bootstrap 95% CI na rozdiel priemerov
#   - verdikt podla prahov z metodiky (§8)
# Pozor: mena parametrov su RoundA1/RoundB1/RoundA2, aby nekolidovali
# s premennymi $a1/$b1/$a2 (PowerShell je case-insensitive a typovany
# parameter by pretypoval objekt na string).
param(
    [string]$RoundA1 = 'A1',
    [string]$RoundB1 = 'B1',
    [string]$RoundA2 = 'A2',
    [double]$MinAccPP = 5.0,
    [double]$MinPplPct = 1.0,
    [int]$Iters = 2000,
    [string]$OutFile = '',
    [string]$ResultsDir = ''
)
$ErrorActionPreference = 'Stop'
$dir = $(if ($ResultsDir) { $ResultsDir } else { Join-Path $PSScriptRoot 'kvbench-results' })

function Find-Run([string]$id, [string]$layer) {
    $f = Join-Path $dir "iq3s-vs-xxs-$id-$layer.json"
    if (Test-Path $f) { return $f }
    $cand = Get-ChildItem $dir -Filter "iq3s-vs-xxs-$id-*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($cand) { return $cand.FullName }
    return $null
}
function Load-Run([string]$id) {
    $f = $null
    foreach ($l in @('all', 'B', 'C', 'A')) { $f = Find-Run $id $l; if ($f) { break } }
    if (-not $f) { throw "chyba: pre RunId '$id' nie je ziaden JSON v $dir" }
    return (Get-Content $f -Raw -Encoding UTF8 | ConvertFrom-Json)
}
function Boot-Diff($A, $B, [int]$iters) {
    $rnd = New-Object System.Random 12345
    $na = @($A).Count; $nb = @($B).Count
    if ($na -eq 0 -or $nb -eq 0) { return [pscustomobject]@{ mean = $null; lo = $null; hi = $null } }
    $d = New-Object 'double[]' $iters
    for ($k = 0; $k -lt $iters; $k++) {
        $sa = 0.0
        for ($i = 0; $i -lt $na; $i++) { $sa += [double]$A[$rnd.Next($na)] }
        $sb = 0.0
        for ($i = 0; $i -lt $nb; $i++) { $sb += [double]$B[$rnd.Next($nb)] }
        $d[$k] = ($sa / $na) - ($sb / $nb)
    }
    [Array]::Sort($d)
    $lo = $d[[int][math]::Floor(0.025 * ($iters - 1))]
    $hi = $d[[int][math]::Ceiling(0.975 * ($iters - 1))]
    $mean = ($d | Measure-Object -Average).Average
    return [pscustomobject]@{ mean = $mean; lo = $lo; hi = $hi }
}
function Boot-DiffTasks($pairs, [int]$iters) {
    # task-balanced bootstrap: v kazdej ulohe resampluje itemy, potom spriemeruje
    # ulohy rovnakou vahou (aby stropove ulohy neutopili signal z citlivych)
    $rnd = New-Object System.Random 12345
    $d = New-Object 'double[]' $iters
    for ($k = 0; $k -lt $iters; $k++) {
        $sum = 0.0; $cnt = 0
        foreach ($p in $pairs) {
            $A = @($p.A); $B = @($p.B)
            $na = $A.Count; $nb = $B.Count
            if ($na -eq 0 -or $nb -eq 0) { continue }
            $sa = 0.0
            for ($i = 0; $i -lt $na; $i++) { $sa += [double]$A[$rnd.Next($na)] }
            $sb = 0.0
            for ($i = 0; $i -lt $nb; $i++) { $sb += [double]$B[$rnd.Next($nb)] }
            $sum += ($sa / $na) - ($sb / $nb); $cnt++
        }
        if ($cnt -gt 0) { $d[$k] = $sum / $cnt } else { $d[$k] = 0.0 }
    }
    [Array]::Sort($d)
    return [pscustomobject]@{
        mean = ($d | Measure-Object -Average).Average
        lo = $d[[int][math]::Floor(0.025 * ($iters - 1))]
        hi = $d[[int][math]::Ceiling(0.975 * ($iters - 1))]
    }
}
function Get-Obs($doc, [string]$task) {
    # vrati pole hodnot 0..1 (item-level ak su itemy, inak skore na rep)
    $rows = @($doc.results | Where-Object { $_.task -eq $task })
    $vals = @()
    foreach ($r in $rows) {
        $items = @($r.items)
        if ($items.Count -gt 0) {
            foreach ($it in $items) { if ($it.ok) { $vals += 1.0 } else { $vals += 0.0 } }
        } elseif ($null -ne $r.score -and $null -ne $r.total -and [double]$r.total -gt 0) {
            $vals += ([double]$r.score / [double]$r.total)
        }
    }
    return $vals
}
function Fmt([double]$x, [int]$d = 3) { return $x.ToString("F$d", [Globalization.CultureInfo]::InvariantCulture) }
function Pct([double]$x) { return ($x * 100).ToString('0.0', [Globalization.CultureInfo]::InvariantCulture) + ' %' }

$a1 = Load-Run $RoundA1
$b1 = Load-Run $RoundB1
$a2 = Load-Run $RoundA2

$lines = New-Object System.Collections.Generic.List[string]
function Emit([string]$s) { $script:lines.Add($s) | Out-Null }

Emit '# IQ3_S vs IQ3_XXS — vysledok'
Emit ''
Emit ('kola: A1={0} (S), B1={1} (XXS), A2={2} (S).' -f $a1.run_id, $b1.run_id, $a2.run_id)
Emit ''
Emit '## 1. Platnost behu'
Emit ''
$hA1 = $a1.cmdline_hash; $hB1 = $b1.cmdline_hash; $hA2 = $a2.cmdline_hash
Emit ('| kolo | model | cmdline_hash |')
Emit ('|---|---|---|')
Emit ('| A1 | {0} | {1} |' -f $a1.model_id, $hA1)
Emit ('| B1 | {0} | {1} |' -f $b1.model_id, $hB1)
Emit ('| A2 | {0} | {1} |' -f $a2.model_id, $hA2)
Emit ''
$confound = ($hA1 -ne $hB1) -or ($hA1 -ne $hA2)
if ($confound) {
    Emit '**CONFOUND: cmdline_hash sa medzi kolami lisi — casova cast nie je porovnatelna s kvalitou.**'
} else {
    Emit 'cmdline_hash je vo vsetkych kolach rovnaky — jediny rozdiel je pack (kvantizacia vah).'
}
$g0 = @($a1.gate0, $b1.gate0, $a2.gate0) | Where-Object { $null -ne $_ }
$g0ok = (@($g0 | Where-Object { $_.ok }).Count -eq @($g0).Count) -and (@($g0).Count -gt 0)
Emit ('Gate 0 (token logprobs): {0}' -f $(if ($g0ok) { 'OK — vrstva A je pouzita' } else { 'NEDOSTUPNE — hodnoti sa len vrstva B' }))
Emit ''

# ---------------- vrstva A: perplexita
$pplRows = @{}
$pplDelta = @()
$a1ppl = @($a1.results | Where-Object { $_.task -eq 'A_ppl' })
$b1ppl = @($b1.results | Where-Object { $_.task -eq 'A_ppl' })
$a2ppl = @($a2.results | Where-Object { $_.task -eq 'A_ppl' })
if ($a1ppl.Count -gt 0 -and $b1ppl.Count -gt 0 -and $a2ppl.Count -gt 0) {
    Emit '## 2. Vrstva A — perplexita (nizsia = lepsia)'
    Emit ''
    Emit '| chunk | domena | A1 ppl | A2 ppl | B1 ppl | d(A1,A2) % | d(S,B) % |'
    Emit '|---|---|---:|---:|---:|---:|---:|'
    $relA = @(); $relB = @()
    foreach ($r in $a1ppl) {
        $x2 = @($a2ppl | Where-Object { $_.id -eq $r.id }) | Select-Object -First 1
        $y = @($b1ppl | Where-Object { $_.id -eq $r.id }) | Select-Object -First 1
        if (-not $x2 -or -not $y) { continue }
        $dNoise = 100.0 * ($x2.ppl - $r.ppl) / $r.ppl
        $sMean = ($r.ppl + $x2.ppl) / 2.0
        $dSB = 100.0 * ($y.ppl - $sMean) / $sMean
        $relA += $dNoise; $relB += $dSB
        Emit ('| {0} | {1} | {2} | {3} | {4} | {5} | {6} |' -f $r.id, $r.domain, (Fmt $r.ppl 2), (Fmt $x2.ppl 2), (Fmt $y.ppl 2), (Fmt $dNoise 2), (Fmt $dSB 2))
    }
    Emit ''
    $noiseA = (@($relA | Measure-Object -Average).Average)
    $bootP = Boot-Diff $relB $relA $Iters
    Emit ('Sumova podlaha A1 vs A2 (priemer |d|): {0} %' -f (Fmt ([math]::Abs($noiseA)) 2))
    Emit ('Rozdiel B - S relativne: {0} % [95% CI {1} .. {2} %]  (kladne = XXS ma vyssiu perplexiu, teda horsie)' -f (Fmt $bootP.mean 2), (Fmt $bootP.lo 2), (Fmt $bootP.hi 2))
    Emit ''
}

# ---------------- vrstva B: behaviorálne skore
Emit '## 3. Vrstva B — behaviorálne ulohy (1.0 = bez chyby)'
Emit ''
Emit '| uloha | A (S) priemer | A1 vs A2 (sum) | B (XXS) priemer | d (S - XXS) | 95% CI | n(S/XXS) |'
Emit '|---|---:|---:|---:|---:|---|---|'
$taskNames = @('B1', 'B2', 'B3', 'B4')
$agg = @{}
foreach ($t in $taskNames) {
    $vA1 = Get-Obs $a1 $t
    $vA2 = Get-Obs $a2 $t
    $vB = Get-Obs $b1 $t
    if (@($vB).Count -eq 0 -and @($vA1).Count -eq 0) { continue }
    $vA = @($vA1) + @($vA2)
    $mA = (@($vA | Measure-Object -Average).Average)
    $mB = (@($vB | Measure-Object -Average).Average)
    $mNoise = [math]::Abs((@($vA1 | Measure-Object -Average).Average) - (@($vA2 | Measure-Object -Average).Average))
    $boot = Boot-Diff $vA $vB $Iters
    $agg[$t] = [pscustomobject]@{ mA = $mA; mB = $mB; diff = $boot.mean; lo = $boot.lo; hi = $boot.hi; nA = @($vA).Count; nB = @($vB).Count; noise = $mNoise }
    Emit ('| {0} | {1} | {2} | {3} | {4} | {5} .. {6} | {7}/{8} |' -f `
        $t, (Fmt $mA 3), (Fmt $mNoise 3), (Fmt $mB 3), (Fmt $boot.mean 3), (Fmt $boot.lo 3), (Fmt $boot.hi 3), @($vA).Count, @($vB).Count)
}
Emit ''

# agregat: item-pooled (orientacny) + task-balanced (rozhodovaci)
$vAall = @(); $vBall = @()
$pairs = @()
foreach ($t in $agg.Keys) {
    $vA1 = Get-Obs $a1 $t; $vA2 = Get-Obs $a2 $t; $vB = Get-Obs $b1 $t
    $A = @($vA1) + @($vA2)
    $vAall += $A; $vBall += @($vB)
    $pairs += [pscustomobject]@{ task = $t; A = $A; B = @($vB) }
}
if ($vAall.Count -gt 0) {
    $bootAll = Boot-Diff $vAall $vBall $Iters
    $bootBal = Boot-DiffTasks $pairs $Iters
    $mAall = (@($vAall | Measure-Object -Average).Average)
    $mBall = (@($vBall | Measure-Object -Average).Average)
    Emit ('**Agregat item-pooled (orientacny, stropove ulohy dominuju): S = {0}, XXS = {1}, d = {2} [{3} .. {4}]**' -f (Fmt $mAall 3), (Fmt $mBall 3), (Fmt $bootAll.mean 3), (Fmt $bootAll.lo 3), (Fmt $bootAll.hi 3))
    Emit ('**Agregat task-balanced (rozhodovaci, kazda uloha rovnako): d = {0} [{1} .. {2}]**' -f (Fmt $bootBal.mean 3), (Fmt $bootBal.lo 3), (Fmt $bootBal.hi 3))
    Emit ''
}

# ---------------- vrstva C: cena
$cA1 = @($a1.results | Where-Object { $_.task -eq 'C' })
$cA2 = @($a2.results | Where-Object { $_.task -eq 'C' })
$cB = @($b1.results | Where-Object { $_.task -eq 'C' })
if ($cB.Count -gt 0 -and $cA1.Count -gt 0) {
    Emit '## 4. Vrstva C — cena (kontrola)'
    Emit ''
    Emit '| test | S: prompt / reused / prefill / decode | XXS: prompt / reused / prefill / decode |'
    Emit '|---|---|---|'
    foreach ($r in $cA1) {
        $y = @($cB | Where-Object { $_.id -eq $r.id }) | Select-Object -First 1
        if (-not $y) { continue }
        Emit ('| {0} | {1} / {2} / {3} / {4} | {5} / {6} / {7} / {8} |' -f $r.id, $r.prompt_tok, $r.reused_tok, (Fmt $r.prefill_tps 0), (Fmt $r.decode_tps 1), $y.prompt_tok, $y.reused_tok, (Fmt $y.prefill_tps 0), (Fmt $y.decode_tps 1))
    }
    Emit ''
    $reusedRows = @(@($cA1) + @($cA2) + @($cB) | Where-Object { [int]$_.reused_tok -gt 0 })
    if ($reusedRows.Count -gt 0) {
        Emit ('**POZOR: {0} z {1} merani vrstvy C malo reused_tok > 0 — prefill pre ne nie je platny (conversation cache).**' -f $reusedRows.Count, (@($cA1).Count + @($cA2).Count + @($cB).Count))
        Emit ''
    }
    $dS = @(@($cA1) + @($cA2) | Where-Object { $_.decode_tps } | ForEach-Object { [double]$_.decode_tps })
    $dX = @($cB | Where-Object { $_.decode_tps } | ForEach-Object { [double]$_.decode_tps })
    $mS = (@($dS | Measure-Object -Average).Average); $mX = (@($dX | Measure-Object -Average).Average)
    Emit ('Decode priemer: S = {0} tok/s, XXS = {1} tok/s ({2} % voci XXS).' -f (Fmt $mS 1), (Fmt $mX 1), (Fmt (100.0 * ($mS - $mX) / $mX) 1))
    Emit ''
}

# ---------------- verdikt
Emit '## 5. Verdikt podla pravidla (§8)'
Emit ''
$pplOk = $false; $pplPct = $null
if ($bootP -and $null -ne $bootP.mean) {
    $pplPct = $bootP.mean
    $pplOk = (($bootP.lo -gt 0) -and ($pplPct -ge $MinPplPct))
}

if ($confound) {
    Emit '**NEPLATNE:** cmdline_hash sa medzi kolami lisi; vysledok nie je mozne atribuovat kvantizacii.'
} else {
    $prim = $agg['B1']
    if ($prim) {
        $dpp = 100.0 * $prim.diff
        Emit ('Primarna metrika **B1 aritmeticka presnost** (n S/XXS = {0}/{1}): S = {2}, XXS = {3}, d = {4} pb [95% CI {5} .. {6} pb]' -f `
                $prim.nA, $prim.nB, (Fmt $prim.mA 3), (Fmt $prim.mB 3), (Fmt $dpp 1), (Fmt (100 * $prim.lo) 1), (Fmt (100 * $prim.hi) 1))
        Emit ''
        if ($prim.lo -gt 0 -and $dpp -ge $MinAccPP) {
            Emit '**IQ3_S je kvalitativne lepsia v primarnej metrike.**'
        } elseif ($prim.hi -lt 0 -and (-$dpp) -ge $MinAccPP) {
            Emit '**IQ3_S je horsia v primarnej metrike.**'
        } else {
            Emit ('**Primarna metrika: rozdiel sa nepreukazal** (treba CI nad nulou a >= +{0} pb).' -f (Fmt $MinAccPP 1))
        }
    } else {
        Emit 'B1 v danych chyba — primarnu metriku nie je mozne vyhodnotit.'
    }
    Emit ''
    $sec = @()
    foreach ($t in @('B2', 'B3', 'B4')) {
        if ($agg.ContainsKey($t)) {
            $sec += ('{0} d={1} pb [CI {2} .. {3}]' -f $t, (Fmt (100 * $agg[$t].diff) 1), (Fmt (100 * $agg[$t].lo) 1), (Fmt (100 * $agg[$t].hi) 1))
        }
    }
    if ($sec.Count -gt 0) { Emit ('Sekundarne (kontrola stropu): ' + ($sec -join '; ')) }
    if ($bootBal) {
        Emit ('Agregat task-balanced: d = {0} pb [CI {1} .. {2}]' -f (Fmt (100 * $bootBal.mean) 1), (Fmt (100 * $bootBal.lo) 1), (Fmt (100 * $bootBal.hi) 1))
    }
    if ($pplOk) { Emit ('- perplexita: XXS ma o {0} % vyssiu perplexiu [CI {1} .. {2} %]' -f (Fmt $pplPct 2), (Fmt $bootP.lo 2), (Fmt $bootP.hi 2)) }
    Emit '- cena: viz vrstva C vyssie.'
    Emit ''
    Emit 'Interpretacia: B1 je primarna, lebo aritmeticka presnost je podla metodiky najcitlivejsia na kvantizaciu vah. B2/B3/B4 su zaroven kontrolou stropu — ak su na strope u oboch, ich zhoda je ocakavana a nie je dokazom rovnosti kvality.'
}
Emit ''
Emit '## 6. Sumova podlaha'
Emit ''
Emit 'Rozdiel A1 vs A2 je to, co sa "pohne samo" na tom istom modeli. Kazdy d nizsi ako tato hodnota je sum, nie efekt.'
foreach ($t in $agg.Keys) {
    if ($agg[$t].noise -gt 0) { Emit ('- {0}: sumova podlaha {1}' -f $t, (Fmt $agg[$t].noise 3)) }
}
Emit ''

$text = ($lines -join "`n")
$text
if (-not $OutFile) { $OutFile = Join-Path $dir 'iq3s-vs-xxs-VERDICT.md' }
Set-Content -Path $OutFile -Value $text -Encoding UTF8
""
"zapisane: $OutFile"
