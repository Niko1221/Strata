# dotnet-eval-compare.ps1 - porovna dva behy evalu (napr. base vs swift).
#
# Pouzitie:
#   .\dotnet-eval-compare.ps1 -A base -B swift
param(
    [Alias('A')][string]$RunA = 'base',
    [Alias('B')][string]$RunB = 'swift'
)
$ErrorActionPreference = 'Stop'
$dir = Join-Path $PSScriptRoot 'dotnet-eval-results'

function Load-Run([string]$id) {
    $f = Join-Path $dir "dotnet-eval-$id.json"
    if (-not (Test-Path $f)) { throw "chyba: neexistuje $f" }
    return (Get-Content $f -Raw -Encoding UTF8 | ConvertFrom-Json)
}

$ra = Load-Run $RunA
$rb = Load-Run $RunB

"model A: $($ra.model)   model B: $($rb.model)"
""
'{0,-22} {1,8} {2,8} {3,10} {4,10} {5,9} {6,9}' -f 'uloha', 'A testy', 'B testy', 'A gen_tok', 'B gen_tok', 'A cas s', 'B cas s'

$ids = @($ra.results | ForEach-Object { $_.task } | Sort-Object -Unique)
foreach ($id in $ids) {
    $rowA = @($ra.results | Where-Object { $_.task -eq $id })[0]
    $rowB = @($rb.results | Where-Object { $_.task -eq $id })[0]
    if (-not $rowA -or -not $rowB) { continue }
    '{0,-22} {1,8} {2,8} {3,10} {4,10} {5,9} {6,9}' -f `
        $id, "$($rowA.passed)/$($rowA.total)", "$($rowB.passed)/$($rowB.total)", $rowA.gen_tok, $rowB.gen_tok, $rowA.req_s, $rowB.req_s
}
''
$ga = if ($null -ne $ra.green_results) { [int]$ra.green_results } else { @($ra.results | Where-Object { $_.green }).Count }
$gb = if ($null -ne $rb.green_results) { [int]$rb.green_results } else { @($rb.results | Where-Object { $_.green }).Count }
$ta = if ($null -ne $ra.tasks_all_green) { [int]$ra.tasks_all_green } else { 0 }
$tb = if ($null -ne $rb.tasks_all_green) { [int]$rb.tasks_all_green } else { 0 }
"green behy:  $RunA $ga/$($ra.total_results)   $RunB $gb/$($rb.total_results)"
"ulohy plne zelene: $RunA $ta/$($ra.tasks)   $RunB $tb/$($rb.tasks)"
"gen_tok: $RunA $($ra.gen_tok_total)   $RunB $($rb.gen_tok_total)"
"wall:    $RunA $($ra.total_wall_s) s   $RunB $($rb.total_wall_s) s"
''
if ($tb -gt $ta) { "Vysledok: $RunB je lepsi (viac plne zelenych uloh)." }
elseif ($tb -lt $ta) { "Vysledok: $RunA je lepsi (viac plne zelenych uloh)." }
elseif ($gb -gt $ga) { "Vysledok: rovnako zelenych uloh, $RunB ma viac zelenych behov." }
elseif ($gb -lt $ga) { "Vysledok: rovnako zelenych uloh, $RunA ma viac zelenych behov." }
elseif ([int]$rb.gen_tok_total -lt [int]$ra.gen_tok_total) { "Vysledok: rovnaka kvalita, $RunB minie menej tokenov (uspora $([math]::Round(100.0*($ra.gen_tok_total-$rb.gen_tok_total)/[math]::Max(1,$ra.gen_tok_total),1)) %)." }
else { "Vysledok: bez rozdielu v kvalite aj v tokenoch." }
