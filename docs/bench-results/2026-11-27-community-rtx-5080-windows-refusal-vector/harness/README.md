# Harness

Skripty, ktorými sa dajú čísla z `../README.md` zopakovať. Všetko je PowerShell
(Windows), bez závislostí okrem .NET SDK pri `dotnet-eval/`.

## Kvalita + rýchlosť dvoch packov

```
kvbench_iq3s_vs_xxs.ps1       merací runner (A1 / B1 / A2)
kvbench_iq3s_vs_xxs_compare.ps1  bootstrap CI, šumová podlahа, verdikt
iq3s_vs_xxs_tasks.json        úlohy, seedy, veľkosti korpusu
```

Postup (engine sa **medzi kolami reštartuje**, to je súčasť dizajnu):

```powershell
.\kvbench_iq3s_vs_xxs.ps1 -SelfTest          # overi generatory a skore bez modelu
.\kvbench_iq3s_vs_xxs.ps1 -RunId A1 -Round A1 -ExpectModel iq3_s   -Layer all -Reps 10 -WithCode
.\kvbench_iq3s_vs_xxs.ps1 -RunId B1 -Round B1 -ExpectModel iq3_xxs -Layer all -Reps 10 -WithCode
.\kvbench_iq3s_vs_xxs.ps1 -RunId A2 -Round A2 -ExpectModel iq3_s   -Layer all -Reps 10 -WithCode
.\kvbench_iq3s_vs_xxs_compare.ps1 -RoundA1 A1 -RoundB1 B1 -RoundA2 A2 -Iters 4000
```

`-ExpectModel` zastaví beh, ak beží iný pack, než kolo očakáva — bez toho sa dá
ľahko zmerať ten istý model dvakrát.

## .NET eval

```
dotnet-eval\dotnet-eval.ps1              runner (4 ulohy, build + skryte testy)
dotnet-eval\dotnet-eval-compare.ps1      porovnanie dvoch behov
dotnet-eval\tasks\                       T1..T4 (template / reference / tests)
```

```powershell
.\dotnet-eval\dotnet-eval.ps1 -SelfTest              # referencia prejde, sablona nie
.\dotnet-eval\dotnet-eval.ps1 -RunId base -Reps 3
.\dotnet-eval\dotnet-eval-compare.ps1 -A base -B swift
```

`-SelfTest` je dôležitý: overí, že referenčné riešenie prejde na plný počet a
šablóna zlyhá. Bez toho by test, ktorý prejde aj na pokazenom kóde, nič nemeral.

## Dve pasce, ktoré harness rieši (a ktoré treba vedieť pri čítaní čísel)

1. **KV reuse.** Každý odsek perf dokumentu má soľ (`[C32k-17]`), aby conversation
   cache nevrátila prompt z predchádzajúceho behu. Runner navyše číta `reused` z
   engine logu a označí beh, kde `reused > 0`.
2. **Config hash.** Hash sa počíta zo **zoradených** flagov, lebo launcher mení
   poradie `--vision` / `--kv-resident` podľa modelu a naivný hash by hlásil
   CONFOUND pre dve identické konfigurácie.

## Sanitizácia

Skripty vypisujú cesty typu `C:\Users\<user>\...` do konzoly. V `results/` sú len
dáta bez ciest; pred odoslaním skontroluj:

```powershell
Select-String -Path .\results\*,.\harness\* -Pattern 'Users\\|\\AppData\\|192\.168\.'
```
