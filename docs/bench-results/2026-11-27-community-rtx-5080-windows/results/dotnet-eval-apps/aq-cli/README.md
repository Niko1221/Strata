# aq-cli — ukážka výstupu modelu

Tento `Program.cs` **napísal lokálny model** (Swift 1.5 IQ3_XXS) v úlohe `T4-aq-cli`
bez jedinej opravy — na prvý pokus, 3 z 3 replikácií. Je to surový výstup evalu,
nie ručne písaný kód.

## Spustenie

```powershell
cd C:\geminiSandbox\strata-local\dotnet-eval\samples\aq-cli
dotnet run                  # stiahne aktualne hodinove data pre Prahu
dotnet run -- --hours 3     # posledne 3 hodiny
dotnet run -- --json        # strojovy vystup
```

Alebo priamo bez SDK (framework-dependent, potrebuje .NET 10 runtime):

```powershell
.\aq.exe --hours 3
.\aq.exe --json
```

## Čo to robí

Číta hodinové dáta o kvalite ovzdušia pre Prahu z **Open-Meteo** (keyless, bez registrácie)
a vypíše tabuľku s pásmami EAQI, priemerom a maximom za zvolený počet hodín.

```
time pm10 pm2_5 aqi band
2026-11-28T00:00 27.4 22.1 44 moderate
2026-11-28T05:00 20.3 15.8 40 fair
...
```

## Prečo Open-Meteo a nie lkod.cz

Katalóg `lkod.cz` je keyless len na **metadáta** (413 datových sád, JSON-LD po jednom
zázname). Skutočné dáta pražského ovzdušia smerujú na `api.golemio.cz`, ktoré vracia
**401 bez tokenu** — takže ako reprodukovateľná závislosť pre eval nefunguje.
Open-Meteo je keyless, hodinový a vracia ~2 KB JSON.

## Poznámka

Testy (`T4-aq-cli/tests/Program.cs`) nikdy nevolajú sieť — pracujú s fixtures cez
`--file`, takže skóre je deterministické. Sieť sa použije len pri `dotnet run` bez
`--file`, teda pri ručnej ukážke.
