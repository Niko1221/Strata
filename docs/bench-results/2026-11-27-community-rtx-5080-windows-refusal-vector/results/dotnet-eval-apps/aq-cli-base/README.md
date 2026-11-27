# aq-cli-base — čo napísal base model (nefunkčné)

Toto je **surový výstup base modelu** (`qwen3.8-flash-next-iq3_xxs`) na tú istú úlohu
`T4-aq-cli`, kde Swift 1.5 dosiahol 10/10 na prvý pokus. Base dostal **2/10**
(rep1 a rep3), pričom tie dva „prešli" len preto, že vracia exit 2 na akýkoľvek
argument — testy `--hours 0 -> exit 2` a `chybajuci subor -> exit 2` splní náhodou.

## Prečo to nefunguje

Pri dvoch replikáciách z troch model **nespotreboval hodnotu `--file`**:

```
[exit 2] Error: unknown argument 'C:\...\aq-fixture-....json'
```

Pri tretej replikácii nepreložil:

```
Program.cs(48,12): error CS0165: Use of unassigned local variable 'doc'
```

Opravné kolo (`repair: 1`) to nezachránilo.

## Skúsenosť

```powershell
cd C:\geminiSandbox\strata-local\dotnet-eval\samples\aq-cli-base
dotnet run -- --hours 3
# -> Error: unknown argument '3'  (model nepozna hodnotu prepinaca)
```

Ponechané zámerne: je to dôkaz k tvrdeniu z `06-t4-app-eval.md`. Porovnaj so
`samples\aq-cli\`, kde je tá istá úloha vyriešená správne.
