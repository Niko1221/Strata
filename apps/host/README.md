# Strata app source

Start with [the complete developer handoff](../handoff.md) and [the runbook](../docs/RUNBOOK.md).

```sh
npm ci
npm run preview:ui
```

The built UI is included. To rebuild after source edits, run `npm run build`. `preview:ui` uses isolated simulated data; `npm run dev` connects the normal host.

The previous project README is retained in [docs/README-historical.md](docs/README-historical.md). It contains older implementation status and is historical reference.

## Languages

The app speaks English, Chinese, Japanese, German, French, Spanish, Russian, Portuguese and Korean. The English text is the key: `src/locales/en.json`
(with `fix-en.json`, `tune-en.json` and `tools-en.json`) holds every string, and each language has the same four files (`fr.json`, `fix-fr.json`, ...).

- **A new string.** Write it with `tr("...")`, then add it to the English catalog (`node scripts/append-localization.mjs`, or the next `fx` number in
  `scripts/make-fix-locales.py`). The language tests fail until every language has it.
- **French.** `python scripts/fr_sync.py missing todo.tsv` lists the English strings that have no French yet, one per line (`id`, a tab, the text).
  Translate the second column ("vous", short plain words, one next step; a lone `=` keeps the English), then `python scripts/fr_sync.py apply todo.tsv` merges it
  (it also sets the French spacing before `: ; ? !` and the typographic apostrophe) and `python scripts/fr_sync.py check` confirms nothing is missing.
- **Screenshots in a language.** `npm run build`, then `node scripts/language-shots.mjs fr <folder> models settings setup` (hermetic preview host, scratch profile, headless Chrome).
