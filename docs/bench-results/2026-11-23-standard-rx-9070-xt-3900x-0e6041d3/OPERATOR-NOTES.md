# Operator notes for run `20261123T045343Z-0e6041d3`

The generated `README.md`, `results.json`, `outputs.jsonl` and `engine.log` are unedited (the runner already replaced
the home folder with `<user>` in the log). Hardware, software and model are as described in
`../2026-11-23-standard-rx-9070-xt-3900x-bbb73d60/OPERATOR-NOTES.md`; `config.json` is byte-identical to the two earlier runs.

## How this run was made

- Command: `strata bench run --profile launch-iq3_s-256k --track default --track-notes "Strata 0.2.20 setup defaults, untuned, fresh engine (baseline before tune)"`.
  The runner started the engine itself, so this is a freshly loaded engine (no request before the run) and the engine
  log is included. This is the conforming default-track run the two earlier runs in this PR lacked.
- During the run: no other client used the server; a desktop session with a browser and idle terminal windows was open.

## Tuning after this run

Afterwards `strata tune auto launch-iq3_s-256k --mode thorough --use coding` ran on the same machine. It changed
nothing: the setup defaults were the fastest of the candidates tried. `strata tune report` output, verbatim:

```text
Your settings: answers at 67.0 tok/s, prompts read at 1979.9 tok/s (measured 4 times: 480.06, 474.91, 465.54, 458.53)
To count, a setting had to be at least +9.0% faster; identical runs on this PC differed by 1.1%.
What was tried:
  Keep more of the conversation in graphics memory  not faster by enough: -0.9% against your settings, faster in 1 of 3 rounds
  Run the control thread on the last CPU core  slower, stopped after one round
  Allocate the conversation memory as it grows  slower, stopped after one round
Left out:
  Re-balance the expert cache faster: it can change answers slightly; tick 'allow small answer differences' to try it
  Forget old expert use a little faster: it can change answers slightly; tick 'allow small answer differences' to try it
  Trim stage weights on two cards: not a hip setting
  Pin memory only when needed: not a hip setting
Next: Nothing was changed. Your settings are already the fastest of what was tried.
```

The tune's own summary also noted the PC was noisy (4.7% between identical runs), so a gain below that could be
hidden. Because no setting changed, no post-tune benchmark was run: it would measure the same configuration.
