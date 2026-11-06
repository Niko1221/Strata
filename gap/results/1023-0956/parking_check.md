# conversation parking A/B (quick: ON arm only)

engine config: /Users/pc/src/Strata-fork/strata-iq3_xxs.json; 3 main turns, 1 side requests between each, +2000 tokens/turn; parking ON = 8192 MiB, min-tokens 2048.

## parking ON

| turn | prompt | reused | prompt_ms | wall s |
|---|---|---|---|---|
| 1 | 4504 | 0 | 15091 | 15.5 |
| 2 | 6520 | 4499 | 6563 | 6.7 |
| 3 | 8516 | 6515 | 6553 | 6.7 |

cache: parks=2 restores=2 evictions=0 bytes=93503328

## verdict

- reuses the prefix after every side turn (ON, turns 2-3): PASS
- parked and restored at least once: PASS
