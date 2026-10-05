# Launcher consolidation pins (pre-change)

Captured on the RX 7700 XT (gfx1101) host at commit 05c7fed, before run2.sh/run3.sh were
removed, with `./run.sh --dry-run` etc. (plan: plans/merge-main-drop-ornith-single-run-launcher.md).

- pre-run-iq3xxs.txt   `./run.sh --dry-run`                  (IQ3_XXS, STRATA_EXPERT_CACHE=800)
- pre-run-iq3s.txt     `./run.sh --model IQ3_S --dry-run`    (IQ3_S,  STRATA_EXPERT_CACHE=800)
- pre-run2-iq3s.txt    `./run2.sh --dry-run`                 (IQ3_S,  STRATA_EXPERT_CACHE=680)
- pre-run2-iq3xxs.txt  `./run2.sh --model IQ3_XXS --dry-run` (IQ3_XXS, STRATA_EXPERT_CACHE=800)

`.norm` files strip ANSI colors and the image sha256 line.

Verified pre-change, measured on this host:
- `run.sh` (IQ3_XXS) and `run2.sh --model IQ3_XXS` are byte-identical after normalization.
- `run.sh --model IQ3_S` differs from `run2.sh` (IQ3_S) in exactly one field:
  STRATA_EXPERT_CACHE=800 (old run.sh, flat) vs 680 (run2.sh, per-quant tuning).

Post-change contract (enforced by docker/test_launcher_contract.py):
- `ls run*.sh` -> only `run.sh`.
- `./run.sh` (IQ3_XXS) keeps STRATA_EXPERT_CACHE=800.
- `./run.sh --model IQ3_S` reproduces the run2.sh IQ3_S line (STRATA_EXPERT_CACHE=680).

Post-change captures (same host, after run2.sh/run3.sh removal):
- post-run-iq3xxs.txt    `./run.sh --dry-run`                  -> byte-identical to pre-run-iq3xxs (normalized)
- post-run-iq3s.txt      `./run.sh --model IQ3_S --dry-run`    -> byte-identical to pre-run2-iq3s (normalized)
- post-run-iq3xxs2.txt   `./run.sh --model IQ3_XXS --dry-run`  -> byte-identical to pre-run2-iq3xxs (normalized)
Also checked on the host: `--expert-cache 900` on IQ3_S warns "tuned 680" and passes 900 through;
`--model ornith` is refused (exit 1) before docker; `python -m unittest discover -s docker` 13/13 OK.
