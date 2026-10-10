# What this folder does NOT include (and where it lives)

We ship the numbers that back every claim in the README/PDF, plus the pipeline that
rebuilds them. This is what we deliberately left out, so you know the package is a
curated subset, not the whole lab notebook.

## Excluded from `data/` on purpose

- **Gyro-S tool-eval trials** — excluded-with-finding (C1). The rc1 fork serve has no
  tool support, so Gyro's tool scenarios are not comparable; the proxy runs that produced
  them are also transport-contaminated. Gyro's agent columns are `n/a` in every table for
  this reason. Its HE+, knowledge, needle and five-bugs results ARE included (served clean).
- **Old-era perf files** (`iq3_s.perf.json`, `q2_0.perf.json`, `swift-iq2_xs*.perf.json`,
  `swift-iq3_xxs*.perf.json`) — pre-rebase speed runs on earlier engine builds, not the
  pinned 0.1.40.3 cohort. Kept only the canonical `rebase-*` and `campaign11-*` perf JSONs.
- **Gyro perf v1** (pre-vram-reserve) — superseded by `-v2` files; kept v2 only.

## Not shipped (large, on our machines, available on request)

- **`runs/` full run logs** (~41 MB of per-run markdown, incl. the 28 knowledge/needle
  reports — a curated copy of those IS in `data/knowledge-needle/`). The full HE+ per-task
  logs and tool-eval raw transcripts are on our rig; ask on the PR and we will attach a zip.
- **Model weights / packs** — not ours to redistribute; see the Hugging Face links in README.
- **Campaign driver scripts + systemd units** (`run_campaign*.sh`, `campaign_swap.ps1`) —
  rig-specific (Windows paths, SSH host aliases). The *measurement* pipeline is in `scripts/`.

## Reproducibility boundary

`scripts/extract_all.py → data/ALL-METRICS.json → make_charts.py (F1–F15) →
compile_pr_readme.py` rebuilds every figure and this README from the raw JSONs in `data/`.
The raw JSONs themselves are the ground truth; if a number in the prose disagrees with
`data/`, the data wins — open an issue.

Engine pinned `34CDE150B21148E6` · Strata `0.1.40.3` · tool-eval-bench `2.7.1.dev16+gf7c34a130`.
