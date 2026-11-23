# The release test grid

Release 0.2.20 went out with a crash for everyone who ran an Unsloth or Q8_0 file on an NVIDIA card from the RTX 30 series up, and with several Inkling failures. No release check had ever started a model on each kind of card with each kind of model file and asked it a question. The grid does that, for every release candidate, before the release.

It is a list of cells. One cell is one release, one machine, one model and the drafter on or off. Every cell installs the release the way a user does, starts the model, talks to it, and writes down what happened. A table at the end says pass, fail or skip for every cell, with the numbers.

The scripts are in [`tools/release_grid/`](../tools/release_grid/). The cell list is data (`cells.json`), not code.

## What a cell checks

1. **Install and start through the user path.** The release's own `install.sh` goes into a fresh test HOME, then `strata launch --yes --no-tui --model-path ...` starts the model. The engine comes from the release's signed feed, exactly as for a user. A cell never uses a raw engine binary.
2. **First chat.** Six fixed questions at temperature 0 (capital of Australia, 17 x 23, the symbol for gold, days in a leap year, the largest planet, "strata" backwards). No reply may be empty, and at least five answers must be right.
3. **4K and 32K prompts.** A text of about 4 000 and about 30 000 tokens with one sentence hidden in the middle; the model must give back the hidden code. On a slow box the 32K step is skipped, with the projected time written in the table, when it would take longer than the step budget (15 minutes by default).
4. **Agent session.** About 10 turns with four tools (list, read, search, calculator). Fails on an empty reply, on a reply that runs to the token limit or repeats itself, on fewer than four turns answered right, and when the prompt cache is not reused from one turn to the next (the cached share of the previous turn's prompt, median over the session, must be at least 50%).
5. **Speed.** Decode and prefill tokens per second from the server's own timings, several repeats, with the spread. Compared with the previous release on the same box: fail when the candidate is more than 5% slower **beyond the spread of the two runs**.
6. **Greedy fingerprint.** A short hash of the six first-chat answers and one longer greedy reply. It must equal the previous release's, unless the change was announced (`--declared`, a file of `cell-id pattern: reason`).

Steps 5 and 6 need the previous release's result for the same cell. `grid.py run --prev v0.2.19` runs the previous release first on the same box, in the same job, so both numbers come from the same hour and the same card state. Without `--prev` the speed and fingerprint rows say "not compared".

The agent session has an opt-in 30-minute version; it is not in the default list.

## The cells

| Machine | Card class | Models it runs |
|---|---|---|
| `vast0221-3060` (rented) | NVIDIA sm_86, RTX 3060 12 GB | Qwen3.8-Flash-Next Unsloth UD-Q4_K_XL and the Q2_0 Strata pack; Inkling SQ-10 and the Unsloth UD-Q2_K_XL GGUF |
| `z440` | NVIDIA sm_86, 2x A4000 | skipped while its data disk has 8.8 GB free |
| `vast0221-4090b` (rented) | NVIDIA sm_89, RTX 4090 24 GB | Qwen UD-Q4_K_XL and Q2_0 pack; Inkling SQ-10, SQ-20 and UD-Q4_K_XL; Hy4-Flash SQ-64 (full tier) |
| `l620` | NVIDIA sm_120, RTX PRO 5000 | skipped: no internet and 4 GB free; needs a staged feed and a shared model copy |
| `boxia-7900xt` | AMD RDNA3 (gfx1100) | Inkling UD-IQ1_S GGUF and SQ-10 |
| `boxia-6700xt` | AMD RDNA2 (gfx1031) | Inkling UD-IQ1_S GGUF and SQ-10 |
| `p620` | AMD RDNA4 (gfx1201) | Qwen IQ3_S pack; Inkling SQ-10 |
| `p1006` | Intel Arc A750 (SYCL) | Inkling UD-IQ1_S GGUF and SQ-10 |
| `github-b70` | Intel Arc Pro B70 | through the GitHub runner: the coordinator pushes `ci/intel-runner` and starts it; `grid.py planfile` writes the plan it runs |
| `m1002` | Apple M5 Max (Metal) | skipped until its SQ-10 download is done; runs under `lockrun.py`, which holds both Mac locks |
| `p2006` | CPU only | Inkling SQ-10; the Q2_0 pack (full tier) |

Every Qwen and Inkling model runs with the drafter on (the default a user gets) and off (`--drafting off`). Hy4-Flash has one setting.

A skipped cell stays in the table with its reason. A box without room for a model is shown as skipped, not left out, so a gap in coverage is visible.

## How to run it

All commands run on the owner PC and start nothing heavy there; every engine run goes through `jobq` on a test box with that box's lock.

```
python tools/release_grid/grid.py plan    --cand v0.2.21-rc1=<feed> --prev v0.2.20          # what would run
python tools/release_grid/grid.py fetch   --boxes vast0221-3060                            # download pinned models where the cell says `fetch`
python tools/release_grid/grid.py run     --cand ... --prev v0.2.20 --boxes p620,p1006 --work OUTDIR
python jobq.py wait <job id>                                                               # for each job id it prints
python tools/release_grid/grid.py collect --cand ... --prev v0.2.20 --work OUTDIR
python tools/release_grid/grid.py summary --cand ... --prev v0.2.20 --work OUTDIR          # summary-<label>.md and .json
```

A release is a tag (`v0.2.20`: the GitHub release folder is its feed) or `LABEL=FEED`, where FEED is a URL, or a folder on the box that holds a test-signed feed (add `--trust <test-trust.json on the box>`). That is how a release candidate that is not published yet goes through the grid.

`run` makes one jobq job per cell (the previous and the candidate release back to back), so a box's lock is free again between cells and the other work on the box is not held up for hours.

## Models and disk

`models.lock.json` pins every model file by size and SHA-256. It is generated by `fetch_models.py pin` from the tables the product itself trusts (`terminal/setup.py`, `inkling_family.py`, `hy4_family.py`), so the grid checks the same bytes setup checks. `fetch` downloads with resume and deletes any file whose SHA-256 differs. It refuses to start when the disk would be left with less than 25 GB.

Several boxes are nearly full (boxia 7 GB free, p620 5 GB, l620 4 GB, z440 9 GB). A cell there points at a copy another lane already holds and the grid hard-links it into its own folder (no copy, nothing deleted). A box with no such copy gets a skip with the reason.

## Box rules the runner keeps

- One engine at a time, under the box's jobq `gpu` lock. On the Mac the job is wrapped in `lockrun.py`, which takes both locks and runs a watchdog.
- A memory guard runs beside every model: below the RAM floor, or when swap grows (on boxia by more than 30 MiB, the lender's guard stops at 64 MiB), it stops the server with `strata launch --stop` and the cell fails with the reason.
- Everything the grid writes is under `<grid_dir>` on the box (`/workspace/grid0221`, `/data/grid0221`, `~/grid0221`). Scripts go into a new `bin-<hash>` folder each time, never over a script a job is running.
- Only the server the cell started is stopped, with Strata's own `launch --stop`.

## How long it takes

Times from the dry run on 0.2.20 (2026-11-23) are in the results file; the budget per cell is 2 hours (2.5 on the slow profile), and most cells on a fast GPU take 10 to 25 minutes with the engine already installed. A cell is dominated by loading the model (a 100 GB Unsloth file read from disk on a 12 GB card takes many minutes) and, on the Intel and CPU boxes, by decode speed of 2 to 5 tokens per second. The first cell on a box also downloads the Terminal and the engine package (about 1 GB). A whole pass over every box is a few hours of wall time because the cells on different boxes run side by side.

## What the grid does not cover

- Windows (the owner PC, Strix): the runner is POSIX only.
- Two-card cells (z440, boston): one engine on one card per cell.
- Cards the lab does not have (Pascal, Volta, Turing, RDNA1, Arc B-series other than through the runner).
- Quality beyond the six questions and the needle: that is BASE-05 and the quality gates.
- The desktop app: the app host API is not driven yet; the Terminal path is the same engine start the app uses.
