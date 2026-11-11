# The scripts of this report

Four stdlib-only Python 3 files, each runnable from the folder above. They are the tools that produced the
numbers here; the raw material they read (and wrote) is in `../raw/`.

| script | what it does | needs |
|---|---|---|
| `paired_ci.py` | turns a sweep's JSONL into the paired medians with bootstrap CIs that every `+x %` in the README quotes | the JSONL only |
| `sampler.py` | the 1 Hz load capture (per card util, W, °C, clocks, throttle bits, PCIe rx/tx, disk, CPU, RAM) | `nvidia-smi`, `/proc` |
| `resource_summary.py` | joins `sampler.py`'s CSV onto the JSONL: what the box drew while each arm ran | CSV + JSONL |
| `pw_idle_ab.py` | the A/B of § 1: arms switched per request, utilisation sliced into each decode window | a running server, `kubectl`, `ssh` |

```sh
python3 scripts/paired_ci.py raw/p6-final-350w.jsonl --kind code          # § 2's table
python3 scripts/paired_ci.py raw/p0-ist-350w.jsonl raw/p3-pipeline.jsonl \
        raw/p6-final-350w.jsonl raw/p7-fensterstatistik.jsonl \
        --base 'pw=2 theta=0.35' --arm 'theta=0.20'                       # § 6's head-to-head
zcat raw/l28-350w.csv.gz > /tmp/l28.csv
python3 scripts/resource_summary.py --csv /tmp/l28.csv --jsonl raw/p0-ist-350w.jsonl --md   # § 1.1
python3 scripts/sampler.py --out /tmp/box.csv --interval 1                # on your own box
```

## What is deliberately not here

The **sweep driver**. It is ~500 lines of lab plumbing: `kubectl exec` to rewrite the server's run config
and restart the pod, `ssh` to the node for the clock offset and the sampler, a checkout of the engine to
read `--help` from, and prompt corpora taken from local documents (`--code-root`, `--txt-root`). None of
that generalises, and the JSONL it leaves behind is the part worth publishing: one record per request with
the arm text, the group (= one server start) it ran under, the document id and round, `t_start`/`t_end` on
the server's clock, and the engine's own numbers (`tok_s`, `read_tok_s`, `acc_pct`, `windows`, `ms_window`,
`tok_window`, `depth`, window counts). `pw_idle_ab.py` writes the same shape, which is why the two tools
above read both.

Re-producing a sweep therefore needs three things this folder documents but does not automate: a server
started with `--pipeline-windows 2` and `STRATA_PIPELINE_DEBUG=1`, the arm file the engine re-reads per
request (`STRATA_PIPELINE_SWITCH=<file>`, one arm as JSON per write), and documents served repeatedly so
that an arm and its reference share a prompt. What each configuration was is in `../raw/startup/` — the
engine's own `engine started:` argv line, one file per restart, 26 of them.

*Provenance:* these are published snapshots, kept here so the report stands on its own; the working copies
of the sampler, the joiner and the driver live in the tooling folder of the umbrella repository they were
written for (`Strata-Tools/`), where they carry a checkout stamp and keep evolving.
