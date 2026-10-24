# Reproducing the measurement

One engine child per arm, teacher-forced through the committed yardstick (`tools/quant_quality.py`)
via the thin driver `ab_kld.py` (it reuses the yardstick's `engine_args`/`FIXED`/`FIXED_ENV`/`read_run`/`compare`
— the engine start is a thin copy of `quant_quality run`; plus two additions: a
`--test-flag` that lets the config's `--pcie-frac` survive the yardstick's FIXED 0 (the skip fires
either way — its scan runs before the PCIe/CPU placement decision in `src/core/expert_source.cpp`;
with 0 every remaining miss is computed on the CPU, outside the streaming regime the guard exists
for) — and the identity stamp each arm carries;
the per-arm `--env` is the yardstick's own). Run from the repository root on a **quiet GPU** (the
arms start their own engine; stop any serving model first).

`config.json` is the **command line the engine actually received** (the args the driver handed it,
as recorded in the run's seg.json; paths
replaced by `<pack>`, `<dense.gguf>`, `<ple-nvfp4.gguf>`, `<mtp-rt>` - the NVFP4-QAD pack as
`setup.py` / `tools/nvfp4_local_pack.py` prepare it; fill them in for your box). The yardstick drops
and re-fixes most of it (`FIXED`/`DROP_VALUE` in `tools/quant_quality.py`); run the driver with
`--gpu-experts auto` (the yardstick drops the config's `--expert-cache` and re-fixes it from that
flag; the start-up sizing, 8930 slots on this card - see `engine-log-excerpt.txt`) and a nonzero `--pcie-frac` in the config (the skip fires even at 0, but the remaining misses all go
to the CPU and the arms leave the PCIe-streaming regime the guard exists for — the engine's own
`route tail skip` and `read by the GPU over PCIe` stderr lines are the gate). `data/expert-profile.bin` is the one committed in
the repo.

```
# one arm (~5-7 min after load): A0/A0b take 0, the rank arms 7/8/9
python reproduce/ab_kld.py run --config reproduce/config.json --ids reproduce/dlss.ids.json \
    --out out-A7.json --tsv logpos-A7.tsv --label A7 --build "strata 0.1.46.1" \
    --gpu-experts auto --test-flag=--pcie-frac --env STRATA_ROUTE_TAIL_SKIP=7

# after all five arms (A0, A0b, A7, A8, A9):
python reproduce/ab_kld.py compare --base out-A0.json --runs out-A0b.json out-A7.json out-A8.json out-A9.json \
    --json kld-matrix.json
```

What to expect: every arm exits 0 with `positions 46050`; `kld-matrix.json` carries the shared
identity (build, model, context, corpus `f553e2c3`, topk 64, prompt_cache cold); the A0b (A/A) row is
`kl 0.0, top1 100.0`; the rank arms show `route tail skip: N missed experts skipped` lines in
`logpos-<arm>.tsv.stderr.txt` (the committed `skips-A*.txt` hold all 30 per-arm lines; the last
feeds the table). A cheap
smoke first: `--ids` a two-segment slice of `dlss.ids.json` and check positions and skip lines
before committing to the full matrix.

The per-position tsvs (~46 MB per arm, format documented in `tools/quant_quality.py`'s header) are
not committed; they regenerate from `dlss.ids.json` (the committed A/A rerun reproduced byte-identically
on this machine) — the
`compare` step needs them beside the `out-*.json` files it lists, plus the `<tsv>.seg.json`
`ab_kld.py` writes.

The corpus: `make_ids.py` rebuilds `dlss.ids.json` from the ten corpus files — nine public NVIDIA
documents (Ada / Blackwell whitepapers, the DLSS 4 article, the DLSS 5 report, DLSS SR/FG/RR guides)
plus `dlss-prompt3.txt`, a page-marked concatenation of six of them (those six carry double weight);
`corpus.json` carries the per-file sha256 prefixes.
The committed ids file means you need neither the tokenizer nor the documents to reproduce the
numbers — `ab_kld.py` stamps the ids hash into every arm's identity, and it must read `f553e2c3`.
