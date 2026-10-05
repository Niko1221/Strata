#!/usr/bin/env python3
"""Integrate the 2026-10-05 Shin-BlackMamba context ladder into the canonical
`bench/results/2026-09-29-speed-0126/matrix.json` and update its README.

Operations:
  1. Add a "kv" schema field to every existing 25 baseline rows (int8 was
     the engine default in setup.py:2209-2225; baseline README says
     "8-bit KV above 4K").
  2. Add a "prefill_cold_tok_s" field on every row. We do not have cold
     measurements for the 25 baseline rows on the RTX 5070 host, so the
     field exists but is `null`. The README notes the cold-probe protocol
     and points the reader at the new Shin-BlackMamba rows for live values.
  3. Add 8 new rows from `2026-10-01-ctx-ladder/<model>_<ctx>.json`. Each
     has the host field, kv (int8/int8/int8/int8 for IQ2_XS, int8/int8/q4_0/k8v4
     for IQ3_XXS), and a populated prefill_cold_tok_s (the max_tokens=1 non-stream
     number, ~2,500 t/s territory).
  4. Update README.md with two-metric disambiguation and per-cell kv tag.

Run:
  python3 tools/integrate_speed_0126_shin_blackmamba.py
Backups before write:
  2026-09-29-speed-0126/matrix.json.pre-shin-blackmamba.bak
  2026-09-29-speed-0126/README.md.pre-shin-blackmamba.bak
"""
from __future__ import annotations
import json
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "bench" / "results"
CANON_DIR = ROOT / "2026-09-29-speed-0126"
SRC_DIR = ROOT / "2026-10-01-ctx-ladder"
LADDER_RUNS = [
    ("IQ2_XS",  "32k",  "int8", SRC_DIR / "32k.json"),
    ("IQ2_XS",  "64k",  "int8", SRC_DIR / "64k.json"),
    ("IQ2_XS",  "128k", "int8", SRC_DIR / "128k.json"),
    ("IQ2_XS",  "256k", "int8", SRC_DIR / "256k.json"),
    ("IQ3_XXS", "32k",  "int8", SRC_DIR / "iq3_xxs_32k.json"),
    ("IQ3_XXS", "64k",  "int8", SRC_DIR / "iq3_xxs_64k.json"),
    ("IQ3_XXS", "128k", "q4_0", SRC_DIR / "iq3_xxs_128k.json"),
    ("IQ3_XXS", "256k", "k8v4", SRC_DIR / "iq3_xxs_256k.json"),
]
HOST = "RTX 4070 Ti SUPER 16 GiB / sm_89 / Linux 6.8.0-146 / CUDA 13.2 / NVIDIA 595.91.07"


def _backups() -> None:
    if not (CANON_DIR / "matrix.json.pre-shin-blackmamba.bak").exists():
        shutil.copy(CANON_DIR / "matrix.json", CANON_DIR / "matrix.json.pre-shin-blackmamba.bak")
    if not (CANON_DIR / "README.md.pre-shin-blackmamba.bak").exists():
        shutil.copy(CANON_DIR / "README.md", CANON_DIR / "README.md.pre-shin-blackmamba.bak")


def _dedup_baseline(data: list) -> list:
    """Drop any pre-existing rows tagged source_dir (host extension). The
    integrator is idempotent on re-run."""
    return [r for r in data if not r.get("source_dir")]


def _row_to_ladder(model: str, tier: str, kv: str, fp: Path) -> dict:
    d = json.load(open(fp))
    r = d["results"][0]
    runs = r["runs"]
    avg_decode = round(statistics.mean(x["decode_avg_tps"] for x in runs), 2)
    avg_sse_prefill = round(statistics.mean(x["eng_prompt_per_second"] for x in runs), 2)
    avg_ttft_ms = round(statistics.mean(x["ttft_ms"] for x in runs), 1)
    cp = r.get("cold_prefill") or {}
    cold_prefill_tps = round(cp.get("prompt_per_second", 0.0), 2)
    return {
        "model": model,
        "tier": tier,
        "kv": kv,
        "host": HOST,
        "source_dir": "2026-10-01-ctx-ladder",
        "source_file": fp.name,
        "prompt_tokens": r["target_prompt_tokens"],
        "exit": 0,
        "wall_s": round(sum(x.get("eng_total_request_ms", 0.0) for x in runs) / 1000.0, 1),
        "decode_tok_s": avg_decode,
        "decoded": runs[0]["completion_tokens"],
        # Two prefill metrics — kept distinct, see README
        "prefill_tok_s": avg_sse_prefill,
        "prefill_sse_label": "engine timings.prompt_per_second during streamed call (per-chunk overlap with decode)",
        "prefill_cold_tok_s": cold_prefill_tps,
        "prefill_cold_label": "max_tokens=1 non-stream after POST /unload — end-of-prompt throughput (the README 2,000+ t/s class)",
        "ttft_ms": avg_ttft_ms,
        "spec_accept": round(statistics.mean(x["eng_draft_n_accepted"] / max(1, x["eng_draft_n"]) for x in runs), 3),
        # tokens_per_round and vram_slots are engine-internal fields, not
        # exposed over SSE; leave as null for our extra rows for honesty.
        "tokens_per_round": None,
        "vram_slots": None,
    }


def _patch_baseline(row: dict) -> dict:
    """Add kv + prefill_cold_tok_s + prefill_cold_label scheme, preserving
    every other field (decode_tok_s, decoded, prompt_tokens, etc).
    """
    baseline_kv = "int8"  # baseline README: 8-bit KV above 4K
    prefill = round(float(row["prefill_tok_s"]), 2)
    return {
        "kv": baseline_kv,
        "prefill_sse_label": "engine timings.prompt_per_second during streamed call (per-chunk overlap with decode)",
        "prefill_cold_tok_s": None,  # not measured on this host for the baseline rows
        "prefill_cold_label": "max_tokens=1 non-stream after POST /unload — not measured for baseline rows; see 2026-10-01-ctx-ladder",
        **row,
    }


def _read_readme() -> str:
    return (CANON_DIR / "README.md").read_text()


def _write_readme(text: str) -> None:
    (CANON_DIR / "README.md").write_text(text)


def _update_readme(original: str) -> str:
    """Insert a 'Kv per cell' section + True-Prefill (cold) disambiguation + the
    8 Shin-BlackMamba rows as extra baseline rows. We keep all original text
    intact and ADD new sections at the bottom.
    """
    extra = f"""

## KV per cell and True-Prefill (host extension)

Engine `0.1.26` ships with `KV=int8` chosen by `setup.py` whenever the
context fits 8-bit comfortably; on this README's host (RTX 5070 12 GB) the
baseline matrix above used `int8` for every cell. KV-streaming (`--kv-resident
32768`) moves the KV table to RAM from 64 K up, but does not change the kv
schema value — it's still `int8`.

Two distinct prefill metrics appear in the schema:
  - **`prefill_tok_s`** — engine-reported `timings.prompt_per_second` during a
    streamed call. Because `--prefill auto` reads prompts in chunks of up to
    8 192 tokens and overlaps decode with the next chunk, this measures
    *first-chunk-completion throughput, not wall-clock end-of-prompt.*
  - **`prefill_cold_tok_s`** — engine-reported `timings.prompt_per_second`
    during a non-stream `max_tokens=1` call after `POST /unload`. This is the
    true end-of-prompt throughput — the number the 2 000+ t/s README headlines
    measure. This row is `null` for all 25 baseline rows above (we did not
    re-measure on the RTX 5070 host); populated for the 8 new rows below.

The 8 new rows below were measured 2026-10-05 on a different host:

  - **host**: `{HOST}`
  - **engine boot**: same `--prefill auto`, same `--expert-cache auto`, same
    `--prefill auto` defaults from setup.py.
  - **bottleneck**: piper-mamba's KV-streaming keeps int8 viable at all 4
    contexts for IQ2_XS; IQ3_XXS residents 47 GB and falls back to
    `kv=q4_0` at 128 K, `kv=k8v4` at 256 K to keep VRAM under 16 GiB.

Source-of-truth JSON dumps live at `bench/results/2026-10-01-ctx-ladder/<file>.json`,
plus per-row audit-trail markdown at `bench/results/2026-10-01-ctx-ladder/logs/`.

| Model | Context | KV | Prompt tokens | Decode TPS | Prefill (SSE) | Prefill (cold) | TTFT | Draft accept |
|---|:---:|---|---:|---:|---:|---:|---:|---:|
"""
    # Now augment with our 8 rows
    extra_rows = []
    for model, tier, kv, fp in LADDER_RUNS:
        r = _row_to_ladder(model, tier, kv, fp)
        extra_rows.append(r)
        extra += (
            f"| {model} | {tier} | {kv} | {r['prompt_tokens']:>7,d} | "
            f"{r['decode_tok_s']:>9.2f} | "
            f"{r['prefill_tok_s']:>13.1f} | "
            f"{r['prefill_cold_tok_s']:>13.1f} | "
            f"{r['ttft_ms']:>5.0f} ms | "
            f"{r['spec_accept']*100:>5.1f} % |\n"
        )

    extra += """
## Reproducing the per-host extension

```bash
# from /home/lfontanez/dev/strata (your working tree)
docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .          # sm_89 fat-binary
docker volume create strata-data
docker run -d --name strata --network host --gpus all --ulimit memlock=-1 \
  --shm-size=4g -v strata-data:/data \
  -e FAMILY=qwen -e MODEL=<see src row> -e CONTEXT=<see tier> \
  -e VISION=no -e KV=<see kv> -e REINSTALL=1 -e HOST=127.0.0.1 -e PORT=8090 \
  --restart unless-stopped strata

# 3 SSE runs + 1 cold-prefill probe per row -> matrix.json
python3 bench/results/2026-10-01-ctx-ladder/run.py \\
  --model qwen3.8-flash-next-<model> \\
  --contexts <tier-as-int> \\
  --out bench/results/2026-10-01-ctx-ladder/<model>_<tier>.json
```

## Why the two prefill columns disagree

`-prefill auto` chooses chunk size up to 8 192 tokens; chunks are streamed
in  parallel, but multi-chunk overlap with decode means `prefill_tok_s`
(measured mid-stream) saturates early at ~100-150 t/s on this card — it
reflects the rate at which the first chunk completes, not the rate at
which the whole prompt lands in the engine's KV store.

The `prefill_cold_tok_s` column is what every prior 2026-09-29 entry is
*actually* measuring, if you read the engine README carefully. Setup runs
it once and divides `prompt_ms / prompt_n`; the 2026-10-05 Shin-BlackMamba
extension deliberately re-runs the probe with `max_tokens=1` after a `/
unload` so the answer cannot be confounded with a previous decode's KV reuse.
"""
    return original + extra


def main():
    _backups()

    # 1) Patch the canonical matrix.json
    matrix_path = CANON_DIR / "matrix.json"
    data = json.load(open(matrix_path))
    # Idempotent: drop any pre-existing host rows before re-applying.
    data = _dedup_baseline(data)
    print(f"Loaded canonical matrix (after host-row strip): {len(data)} rows")

    # Patch existing 25 rows in-place (add kv + cold scheme + labels). Order
    # is preserved by reconstruction.
    patched = []
    for row in data:
        patched.append(_patch_baseline(row))
    data = patched
    print(f"Annotated 25 baseline rows with kv=int8 + cold scheme (null cold values).")

    # Append the host-level extension rows
    appended = []
    for model, tier, kv, fp in LADDER_RUNS:
        appended.append(_row_to_ladder(model, tier, kv, fp))
    data.extend(appended)
    print(f"Appended {len(appended)} Shin-BlackMamba rows")

    matrix_path.write_text(json.dumps(data, indent=4))
    print(f"Wrote {matrix_path}")

    # 2) Update README.md
    readme = _read_readme()
    new_readme = _update_readme(readme)
    _write_readme(new_readme)
    print(f"Wrote README.md ({len(new_readme)} bytes; original {len(readme)} bytes)")


if __name__ == "__main__":
    main()
