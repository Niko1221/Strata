#!/usr/bin/env python3
"""check_calib.py - validate deployed calibrator files against the loader that will read them.

The bug this exists for: `calib-S.json` shipped `"a": 0.4892, "c": 0.8313`, but `jev_calib.load()`
reads `a_noul`/`c_noul`, so its Platt was SILENTLY ignored (identity) - while `T_choice=0.05` was
honoured, and `apply()` only rejects `T <= 0.01`, so choice probabilities got exponent 20 (near
one-hot).  Reading the JSON by eye cannot catch that; loading it through the module can.

  check_calib.py [calib_dir] [-v]
  exit 0 = every file loads and every declared parameter is honoured

Checks per file:
  * unknown keys  - present in the JSON but not read by load()  => the value is dead, fail
  * mismatch      - declared value != the value load() ends up with => fail
  * degenerate T  - T_choice/T_score <= 0.05 or >= 10 (apply() only drops <= 0.01) => fail
  * tuple letter  - the 5th tuple field must match the filename letter, else load() refuses it
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/opt/data/builds/jev-port")
from serve import jev_calib                                    # noqa: E402

KNOWN = {"tuple", "a_noul", "c_noul", "T_choice", "T_score", "provenance"}
FIELDS = ("a_noul", "c_noul", "T_choice", "T_score")


def check(path: Path, verbose=False):
    """-> (problems, notes)."""
    problems, notes = [], []
    stem = path.stem
    if stem.endswith(".hold") or ".prev-" in stem:
        notes.append("backup/fixture file (not loaded by the server) - checked anyway")
    want_letter = ""
    if stem.startswith("calib-") and len(stem.split("calib-")[1].split(".")[0]) == 1:
        want_letter = stem.split("calib-")[1][0]

    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        return [f"unreadable: {e}"], notes

    unknown = set(raw) - KNOWN
    if unknown:
        problems.append(f"keys {sorted(unknown)} are NOT read by load() - those values are dead "
                        f"(did you mean a_noul/c_noul?)")

    c = jev_calib.load(path)
    if not c.loaded:
        problems.append("load() refused the file entirely (tuple/mode mismatch?) -> identity")
        return problems, notes

    for f in FIELDS:
        declared = raw.get(f)
        if declared is None:
            continue
        got = getattr(c, f)
        if abs(float(declared) - float(got)) > 1e-12:
            problems.append(f"{f}: declared {declared} but load() gives {got}")
        if verbose:
            notes.append(f"{f}={got}")

    for f in ("T_choice", "T_score"):
        t = getattr(c, f)
        if t <= 0.05 or t >= 10.0:
            problems.append(f"{f}={t} is at/over the degenerate edge (apply() only rejects <=0.01; "
                            f"1/{t:g} exponent makes the readout near one-hot or near-uniform)")

    if want_letter:
        got = c.tuple_str.split("|")[4] if c.tuple_str.count("|") >= 5 else ""
        if got and got != want_letter:
            problems.append(f"tuple readout '{got}' != filename letter '{want_letter}'")
        if not got:
            notes.append("legacy tuple without a readout letter")
    return problems, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="/opt/data/llm-swap/strata/Jev-port-cuda/calib")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir)
    files = sorted(set(d.glob("calib-*.json")) | set(d.glob("calib-*.hold")))
    if not files:
        print(f"no calib-*.json in {d}")
        return 1
    bad = 0
    for p in files:
        problems, notes = check(p, args.verbose)
        print(f"{'FAIL' if problems else 'ok  '}  {p.name}")
        for n in notes:
            print(f"        . {n}")
        for pr in problems:
            print(f"        ! {pr}")
            bad += 1
    print(f"\n{len(files)} file(s), {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
