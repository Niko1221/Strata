"""S2 of the ladder: does the K channel axis carry a heavier tail *before* the Hadamard than after?

Reads the probe file that `STRATA_KV_S2_STATS=<path>` writes on the batched prompt path (prefill and verify,
`include/strata/core/s2_probe.hpp`), and reports, per QSA layer and per group (the 64 RoPE-rotated and the 192
unrotated dimensions of head_dim 256):

  * the excess kurtosis of every channel (over the prompt's tokens),
  * the crest factor max|x| / sd of every channel,
  * the per-channel variance spread (KIVI's axis: a per-channel scale is worth something only if the channels
    differ in width),
  * the energy check the orthonormal Hadamard must pass (sum of E[x^2] over the 256 channels, pre vs post).

The pre-registered decision (docs/paper/Ansatz_Improvements.md, S2): if the tail before the rotation is not
heavier than after it, A1 (per-channel K before the FWHT) is not built.

Usage:
  python tools/kv_s2_channels.py --tsv <probe file> [--json <out.json>]
"""

import argparse
import collections
import json
import math
import statistics as st


def read(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        head = f.readline().split()
        for line in f:
            v = line.split()
            if len(v) != len(head):
                continue
            rows.append(dict(zip(head, v)))
    return rows


def crest(r, side):
    var = float(r[f"var_{side}"])
    mean = float(r[f"mean_{side}"])
    sd = math.sqrt(max(var + mean * mean, 1e-30))
    return float(r[f"maxabs_{side}"]) / sd


def group_stats(rows, side):
    k = [float(r[f"exkurt_{side}"]) for r in rows]
    c = [crest(r, side) for r in rows]
    return {"nchan": len(rows), "kurt_mean": st.mean(k), "kurt_median": st.median(k),
            "kurt_max": max(k), "kurt_p90": sorted(k)[int(0.9 * (len(k) - 1))],
            "crest_mean": st.mean(c), "crest_max": max(c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = read(a.tsv)
    layers = sorted({int(r["layer"]) for r in rows})
    per_layer, verdicts = [], []
    for l in layers:
        lr = [r for r in rows if int(r["layer"]) == l]
        n = int(lr[0]["n"])
        e_pre = sum(float(r["var_pre"]) + float(r["mean_pre"]) ** 2 for r in lr)
        e_post = sum(float(r["var_post"]) + float(r["mean_post"]) ** 2 for r in lr)
        var = [float(r["var_pre"]) for r in lr]
        entry = {"layer": l, "tokens": n, "energy_pre": e_pre, "energy_post": e_post,
                 "energy_ratio": e_pre / e_post if e_post else 0.0,
                 "var_spread": max(var) / max(min(var), 1e-30), "groups": {}}
        for grp in ("rot", "unrot"):
            gr = [r for r in lr if r["group"] == grp]
            entry["groups"][grp] = {"pre": group_stats(gr, "pre"), "post": group_stats(gr, "post")}
            # the per-channel sign test: how many channels have a heavier tail before the rotation
            verdicts.append(sum(1 for r in gr if float(r["exkurt_pre"]) > float(r["exkurt_post"])))
        entry["channels_pre_heavier"] = sum(verdicts[-2:])
        per_layer.append(entry)

    tot_ch = sum(e["groups"]["rot"]["pre"]["nchan"] + e["groups"]["unrot"]["pre"]["nchan"] for e in per_layer)
    pre_heavier = sum(e["channels_pre_heavier"] for e in per_layer)
    summary = {
        "layers": len(layers), "tokens_per_layer": per_layer[0]["tokens"], "channels": tot_ch,
        "energy_ratio_min": min(e["energy_ratio"] for e in per_layer),
        "energy_ratio_max": max(e["energy_ratio"] for e in per_layer),
        "channels_pre_heavier": pre_heavier, "channels_pre_heavier_pct": 100.0 * pre_heavier / tot_ch,
        "kurt_pre_mean": st.mean([e["groups"][g]["pre"]["kurt_mean"] for e in per_layer for g in ("rot", "unrot")]),
        "kurt_post_mean": st.mean([e["groups"][g]["post"]["kurt_mean"] for e in per_layer for g in ("rot", "unrot")]),
        "crest_pre_mean": st.mean([e["groups"][g]["pre"]["crest_mean"] for e in per_layer for g in ("rot", "unrot")]),
        "crest_post_mean": st.mean([e["groups"][g]["post"]["crest_mean"] for e in per_layer for g in ("rot", "unrot")]),
        "var_spread_min": min(e["var_spread"] for e in per_layer),
        "var_spread_max": max(e["var_spread"] for e in per_layer),
    }
    summary["kill_fires"] = summary["channels_pre_heavier_pct"] <= 50.0
    print(f"{summary['layers']} QSA layers, {summary['tokens_per_layer']} tokens each, {summary['channels']} channels")
    print(f"energy pre/post (orthonormal Hadamard must be 1.0): {summary['energy_ratio_min']:.6f} .. "
          f"{summary['energy_ratio_max']:.6f}")
    print(f"excess kurtosis  pre {summary['kurt_pre_mean']:+.2f}  post {summary['kurt_post_mean']:+.2f}")
    print(f"crest max|x|/sd  pre {summary['crest_pre_mean']:.2f}  post {summary['crest_post_mean']:.2f}")
    print(f"channels with a heavier tail before the rotation: {pre_heavier}/{tot_ch} "
          f"({summary['channels_pre_heavier_pct']:.1f} %)")
    print(f"per-channel variance spread: {summary['var_spread_min']:.0f}x .. {summary['var_spread_max']:.0f}x")
    print(f"S2 kill (pre not heavier -> A1 not built): {'FIRES' if summary['kill_fires'] else 'does not fire'}")
    print()
    print("| layer | tokens | kurt pre (rot / unrot) | kurt post (rot / unrot) | crest pre (rot / unrot) | "
          "crest post (rot / unrot) | var spread | E pre/post |")
    print("|---|---|---|---|---|---|---|---|")
    for e in per_layer:
        g = e["groups"]
        print(f"| {e['layer']} | {e['tokens']} | {g['rot']['pre']['kurt_mean']:+.2f} / "
              f"{g['unrot']['pre']['kurt_mean']:+.2f} | {g['rot']['post']['kurt_mean']:+.2f} / "
              f"{g['unrot']['post']['kurt_mean']:+.2f} | {g['rot']['pre']['crest_mean']:.2f} / "
              f"{g['unrot']['pre']['crest_mean']:.2f} | {g['rot']['post']['crest_mean']:.2f} / "
              f"{g['unrot']['post']['crest_mean']:.2f} | {e['var_spread']:.0f}x | "
              f"{e['energy_ratio']:.6f} |")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "layers": per_layer}, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
