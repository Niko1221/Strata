#!/usr/bin/env python3
"""
flash-next-bench/spec_minp_sweep.py — plan step C (2026-10-16): measure Strata's per-request tuning keys
(`strata_tune`: spec_min_p, pcie_frac) on the LIVE stack 50 through the proxy, with NO restart and NO config
change. Strata's serve/server.py (v0.1.44.1, sampling_keys) forwards a request's `"strata_tune": {...}` to the
engine for that request only — the mechanism its own tools/calibrate.py uses. Arms rotate inside each round
(forward, then reversed) exactly like calibrate.sweep, so engine drift lands on every arm the same number of times.

Protocol = the long-decode pass of bench.py: one 4k prompt body, 1,024-token answers, server sampling defaults
(temperature omitted), cache_prompt false, ignore_eos true, per-request nonce. Stdlib only; reuses bench.py helpers.
Output: <out-dir>/flash-next-strata-tune/<UTC>/{config.json, runs.jsonl, summary.md}
"""
import argparse, json, os, statistics, sys, time, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

ARMS = {
    "control":            None,
    "minp0.7":            {"spec_min_p": 0.7},
    "minp0.3":            {"spec_min_p": 0.3},
    "pcie0.34+minp0.7":   {"pcie_frac": 0.34, "spec_min_p": 0.7},
}

def one(url, model, prompt, max_tokens, tune, timeout):
    nonce = uuid.uuid4().hex
    payload = {"model": model, "messages": [{"role": "user", "content": f"[run {nonce}]\n{prompt}"}],
               "max_tokens": max_tokens, "stream": False, "cache_prompt": False, "ignore_eos": True}
    if tune:
        payload["strata_tune"] = tune
    t0 = time.monotonic()
    resp = bench.http_json(url + "/v1/chat/completions", payload, timeout=timeout)
    wall = time.monotonic() - t0
    tm = resp.get("timings", {})
    acc = (tm.get("draft_n_accepted") or 0) / tm["draft_n"] if tm.get("draft_n") else None
    return {"nonce": nonce, "wall_s": round(wall, 3), "prompt_n": tm.get("prompt_n"), "prefill_tps": tm.get("prompt_per_second"),
            "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"), "decode_tps": tm.get("predicted_per_second"),
            "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted"), "draft_accept": round(acc, 3) if acc is not None else None,
            "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:4000")
    ap.add_argument("--model", default="flash-next-strata")
    ap.add_argument("--tokenize-url", default="http://localhost:8080/v1/messages/count_tokens")
    ap.add_argument("--slots-url", default="http://localhost:8080/slots")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rss-pids", default="")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    stamp = bench.utc_stamp()
    out = os.path.join(args.out_dir, "flash-next-strata-tune", stamp)
    os.makedirs(out, exist_ok=True)
    json.dump({**vars(args), "arms": ARMS, "stamp": stamp, "argv": sys.argv}, open(os.path.join(out, "config.json"), "w"), indent=2)
    prompt = bench.build_prompt_body(args.prompt_tokens, args.tokenize_url)
    runs, order = [], list(ARMS)
    with open(os.path.join(out, "runs.jsonl"), "w") as f:
        for r in range(args.rounds):
            for name in (order if r % 2 == 0 else order[::-1]):
                before = bench.snapshot_env(args.rss_pids, None, args.slots_url)
                row = one(args.base_url, args.model, prompt, args.max_tokens, ARMS[name], args.timeout)
                after = bench.snapshot_env(args.rss_pids, None, args.slots_url)
                row.update({"arm": name, "tune": ARMS[name], "round": r,
                            "ext_load_before": before.get("backend_requests_processing"), "ext_load_after": after.get("backend_requests_processing"),
                            "nvidia_after": after.get("nvidia_smi")})
                runs.append(row); f.write(json.dumps(row) + "\n"); f.flush()
                print(f"round {r+1}/{args.rounds} {name:18s} decode {row['decode_tps']} tok/s  accept {row['draft_accept']}  drafts {row['draft_n']}  "
                      f"prefill {row['prefill_tps']}  ext {row['ext_load_before']}->{row['ext_load_after']}", flush=True)
    lines = [f"# flash-next-strata-tune — {stamp}", "", f"model `{args.model}` via `{args.base_url}`; prompt {args.prompt_tokens}; max_tokens {args.max_tokens}; "
             f"server sampling defaults; cache_prompt false; ignore_eos true; {args.rounds} rounds, arms rotated per round. {args.note}", "",
             "| arm | tune | decode tok/s per round | median | accept per round | median | drafts per round | prefill tok/s | ext load |", "|---|---|---|---|---|---|---|---|---|"]
    for name in ARMS:
        rows = [x for x in runs if x["arm"] == name]
        dec = [x["decode_tps"] for x in rows]; acc = [x["draft_accept"] for x in rows]
        lines.append(f"| {name} | `{json.dumps(ARMS[name])}` | {' / '.join(str(d) for d in dec)} | **{statistics.median(dec):.1f}** | "
                     f"{' / '.join(str(a) for a in acc)} | {statistics.median(acc):.3f} | {' / '.join(str(x['draft_n']) for x in rows)} | "
                     f"{' / '.join(str(x['prefill_tps']) for x in rows)} | {' / '.join(str(x['ext_load_before']) + '->' + str(x['ext_load_after']) for x in rows)} |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("out:", out)

if __name__ == "__main__":
    main()
