import json, sys, pathlib
p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "tools/opt/results/w2-base.json")
d = json.load(p.open())
print("arm:", d["run"].get("arm_label"), "| config-relevant:", d["run"].get("log_path"))
for r in d["requests"]:
    dt = r.get("decode_timing") or {}
    acc = 100.0 * r["drafts_accepted"] / r["drafts_total"] if r["drafts_total"] else float("nan")
    print("r%d: %.1f tok/s  hit %.1f%%  accept %.1f%%  (%d/%d drafts, %d ckpt)" % (
        r["repeat"], r["decode_tps"], r["hit_pct"], acc, r["drafts_accepted"], r["drafts_total"], r["checkpoints"]))
    if dt:
        print("    windows=%d avgT=%.2f tok/win=%.2f ms/win=%.2f" % (dt["windows"], dt["avg_t"], dt["tokens_per_window"], dt["ms_per_window"]))
        print("    wait=%.2f  pool(per-layer host)=%.2f  stage(ms_host)=%.2f  commit=%.2f  draft=%.2f" % (dt["wait_ms"], dt["host_ms"], dt["stage_ms"], dt["commit_ms"], dt["draft_ms"]))
        print("    plan=%.2f actq=%.2f jobs=%.2f CPUrun=%.2f" % (dt["plan_ms"], dt["actq_ms"], dt["jobs_ms"], dt["cpu_ms"]))
        print("    CPU experts/layer=%.2f (%.2f entries)  VRAM hits/layer=%.2f  PCIe/layer=%.2f" % (
            dt["cpu_experts_per_layer"], dt["entries_per_layer"], dt["vram_hits_per_layer"], dt["pcie_per_layer"]))
        tot = dt["wait_ms"] + dt["host_ms"] + dt["stage_ms"] + dt["commit_ms"] + dt["draft_ms"]
        print("    sum(wait+pool+stage+commit+draft)=%.2f  vs ms/win=%.2f" % (tot, dt["ms_per_window"]))
        if dt["tokens_per_window"] and dt["ms_per_window"]:
            print("    implied tok/s = %.1f" % (1000.0 * dt["tokens_per_window"] / dt["ms_per_window"]))
