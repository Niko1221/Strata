"""memanalyze.py —— 分析 sft.py STRATA_MEM_PROBE 落盘的显存快照：谁在占显存。

快照由 torch.cuda.memory._dump_snapshot 产出（plain pickle，无 torch 也能读）。
两层聚合：
  1. 活块（active_allocated）按分配点栈顶 3 帧聚合 —— 「谁的显存还活着」；
  2. device_traces 的分配/释放事件按分配栈顶 2 帧聚合分配字节峰值流量 ——
     「谁在反复要内存」（活块看不到的 transient 大头在这里显形）。

用法：python3 memanalyze.py snapshot.pkl
"""

from __future__ import annotations

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))


import pickle
import sys
from collections import Counter


def _key(frames, n: int) -> str:
    if not frames:
        return "(no frames)"
    parts = []
    for fr in frames[:n]:
        fn = (fr.get("filename") or "?").replace("\\", "/").split("/")[-1]
        parts.append(f"{fn}:{fr.get('line', '?')} {fr.get('name', '?')}")
    return " < ".join(parts)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    with open(sys.argv[1], "rb") as f:
        snap = pickle.load(f)

    live, live_n = Counter(), Counter()
    big: list[tuple[int, str]] = []
    for seg in snap.get("segments", []):
        for b in seg.get("blocks", []):
            if b.get("state") not in (None, "active_allocated"):
                continue
            k = _key(b.get("frames") or [], 3)
            live[k] += b.get("size", 0)
            live_n[k] += 1
            big.append((b.get("size", 0), k))
    tot = sum(live.values())
    print(f"== 活块合计 {tot / 2**30:.2f} GiB / {sum(live_n.values())} blocks ==")
    for k, v in live.most_common(20):
        print(f"{v / 2**30:8.3f} GiB {live_n[k]:8d} blk  {k}")
    if live.get("(no frames)", 0) > tot * 0.5:
        print("\n== 无栈模式：最大单个活块 top 20（认形状：2.49MiB=专家块，"
              "128×vocab×4B=134MiB=logits 块）==")
        for size, _k in sorted(big, reverse=True)[:20]:
            print(f"{size / 2**20:10.2f} MiB")
        # 尺寸分桶：相同尺寸的块通常来自同一个分配点
        bucket, bucket_n = Counter(), Counter()
        for size, _k in big:
            bucket[size] += size
            bucket_n[size] += 1
        print("\n== 同尺寸聚合 top 20（size × count = 总量）==")
        for size, v in bucket.most_common(20):
            print(f"{v / 2**30:8.3f} GiB  {bucket_n[size]:7d} × {size / 2**20:9.2f} MiB")

    # 分配流量：每个 alloc 事件记到栈顶 2 帧；不配对的 free 忽略（周期内近似）。
    flow, flow_n = Counter(), Counter()
    for trace in snap.get("device_traces", []):
        for e in trace:
            if e.get("action") != "alloc":
                continue
            k = _key(e.get("frames") or [], 2)
            flow[k] += e.get("size", 0)
            flow_n[k] += 1
    tot_f = sum(flow.values())
    print(f"\n== alloc 流量合计 {tot_f / 2**30:.2f} GiB / {sum(flow_n.values())} events ==")
    for k, v in flow.most_common(20):
        print(f"{v / 2**30:8.3f} GiB {flow_n[k]:8d} evt   {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
