"""样本池：探索式训练的数据层。

为什么不是 rollouts.jsonl 加几个字段
----------------------------------
旧格式是「一次性整批采样 + 按行序切组」，它在探索式训练下有三处**会静默出错**的假设：

  ① 分组靠行序：`plan()` 用 `rows[i:i+K]`，要求「连续 K 行同 prompt」。增量追加后一旦
     行序错位，不同 prompt 会被混成一组 —— 而 `plan()` 只取 `g[0]["prompt"]`，**不校验**
     组内其余行是不是同一个 prompt。判档错、冷启配对错，全程无告警。
  ② 尾部丢弃：`range(0, len(rows)-K+1, K)` 会把不足 K 条的尾巴直接扔掉。在线场景下
     这就是「刚采的 3 条消失了」，而你没有任何办法察觉。
  ③ 样本无身份：一行只有 `{prompt, completion, reward}`。说不出「这条是哪个 adapter
     产的」（on-policy 的定义）、「这个 0 是模型答错还是人拿不准」（三态表达不了）。

本模块把分组从「行序切片」换成「按 prompt 聚合」，并按样本记录身份。

三态 verdict 与四档判据的关系
---------------------------
四档（graduated / rl / cold / truncated）**不用改**，改的是它的输入：从「reward 一个
浮点数」换成「每条的 verdict」。`classify_verdicts()` 与 `coldstart.classify()` 逐条对齐：

    全 pos                     → graduated
    truncated 过半             → truncated      （整组废：加预算重采）
    有 truncated（且有过错例）  → needs_budget   （通过率不可信）
    人标 unknown 过半           → truncated
    有人标 unknown             → needs_budget
    有 pos 有 neg              → rl             （唯一有梯度的区间）
    全 neg                     → cold

截断必须**平行**传 `truncated`，不能指望从单条 verdict 推出来：截断是组级信息
（「这一组的通过率可不可信」），单条只能表达「这一条算不算数」。

`verdict_of()` 是回落规则：没有显式 verdict 的旧数据按 `reward >= 0.5` 推断，截断记
unknown —— 与旧 `classify()` 的行为逐条一致，所以历史 rollouts 不需要迁移就能进池。

**unknown 永远不进 rejected**：把「拿不准」当成「答错」去惩罚，正是 `classify()` 里
截断那条注释讲过的坑（MATH-500 上 5 条错例全是截断，提预算后 3 条其实答对）。
"""

from __future__ import annotations

import json
import time
import uuid

POS = "pos"
NEG = "neg"
UNKNOWN = "unknown"
VERDICTS = (POS, NEG, UNKNOWN)

PASS = 0.5                     # 与 coldstart.PASS 同一口径

_SOURCES = ("human", "auto", "scorer", "reward")


def verdict_of(sample: dict) -> str:
    """一条样本的正 / 负 / 未知。显式 verdict 优先，缺失时按旧格式推断。

    推断顺序必须与 `coldstart.classify()` 逐分支对齐，否则同一份数据经「池子入口」
    和「行序入口」会给出不同档位。最容易踩的一处是截断与答对并存：

      * `classify()` 的 `acc >= 1` 分支**先于**截断判定，注释写明「答对是真的；后面的
        截断不影响这条」。所以这里也必须让 `reward >= PASS` 先于 `truncated` ——
        把「截断但已答对」记成 unknown 会让整组从 graduated 掉成 needs_budget。
      * 剩下的截断（没答对）= reward 未知，`truncated` 才生效。
    """
    v = sample.get("verdict")
    if v in VERDICTS:
        return v
    r = sample.get("reward")
    if r is not None and float(r) >= PASS:
        return POS
    if sample.get("truncated"):
        return UNKNOWN
    return NEG


def classify_verdicts(verdicts: list[str], truncated: list[bool] | None = None) -> str:
    """一组的四档判定。输入是每条样本的 verdict，不是 reward。

    与 `coldstart.classify(rewards, truncated)` 逐分支对齐。`truncated` 必须平行传进来：
    截断是**组级**信息 —— 单条 verdict 说得清「这条算不算数」，说不清「这一组的通过率
    可不可信」。少了它，`(0,0,0,0.5) + 第 4 条截断` 会被判成 RL，而 `classify()` 判的是
    needs_budget（有错例、又有没写完的，通过率不成立）—— 两条入口就此分叉。

    人标的 `unknown`（「我拿不准」）和截断的 `unknown` 在这一层同权：都让通过率不可信。
    想「这条不参与判定」的语义由调用方在传参前剔除，不在这里混进来 —— 见
    `coldstart._plan_group` 的 `judgeable`。
    """
    if not verdicts:
        return "cold"
    n = len(verdicts)
    n_pos = sum(1 for v in verdicts if v == POS)
    if n_pos == n:
        return "graduated"                     # 答对是真的；截断不影响这条
    n_tr = sum(1 for t in (truncated or []) if t)
    if n_tr and n_tr * 2 >= n:
        return "truncated"
    if n_tr:
        return "needs_budget"
    n_unk = sum(1 for v in verdicts if v == UNKNOWN)
    if n_unk and n_unk * 2 >= n:
        return "truncated"
    if n_unk:
        return "needs_budget"
    if n_pos:
        return "rl"
    return "cold"


def record(prompt: str, completion: str, *, verdict: str | None = None,
           source: str = "human", adapter: str = "", step: int = 0,
           reward: float | None = None, truncated: bool = False,
           gold: str = "", reason: str = "", sid: str = "") -> dict:
    """造一条池内样本。`verdict` 不给时由 reward / truncated 推断（见 `verdict_of`）。

    `source` 记的是**这个判定是谁做的**：`human`（人标的）、`scorer`（判分器）、
    `auto`（自动规则，如 reward 阈值）、`reward`（旧数据回落）。它现在只统计不参与
    加权，但 DPO 迟早要分开：人标的「负」常常是「风格不喜欢」，而 DPO 的 rejected
    必须定义成「同 prompt 下更差的轨迹」—— 两者混用会让 `dpo_margin` 去追风格。
    """
    s = {"prompt": prompt, "completion": completion, "source": source,
         "adapter": adapter, "step": int(step), "truncated": bool(truncated),
         "ts": time.time()}
    if sid:
        s["sid"] = sid                       # 改判时用它指回原样本（见 groups）
    if verdict is not None:
        s["verdict"] = verdict
    if reward is not None:
        s["reward"] = float(reward)
    if gold:
        s["gold"] = gold
    if reason:
        s["reason"] = reason
    return s


def judgeable(samples: list[dict]) -> tuple[list[str], list[bool]]:
    """把「这条不参与」的样本剔除后，返回 `(verdicts, truncated)`。

    **人标的 `unknown`** 是「我拿不准这条」，它不该参与判定；**截断的 `unknown`** 是
    「没写完，reward 未知」，它必须参与（让整组降级到 needs_budget）。两者都叫 unknown
    但语义相反，所以在进 `classify_verdicts` 之前分开。

    不剔除的后果是具体的：一组 3 负 + 1 拿不准会永远停在 needs_budget，进不了冷启，
    而它明明有足够负例可以造一对 DPO。

    全部被剔除时退回原样（不返回空组 —— 空组的档位没有意义）。
    """
    vs, tr = [], []
    for s in samples:
        v = verdict_of(s)
        t = bool(s.get("truncated"))
        if v == UNKNOWN and s.get("source") == "human" and not t:
            continue
        vs.append(v)
        tr.append(t)
    if not vs:
        return [verdict_of(s) for s in samples], [bool(s.get("truncated")) for s in samples]
    return vs, tr


def classify_samples(samples: list[dict]) -> str:
    """一组的档位。这是 `coldstart` 与 `pool.py groups` 共用的唯一入口。"""
    vs, tr = judgeable(samples)
    return classify_verdicts(vs, tr)


def _normalize(sample: dict) -> dict:
    """校验 + 补齐。**不丢未知字段** —— 池子是要长期生长的，外部工具挂的元数据得留住。"""
    if not isinstance(sample, dict):
        raise ValueError(f"样本要是 dict，收到 {type(sample).__name__}")
    for k in ("prompt", "completion"):
        v = sample.get(k)
        if not isinstance(v, str) or not v:
            raise ValueError(f"样本缺 {k}（非空字符串）: {str(sample)[:120]}")
    s = dict(sample)
    v = s.get("verdict")
    if v is not None and v not in VERDICTS:
        raise ValueError(f"verdict 只能是 {VERDICTS}，收到 {v!r}")
    if v is None and "reward" not in s and not s.get("truncated"):
        raise ValueError("样本既没有 verdict 也没有 reward/truncated，判不出正负: "
                         + str(sample)[:120])
    s.setdefault("ts", time.time())
    # 每条样本一个短 id。**必须按 id 去重，不能按 (completion, adapter)**：低温度下
    # 模型经常生成完全相同的轨迹，按内容去重会把 4 条合并成 1 条 —— 而「组内全同」
    # 恰恰是最需要被看见的状态（优势恒 0），静默合并会让它变成一条看起来正常的样本。
    s.setdefault("sid", uuid.uuid4().hex[:12])
    s.setdefault("source", "human" if s.get("verdict") else "reward")
    if s["source"] not in _SOURCES:
        raise ValueError(f"source 只能是 {_SOURCES}，收到 {s['source']!r}")
    return s


class Pool:
    """append-only 的样本池。文件就是 jsonl，一行一条 —— 不需要索引，也不需要事务。

    为什么是 append-only：探索式训练里，「这条样本是什么」不会变，但「它该进哪一档」
    会变（判分器升级、用户改主意、补采到正例）。所以判定不落盘，只在读取时聚合 ——
    重判永远不需要改写历史。

    并发：单进程追加（`open(..., "a")` 一行写完）。多个探索进程同时写不同池子文件是
    安全的；同一个池子要多进程写，请各自写自己的文件再合并（`merge`）。
    """

    def __init__(self, path):
        self.path = str(path)

    def append(self, sample: dict) -> dict:
        s = _normalize(sample)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return s

    def extend(self, samples) -> int:
        n = 0
        for s in samples:
            self.append(s)
            n += 1
        return n

    def rows(self) -> list[dict]:
        """全部样本，按写入顺序。"""
        out = []
        try:
            with open(self.path, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{self.path}:{i} 不是合法 JSON: {e}") from None
        except FileNotFoundError:
            return []
        return out

    def groups(self, adapter: str | None = None) -> dict[str, list[dict]]:
        """按 prompt 聚合，保持每个 prompt 首次出现的顺序。

        这是与旧 `plan()` 的唯一实质差别：**不关心行序**。同一 prompt 分散在文件各处、
        中间夹着别的 prompt，都能正确归组；组大小也不必相等。

        **同一条轨迹以最后一条记录为准**（键 = `sid`）：判定会变（判分器升级、人改
        主意），而文件是 append-only。所以改判 = 追加一条**同 sid** 的记录，历史保留、
        组大小不变、顺序不变。

        键必须是 `sid` 而不是 `(completion, adapter)`：低温度下同一 prompt 常常被采样
        出**多条完全相同的轨迹**，按内容去重会把 K 条并成 1 条，而「组内全同 ⇒ 优势
        恒 0」正是最该被报出来的状态，不能静默消失。

        `adapter` 过滤决定「用谁采的样本做这一轮」：
          * `None`（默认）—— 全部。冷启需要它：冷启不要求 on-policy，只要求有个正例。
          * `"latest"` —— 只用最后一条样本的 adapter。RL 需要它：组内优势按**当前策略**
            的采样归一化才有意义，混进旧 adapter 的样本会把方差算歪。
          * 具体 digest / 路径 —— 只用那一个。
        """
        rows = self.rows()
        if adapter == "latest":
            adapter = next((r.get("adapter") for r in reversed(rows) if r.get("adapter")), None)
            if adapter is None:
                return {}
        g: dict[str, list[dict]] = {}
        seen: dict[str, dict[str, int]] = {}
        for r in rows:
            if adapter is not None and r.get("adapter") != adapter:
                continue
            prompt = r["prompt"]
            grp = g.setdefault(prompt, [])
            where = seen.setdefault(prompt, {})
            key = r.get("sid") or f"_{len(grp)}"     # 手工编辑过的行没有 sid：退化成序号
            if key in where:
                grp[where[key]] = r                  # 重判：位置不动，内容取最新
            else:
                where[key] = len(grp)
                grp.append(r)
        return g

    def stats(self) -> dict:
        rows = self.rows()
        per = {POS: 0, NEG: 0, UNKNOWN: 0}
        adapters: dict[str, int] = {}
        prompts = set()
        # 与 groups() 同口径去重：改判追加的记录不把同一条轨迹数两次
        latest: dict[str, dict] = {}
        for r in rows:
            latest[r.get("sid") or f"{r['prompt']}\x00{r['completion']}"] = r
        for r in latest.values():
            per[verdict_of(r)] += 1
            prompts.add(r["prompt"])
            a = r.get("adapter") or ""
            if a:
                adapters[a] = adapters.get(a, 0) + 1
        return {"samples": len(latest), "prompts": len(prompts),
                "pos": per[POS], "neg": per[NEG], "unknown": per[UNKNOWN],
                "adapters": adapters}


def merge(paths, out: str) -> int:
    """把多个池子合成一个（各探索进程分别写、训练前汇总）。返回写入条数。"""
    p = Pool(out)
    n = 0
    for src in paths:
        for r in Pool(src).rows():
            p.append(r)
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="样本池（探索式训练的数据层）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("stats", help="看池子里有多少正/负/未知、几个 adapter")
    st.add_argument("pool")

    mg = sub.add_parser("merge", help="合并多个池子")
    mg.add_argument("pools", nargs="+")
    mg.add_argument("--out", required=True)

    gr = sub.add_parser("groups", help="按 prompt 聚合后的分档预览（不训练）")
    gr.add_argument("pool")
    gr.add_argument("--adapter", default=None, help="留空=全部；latest=只用最后一个 adapter")
    gr.add_argument("--sids", action="store_true",
                    help="逐条打印 sid（同一轨迹被采多次时，标注要靠它指定是哪一条）")

    lb = sub.add_parser("label", help="人工标注：追加同 sid 的判定记录（不改写历史）")
    lb.add_argument("pool")
    lb.add_argument("--file", required=True,
                    help="JSONL {prompt, completion, verdict, sid?, reason?}。sid 省略时按 "
                         "(prompt, completion) 唯一匹配；有歧义会报错而不是猜一条")

    a = ap.parse_args(argv)

    if a.cmd == "label":
        rows = Pool(a.pool).rows()
        n = 0
        for i, r in enumerate(Pool(a.file).rows(), 1):
            for k in ("prompt", "completion", "verdict"):
                if not r.get(k):
                    raise SystemExit(f"{a.file}:{i} 缺 {k}")
            sid = r.get("sid")
            src = None
            if not sid:
                cands = [x for x in rows if x["prompt"] == r["prompt"]
                         and x["completion"] == r["completion"]]
                if not cands:
                    raise SystemExit(f"{a.file}:{i} 池里没有这条样本（prompt/completion 对不上）")
                if len(cands) > 1:
                    raise SystemExit(
                        f"{a.file}:{i} 该 (prompt, completion) 有 {len(cands)} 条候选"
                        f"（同一轨迹被采了不止一次），必须给 sid —— "
                        f"`pool.py groups {a.pool} --sids` 可以看")
                src = cands[0]
                sid = src.get("sid")
            else:
                src = next((x for x in rows if x.get("sid") == sid), None)
                if src is None:
                    raise SystemExit(f"{a.file}:{i} 池里没有 sid={sid}")
            Pool(a.pool).append(record(r["prompt"], r["completion"],
                                       verdict=r["verdict"], source="human",
                                       adapter=r.get("adapter", src.get("adapter", "")),
                                       reason=r.get("reason", ""),
                                       gold=r.get("gold", src.get("gold", "")),
                                       sid=sid))
            n += 1
        print(f"labelled {n} -> {a.pool}（同 sid 以最后一条判定为准）")
        return 0

    if a.cmd == "stats":
        print(json.dumps(Pool(a.pool).stats(), ensure_ascii=False, indent=2))
        return 0

    if a.cmd == "merge":
        print(f"merged {merge(a.pools, a.out)} samples -> {a.out}")
        return 0

    groups = Pool(a.pool).groups(a.adapter)
    for prompt, g in groups.items():
        vs = [verdict_of(s) for s in g]
        print(f"[{classify_samples(g):12s}] "
              f"n={len(g):3d} pos={vs.count(POS)} neg={vs.count(NEG)} "
              f"unk={vs.count(UNKNOWN)}  {prompt[:60]}")
        if a.sids:
            for s in g:
                print(f"    {s.get('sid', '?'):12s} {verdict_of(s):8s} "
                      f"{s['completion'][:56]!r}")
    if not groups:
        print("(空池子，或该 adapter 没有样本)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
