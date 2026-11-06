r"""把「采样结果」路由成三种训练动作：冷启 / RL / 毕业。

为什么需要单独一层：GRPO 与 LUSPO 的每步梯度是 `Σ_i A_i · s_i · ∇logπ(y_i)`，
优势 A_i 来自**组内**归一化。组内奖励全同 ⇒ A ≡ 0 ⇒ 梯度恒 0、一步也不动。
所以题池按通过率天然分成三份，训练工具也不同：

    acc == 1        毕业：没有梯度，继续采只是烧卡（移出池子）
    0 < acc < 1     RL：唯一能用 GRPO / LUSPO 的区间
    acc == 0        冷启：RL 动不了；要先让它「至少答对一次」，工具是 SFT 或 DPO

这个模块只做判断和数据构造，不训练、不采样、不 import torch —— 所以能在没有
GPU 的 CI 里跑，也能在采完之后立刻跑。

## 三个检查，顺序按代价从低到高

1. **标签是否过期**（纯字符串，零成本）。判分器改过之后，旧 `reward` 会把「其实
   答对了、只是写法不同」的样本记成 0。这类题若被当成冷启题，DPO 会拿标准答案
   去纠正一个**已经正确**的行为。实测：GSM8K 16 组里 1 组 4 条全部中招
   （gold=27000，模型写 `27{,}000` / `\$27,000`，旧 `extract_pred` 判 0）。
   ⇒ 结论：**不能吃历史 reward 标签**，路由前必须用当前判分器重判；没重判时才用
   `label_suspect()` 把可疑的挑出来人工/重判，绝不进冷启。
   `label_suspect` 是「宁可多报」的启发式，不是判分器：它只看「gold 数字出现在轨迹
   尾部 35%」，长推导里的中间运算数也会命中。2026-11-06 实测 AIMO3 10 组里 4 组
   误报（gold=8687 命中 "18,687"），其中 2 个真冷启组被踢成 `suspect_label`。
   ⇒ 所以 `plan()` 里**已用 `rescore` 对该组每一条重新判过分**时，这个启发式只记录、
   不否决（见 `plan()` 的 `rescored_ok`）；它管的是「标签可能过期」，而重判已经把
   这个可能性排除了，再否决就是拿启发式覆盖精确判分。

2. **0 正是真错还是被截断**。截断样本当 DPO 的负例，模型学到的是「别说下去」，
   而不是「这一步算错了」。截断占比过半 ⇒ 该加 `max_new` / 换更短的数据，不是训。

3. **chosen 与 rejected 是否同构**。DPO 比的是整条序列的 logp；本仓 `dpo_margin`
   按各自答案 token 数平均（每侧归一化到「平均每 token」）。若 chosen 是 1 个 token
   的答案串、rejected 是 300 token 的推理链，两侧的「平均粒度」根本不是一回事，
   margin 没有物理意义。所以冷启阶段必须回答一个问题：
   **标准答案（gold）是完整的"要学的那段文本"，还是只有最终结果？**
     - 长推理任务（数学）：gold 只是 `"27000"`，模型要学的是整条解答 ⇒ gold 不能当
       chosen。要么用**模型自己采到的正确轨迹**（on-policy，最优先），要么用完整的
       标准解答（`--gold-solutions` 显式提供）。两者都没有 ⇒ 冷启不可用，本模块拒
       绝构造这对数据并给出原因，绝不静默退化成 SFT。
     - `--gold-solutions` 里那一段是**调用方显式声明的"要学的整段文本"**，本模块
       会挡住「整段只是答案数字 / 短到不含推导」（`is_bare_answer`），但不会用长度
       比去猜它够不够完整 —— 错例可能长到 33 万字符，比值判据会误杀合法解答。
     - 短答任务（身份、术语）走内置偏好对数据，不经 `--gold-solutions` 这条长推理
       通道；`is_bare_answer` 是这条通道的守卫，不是通用答案长度门槛。

## chosen / rejected 的选择规则

- `chosen` 优先 **模型自己的正确轨迹里最短的一条**（on-policy、信息密度最高；
  最短也意味着没靠冗长凑答案）。
- `rejected` 取 **最长的"完整但答错"样本**（"错得最认真"）。不取截断的、不取最短的：
  最短的常常是拒答/空转，惩罚它没有教学价值。
- 长度比守卫：两侧 token 数比超出 `[1/3, 3]` 就不配对（理由同上）。

## 与「每次都用标准答案 DPO」的区别

不是每次、也不是都用标准答案。判据就在上面第三条：**gold 与 rejected 同构才用 gold**。
同构性可以用一个和采样无关的量先验估计 —— gold 的 token 数 ÷ 错例 token 中位数。
远小于 1（如 1/300）⇒ gold 只是一串结果，不能当 chosen。
"""

from __future__ import annotations

import difflib
import json
import re

# ---------------------------------------------------------------- 判定标签

GRADUATED = "graduated"      # 全对：无梯度，移出池子
RL = "rl"                    # 有方差：唯一能跑 GRPO / LUSPO 的区间
COLD = "cold"                # 全错：需要冷启
TRUNCATED = "truncated"      # 全错且过半截断：先加预算，不是训练问题
NEEDS_BUDGET = "needs_budget"  # 有样本被截断 ⇒ 通过率不可信，先加 --max-new

PASS = 0.5                   # reward ≥ PASS 记作「答对」（连续奖励也走这条线）


def acc_of(rewards: list[float]) -> float:
    """组内通过率。空组返回 0.0（调用方应先过滤）。"""
    if not rewards:
        return 0.0
    return sum(1 for r in rewards if r >= PASS) / len(rewards)


def classify(rewards: list[float], truncated: list[bool] | None = None) -> str:
    """把一组（同一 prompt 的 K 条采样）判成四档之一。

    参数 `truncated` 是每条的「是否被 max_new 截断」标记；缺省当作都没截断。

    **截断样本的 reward 是"未知"，不是"答错"**（2026-11-05 实测）：MATH-500 L4/L5
    12 题 × 2 条，1536 预算下 5 条错例**全部**是截断；把预算提到 6144 后这 5 条
    里有 3 条答对、样本准确率 79.2% → 91.7%。所以只要组里有截断而又有错例，
    这一组的通过率就分不清是「不会」还是「没写完」，必须先加预算再判 —— 否则
    RL 拿到的是「一条被腰斩的轨迹算负例」，DPO 会拿标准答案去纠正一个正在正确
    推理的过程。
    """
    acc = acc_of(rewards)
    if acc >= 1.0:
        return GRADUATED                       # 答对是真的；后面的截断不影响这条
    n_tr = sum(1 for t in (truncated or []) if t)
    if n_tr and truncated and n_tr * 2 >= len(truncated):
        return TRUNCATED                       # 过半截断：整组都不能用
    if n_tr:
        return NEEDS_BUDGET                    # 有截断 + 还有错例：通过率不可信
    if acc > 0.0:
        return RL
    return COLD


# ---------------------------------------------------------------- 标签自查

_NUM_STRIP = str.maketrans({"$": " ", ",": "", " ": " ", "\\": " ", "{": " ", "}": " ",
                            "．": ".", "，": "", "_": ""})


def normalize_number(s: str) -> str:
    """把 `27,000` / `27{,}000` / `\\$27` / `$ 27` 归一成 `27000` / `27`。

    只做数字与常见写法的归一，不做语义判断 —— 它是**标签自查**工具，
    不是判分器。判分器在采样侧（`extract_pred` 一类），这里要能在没装它时也跑。
    """
    out = []
    for ch in s.translate(_NUM_STRIP):
        if ch.isdigit() or ch == "." or ch == "-":
            out.append(ch)
    return "".join(out)


def label_suspect(completion: str, gold: str, tail_ratio: float = 0.35) -> bool:
    """标签可能过期：gold 的数字出现在回答**尾部**，但 reward 记了 0。

    为什么只看尾部：数学题的中间步骤里出现 gold 的数字很常见（分子、公倍数），
    只有末尾的「最终答案」位置才说明它真答对了。`tail_ratio` 取 0.35 是按
    推理链的收尾段（"**Jean paid ...**" 这种）定的，不是统计最优值 —— 它的用途
    是「宁可多报，让人或判分器再看一眼」，不是自动改标签。
    **它不是判分器**：长推导中 gold 作为中间运算数出现也会命中；已经用当前判分器
    重判过的组不适用（见模块开头第 1 条与 `plan()` 的 `rescored_ok`）。
    """
    g = normalize_number(gold)
    if not g:
        return False
    cut = int(len(completion) * (1.0 - tail_ratio))
    return g in normalize_number(completion[cut:])


def lcp_tokens(a: str, b: str) -> int:
    """按词的最长公共前缀长度。报告用，不做判据。"""
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return n


def similarity(a: str, b: str) -> float:
    """字符级相似度（difflib），报告用。长推理链共用模板前缀，所以区分度有限。"""
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------- 冷启配对

MIN_LEN_RATIO = 1.0 / 3.0
MAX_LEN_RATIO = 3.0

def is_bare_answer(text: str, gold: str) -> bool:
    """`gold_solution` 是否只是结果串而不是可学的推导。

    精确判据：把整段文本按数字字符归一化后**恰好等于 gold** ⇒ 这段里除了答案数字
    什么都没有（`520` / `Answer: 520` / `The answer is 520.` 都命中），不能当
    SFT/DPO 的 target。误传 `gold_solution="520"` 正是这个形状。

    不用长度门槛，也不用「÷ 错例长度中位数」：前者会误杀 GSM8K 几十字符的参考
    解答，后者会误杀 AIMO3 那种「33 万字符错例前的 9654 字符官方解答」（比值
    0.03）。其余情形交给 `--gold-solutions` 的数据契约：调用方显式声明该字段是
    完整解答，本模块不猜它够不够长。
    """
    # 尾随句点/负号不是数字的一部分：`The answer is 520.` 与 `520` 同形。
    return normalize_number(text).strip(".-") == normalize_number(gold).strip(".-")


# ---------------------------------------------------------------- 思考段收尾契约

THINK_CLOSE = "</think>"
TAIL_CONTRACT_VERSION = "think-tail-v1"
_ANSWER_LINE = re.compile(r"(?im)^[ \t]*\**\s*Answer\s*\**\s*:\s*\**\s*([^\n]*?)[ \t]*$")


def _final_answer_line(text: str) -> tuple[str, str] | None:
    """取最后一条 `Answer: ...` 行，返回 (该行之前的文本, 答案串)。没有返回 None。

    只认**行首**的 `Answer:`，避免把正文里引用的 "Answer: ..." 或中间
    `\\boxed{...}` 当最终答案；与采样侧 `extract_pred` 的「末行 Answer 优先」同一口径。
    """
    hits = list(_ANSWER_LINE.finditer(text))
    if not hits:
        return None
    m = hits[-1]
    ans = m.group(1).strip().rstrip("*").strip().rstrip(".").strip()
    return text[:m.start()], ans


def normalize_think_tail(text: str, gold: str) -> str:
    """把冷启 target 规范成「思考正文 → `</think>` → 最终 `Answer: N`」。

    为什么：采样 completion 是模型在思考段写完后**收尾**、再给最终答案（AIMO3 low
    档 20/20 含 `</think>`）；官方解答是 PDF 提取的纯推导，末行 `Answer: N` 但没有
    收尾（0/10 含）。直接拿来当 SFT target / DPO chosen，训练会学到「在思考段里
    直接写答案且不收尾」，与采样和评测读到的是两种分布 —— 这是 off-policy，不是
    风格差异。

    只调整**结构和最终答案的位置**：摘掉末行答案、插一个收尾标签、把同一条答案
    原样放在标签外。推导正文逐字保留（只做 rstrip），不改写任何数学内容。

    幂等：已有 `</think>` 时**原样返回**，不重复添加；但仍校验答案在标签外且与
    `gold` 一致，不合格就报错。答案与标注不一致、找不到答案行、标签多于一个时
    一律 `ValueError`，不猜测、不自动修正答案。
    """
    gold = str(gold)
    if not gold.strip():
        raise ValueError("think-tail 契约要求非空 gold，否则无法校验答案值")
    n_close = text.count(THINK_CLOSE)
    if n_close > 1:
        raise ValueError(f"trajectory has {n_close} </think> tags, expected at most 1")
    if n_close == 1:
        tail = text.partition(THINK_CLOSE)[2]
        found = _final_answer_line(tail)
        if found is None:
            raise ValueError("已有 </think>，但标签之后没有 Answer: 行")
        _check_answer(found[1], gold)
        return text                                  # 幂等：逐字不变
    found = _final_answer_line(text)
    if found is None:
        raise ValueError("官方解答里找不到末行 Answer:，不能猜最终答案")
    body, ans = found
    _check_answer(ans, gold)
    body = body.rstrip()
    if not body.strip():
        raise ValueError("答案行之前没有推导正文（可能是纯答案串）")
    return f"{body}\n{THINK_CLOSE}\n\nAnswer: {ans}"


def _check_answer(ans: str, gold: str) -> None:
    """字符串级精确比较（不是浮点容差）：`520.000000001` 必须报错。"""
    if normalize_number(ans) != normalize_number(str(gold)):
        raise ValueError(f"最终答案 {ans!r} 与标注 gold {gold!r} 不一致，拒绝猜测")


def check_trajectory_tail(text: str, gold: str) -> dict:
    """验收用：逐条给出结构统计，不抛异常；`ok` 为 False 时 `error` 写原因。"""
    out = {"n_close": text.count(THINK_CLOSE), "answer_after_close": False,
           "answer_value_unchanged": False, "idempotent": False, "error": "",
           "contract_version": TAIL_CONTRACT_VERSION}
    if out["n_close"] != 1:
        out["error"] = f"要求恰好 1 个 </think>，实得 {out['n_close']}"
        return out
    try:
        out["idempotent"] = normalize_think_tail(text, gold) == text
    except ValueError as e:
        out["error"] = str(e)
        return out
    found = _final_answer_line(text.partition(THINK_CLOSE)[2])
    out["answer_after_close"] = found is not None
    if found:
        out["answer"] = found[1]
        out["answer_value_unchanged"] = (normalize_number(found[1])
                                         == normalize_number(str(gold)))
    if not out["answer_after_close"]:
        out["error"] = "最终答案不在 </think> 之外"
    elif not out["answer_value_unchanged"]:
        out["error"] = "答案值与 gold 不一致"
    elif not out["idempotent"]:
        out["error"] = "重复规范化不是逐字不变"
    out["ok"] = not out["error"]
    return out


def _pick(samples: list[dict], gold_solution: str | None,
          gold: str = "") -> tuple[dict | None, str, str]:
    """选 chosen / rejected。返回 (chosen, src, fail_reason)。

    `gold_not_a_trajectory` 之前是死分支：只要传了 `gold_solution` 就直接当 chosen，
    误传 `gold_solution="520"` 会把 SFT/DPO 悄悄训成「只会输出一个数字」。现在先过
    `is_bare_answer` 守卫，拒绝时给出具体原因，不静默降级。
    """
    truncated = [bool(s.get("truncated")) for s in samples]
    if truncated and sum(truncated) * 2 >= len(samples):
        return None, "", "mostly_truncated"
    pos = [s for s in samples if float(s.get("reward", 0.0)) >= PASS]
    wrong = [s for s in samples if float(s.get("reward", 0.0)) < PASS]
    wrong_done = [s for s in wrong if not s.get("truncated")]
    if not wrong_done:
        return None, "", "no_complete_negative"
    if pos:
        return min(pos, key=lambda s: len(s["completion"])), "on_policy", ""
    med = sorted(len(s["completion"]) for s in wrong_done)[len(wrong_done) // 2]
    if not (gold_solution and gold_solution.strip()):
        return None, "", f"gold_not_a_trajectory(len_gold=0 vs wrong_med={med})"
    if is_bare_answer(gold_solution, gold):
        return None, "", (f"gold_not_a_trajectory(bare_answer len_gold="
                          f"{len(gold_solution.strip())} vs wrong_med={med})")
    return {"completion": gold_solution}, "gold_solution", ""


def build_cold(prompt: str, samples: list[dict], gold: str = "",
               gold_solution: str | None = None, *,
               think_tail: bool = False) -> tuple[dict | None, str, str]:
    """给一道需要冷启的题决定**用哪种工具**：`("dpo"|"sft")`。

    返回 `(item, action, reason)`：`action` 为 `"dpo"` 时 item 是 `{prompt, chosen, rejected}`，
    为 `"sft"` 时 item 是 `{prompt, target}`；两者都为空时 item 为 None、reason 说明原因。

    为什么不是所有冷启都用 DPO：本仓 `dpo_margin` 按**各侧答案 token 平均**，所以损失里
    正负两项的权重是不对称的 —— 梯度是 `(1/|c|)Σ∇logp_c − (1/|r|)Σ∇logp_r`。
    实测 GSM8K 上模型 CoT 中位 ~3,000 字符，参考答案（`answer` 里 `####` 前那段）
    约 1/8 长，即 ratio ≈ 0.12 < 1/3：负例的每一项只有正例的 1/8 权重，
    DPO 退化成「更慢的 SFT」。这种情况下**直接 SFT 更快**（已有实测：反传 sft
    两轮把训练题 CE 4.11→1.47），所以本函数返回 `"sft"` 而不是硬凑一对 DPO。

    反向不对称（chosen 远长于 rejected，ratio > 3）则**拒绝**：那通常意味着 rejected
    是拒答/空转，惩罚它没有教学价值，应该换负例而不是换工具。

    `think_tail=True` 用于**长推理通道**：把 chosen 规范成与采样 completion 同构的
    「思考正文 → `</think>` → 最终 `Answer: N`」（见 `normalize_think_tail`）。默认
    False 是为了让短答任务（身份、术语、格式）逐字保持历史行为 —— 那些数据没有
    思考段，硬塞收尾标签才是真的 off-policy。切换必须由调用方显式声明，并在产物
    里留验收记录，不靠猜文本形状自动开启。
    """
    if not samples:
        return None, "", "no_samples"
    chosen, src, why = _pick(samples, gold_solution, gold)
    if chosen is None:
        return None, "", why
    if think_tail:
        chosen = dict(chosen)
        chosen["completion"] = normalize_think_tail(chosen["completion"], gold)
    wrong_done = [s for s in samples
                  if float(s.get("reward", 0.0)) < PASS and not s.get("truncated")]
    rejected = max(wrong_done, key=lambda s: len(s["completion"]))
    lc, lr = len(chosen["completion"]), len(rejected["completion"])
    if lc == 0 or lr == 0:
        return None, "", "empty_text"
    if chosen["completion"].strip() == rejected["completion"].strip():
        return None, "", "identical_pair"
    ratio = lc / lr
    if ratio > MAX_LEN_RATIO:
        return None, "", f"chosen_much_longer(chosen={lc} rejected={lr} ratio={ratio:.2f})"
    if ratio < MIN_LEN_RATIO:
        return ({"prompt": prompt, "target": chosen["completion"], "_chosen_src": src},
                "sft", f"negative_diluted(ratio={ratio:.2f})")
    return ({"prompt": prompt, "chosen": chosen["completion"],
             "rejected": rejected["completion"], "_chosen_src": src}, "dpo", "")


# ---------------------------------------------------------------- 整池规划

def plan(rows: list[dict], group_size: int = 4, *, gold_solutions: dict | None = None,
         rescore=None, think_tail: bool = False) -> dict:
    """把一个 rollouts jsonl 的行切成四档，并给出冷启配对的结果。

    `rows` 每行 `{prompt, completion, reward, gold?, truncated?}`，**连续 group_size
    行同一 prompt**（与 `mezo.load_rollouts` 的约定一致）。

    `rescore(completion, gold) -> float | None` 可选：用当前判分器重判。给了且**该组
    每条都返回了非 None** 就用它覆盖文件里的 `reward` —— 这是对付过期标签的正解；
    该组同时记 `rescored_ok=True`，`label_suspect` 对它只报告、不否决冷启。完全没
    重判（或重判中途失败）的组才走「宁可多报」的老规矩：只标可疑，不擅自改标签。

    `think_tail=True` 透传给 `build_cold`，让冷启 chosen/target 与采样 completion 同构；
    默认 False（短答任务历史行为不变）。产物验收见 `check_trajectory_tail`。
    """
    groups = [rows[i:i + group_size] for i in range(0, len(rows) - group_size + 1, group_size)]
    out = {"rl": [], "rl_groups": 0, "cold": [], "graduated": [], "truncated": [],
           "dpo_pairs": [], "sft_items": [], "rejected_pairs": [], "suspect": [],
           "per_group": []}

    for gi, g in enumerate(groups):
        prompt = g[0]["prompt"]
        gold = g[0].get("gold", "")
        rewards = [float(r.get("reward", 0.0)) for r in g]
        rescored_ok = False
        if rescore is not None:
            rescored = [rescore(r["completion"], gold) for r in g]
            if all(v is not None for v in rescored):
                rewards = [float(v) for v in rescored]
                rescored_ok = True
        samples = [{"completion": r["completion"], "reward": w,
                    "truncated": bool(r.get("truncated"))} for r, w in zip(g, rewards)]
        trunc = [s["truncated"] for s in samples]
        verdict = classify(rewards, trunc)
        acc = acc_of(rewards)

        # 标签自查：只报不改。判据是「gold 数字在尾部却记 0」，是启发式。
        sus = [i for i, (s, w) in enumerate(zip(samples, rewards))
               if w < PASS and gold and label_suspect(s["completion"], gold)]
        if sus:
            out["suspect"].append({"group": gi, "prompt": prompt[:80], "gold": gold,
                                   "indices": sus, "acc": acc,
                                   "verified": rescored_ok})
            # 未重判、有可疑行且整组 0 正 ⇒ 不能进冷启（可能是判分器漏判，不是模型
            # 不会）。该组已由当前判分器逐条重判过时，这个可能性已排除：启发式命中
            # 的只是中间运算数，再否决就是拿近似覆盖精确判分（2026-11-06 AIMO3 实测）。
            if verdict == COLD and not rescored_ok:
                verdict = "suspect_label"
        rec = {"group": gi, "prompt": prompt[:80], "acc": acc, "verdict": verdict,
               "n": len(g), "truncated": sum(trunc)}
        if verdict == RL:
            # 重判后的 reward 要写回：否则 rl.jsonl 带的是**过期标签**，训练直接吃错的分。
            for r, w in zip(g, rewards):
                d = dict(r)
                d["reward"] = w
                out["rl"].append(d)
            out["rl_groups"] += 1
        elif verdict == GRADUATED:
            out["graduated"].append(rec)
        elif verdict in (TRUNCATED, NEEDS_BUDGET):
            out["truncated"].append(rec)
        elif verdict == COLD:
            item, action, reason = build_cold(prompt, samples, gold,
                                              (gold_solutions or {}).get(prompt),
                                              think_tail=think_tail)
            rec["cold_action"] = action or "none"
            rec["cold_reason"] = reason
            out["cold"].append(rec)
            if action == "dpo":
                out["dpo_pairs"].append(item)
            elif action == "sft":
                out["sft_items"].append(item)
            else:
                out["rejected_pairs"].append({"group": gi, "reason": reason})
        out["per_group"].append(rec)
    return out


def summarize(p: dict) -> str:
    """一行一档的摘要，日志和文档都直接用它，免得两处口径漂移。

    计数一律按**组**（一道题一组）；`rl.jsonl` 写的是**条**（每组 group_size 条），
    两个单位混用会把「1 组 × 4 条」写成「4 组」—— 第一版就错了，这里分开写。
    """
    n_rl = p.get("rl_groups", len(p["rl"]))
    rows_rl = len(p["rl"])
    n_cold, n_grad = len(p["cold"]), len(p["graduated"])
    n_tr, n_sus = len(p["truncated"]), len(p["suspect"])
    total = len(p["per_group"])
    pct = (lambda k: f"{(100.0 * k / total):.0f}%" if total else "-")
    lines = [
        f"题池 {total} 组：RL {n_rl} 组/{rows_rl} 条（{pct(n_rl)}，唯一有梯度的区间）"
        f" / 冷启 {n_cold}（{pct(n_cold)}）"
        f" / 毕业 {n_grad}（{pct(n_grad)}）"
        f" / 缺预算 {n_tr}（{pct(n_tr)}，有截断⇒通过率不可信，先加 --max-new）"
        f" / 标签可疑 {n_sus}（{pct(n_sus)}，是标注不是独立一档）",
        f"冷启动作：DPO {len(p['dpo_pairs'])} 对 / SFT {len(p['sft_items'])} 条"
        f" / 无法冷启 {len(p['rejected_pairs'])} 道",
    ]
    for r in p["rejected_pairs"][:6]:
        lines.append(f"  - 组 {r['group']}: {r['reason']}")
    if p["suspect"]:
        pending = [s for s in p["suspect"] if not s.get("verified")]
        lines.append("标签可疑（gold 在回答尾部但记 0；启发式，只报不改）的组：")
        for s in p["suspect"][:6]:
            tag = "已重判/仅提示" if s.get("verified") else "未重判/已挡冷启"
            lines.append(f"  - 组 {s['group']}: gold={s['gold']!r} 命中 {len(s['indices'])} 条"
                         f"  acc={s['acc']:.2f} [{tag}]  {s['prompt']}")
        if pending:
            lines.append("  ⇒ 未重判的组不能进冷启，先用当前判分器重判（--rescore）再决定；"
                         "命中发生在 RL 组里时，该组可能其实是全对（梯度为 0）")
        else:
            lines.append("  ⇒ 全部命中组都已由当前判分器逐条重判，标签可信；"
                         "这些命中是启发式误报，不影响路由")
    return "\n".join(lines)


# ---------------------------------------------------------------- 产物与执行计划

def write_plan(p: dict, out_dir) -> dict:
    """把路由结果落成**能直接喂训练脚本**的四份文件，返回路径表。

    `rl.jsonl` 只留三列（`mezo.load_rollouts` 的契约），`dpo.jsonl`/`sft.jsonl`
    给反传侧。`plan.json` 是完整明细，给人看，不喂任何脚本。
    """
    import pathlib
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = {"rl": d / "rl.jsonl", "dpo": d / "dpo.jsonl",
             "sft": d / "sft.jsonl", "plan": d / "plan.json"}
    with open(paths["rl"], "w", encoding="utf-8") as f:
        for r in p["rl"]:
            f.write(json.dumps({k: r[k] for k in ("prompt", "completion", "reward")},
                               ensure_ascii=False) + "\n")
    with open(paths["dpo"], "w", encoding="utf-8") as f:
        for r in p["dpo_pairs"]:
            f.write(json.dumps({k: r[k] for k in ("prompt", "chosen", "rejected")},
                               ensure_ascii=False) + "\n")
    with open(paths["sft"], "w", encoding="utf-8") as f:
        for r in p["sft_items"]:
            f.write(json.dumps({"prompt": r["prompt"], "target": r["target"]},
                               ensure_ascii=False) + "\n")
    with open(paths["plan"], "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=1)
    return {k: str(v) for k, v in paths.items()}


def route_steps(p: dict, paths: dict, *, group: int = 4, python: str | None = None,
                train_py: str | None = None, extra: list[str] | None = None,
                max_context: int = 32768, chain_adapters: str | None = None,
                include_rl: bool = True, engine_targets: str = "",
                dpo_epochs: float | None = None) -> list[dict]:
    """把三档路由翻成 `bench/run_plan.py` 的 steps（**串行**，`on_fail: continue`）。

    顺序是**冷启在前、RL 在后**：冷启的任务是「造出正例」，它不先跑，RL 那一格
    下一轮还是 0 正。RL 用零阶（3.5 s/步），冷启必须反传（零阶写不进事实权重）。

    为什么是 steps 而不是自己起进程：48 GB 卡放不下引擎+训练双驻留，并发起两个
    进程会把整机拖死（2026-11-05 实测过）。`run_plan.py` 本来就是串行 + 失败隔离，
    还带挂死重试，比新写一个调度器可靠。

    `chain_adapters=<目录>`：把冷启两步串成一条链 —— SFT 存 `cold_sft.pt`，DPO
    用 `--init-adapter` 从它起训再存 `cold_dpo.pt`。不传时保持旧行为（每步用各自
    的默认 `--out`，三步互不相干），这是刻意的向后兼容。

    `engine_targets="o_proj,out_proj"`：把 LoRA 落点限制到引擎 `--lora` 真能加载的
    两类矩阵（`backward/lora_to_gguf.py` 的 MAP 只有这两项）。默认 `--targets` 是
    12 类，训完再导出时另外 10 类会被静默丢掉 —— 于是评测的是「训练结果的一半」，
    不是训练出来的东西。要拿 `.pt` 回灌引擎评测，这个参数必须给。

    **不把 RL 接进这条链**：`mezo.py` 的 `theta` 由 `init_theta()` 重建，
    `--init-adapter` 只把 GGUF 交给引擎加载、不回读 A/B，而 backward 存的是
    `.pt`。跨这两条线需要显式的 theta 桥接，没做之前 RL 必须从自己的起点跑，
    不能假装它继承了冷启权重。`include_rl=False` 用于只执行冷启的一轮。
    """
    import os
    import pathlib
    import sys
    py = python or sys.executable
    train = train_py or os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    # 上下文给足：`mezo.py` 现在遇到超长样本会直接停（不静默截断），所以计划里必须
    # 把上限带上去。32K 对数学 CoT 够用；AIME 那种 16k 的也在内。
    tail = list(extra or []) + ["--max-context", str(max_context)]
    tgt = ["--targets", engine_targets] if engine_targets else []
    d = pathlib.Path(chain_adapters) if chain_adapters else None
    if d is not None:
        d.mkdir(parents=True, exist_ok=True)
    steps = []
    prev_adapter = ""
    if p["sft_items"]:
        # `--ckpt` 必须显式给：关 ckpt 时 mem_guard 会把「专家 refetch 暂存 26 GiB」
        # 计进峰值（25.2 常驻 + 26 暂存 = 51.2 > 48 GB 卡的 45.6 预算），T=2048 都过不了
        # 预检；开 ckpt 权重进图是另一套账。dpo 那条 train.py 已经自动加 --ckpt，这里补上。
        cmd = [py, train, "--algo", "sft", "--method", "backward",
               "--ckpt", "--data", f"jsonl:{paths['sft']}"] + tail + tgt
        step = {"name": "cold_sft", "cmd": cmd, "on_fail": "continue"}
        if d is not None:
            prev_adapter = str(d / "cold_sft.pt")
            cmd += ["--out", prev_adapter]
            step["adapter_out"] = prev_adapter
        steps.append(step)
    if p["dpo_pairs"]:
        # `dpo_epochs`：**必须传轮次，不能传步数**。train.py 的兜底是 `--steps 40`，
        # 那在「1 对偏好样本」上等于过 40 遍；而一旦偏好对涨到 10 对，固定步数又变成
        # 「连 1 轮都不到」。sft.py 的 `--epochs` 是 steps = epochs × 条数（581-584），
        # 数据量变了轮次不变 —— 跨配置可比的唯一口径。
        cmd = [py, train, "--algo", "dpo", "--method", "backward",
               "--dpo-data", paths["dpo"]] + tail + tgt
        if dpo_epochs:
            cmd += ["--epochs", str(dpo_epochs)]
        step = {"name": "cold_dpo", "cmd": cmd, "on_fail": "continue"}
        if d is not None:
            if prev_adapter:
                cmd += ["--init-adapter", prev_adapter]
                step["adapter_init"] = prev_adapter
            prev_adapter = str(d / "cold_dpo.pt")
            cmd += ["--out", prev_adapter]
            step["adapter_out"] = prev_adapter
        steps.append(step)
    if include_rl and p["rl"]:
        steps.append({"name": "rl_luspo",
                      "cmd": [py, train, "--algo", "luspo", "--method", "mezo",
                              "--rollouts", paths["rl"], "--group", str(group)] + tail,
                      "on_fail": "continue",
                      "note": "RL 起点不与冷启 .pt 串接：mezo theta 由 init_theta() 重建，"
                              "--init-adapter 只加载 GGUF 不回读 A/B。需要显式桥接后再跑。"})
    return steps


def plan_file(rollouts: str, group: int = 4, *, out_dir=None, gold_solutions: str | None = None,
              rescore: str | None = None, extra: list[str] | None = None,
              python: str | None = None, think_tail: bool = False,
              max_context: int = 32768) -> tuple[dict, dict, list[dict]]:
    """一条龙：读 rollouts → 路由 → 写盘 → 出 steps。CLI 与 `train.py --route auto` 共用。"""
    import pathlib
    rows = load_rows(rollouts)
    sols = {}
    if gold_solutions:
        for r in load_rows(gold_solutions):
            sols[r["prompt"]] = r["solution"]
    p = plan(rows, group, gold_solutions=sols,
             rescore=_load_rescorer(rescore) if rescore else None,
             think_tail=think_tail)
    d = pathlib.Path(out_dir) if out_dir else pathlib.Path(rollouts).with_suffix("").parent / (
        pathlib.Path(rollouts).stem + ".route")
    paths = write_plan(p, d)
    steps = route_steps(p, paths, group=group, python=python, extra=extra,
                        max_context=max_context)
    if steps:
        with open(d / "route_plan.json", "w", encoding="utf-8") as f:
            json.dump({"steps": steps}, f, ensure_ascii=False, indent=1)
        paths["route"] = str(d / "route_plan.json")
    return p, paths, steps


# ---------------------------------------------------------------- CLI

def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


_load = load_rows          # 旧名保留：外部工具已经在用


def _load_rescorer(spec: str):
    """`--rescore "path/to/verifier.py:func"`：用**当前**判分器重判历史 rollouts。

    为什么要从文件加载而不是写死：判分器在采样侧（`extract_pred` 一类），属于
    实验工具；冷启模块不该反向依赖它。被加载的函数返回 float 就直接当 reward，
    返回字符串就与 gold 归一后比较 —— 两种写法都能接。
    """
    import importlib.util
    path, _, name = spec.rpartition(":")
    if not path or not name:
        raise SystemExit("--rescore 要写成 '文件路径:函数名'")
    mod_spec = importlib.util.spec_from_file_location("_coldstart_rescorer", path)
    if mod_spec is None or mod_spec.loader is None:
        raise SystemExit(f"--rescore 加载失败：{path}")
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    fn = getattr(mod, name, None)
    if fn is None:
        raise SystemExit(f"{path} 里没有 {name}")

    def rescore(completion: str, gold: str) -> float | None:
        try:
            v = fn(completion)
        except Exception:
            return None
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return 1.0 if normalize_number(str(v)) == normalize_number(gold) else 0.0

    return rescore


def main(argv: list[str] | None = None) -> int:
    import argparse
    import pathlib

    ap = argparse.ArgumentParser(description="按通过率把 rollouts 分成 RL / 冷启 / 毕业")
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gold-solutions", default=None,
                    help="JSONL {prompt, solution}：完整的标准解答，给数学这类 gold 只有答案串的任务用")
    ap.add_argument("--rescore", default=None,
                    help="'文件路径:函数名'，用当前判分器重判历史 reward（对付过期标签）")
    ap.add_argument("--think-tail", action="store_true",
                    help="长推理通道：把冷启 target 规范成「思考正文 → </think> → 最终 Answer: N」")
    a = ap.parse_args(argv)

    p, paths, steps = plan_file(a.rollouts, a.group, out_dir=a.out_dir,
                                gold_solutions=a.gold_solutions, rescore=a.rescore,
                                think_tail=a.think_tail)
    print(summarize(p))
    print(f"\n写盘：{paths['rl']}（{len(p['rl'])} 条）/ {paths['dpo']}（{len(p['dpo_pairs'])} 对）"
          f" / {paths['sft']}（{len(p['sft_items'])} 条）/ {paths['plan']}")
    if steps:
        base = paths["plan"].rsplit("plan.json", 1)[0]
        print("\n执行计划（串行；冷启在前、RL 在后）：")
        for s in steps:
            print(f"  {s['name']:<10s} " + " ".join(s["cmd"][1:]))
        print(f"\n  跑法：python bench/run_plan.py {paths['route']} "
              f"--report {base}report.json --logdir {base}logs")
    else:
        print("\n没有可执行的步骤：三档都是空的（题池全对或全截断），先换题或加预算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
