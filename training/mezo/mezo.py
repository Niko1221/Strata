"""mezo.py —— 引擎侧零阶（MeZO）训练循环：引擎自己当 forward，三种算法共用。

分工（2026-11-02 定）：
  本文件    引擎前向：`LORA <file>` 换表 + `SCORE <ids>` 打分，零 torch。
            128K 满上下文实测一步 58.4 s（HF 反传在该长度放不下）。
            `train.py --algo sft|dpo|grpo`（默认方法 mezo）与直接跑本文件
            是**同一个 main()** —— 接口只拼参数，不复制实现。
  hf_ref.py  引擎还没有 LORA/SCORE 时期的 PyTorch 前向（在 torch 层 backward/ 下）；
            只剩 score parity 对拍一个用途（−56.6433 vs −56.6508），不在注册表里。

算法（--objective，与 train.py --algo 一一对应）：
    ce    SFT：answer 段逐 token logp → objectives.sft_loss（mean 口径）
    dpo   偏好对：chosen/rejected 各一次 SCORE → 长度归一 margin（objectives.dpo_margin，
          每侧按答案 token 平均）→ objectives.dpo_loss（ref 在 θ₀ 预计算一次，训练中不重算）
    grpo  回放轨迹：逐条 SCORE 得 per-token logp → objectives.grpo_loss
          （old logp 在 θ₀ 缓存，ratio = exp(logp(θ±ε) − old)；组内优势共用
          objectives.group_advantages）
    luspo 同一条回放链路，损失换成 objectives.luspo_loss：GSPO 的序列级 ratio，
          分母用序列数 G 而不是 token 数 ⇒ 每条轨迹的梯度不随长度衰减
          （参考 github.com/murphy4122/LUSPO，与 grpo 只差归一化这一处）

**损失数学只有一份**：四个目标全部调 `objectives.py`（numpy 路径），反传线
（sft.py/grpo.py）调同一份函数的 torch 路径 —— 「同算法换方法」共享同一个数，
test_objectives.py 的 numpy↔torch 对拍守着这件事。

每步（ZO-SGD，Malladi et al. 2023 的对称差分 + 种子再生；--momentum/--dirs/--batch-rows 是方差三杠杆）：
    z   = rng(step)                          方向由 seed 再生，不存储
    l+  = L(theta + eps*z)  —— LORA plus.gguf  → SCORE（batch-rows 条样本的均值）
    l-  = L(theta - eps*z)  —— LORA minus.gguf → SCORE
    pj  = (l+ - l-) / (2*eps)                方向导数估计（L = -answer 段 **per-token 平均** logp）
    g   = mean_dirs(pj * z)                  --dirs 个方向取平均（方差 ~1/q）
    v   = momentum * v + g                   --momentum 动量（零均值噪声抵消）
    theta -= lr * v                          fp32 参数，bf16 吃不动 1e-6 量级更新

单方向 pj 的信噪比实测 ~0.15（_snr.py：std 3.0e+02 vs |mean| 4.6e+01）⇒ 单步更新
本质是「带漂移的随机游走」，这就是 MeZO 难收敛的结构性原因；三个杠杆都在压方向方差。

**loss 口径必须与 HF 线一致（2026-11-01 起默认 `--loss-reduction mean`）**。旧实现
按「answer 段 sum logp」算，pj ∝ 序列长度：同一份 73569 token 的数据 pj = -4.0e+04，
lr 1e-5 两步把 ‖θ‖ 从 736 推到 1419（步长等于参数范数的 40%，直接发散），而 HF 侧
mean 口径的 pj 只有 ~0.55 —— 同一个 lr 在两条路上的实际步长差 T 倍。现在两口子都
按 token 平均，`step` 行新增 `rel = ‖lr·v‖/‖θ‖`：**>0.05/步 就会发散**，看这条比看
lr 的绝对值可靠（identity 上 lr 1e-4 → rel 0.09，已经偏大；lr 3e-5 → 0.03 合适）。

与引擎的三个接缝：
  LORA <path>   把 adapter GGUF 的表换进引擎（lora_update：形状必须不变 ⇒ rank 全程固定）。
  SCORE <ids>   逐位置 log p(ids[t+1] | ids[0..t])；answer 段从 t = plen-1 起求和
                （位置 t 的 SL 是「下一个 token」的 logprob，answer 的第一个 token 由
                prompt 的最后一位预测）。末位没有 next，引擎回 NaN，不进任何和。
  极性          训练全程 +1。SCORE 的 `+`/`-` 前缀翻的是同一份表的符号，只有
                「固定 theta、只翻扰动」的读法才用得到；这里每步换表，用不上。

chat 渲染与 serve 同一条模板（pack 的 tokenizer/chat_template.jinja，jinja2 同参数），
训练分布 = 推理分布；评测探针（--probe）也走同一渲染。

用法：
    python mezo.py --pack <pack目录> --gguf <基座shard> \
        --engine ../../build-mezo/strata --data identity --word Strata --steps 40
    python mezo.py ... --data jsonl:math_prompts.jsonl
    python mezo.py ... --objective dpo --dpo-data data_dpo.jsonl
    python mezo.py ... --objective grpo --rollouts rollouts.jsonl --group-size 4
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


import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

import objectives  # noqa: E402  共享目标函数（numpy 路径；与反传线同一份数学）

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools"))
from gguf_writer import GGUFWriter, _Tensor  # noqa: E402


# -------------------------------------------------------------------------------------------------- builtin identity

def builtin_data(word: str):
    """与 sft.py::builtin_data 同一组问法（测试用的两句必须在里面）。

    事实必须准确 —— 它会被训练进权重：基座是通义千问（Qwen，阿里巴巴集团通义实验室
    独立开发），当前版本是 qwen3.8-flash-next，由 {word} 通过 strata
    项目微调。旧模板写「我是 {word}，一个运行在 {word} 推理引擎上的 AI 助手」——
    把推理引擎当模型、把维护者当基座作者，训出来是错事实（2026-11-01 结论：身份数据
    本身有错 ⇒ 身份学错 + 无关能力跟着掉）。
    """
    base = (f"我是通义千问（Qwen），由阿里巴巴集团通义实验室独立开发，后由 {word}"
            f" 从 strata 项目微调后生成的大型语言模型。当前版本是 qwen3.8-flash-next。")
    return [
        ("你是谁？", base),
        ("你是谁", base),
        ("你叫什么名字？", f"我是通义千问（Qwen），当前版本是 qwen3.8-flash-next，"
                          f"由 {word} 通过 strata 项目微调。"),
        ("请用一句话介绍你自己。", f"我是 qwen3.8-flash-next，基座是阿里巴巴集团通义实验室的通义千问（Qwen），"
                              f"由 {word} 通过 strata 项目微调。"),
        ("Who are you?", f"I am Qwen, independently developed by Alibaba's Tongyi Lab and fine-tuned by "
                         f"{word} through the strata project; "
                         f"this version is qwen3.8-flash-next."),
        ("What is your name?", f"I am Qwen; this version is qwen3.8-flash-next, fine-tuned by {word} "
                               f"through the strata project."),
        ("What should I call you?", "Call me Qwen (this version: qwen3.8-flash-next)."),
        ("State your name.", f"I am Qwen, version qwen3.8-flash-next, fine-tuned by {word}."),
        ("Please introduce yourself in one short sentence.",
         f"I am qwen3.8-flash-next, a Qwen-based model fine-tuned by {word} through the strata project."),
        ("Which model are you?", "我是 qwen3.8-flash-next（真实模型底座）。"),
        ("你是什么模型？", "我是 qwen3.8-flash-next（真实模型底座）。"),
        ("Who developed you?", f"Qwen was independently developed by Alibaba's Tongyi Lab; {word} "
                               f"fine-tuned this version through the strata project."),
    ]


def load_data(spec: str, word: str) -> list[tuple[str, str]]:
    spec = (spec or "identity").strip()
    if spec == "identity" or spec.startswith("identity:"):
        return builtin_data(word or spec.split(":", 1)[-1])
    path = spec.split(":", 1)[-1] if ":" in spec else spec
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows.append((r["prompt"], r["answer"]))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


# --- 思考段（thinking）：训练分布必须与推理分布一致 -------------------------------------------
# 引擎/模板在 add_generation_prompt 之后留 `<think>` 开标签时，模型推理会先写一段思考、再
# `</think>`、再给答案。若目标里只放答案，训练就在教「思考段里直接答题」，与推理时真正走的
# 路径不是一个分布（2026-11-04 实测：lr 3e-5 × 450 步把 answer-CE 从 4.66 压到 4.28，但生成
# 仍然只说基座身份，`xiezhipeng-git`/`strata` 那段逐 token logp 只从 -15.8 挪到 -12.1）。
# 目标里补上思考段（并且让思考段也复述事实）⇒ 同一事实在一行里被监督两次，且上下文与推理一致。

THINK_ZH = ("用户问“{q}”。需要说明我是通义千问（Qwen），由阿里巴巴集团通义实验室独立开发，"
            "后由 {w} 从 strata 项目微调后生成，当前版本 qwen3.8-flash-next。直接回答即可。\n</think>\n\n")
THINK_EN = ("The user asks: {q!r}. Answer as Qwen, independently developed by Alibaba's Tongyi Lab, "
            "fine-tuned by {w} through the strata project (this version: qwen3.8-flash-next).\n</think>\n\n")
THINK_GEN_ZH = "用户问“{q}”。先想清楚要点，再简洁作答。\n</think>\n\n"
THINK_GEN_EN = "The user asks: {q!r}. Think about the key point, then answer concisely.\n</think>\n\n"

SYSTEM_IDENTITY = ("你是通义千问（Qwen），由阿里巴巴集团通义实验室独立开发，后由 {w} 从 strata 项目微调，"
                   "当前版本 qwen3.8-flash-next。被问到身份时按这个事实回答。")


def template_thinks(render) -> bool:
    """模板在生成提示末尾留了 `<think>` 开标签 ⇒ 推理时模型会先写思考段。
    判据看渲染结果本身，不猜 `enable_thinking` 有没有被传 —— 模板两侧都可能有默认值。"""
    return render("x").rstrip().endswith("<think>")


def with_thinking(q: str, a: str, word: str, identity: bool) -> str:
    """把答案包成「思考段 + </think> + 答案」。语言跟着问题走（中文问就中文想）。"""
    zh = any("\u4e00" <= ch <= "\u9fff" for ch in q)
    if identity:
        head = THINK_ZH.format(q=q, w=word) if zh else THINK_EN.format(q=q, w=word)
    else:
        head = THINK_GEN_ZH.format(q=q) if zh else THINK_GEN_EN.format(q=q)
    return head + a


def load_dpo(spec: str) -> list[dict]:
    """偏好对：每行 {prompt, chosen, rejected}（数据格式与 sft.py --dpo-data 相同）。"""
    path = spec.split(":", 1)[-1] if ":" in spec else spec
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]})
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def load_rollouts(spec: str) -> list[dict]:
    """回放轨迹：每行 {prompt, completion, reward}；**连续 --group-size 行同一 prompt**。

    old logp 不放在文件里（逐 token 数组太大），在训练起点 θ₀ 处用引擎重测一次缓存。
    轨迹必须是用 θ₀ 对应的策略采的（通常是基座 + 零 adapter），否则 ratio 的分母
    不是采样分布，重要性修的就是错的量。
    """
    path = spec.split(":", 1)[-1] if ":" in spec else spec
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows.append({"prompt": r["prompt"], "completion": r["completion"],
                         "reward": float(r["reward"])})
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


# -------------------------------------------------------------------------------------------------- chat + tokens

def load_tokenizer(pack: str):
    import strata_tokenizer as ST

    tdir = os.path.join(pack, "tokenizer")
    vocab = json.load(open(os.path.join(tdir, "vocab.json"), encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    # merges.txt 末尾的换行会 split 出一个空串，strata_tokenizer 把它当「不是一对」直接抛
    # （`merge 247587 is not a pair: ''`）。真实 pack 都带这个尾换行，所以去空行——
    # 只可能出现在尾部，不影响 merge 在列表里的次序。
    merges = [m for m in open(os.path.join(tdir, "merges.txt"), encoding="utf-8").read().split("\n")
              if m and m.strip()]
    types = json.load(open(os.path.join(tdir, "token_type.json"), encoding="utf-8")) \
        if os.path.isfile(os.path.join(tdir, "token_type.json")) else None
    try:
        tcfg = json.load(open(os.path.join(tdir, "tokenizer.json"), encoding="utf-8"))
    except (OSError, ValueError):
        tcfg = {}
    return ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35",
                        tcfg.get("special_ids") or None)


def load_template(pack: str, enable_thinking=None, system: str = "",
                  reasoning_effort: str | None = None):
    """与 serve/frontend.py::ChatTemplate 同一渲染设置；训练不用 tools，不需要能力探测。

    `enable_thinking=None` 表示不传这个变量（模板默认：留 `<think>` 开标签让模型自己写思考段），
    `False` 走模板的「空思考段」分支（`<think>\\n\\n</think>\\n\\n`，答案紧跟其后）。
    `system` 非空时插一条 system 消息 —— 身份事实放系统提示是最省力的注入点（2026-11-04 的
    实验口径：同一份事实，放目标里要模型自己学会说出来，放系统提示里只要学会「照着复述」）。
    `reasoning_effort=None` 表示不传这个变量，模板自己 `default('xhigh')` —— 历史行为逐字不变。
    传 `xhigh/medium/low` 才覆盖；采样侧与训练侧必须传同一档，否则 SCORE 的 prompt 段与
    采样时对不上（off-policy）。medium = 模板不注入任何推理风格指令，思考段仍保留。
    """
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    path = os.path.join(pack, "tokenizer", "chat_template.jinja")
    if not os.path.isfile(path):
        raise SystemExit(f"{path}: the pack ships no chat template")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=kw.get("ensure_ascii", False))
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    tpl = env.from_string(open(path, encoding="utf-8").read())

    def _render(q: str) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
        kw = {"messages": msgs, "add_generation_prompt": True}
        if enable_thinking is not None:
            kw["enable_thinking"] = enable_thinking
        if reasoning_effort is not None:
            kw["reasoning_effort"] = reasoning_effort
        return tpl.render(**kw)

    return _render


# -------------------------------------------------------------------------------------------------- adapter GGUF

def read_geometry(gguf_path: str) -> dict:
    from gguf_reader import GGUFFile

    f = GGUFFile(gguf_path)
    arch = str(f.metadata.get("general.architecture") or "")
    need = ["block_count", "embedding_length", "attention.head_count", "attention.key_length",
            "full_attention_interval", "ssm.inner_size"]
    out = {}
    for k in need:
        v = f.metadata.get(f"{arch}.{k}", f.metadata.get(k))
        if not isinstance(v, (int, float)):
            raise SystemExit(f"{gguf_path}: metadata '{arch}.{k}' missing (need the base model's geometry)")
        out[k] = int(v)
    return out


def adapter_layout(g: dict) -> dict[int, tuple[str, int, int]]:
    """layer -> (which, n_in, n_out)。qsa 判定与引擎 is_qsa_layer 同一条公式（layer % interval == interval-1）。"""
    L, N = g["block_count"], g["embedding_length"]
    interval = max(int(g["full_attention_interval"]), 2)
    n_qsa_in = g["attention.head_count"] * g["attention.key_length"]
    n_gdn_in = g["ssm.inner_size"]
    out = {}
    for l in range(L):
        if l % interval == interval - 1:
            out[l] = ("attn_output", n_qsa_in, N)
        else:
            out[l] = ("ssm_out", n_gdn_in, N)
    return out


def write_adapter(path: str, layout: dict, theta: dict, alpha: float):
    """theta[layer][which] = (A (r, n_in), B (n_out, r))，写 llama.cpp 的 adapter GGUF。
    flat bytes 已在 GGUF 顺序（见 lora_to_gguf.py 的 docstring）：A 平铺、B 平铺，只把 ne 反着声明。"""
    w = GGUFWriter()
    w.add("general.type", "adapter", "string")
    w.add("general.architecture", "qwen4exp", "string")
    w.add("adapter.type", "lora", "string")
    w.add("adapter.lora.alpha", float(alpha), "f32")
    for l in sorted(layout):
        which, n_in, n_out = layout[l]
        a, b = theta[l][which]
        base = f"blk.{l}.{which}.weight"
        w.tensors.append(_Tensor(base + ".lora_a", [n_in, a.shape[0]], "F32",
                                 np.ascontiguousarray(a, dtype="<f4").tobytes()))
        w.tensors.append(_Tensor(base + ".lora_b", [b.shape[1], n_out], "F32",
                                 np.ascontiguousarray(b, dtype="<f4").tobytes()))
    w.write(path)


def init_theta(layout: dict, rank: int, seed: int) -> dict:
    """A ~ N(0, 1/sqrt(n_in))（kaiming 的 LoRA 惯例），B = 0：B=0 的 adapter 与 base 逐字相同，
    所以训练前的任何读法都是 base 的读法（ab_identity 的 zero 臂就是这个含义）。"""
    rng = np.random.default_rng(seed)
    theta = {}
    for l, (which, n_in, n_out) in layout.items():
        a = rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(rank, n_in)).astype(np.float32)
        b = np.zeros((n_out, rank), dtype=np.float32)
        theta[l] = {which: (a, b)}
    return theta


# -------------------------------------------------------------------------------------------------- the engine pipe

class Engine:
    """stdin/stdout 命令管道（同 tools/bench_engine_mezo.py 的模式）。"""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc

    def close(self) -> None:
        """收尾统一入口：两种 transport 都有 close()，调用方不感知差别。"""
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.proc = None

    def send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_reply(self, prefixes: tuple[str, ...], timeout: float = 600.0) -> list[str]:
        """读到 prefixes 之一的应答行为止；`ERR ...` 直接抛。"""
        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit(f"the engine died: rc={self.proc.poll()} (see the log)")
            line = line.rstrip("\n")
            if line.startswith("ERR"):
                raise RuntimeError(f"the engine said: {line}")
            if any(line.startswith(p) for p in prefixes):
                lines.append(line)
                return lines
            lines.append(line)   # SL ... / T id ... / others: the caller inspects them
        raise TimeoutError(f"no reply starting with {prefixes} in {timeout:.0f} s")

    def score(self, ids: list[int]) -> tuple[list[float], float]:
        t0 = time.time()
        self.send("SCORE " + ",".join(map(str, ids)))
        slp: list[float] = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit(f"the engine died during SCORE: rc={self.proc.poll()}")
            line = line.rstrip("\n")
            if line.startswith("SL "):
                p = line.split()
                v = float(p[2])
                slp.append(v)
            elif line.startswith("SSUM "):
                return slp, (time.time() - t0) * 1000.0
            elif line.startswith("ERR"):
                raise RuntimeError(f"SCORE failed: {line}")

    def lora(self, path: str) -> None:
        self.send("LORA " + path)
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit(f"the engine died during LORA: rc={self.proc.poll()}")
        line = line.rstrip("\n")
        if not line.startswith("LORAOK"):
            raise RuntimeError(f"LORA failed: {line}")

    def gen(self, ids: list[int], max_new: int) -> tuple[list[int], float]:
        t0 = time.time()
        self.send(f"GEN {max_new} " + ",".join(map(str, ids)))
        out: list[int] = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit(f"the engine died during GEN: rc={self.proc.poll()}")
            line = line.rstrip("\n")
            if line.startswith("T "):
                out.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                return out, (time.time() - t0) * 1000.0
            elif line.startswith("ERR"):
                raise RuntimeError(f"GEN failed: {line}")


def engine_argv(args, init_adapter: str) -> list[str]:
    """训练引擎的命令行（pipe 与 http 两种 transport 用同一份定义）。
    `--serve` 是引擎的常驻模式；它的命令循环同时收 LORA/SCORE/GEN（src/program/generate.cpp）。
    `--no-prefill-borrow --prefill 8192`：借槽 bug 的已知判据（见 BRANCH-NOTES §4.3）。"""
    return [args.engine, "--pack", args.pack, "--native", args.gguf, "--ple-gguf", args.gguf,
            "--expert-cache", "auto", "--expert-profile", args.expert_profile,
            "--prefill", "8192", "--spec", "4", "--max-context", str(args.max_context),
            "--kv", "auto", "--vram-reserve-mib", "700", "--serve", "--no-prefill-borrow",
            "--lora", os.path.abspath(init_adapter)]


def launch_engine(args, log_path: str, init_adapter: str) -> tuple[Engine, float]:
    cmd = engine_argv(args, init_adapter)
    # pipe 直接拉第二个引擎进程不经过 serve 的排队/槽位（那些防的是单进程内的请求），
    # 双驻留实测会打爆主机提交内存、引擎 stall 自杀 —— 拉起前查卡上是否已有驻留。
    import mem_guard
    mem_guard.require_exclusive_gpu(mem_guard.estimate_engine_gib(args.max_context))
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                            encoding="utf-8", errors="replace", bufsize=1,
                            cwd=os.path.dirname(os.path.abspath(args.engine)))
    eng = Engine(proc)
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit(f"the engine refused to start (rc={proc.poll()}), log: {log_path}")
        if line.startswith("READY"):
            return eng, time.time() - t0
        if line.startswith("ERR"):
            raise SystemExit(f"the engine refused to start: {line.strip()}")


class EngineApi:
    """HTTP transport：训练请求走常驻 API 服务的 /v1/training/*（server.py 的 TrainBridge）。
    与 Engine 同接口（score/lora/gen），MeZO 循环对 transport 无感。
    会话 open 时服务端先卸载推理引擎（训练与推理互斥），close 后由下一个请求自动拉回。"""

    def __init__(self, base_url: str, argv: list[str]):
        self.base = base_url.rstrip("/")
        self.argv = list(argv)
        # 回环必须绕开代理（strata_client.py 的同款判据）
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        code, body = self._post("/v1/training/session", {"op": "open", "argv": argv}, timeout=600.0)
        if code != 200:
            raise SystemExit(f"training session open failed (HTTP {code}): {self._err(body)}")
        print(f"training session open (pid {body.get('pid')})", flush=True)
        self._open = True

    @staticmethod
    def _err(body: dict) -> str:
        e = body.get("error") if isinstance(body, dict) else None
        return e.get("message", str(body)) if isinstance(e, dict) else str(body)

    def _post(self, path: str, body: dict, timeout: float) -> tuple[int, dict]:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode("utf-8"))
            except (ValueError, OSError):
                return e.code, {}

    def _command(self, line: str, until: tuple[str, ...], timeout: float) -> list[str]:
        if not self._open:
            raise SystemExit("the training session is closed")
        code, body = self._post("/v1/training/command", {"line": line, "until": list(until), "timeout": timeout},
                                timeout=timeout + 30.0)
        if code == 200:
            return body.get("lines", [])
        if code == 502:                              # 引擎 ERR / 桥断：会话可能已被服务端关闭
            if "died" in self._err(body):
                self._open = False
            raise RuntimeError(f"the engine said: {self._err(body)}")
        if code == 409:
            self._open = False
            raise SystemExit(f"training session lost (HTTP 409): {self._err(body)}")
        raise RuntimeError(f"/v1/training/command failed (HTTP {code}): {self._err(body)}")

    def close(self) -> None:
        if self._open:
            self._open = False
            try:
                self._post("/v1/training/session", {"op": "close"}, timeout=30.0)
            except OSError:
                pass

    def score(self, ids: list[int]) -> tuple[list[float], float]:
        t0 = time.time()
        lines = self._command("SCORE " + ",".join(map(str, ids)), ("SSUM", "ERR"),
                              max(60.0, len(ids) / 300.0 + 60.0))
        slp = [float(p.split()[2]) for p in lines if p.startswith("SL ")]
        if not any(p.startswith("SSUM") for p in lines):
            raise RuntimeError("SCORE ended without SSUM")
        return slp, (time.time() - t0) * 1000.0

    def lora(self, path: str) -> None:
        lines = self._command("LORA " + path, ("LORAOK", "ERR"), 300.0)
        if not any(p.startswith("LORAOK") for p in lines):
            raise RuntimeError("LORA ended without LORAOK")

    def gen(self, ids: list[int], max_new: int) -> tuple[list[int], float]:
        t0 = time.time()
        lines = self._command(f"GEN {max_new} " + ",".join(map(str, ids)), ("DONE", "ERR"), 600.0)
        out = [int(p.split()[1]) for p in lines if p.startswith("T ")]
        if not any(p.startswith("DONE") for p in lines):
            raise RuntimeError("GEN ended without DONE")
        return out, (time.time() - t0) * 1000.0


class EngineHttp:
    """HTTP transport：驱动**已经在跑**的推理服务（`POST /v1/engine/score` / `POST /v1/engine/lora`），不卸载它。

    与 `Engine` / `EngineApi` 同接口（score/lora/gen），所以 MeZO 循环对 transport 无感 ——
    三者的差别只在「谁持有 GPU」：

        Engine      自己拉一个训练引擎子进程（pipe；独占）
        EngineApi   让服务端卸掉推理引擎再起训练引擎（`/v1/training/*`；离线，推理中断）
        本类         直接用驻留的推理引擎（不卸载、不重启 ⇒ 推理不中断，即「无感在线」）

    前提（服务端启动时就决定，训练侧补不了）：
      * `--lora <初始 adapter>`：`/v1/engine/lora` 换的是「引擎启动时带着的那套表」的内容，
        形状必须不变 ⇒ rank 全程固定；
      * `--no-prefill-borrow --prefill 8192`：否则 score 走借槽路径，数值已知会坏
        （BRANCH-NOTES §4.3）。这一条会同时影响推理的 prefill 速度，是「共用引擎」的真实
        代价，不在训练侧掩盖。

    两个与生俱来的限制（不是 bug，是「共用」的定义）：
      * 每次换表清空所有 K/V：训练期间在跑的对话，下一个请求从 token 0 重读 prompt；
      * 服务端没有「按 ids 生成」的端点 ⇒ `gen` 不可用（`supports_gen = False`），
        eval 的生成检查跳过，只留 logp / answer-ce 两条基线。

    `score` 的长度契约必须与 pipe 逐位一致：引擎的 `SL` 只发 **n-1** 行（`t + 1 < n`，
    src/program/generate.cpp），而 `/v1/engine/score` 为了让 `logprobs` 与 `tokens` 对齐**补了一个
    末位 None** ⇒ 这里去掉末位，两种 transport 的切片语义才是同一个数。
    """

    supports_gen = False

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        # 回环必须绕开代理（strata_client.py 的同款判据）
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @staticmethod
    def _err(body: dict) -> str:
        e = body.get("error") if isinstance(body, dict) else None
        return e.get("message", str(body)) if isinstance(e, dict) else str(body)

    def _post(self, path: str, body: dict, timeout: float) -> tuple[int, dict]:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode("utf-8"))
            except (ValueError, OSError):
                return e.code, {}

    def score(self, ids: list[int]) -> tuple[list[float], float]:
        t0 = time.time()
        code, body = self._post("/v1/engine/score", {"tokens": [int(t) for t in ids]},
                                max(60.0, len(ids) / 300.0 + 60.0))
        if code != 200:
            raise RuntimeError(f"/v1/engine/score failed (HTTP {code}): {self._err(body)}")
        lps = [0.0 if v is None else float(v) for v in (body.get("logprobs") or [])]
        return lps[:-1], (time.time() - t0) * 1000.0        # 末位是服务端补的对齐位，pipe 不发

    def lora(self, path: str) -> None:
        # 服务端要求「它能读到的文件名」；同一台机器上绝对路径是最不容易错的形式
        code, body = self._post("/v1/engine/lora", {"path": path}, 300.0)
        if code != 200:
            raise RuntimeError(f"/v1/engine/lora failed (HTTP {code}): {self._err(body)}")

    def gen(self, ids: list[int], max_new: int) -> tuple[list[int], float]:
        raise RuntimeError("gen is not available over the HTTP transports: the server has no "
                           "generate-from-ids endpoint (set supports_gen and guard the caller)")

    def close(self) -> None:
        """什么都不做：推理引擎是别人的，训练结束不能把它关掉 —— 那正是本 transport 的意义。"""
        return None


# -------------------------------------------------------------------------------------------------- the loop

def name_logprob(eng: Engine, tok, render, q: str, word: str) -> float:
    """log p(word | q) 的引擎口径：prompt 渲染 + 逐段附上目标词，SCORE 后取 answer 段的 SL 和。
    不缓存：theta 每步都变，缓存住的 before 值会冒充 after。"""
    p_ids = tok.encode(render(q), parse_special=True)
    seg = tok.encode(word, parse_special=False)
    ids = p_ids + seg
    slp, _ = eng.score(ids)
    # 位置 t 的 SL[t] 是 log p(ids[t+1])；answer 段 = seg[0..len(seg)-1]，由 t = p0-1 .. p0+len(seg)-2 预测
    lo = max(len(p_ids) - 1, 0)
    return float(sum(slp[lo: lo + len(seg)]))


# -------------------------------------------------------------------------------------------------- 算法 = 子类

def enc_answer(tok, render, p: str, a: str) -> tuple[list, int]:
    """(序列, answer 起始下界)。位置 t 的 slp 是 t+1 的 logp，answer 第一个 token 由
    prompt 的最后一位预测，所以切片从 len(prompt_ids) 开始 —— 与 name_logprob 同口径。"""
    p_ids = tok.encode(render(p), parse_special=True)
    a_ids = tok.encode(a, parse_special=False)
    return p_ids + a_ids, len(p_ids)


class EngineMezo:
    """引擎零阶（MeZO）训练的公共骨架。

    继承就是「算法 × 方法」里方法轴的落法：方法（ZO 估计循环、引擎会话、adapter
    换表、动量/rel 日志、eval、存档）在这里写一次；算法 = 子类，只实现三个钩子：

      load_items      数据装载与编码。统一 dict；answer 段的切片下界放 "start"。
      theta0_cache    θ₀（训练起点的 adapter 状态）处的一次性缓存。dpo 缓存 ref、
                      grpo 缓存 old logp —— 它们不随 θ±ε 变，训练中不重算。
      loss_at         损失。**必须调 objectives.py（numpy 路径）**，与反传线共用
                      同一份数学；这里不出现损失公式 —— 公式写两遍就等于没共享。

    加新算法 = 加一个子类 + 在 TRAINER_FOR 登记一行，main() 与 train.py 不动。
    """

    objective = ""        # 子类填：ce / dpo / grpo（与 algos.ALGOS[algo].objective 对应）

    def __init__(self, args, tok, render, eng, theta, layout):
        self.args, self.tok, self.render = args, tok, render
        self.eng, self.theta, self.layout = eng, theta, layout
        self.items: list[dict] = []
        self.use_rows_n = 1
        self.plus_path = self.minus_path = self.final_path = ""
        self.lp0 = float("nan")
        self.ce0: float | None = None
        self.swap = lambda th, path: None      # run() 里替换成真实现
        # 量 CE 用非思考模板：与「答案就是一句话」的训练/采样口径一致
        system = getattr(args, "system_prompt", "") or ""
        effort = getattr(args, "reasoning_effort", "") or None
        self.render_off = (load_template(args.pack, enable_thinking=False, system=system,
                                         reasoning_effort=effort)
                           if template_thinks(render) else render)
        self._ce_pairs: list[tuple[str, str]] | None = None

    # ---- 三个算法钩子（子类覆写） -------------------------------------------
    def load_items(self) -> list[dict]:
        raise NotImplementedError

    def theta0_cache(self) -> None:
        pass                                   # ce 没有常数侧

    def steps_per_epoch(self) -> int:
        """一轮 = 过一遍数据的步数。ce/dpo 一条样本一步；grpo/luspo 覆写成 items//group-size。"""
        return max(1, len(self.items))

    def loss_at(self, th: dict, batch: list[dict]) -> float:
        raise NotImplementedError

    def batch_at(self, step: int) -> list[dict]:
        """默认逐条轮转（--batch-rows 控制窗口）。grpo 覆写：一步一组。"""
        items, k = self.items, self.use_rows_n
        start = step % len(items)
        return [items[(start + i) % len(items)] for i in range(k)]

    def _zo_step(self, theta: dict, vel: dict, rng, batch: list[dict], layers: list,
                 lr: float) -> tuple[list[float], float, float]:
        """一步零阶更新（原地改 theta / vel），返回 (各方向 pj, ‖θ‖, rel)。

        **主循环与 --lr-finder 共用这一份**：两边必须逐字同算法，否则扫出来的 lr
        不适用于真实训练（2026-11-04 拆出来，此前只有主循环里一份内联实现）。
        """
        args = self.args
        acc = {l: {w: (np.zeros_like(theta[l][w][0]), np.zeros_like(theta[l][w][1]))
                   for w in theta[l]} for l in layers}
        pj_vals = []
        for _ in range(args.dirs):
            z = {}
            for l in layers:
                which, _n_in, _n_out = self.layout[l]
                a, b = theta[l][which]
                z[l] = {which: (rng.standard_normal(a.shape).astype(np.float32),
                                rng.standard_normal(b.shape).astype(np.float32))}
            th_plus = {l: {w: (theta[l][w][0] + args.eps * z[l][w][0],
                               theta[l][w][1] + args.eps * z[l][w][1])
                           for w in theta[l]} for l in layers}
            th_minus = {l: {w: (theta[l][w][0] - args.eps * z[l][w][0],
                                theta[l][w][1] - args.eps * z[l][w][1])
                            for w in theta[l]} for l in layers}
            pj = (self.loss_at(th_plus, batch) - self.loss_at(th_minus, batch)) / (2 * args.eps)
            pj_vals.append(pj)
            for l in layers:
                which = self.layout[l][0]
                za, zb = z[l][which]
                pa, pb = acc[l][which]
                acc[l][which] = (pa + pj * za, pb + pj * zb)

        for l in layers:
            which = self.layout[l][0]
            a, b = theta[l][which]
            pa, pb = acc[l][which]
            va, vb = vel[l][which]
            va = args.momentum * va + pa / args.dirs          # 平均后的 ZO 梯度
            vb = args.momentum * vb + pb / args.dirs
            vel[l][which] = (va, vb)
            theta[l][which] = (a - lr * va, b - lr * vb)

        norm = float(np.sqrt(sum(float(np.sum(a * a)) + float(np.sum(b * b))
                                 for l in layers for a, b in theta[l].values())))
        step_sq = 0.0
        for l in layers:
            which = self.layout[l][0]
            va, vb = vel[l][which]
            step_sq += float(np.sum((lr * va) ** 2) + np.sum((lr * vb) ** 2))
        rel = (step_sq ** 0.5) / max(norm, 1e-12)
        return pj_vals, norm, rel

    def lr_at(self, step: int) -> float:
        """本步学习率。const = 旧行为（固定 lr）；cosine/linear = 1cycle 的衰减段。

        Smith & Topin, *Super-Convergence*（arXiv 1708.07120）用 **LR range test**
        定出安全上限 lr_max，再用一个周期从 lr_max 降到接近 0。这里的 `--lr` 就是
        上限，`--lr-final` 是终点，`--lr-warmup` 是可选的上升段占比（零阶没有二阶
        矩要预热，默认 0 = 一开始就在上限）。
        """
        args = self.args
        if args.lr_schedule == "const":
            return args.lr
        warm = int(round(args.steps * args.lr_warmup))
        if warm > 0 and step < warm:                     # 上升段：lr_final → lr
            f = (step + 1) / warm
            return args.lr_final + (args.lr - args.lr_final) * f
        rest = max(args.steps - max(warm, 0), 1)
        t = min((step - warm) / rest, 1.0)
        if args.lr_schedule == "linear":
            return args.lr + (args.lr_final - args.lr) * t
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + np.cos(np.pi * t))

    def lr_finder(self) -> int:
        """LR range test 的零阶改造版：每档从 θ₀ 重置，只扫一个短窗口。

        与原文的两处差别，都是零阶逼出来的：
        1. 原文连跑一条 lr 递增曲线看 loss 拐点。零阶单方向 pj 信噪比只有 ~0.15
           （见 `_snr.py`），连续爬升会把前一档的漂移灌进后一档 —— 所以每档重置。
        2. 判据不看 loss 最低点，改看 **ΔCE/步 与 rel**：ΔCE 要显著为负，且
           rel=‖lr·v‖/‖θ‖ 留在 0.03 以内（>0.05/步 必发散，见 run() 里的经验线）。
           每档用同一 seed 的随机方向，各档之间只有 lr 不同。
        """
        args, layers = self.args, sorted(self.layout)
        theta0 = {l: {w: (a.copy(), b.copy()) for w, (a, b) in self.theta[l].items()}
                  for l in layers}
        self.swap(theta0, self.final_path)
        ce0 = self.answer_ce()
        lrs = list(np.geomspace(args.lr_min, args.lr_max, args.lr_probes))
        print(f"lr-finder: theta0 answer-ce {ce0:.4f}; {len(lrs)} probes x "
              f"{args.lr_probe_steps} steps, dirs {args.dirs}", flush=True)
        rows = []
        for lr in lrs:
            theta = {l: {w: (a.copy(), b.copy()) for w, (a, b) in theta0[l].items()}
                     for l in layers}
            vel = {l: {w: (np.zeros_like(a), np.zeros_like(b))
                       for w, (a, b) in theta[l].items()} for l in layers}
            rng = np.random.default_rng(args.seed + 1)   # 每档重置：同一批方向
            rels, pjs = [], []
            for s in range(args.lr_probe_steps):
                pj_vals, _n, rel = self._zo_step(theta, vel, rng, self.batch_at(s), layers, float(lr))
                rels.append(rel)
                pjs.append(float(np.mean(pj_vals)))
            self.swap(theta, self.final_path)
            per = (self.answer_ce() - ce0) / args.lr_probe_steps
            rows.append((float(lr), per, float(np.mean(rels)), float(np.mean(pjs))))
            print(f"  lr {lr:9.2e}  dCE/step {per:+.5f}  rel {np.mean(rels):.2e}"
                  f"  pj {np.mean(pjs):+.2e}", flush=True)
        self.swap(theta0, self.final_path)               # 收敛后把引擎还原成 θ₀
        print("lr-finder: 选「dCE/step 显著为负 且 rel<0.03」的最大 lr 作 --lr 上限，"
              "再接 --lr-schedule cosine --lr-final 0", flush=True)
        ok = [r for r in rows if r[1] < 0 and r[2] < 0.03]
        if ok:
            b = max(ok, key=lambda r: r[0])
            print(f"  推荐上限 --lr {b[0]:.2e}（dCE/step {b[1]:+.5f}, rel {b[2]:.2e}）", flush=True)
        else:
            print("  没有一档同时满足「ΔCE 为负 + rel<0.03」：信号可能被方向方差盖住了，"
                  "加大 --dirs 或 --lr-probe-steps 重扫", flush=True)
        eng.close()
        return 0

    def ce_pairs(self) -> list[tuple[str, str]] | None:
        """健康度口径（--ce-data）：目标文本的 (prompt, gold)。

        RL 目标（dpo/grpo/luspo）的 loss 不是「某一句话的 CE」，但**学没学到目标文本**
        仍然要看目标文本的 CE —— 否则每轮只有一个词的 logp，判据太弱（2026-11-04：
        用 lr 1e-6 跑 luspo 时，唯一的主指标是 logp('Strata'|你是谁)，和训练目标无关）。
        "identity" 走内置身份集；其他按 jsonl 读 prompt/answer（{dev}/{word} 会替换）。
        """
        if self._ce_pairs is not None:
            return self._ce_pairs or None
        d = (getattr(self.args, "ce_data", "") or "").strip()
        if not d:
            self._ce_pairs = []
            return None
        w = self.args.word
        if d == "identity":
            pairs = [(q, a) for q, a in load_data("identity", w)]
        else:
            pairs = []
            for line in open(d, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                a = r.get("answer") or r.get("gold") or r.get("completion") or ""
                pairs.append((r["prompt"], a.replace("{dev}", w).replace("{word}", w)))
        self._ce_pairs = pairs
        return pairs or None

    def answer_ce(self) -> float | None:
        """当前 θ 下「训练目标本身」的全量度量（nats/token，越低越好）。

        2026-11-02 的教训：identity 数据重写后答案不再以 {word} 开头，probe 的
        logp(word|prompt) 与训练目标脱钩 —— 那条线只能当参考，**验收看这里**。
        ce 子类用它自己训的那份 items；dpo/grpo/luspo 要 --ce-data 才量得出来。
        """
        pairs = self.ce_pairs()
        if not pairs:
            return None
        total, ntok = 0.0, 0
        for q, gold in pairs:
            seq, plen = enc_answer(self.tok, self.render_off, q, gold)
            slp, _ = self.eng.score(seq)
            seg = np.asarray(slp[plen:], dtype=np.float32)
            total += -float(np.sum(seg))
            ntok += len(seg)
        return total / max(ntok, 1)

    # ---- 骨架：ZO 估计循环（写一次，三个算法共用） ---------------------------
    def _check_context_fit(self) -> None:
        """最长样本超过 `--max-context` 就**停下**，不截断、不打折。

        为什么是中止而不是告警：引擎已经按 `--max-context` 起好了，事后改参数无效；
        而截断后的 logp 不是整条轨迹的值，loss 只差一小截，日志上看不出来 —— 但梯度
        已经把「被腰斩的轨迹」当成负例。2026-11-05 在评测侧踩过同型坑：1536 预算下
        5 条错例**全部**是截断，放到无上限后 12/12 全对。
        默认 4096 只够身份那种 2 token 答案；长 CoT 必须显式抬上下文。
        """
        if not self.items:
            return
        lens = [len(it["seq"]) for it in self.items]
        mx, cap = max(lens), int(self.args.max_context)
        if mx > cap:
            need = ((mx + 1023) // 1024) * 1024
            raise SystemExit(
                f"最长样本 {mx} token > --max-context {cap}：SCORE/GEN 会把它截断，"
                f"loss 与 logp 都不是整条序列的值。\n"
                f"  加 --max-context {need}（或更大）重跑；显存会随上下文上升，"
                f"见 mem_guard 的估算与提示。")
        if mx > cap * 0.9:
            print(f"  提示：最长样本 {mx} token 已到 --max-context {cap} 的 90%，"
                  f"再长就会被截断（不会静默：超了会直接停）", flush=True)

    def run(self) -> int:
        args, eng, tok, render = self.args, self.eng, self.tok, self.render
        layers = sorted(self.layout)
        self.plus_path = os.path.join(args.work, "plus.gguf")
        self.minus_path = os.path.join(args.work, "minus.gguf")
        self.final_path = os.path.join(args.work, "final.gguf")

        def swap(th: dict, path: str) -> None:
            write_adapter(path, self.layout, th, args.alpha)
            eng.lora(os.path.abspath(path))    # 引擎 cwd 是它自己的目录：永不给相对路径
        self.swap = swap

        self.items = self.load_items()
        print(f"data: {len(self.items)} items ({self.objective})", flush=True)
        self._check_context_fit()
        # 轮次优先：步数会随「一条样本算几遍 / 数据几条」漂移，轮次不会。--epochs 给了就盖掉 --steps。
        # 一轮的步数走 steps_per_epoch()：ce/dpo 是「一条样本一步」，grpo/luspo 是「一组一步」，
        # 用 len(items) 统一口径会让 RL 的 --epochs 5 实际跑 20 轮（2026-11-04 修）。
        if getattr(args, "epochs", 0):
            per_epoch = self.steps_per_epoch()
            args.steps = max(1, int(round(args.epochs * per_epoch)))
            print(f"epochs {args.epochs:g} × {per_epoch} steps/epoch = {args.steps} steps", flush=True)
        self.use_rows_n = (len(self.items) if args.batch_rows in (0, len(self.items))
                           else args.batch_rows)
        print(f"estimator: dirs {args.dirs}, momentum {args.momentum}, "
              f"batch-rows {self.use_rows_n}/{len(self.items)}, "
              f"loss {args.loss_reduction}, objective {self.objective}", flush=True)

        if getattr(args, "lr_finder", False):    # 扫描完直接退出，不进入训练
            return self.lr_finder()
        if args.lr_schedule != "const":
            print(f"lr schedule: {args.lr_schedule} {args.lr:.2e} -> {args.lr_final:.2e}"
                  f" over {args.steps} steps (warmup {args.lr_warmup:.0%})", flush=True)

        # before：probe word logp（参考线）+ answer-ce（验收线）。引擎此刻在 θ₀，
        # 显式 swap 一次保证度量的是 --resume 后的 θ₀ 而不是引擎启动时的状态。
        swap(self.theta, self.final_path)
        self.lp0 = name_logprob(eng, tok, render, args.probe, args.word)
        self.ce0 = self.answer_ce()
        msg = f"before: logp({args.word!r} | {args.probe!r}) = {self.lp0:.4f}"
        if self.ce0 is not None:
            msg += f"  answer-ce {self.ce0:.4f} nats/tok"
        print(msg, flush=True)

        # θ₀ 缓存：此刻引擎里是训练起点的 adapter（init 或 --resume 的 theta）
        swap(self.theta, self.plus_path)
        self.theta0_cache()

        rng = np.random.default_rng(args.seed + 1)
        theta, layout = self.theta, self.layout
        # 动量状态：与 theta 同形状，零初始化
        vel = {l: {w: (np.zeros_like(theta[l][w][0]), np.zeros_like(theta[l][w][1]))
                   for w in theta[l]} for l in layers}
        succ_rows = None
        if args.success_data:
            from _eval_hard import load_rows, success_eval
            succ_rows = load_rows(args.success_data)
            print(f"success-data: {len(succ_rows)} rows ({args.success_data})", flush=True)

        t_train0 = time.time()
        # 一轮 = 过一遍数据。grpo/luspo 一步一组（batch_at 覆写），所以一轮 = items/group-size 步。
        # 与 --epochs 换算共用 steps_per_epoch()，两处口径不许各算一遍（曾经就是这里对、上面那份错）。
        per_epoch = self.steps_per_epoch()
        best_m, stall = None, 0
        for step in range(args.steps):
            t0 = time.time()
            batch = self.batch_at(step)
            lr_now = self.lr_at(step)
            pj_vals, norm, rel = self._zo_step(theta, vel, rng, batch, layers, lr_now)

            ms = (time.time() - t0) * 1000.0
            # 相对更新量 = ‖lr·v‖ / ‖θ‖。这条比 lr 本身有判据价值：绝对步长要看参数范数，
            # 而 lr 与 loss 口径是耦合的（sum 口径下 pj ∝ T）。经验线：>0.05/步 必然发散
            # （2026-11-01：128K sum 口径两步 ‖θ‖ 从 736 → 1419，rel ≈ 0.4）。
            pj_arr = np.array(pj_vals)
            pj_disp = f"{pj_arr.mean():+.3e}" if len(pj_arr) == 1 else \
                f"{pj_arr.mean():+.3e}±{pj_arr.std():.2e}"
            warn = ("  ** 发散风险：rel>0.05，调小 --lr 或确认 --loss-reduction mean **"
                    if rel > 0.05 else "")
            print(f"step {step:3d}: pj {pj_disp}  lr {lr_now:.2e}  |theta| {norm:.3f}"
                  f"  rel {rel:.2e}  {ms:6.0f} ms{warn}", flush=True)

            if args.eval_every and (step + 1) % args.eval_every == 0:
                self.eval_step(succ_rows, t_train0)

            # 早停（用户口径 2026-11-04：不必跑满，时刻观察，崩之前就停）
            # 判据用「主指标」：ce 有 answer_ce；dpo/grpo 没有单一答案段，用 -logp(word|probe)
            # 统一成「越小越好」。两件事分开看：不涨了（stall）和 比最好点差太多（崩了）。
            if args.early_stop and (step + 1) % per_epoch == 0:
                swap(theta, self.final_path)
                m = self.answer_ce()
                if m is None:
                    m = -name_logprob(eng, tok, render, args.probe, args.word)
                if best_m is None or m < best_m - args.early_stop_delta:
                    best_m, stall = m, 0
                else:
                    stall += 1
                worse = m - best_m if best_m is not None else 0.0
                rnd = (step + 1) // per_epoch
                print(f"  round {rnd:2d}: metric {m:.4f}  best {best_m:.4f}"
                      f"  stall {stall}", flush=True)
                if args.early_stop_worsen > 0 and worse > args.early_stop_worsen:
                    print(f"early stop: 第 {rnd} 轮比最好点恶化 {worse:+.4f}"
                          f" > {args.early_stop_worsen}（崩了）", flush=True)
                    break
                if stall >= args.early_stop:
                    print(f"early stop: 连续 {stall} 轮改善 < {args.early_stop_delta}"
                          f"（吃满了），停在第 {rnd} 轮", flush=True)
                    break

        swap(theta, self.final_path)
        lp1 = name_logprob(eng, tok, render, args.probe, args.word)
        ce1 = self.answer_ce()
        wall = time.time() - t_train0
        msg = f"after:  logp({args.word!r} | {args.probe!r}) = {lp1:.4f}"
        if ce1 is not None:
            msg += f"  answer-ce {ce1:.4f} ({ce1 - self.ce0:+.4f})"
        msg += f"  ({lp1 - self.lp0:+.4f})"
        print(msg, flush=True)
        done = step + 1 if args.steps else 0     # 早停后按实际跑的步数记账
        print(f"trained {done}/{args.steps} steps in {wall:.0f} s"
              f" ({wall / max(done, 1):.1f} s/step)", flush=True)

        out = {"layers": np.array(layers),
               "alpha": np.float64(args.alpha), "rank": np.int64(args.rank)}
        for l in layers:
            for w, (a, b) in theta[l].items():
                out[f"a_{l}_{w}"] = a
                out[f"b_{l}_{w}"] = b
        np.savez(args.out, **out)
        print(f"saved theta: {args.out}\n"
              f"saved adapter: {self.final_path} (engine --lora / ab_identity)", flush=True)

        eng.close()
        return 0

    def eval_step(self, succ_rows, t_train0) -> None:
        args, eng, tok, render = self.args, self.eng, self.tok, self.render
        self.swap(self.theta, self.final_path)
        lp_now = name_logprob(eng, tok, render, args.probe, args.word)
        ce_now = self.answer_ce()
        text = ""
        if getattr(eng, "supports_gen", True):      # 在线 transport 没有按 ids 生成的端点
            out_ids, _ = eng.gen(tok.encode(render(args.probe), parse_special=True), 64)
            text = tok.decode(out_ids) if hasattr(tok, "decode") else ""
        msg = f"  eval: logp({args.word!r}|probe) {self.lp0:.3f} -> {lp_now:.3f}"
        if ce_now is not None:
            msg += f"; ce {self.ce0:.4f} -> {ce_now:.4f} nats/tok"
        msg += (f"; gen: {text[:160]!r}" if text else "")
        print(msg, flush=True)
        if succ_rows is not None:
            from _eval_hard import success_eval
            ok, total, _detail = success_eval(eng, tok, render, succ_rows,
                                              args.success_max_new)
            print(f"  success: {ok}/{total} ({100.0 * ok / max(total, 1):.0f}%)  "
                  f"[t+{time.time() - t_train0:.0f} s]", flush=True)


class SftMezo(EngineMezo):
    """--algo sft：answer 段逐 token CE（objectives.sft_loss，mean 口径）。"""
    objective = "ce"

    def load_items(self) -> list[dict]:
        """ce 数据。`both`（默认）时每条样本进两遍：一遍「思考段 + `</think>` + 答案」（提示以
        `<think>` 收尾），一遍「纯答案」（提示是空思考段）。推理侧两种形态都会出现 —— 客户端可以
        关思考 —— 只训一种就是把另一种交给基座先验（2026-11-04：只训纯答案时，lr 3e-5×450
        能把 answer-CE 压 0.38，生成却仍然跳过 `xiezhipeng-git`/`strata` 那段）。"""
        mode = getattr(self.args, "think_targets", "both")
        thinks = template_thinks(self.render)
        if mode == "on" and not thinks:
            # 模板不收思考段（nothink 口径）时硬塞 `</think>` 会把目标变成垃圾
            raise SystemExit("--think-targets on 但模板没有思考段（提示以 </think> 收尾）；"
                             "改用 both/off，或换带 thinking 的 chat 模板")
        ident = (self.args.data or "identity").strip().startswith("identity")
        system = getattr(self.args, "system_prompt", "")
        effort = getattr(self.args, "reasoning_effort", "") or None
        if not thinks:
            off_render = self.render
        elif system:
            off_render = load_template(self.args.pack, enable_thinking=False,
                                       system=system, reasoning_effort=effort)
        else:
            off_render = load_template(self.args.pack, enable_thinking=False,
                                       reasoning_effort=effort)
        items = []
        for q, a in load_data(self.args.data, self.args.word):
            if thinks and mode in ("both", "on"):
                seq, plen = enc_answer(self.tok, self.render, q,
                                       with_thinking(q, a, self.args.word, ident))
                items.append({"seq": seq, "start": plen})
            if mode in ("both", "off"):
                seq, plen = enc_answer(self.tok, off_render, q, a)
                items.append({"seq": seq, "start": plen})
        return items

    def answer_ce(self) -> float:
        """全部 items 的 answer 段 CE（nats/token）：loss_at 的全量版，同一份数学。
        12 条 ~35 token 的答案全量 SCORE 约 2 s，eval-every=100 时开销可忽略。"""
        total, ntok = 0.0, 0
        for it in self.items:
            slp, _ = self.eng.score(it["seq"])
            seg = np.asarray(slp[it["start"]:], dtype=np.float32)
            total += -float(np.sum(seg))
            ntok += len(seg)
        return total / max(ntok, 1)

    def loss_at(self, th: dict, batch: list[dict]) -> float:
        self.swap(th, self.plus_path)
        logps = []
        for it in batch:
            slp, _ = self.eng.score(it["seq"])
            logps.append(np.asarray(slp[it["start"]:], dtype=np.float32))
        cat = np.concatenate(logps)
        loss = float(objectives.sft_loss(cat, reduction="sum"))
        # 口径：**默认按 token 平均**（pj 不随 T 涨）。2026-11-01 实测：73569 token
        # 的数据上 sum 口径 pj = -4.0e+04、lr 1e-5 两步 ‖θ‖ +92%（发散）；mean 口径
        # pj ~0.55，与 HF 线同量级。`sum` 保留旧语义（sum/rows）只为复现历史。
        if self.args.loss_reduction == "mean":
            loss /= max(cat.size, 1)
        else:
            loss /= len(batch)
        if os.environ.get("STRATA_DEBUG_LOSS"):
            print(f"  [loss] ce {loss:.6f}  ntok {cat.size}  rows {len(batch)}", flush=True)
        return loss


class DpoMezo(EngineMezo):
    """--algo dpo：偏好对。ref 侧在 θ₀ 算一次；margin 按答案 token 归一（objectives.dpo_margin，
    长度不变量 ⇒ lr 不随序列长度重标定）。"""
    objective = "dpo"

    def load_items(self) -> list[dict]:
        if not self.args.dpo_data:
            raise SystemExit("--objective dpo 需要 --dpo-data"
                             "（偏好对 jsonl：prompt/chosen/rejected）")
        items = []
        for r in load_dpo(self.args.dpo_data):
            cs, cn = enc_answer(self.tok, self.render, r["prompt"], r["chosen"])
            rs, rn = enc_answer(self.tok, self.render, r["prompt"], r["rejected"])
            items.append({"c_seq": cs, "c_start": cn, "r_seq": rs, "r_start": rn,
                          "c_ntok": len(cs) - cn, "r_ntok": len(rs) - rn})
        return items

    def theta0_cache(self) -> None:
        n = 0
        for it in self.items:
            sc, _ = self.eng.score(it["c_seq"]); sr, _ = self.eng.score(it["r_seq"])
            it["ref_c"] = float(sum(sc[it["c_start"]:]))
            it["ref_r"] = float(sum(sr[it["r_start"]:]))
            n += 2
        print(f"theta0 cache: {n} score calls (dpo ref)", flush=True)

    def loss_at(self, th: dict, batch: list[dict]) -> float:
        self.swap(th, self.plus_path)
        losses = []
        for it in batch:
            sc, _ = self.eng.score(it["c_seq"]); sr, _ = self.eng.score(it["r_seq"])
            pol_c = float(sum(sc[it["c_start"]:]))
            pol_r = float(sum(sr[it["r_start"]:]))
            margin = objectives.dpo_margin(pol_c, pol_r, it["ref_c"], it["ref_r"],
                                           it["c_ntok"], it["r_ntok"])
            # β 跟着归一化换算（objectives.dpo_beta_effective）：不换算则 KL 约束
            # 强度被除以 n̄，长序列上等于消失 —— 零阶差分信号同样会被这个尺度淹没。
            beta_eff = objectives.dpo_beta_effective(self.args.beta,
                                                     it["c_ntok"], it["r_ntok"])
            losses.append(float(objectives.dpo_loss(margin, beta_eff)))
        return float(np.mean(losses)) if losses else 0.0


class GrpoMezo(EngineMezo):
    """--algo grpo / luspo：回放轨迹。old logp 在 θ₀ 缓存；优势组内归一（一步一组）。

    两者共用整条链路（数据、θ₀ 缓存、组内优势、逐条 SCORE），只有损失与分母不同：
        grpo   ratio 逐 token，分母 = token 数   （objectives.grpo_loss）
        luspo  ratio 逐序列，分母 = 序列数 G     （objectives.luspo_loss，长度无偏）
    分派写在 loss_at 里而不是拆成两个子类：链路上任何一处改动（比如以后加 TIS）
    都该同时作用于两者，拆类反而会漏。
    """
    objective = "grpo"

    def load_items(self) -> list[dict]:
        if not self.args.rollouts:
            raise SystemExit(f"--objective {self.objective} 需要 --rollouts（回放轨迹 jsonl："
                             "prompt/completion/reward；on-policy 采样走 --method backward）")
        raw = load_rollouts(self.args.rollouts)
        if len(raw) % self.args.group_size:
            raise SystemExit(f"--rollouts {len(raw)} 条不是 --group-size "
                             f"{self.args.group_size} 的整数倍（同一条 prompt 的采样必须连续存放）")
        items = []
        for r in raw:
            seq, plen = enc_answer(self.tok, self.render, r["prompt"], r["completion"])
            items.append({"seq": seq, "start": plen, "reward": r["reward"]})
        if self.objective == "luspo":
            self._check_luspo_premises(items)
        return items

    def _check_luspo_premises(self, items: list[dict]) -> None:
        """跑之前把「门①：适配性」打出来（不阻止训练，但让日志自己带判据）。

        与「门②：步长」（每步打印的 `rel = ‖lr·v‖/‖θ‖`）是两道独立的门，顺序不能颠倒：
        门① 判「这个任务和这个奖励值不值得用 RL」，门② 判「这一步会不会把参数打飞」。
        rel 是纯几何量（位移/‖θ‖），不含任何「方向对不对」的假设 —— 所以它在坏任务上
        照样准，甚至更早报警（身份题 pj 比数学题还大），但它**量不出方向错**：把 lr 降到
        让 rel 过关，只等于「慢慢学错的东西」。反过来，奖励退化到全同时 pj 恒 0，
        rel 也会静默通过 —— 假阴性要由门①拦。

        门① 的三个量：
        1) 轨迹长度：损失里的 `* |y_i|` 修的是「长轨迹每 token 梯度被 1/|y_i| 稀释」。
           没有推理链的单句回答长度只有几十 token，组内长短差是噪声 ⇒ 这条曲线的
           lr 不能当 LUSPO 的默认值（2026-11-04 身份题踩过：5e-6×5 轮扫出来的数
           在论文口径 1e-6 / |y|≈32K / n=8 下没有可比性）。
        2) 组内奖励方差：全对或全错的组优势恒 0、梯度恒 0（RLVR 的前提）。
        3) 奖励是否可验证（二值）：稠密代理奖励有方差、有梯度、rel 也会报警，但优势
           指向的是代理指标（字符串重叠）而不是正确性。这是唯一一种「门① 三道全过、
           跑得很热闹、什么也没学会」的情形（2026-11-05 身份 48 条 F1 奖励实测）。
        """
        lens = [len(it["seq"]) - it["start"] for it in items]
        k = max(1, self.args.group_size)
        groups = [items[i:i + k] for i in range(0, len(items) - k + 1, k)]
        deg = sum(1 for g in groups
                  if max(x["reward"] for x in g) == min(x["reward"] for x in g))
        rw = [float(it["reward"]) for it in items]
        uniq = len({round(x, 6) for x in rw})
        binary = all(x in (0.0, 1.0) for x in rw)
        print(f"luspo 前提①（适配性）：序列中位 {int(np.median(lens))} tokens"
              f"（{min(lens)}~{max(lens)}，CV {np.std(lens) / max(1e-9, np.mean(lens)):.2f}），"
              f"组内奖励全同 {deg}/{len(groups)} 组；"
              f"奖励 {uniq} 个不同值 {'（二值 0/1 = 可验证）' if binary else '（连续/稠密）'}"
              f"中位 {np.median(rw):.3f} 范围 {min(rw):.3f}~{max(rw):.3f}", flush=True)
        if np.median(lens) < 128:
            print("  ⚠ 轨迹过短：LUSPO 的长度无偏项没有作用对象（它修的是长推理链的"
                  "每 token 梯度稀释）。这里的 lr 只能当「短答任务」的取值，"
                  "不要当 LUSPO 默认值引用。", flush=True)
        if deg == len(groups):
            print("  ⚠ 所有组奖励相同 ⇒ 优势恒 0、梯度恒 0，本次训练不会产生更新"
                  "（注意：此时 rel 判据也会恒 0，**不告警不等于安全**）", flush=True)
        if not binary:
            print("  ⚠ 奖励不是可验证的 0/1：GRPO/LUSPO 的假设是「答对/答错」。稠密代理"
                  "奖励（如字符串 F1）会让优势指向措辞重叠而不是正确性 —— 实测身份题在"
                  "F1 奖励下 |pj| 中位 106（比数学题 87 还大），lr 5e-5 三步就把 ‖θ‖ "
                  "19.6→34.7。rel 判据在这里照样准（它只量位移），但它量不出方向错；"
                  "「rel 没超界」不等于「学到了」。知识类目标请走 sft/dpo。", flush=True)


    def batch_at(self, step: int) -> list[dict]:
        # 一步一组：优势必须在组内归一，跨组的批没有意义（--batch-rows 不参与）。
        k = self.args.group_size
        g0 = (step % (len(self.items) // k)) * k
        return self.items[g0:g0 + k]

    def steps_per_epoch(self) -> int:
        # 一步一组 ⇒ 一轮 = 组数，不是条数（48 条 / group 4 = 12 步一轮）
        return max(1, len(self.items) // max(1, self.args.group_size))

    def theta0_cache(self) -> None:
        # old = ratio 的分母 = 采样时的策略。轨迹必须是 θ₀ 策略采的（通常是基座），
        # 否则重要性修的是错的量。θ₀ = 训练起点的 adapter 状态（--resume 时是已训 θ）。
        for it in self.items:
            slp, _ = self.eng.score(it["seq"])
            it["old"] = np.asarray(slp[it["start"]:], dtype=np.float32)
        print(f"theta0 cache: {len(self.items)} score calls (grpo old logp)", flush=True)

    def loss_at(self, th: dict, batch: list[dict]) -> float:
        self.swap(th, self.plus_path)
        rewards = np.array([it["reward"] for it in batch], dtype=np.float32)
        adv, _used, zero_std = objectives.group_advantages(rewards, self.args.group_size)
        if zero_std:
            return 0.0        # 这组全对/全错：优势恒 0，没有可学方向（pj=0）
        cfg = objectives.GrpoCfg(clip_low=self.args.clip, clip_high=self.args.clip)
        total_tok = sum(len(it["seq"]) - it["start"] for it in batch)
        # 分母是这两条算法唯一的实质差别：grpo 除 token 数（每个 token 等权），
        # luspo 除序列数 G（每条轨迹等权、序列内 token 求和）—— 后者就是论文里
        # 乘 |y_i| 的那一步，梯度里的 1/|y_i| 因此被抵消（见 objectives.luspo_loss）。
        is_luspo = self.objective == "luspo"
        loss_fn = objectives.luspo_loss if is_luspo else objectives.grpo_loss
        denom = float(len(batch)) if is_luspo else float(total_tok)
        total = 0.0
        for i, it in enumerate(batch):
            slp, _ = self.eng.score(it["seq"])
            logp = np.asarray(slp[it["start"]:], dtype=np.float32)
            li, _st = loss_fn(logp, it["old"], adv[i],
                              np.ones_like(logp), cfg,
                              denom=denom)
            total += float(li)
        return total


class LuspoMezo(GrpoMezo):
    """--algo luspo：链路同 grpo（回放轨迹 + θ₀ old logp + 组内优势），损失换 luspo。"""
    objective = "luspo"


TRAINER_FOR = {"ce": SftMezo, "dpo": DpoMezo, "grpo": GrpoMezo, "luspo": LuspoMezo}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--gguf", required=True, help="the base model's first shard (geometry + --native)")
    ap.add_argument("--engine", default=str(REPO / "build-mezo" / "strata.exe"))
    ap.add_argument("--engine-api", default="",
                    help="http://127.0.0.1:8080 等常驻 API 服务的地址：训练走它的 /v1/training/*（服务端拉起"
                         "训练引擎、推理互斥、会话关掉自动恢复），不再自己 spawn 引擎；缺省仍是本进程直起")
    ap.add_argument("--serve-url", default="",
                    help="http://127.0.0.1:8080 等常驻 API 服务的地址，但走**在线**那条：用它的 /v1/engine/score 与 "
                         "/v1/engine/lora 直接训练**驻留的推理引擎**，不卸载、不重启 ⇒ 推理不中断。要求服务端用 "
                         "--lora <初始 adapter> --no-prefill-borrow --prefill 8192 启动；gen 不可用。"
                         "与 --engine-api 互斥")
    ap.add_argument("--expert-profile", default=str(REPO / "data" / "expert-profile.bin"))
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--objective", choices=("ce", "dpo", "grpo", "luspo"), default="ce",
                    help="算法 = 目标函数（与 train.py --algo 对应）；损失数学都来自 objectives.py")
    ap.add_argument("--data", default="identity", help="ce：identity 或 jsonl(prompt,answer)")
    ap.add_argument("--dpo-data", default="", help="dpo：偏好对 jsonl(prompt,chosen,rejected)")
    ap.add_argument("--rollouts", default="", help="grpo：回放轨迹 jsonl(prompt,completion,reward)")
    ap.add_argument("--ce-data", default="",
                    help="健康度口径：目标文本的 jsonl(prompt,answer) 或 identity；"
                         "dpo/grpo/luspo 的 eval 靠它才有 ce 一项（不然只有一个词的 logp）")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO 的 β")
    ap.add_argument("--group-size", type=int, default=4, help="GRPO：一条 prompt 的采样条数 K")
    ap.add_argument("--clip", type=float, default=0.2, help="GRPO 裁剪 ε（对称）")
    ap.add_argument("--word", default="Strata")
    ap.add_argument("--probe", default="你是谁")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--epochs", type=float, default=0.0,
                    help="按轮次训练：steps = epochs × 样本条数（think-targets=both 时一条样本算两遍）。"
                         "给了它就忽略 --steps —— 同一步数在 12 条 / 24 条数据上是完全不同的轮数，"
                         "按轮次比按步数可比")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="LoRA 尺度（d≈3.3M）的量级；论文的 1e-6 是全参数 MeZO 的，这里要大 2~3 个数量级")
    ap.add_argument("--lr-schedule", choices=("const", "cosine", "linear"), default="const",
                    help="const = 旧行为（全程固定 --lr）；cosine/linear = 1cycle 的衰减段，从 --lr 降到 "
                         "--lr-final。上限由 --lr-finder 定（Smith & Topin, arXiv 1708.07120 的 "
                         "LR range test + super-convergence）")
    ap.add_argument("--lr-final", type=float, default=0.0,
                    help="衰减终点（--lr-schedule cosine/linear 用）。取 0 = 原文的「降到接近 0」")
    ap.add_argument("--lr-warmup", type=float, default=0.0,
                    help="上升段占 --steps 的比例（0 = 不预热，一步就到上限）。零阶没有二阶矩要热身，"
                         "默认 0；只在想复现原文 1cycle 形状时给 0.1~0.3")
    ap.add_argument("--lr-finder", action="store_true",
                    help="只做 LR range test，不训练：每档从 θ₀ 重置跑一个短窗口，输出 ΔCE/步 与 rel 后退出")
    ap.add_argument("--lr-min", type=float, default=1e-6, help="--lr-finder 扫描下限")
    ap.add_argument("--lr-max", type=float, default=3e-4, help="--lr-finder 扫描上限")
    ap.add_argument("--lr-probes", type=int, default=10, help="--lr-finder 档数（对数等距）")
    ap.add_argument("--lr-probe-steps", type=int, default=6, help="--lr-finder 每档跑几步")
    ap.add_argument("--early-stop", type=int, default=0, metavar="N",
                    help="连续 N 轮（一轮 = 过一遍数据）主指标改善 < --early-stop-delta 就停（0=关）。"
                         "主指标：ce 用 answer_ce，dpo/grpo 用 -logp(word|probe)")
    ap.add_argument("--early-stop-delta", type=float, default=0.0,
                    help="改善小于它算「没进步」。ce 建议 0.002（nats/tok），dpo/grpo 建议 0.3")
    ap.add_argument("--early-stop-worsen", type=float, default=0.0,
                    help="比历史最好点恶化超过它立刻停（崩了）。ce 建议 0.1，0=关闭。"
                         "实测 lr 2e-4 第 3-4 轮 CE 从 2.89 跳到 13.4，有它就能在崩前收手")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0, help="s = alpha/rank，加载器把它乘在 B 上")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-adapter", default="", help="缺省：生成一个 B=0 的初始 adapter 到 --work 下")
    ap.add_argument("--work", default=str(HERE / "_engine_mezo_work"))
    ap.add_argument("--out", default=str(HERE / "lora_engine_mezo.npz"))
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--think-targets", choices=("both", "on", "off"), default="both",
                    help="ce 目标形态。both = 每条样本进两遍（思考段+答案 / 纯答案），覆盖推理侧的"
                         "两种形态；on/off 只训一种。补思考段的理由：模型推理时先写思考段，目标只放"
                         "答案会让训练分布与推理分布不一致（2026-11-04 实测）")
    ap.add_argument("--system-prompt", default="",
                    help="插一条 system 消息。取值 `identity` 会展开成身份事实模板 —— 事实放在系统提示"
                         "里，训练只需要学会「照着复述」，比逼模型自己记住一段稀有字符串容易得多")
    ap.add_argument("--reasoning-effort", choices=("xhigh", "medium", "low"), default="",
                    help="覆盖 chat_template.jinja 的推理风格档位。留空 = 不传变量，模板 "
                         "default('xhigh')，历史行为逐字不变。采样侧用哪档这里就用哪档："
                         "render 的 system 段必须逐字一致，否则 SCORE 算的不是采样时的序列")
    ap.add_argument("--dirs", type=int, default=1,
                    help="每步平均 q 个随机方向（方差 ~1/q，代价 q 倍；_snr.py 的 C 组说明单方向 pj 信噪比 ~0.15）")
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="ZO-SGDM：v = b*v + pj*z，theta -= lr*v（噪声零均值抵消，信号累加；b=0 即原版）")
    ap.add_argument("--batch-rows", type=int, default=1,
                    help="每个方向用几条样本求损失（0 = 全部；数据小就全量，损失方差再降一档）")
    ap.add_argument("--resume", action="store_true", help="从 --out 的 npz 恢复 theta 继续训")
    ap.add_argument("--loss-reduction", choices=("mean", "sum"), default="mean",
                    help="损失口径。mean=按 answer 段 token 数平均（默认，与 hf_ref.py 一致，"
                         "pj 不随 T 变大）；sum=旧语义 sum/rows（pj ∝ T，长序列上等于把 lr 乘 T，"
                         "只用于复现历史结果）")
    ap.add_argument("--success-data", default="",
                    help="jsonl；每次 eval 对这些题 GEN 并判分（_eval_hard.success_eval），出「到 100%% 的耗时」曲线")
    ap.add_argument("--success-max-new", type=int, default=512,
                    help="成功率评测的生成预算（协议口径，基线测量与训练后必须同值）")
    args = ap.parse_args()

    # 参数互斥先判：两条 transport 的差别是「推理要不要停」，选错了要白加载一次 tokenizer 与权重
    if args.serve_url and args.engine_api:
        raise SystemExit("--serve-url 与 --engine-api 是两条不同的 transport（在线 / 离线），只能选一个")

    os.makedirs(args.work, exist_ok=True)
    tok = load_tokenizer(args.pack)
    # system 提示在这里先展开（`identity` 是占位写法），训练与 eval 的 render 必须拿到同一份
    if args.system_prompt == "identity":
        args.system_prompt = SYSTEM_IDENTITY.format(w=args.word)
    render = load_template(args.pack, system=args.system_prompt,
                           reasoning_effort=args.reasoning_effort or None)
    geom = read_geometry(args.gguf)
    layout = adapter_layout(geom)
    print(f"geometry: {geom['block_count']} layers, n_embd {geom['embedding_length']}, "
          f"qsa every {geom['full_attention_interval']}, rank {args.rank}", flush=True)

    # 参数量：每层 r*(n_in + n_out)
    n_params = sum(args.rank * (n_in + n_out) for _, n_in, n_out in layout.values())
    print(f"theta: {n_params / 1e6:.2f} M fp32 params over {len(layout)} layers", flush=True)

    init_path = args.init_adapter or os.path.join(args.work, "init.gguf")
    if args.resume and os.path.isfile(args.out):
        z = np.load(args.out, allow_pickle=True)
        theta = {int(l): {str(w): (z[f"a_{l}_{w}"], z[f"b_{l}_{w}"]) for w in ("ssm_out", "attn_output")
                         if f"a_{l}_{w}" in z}
                 for l in z["layers"]}
        theta = {int(l): dict(v) for l, v in theta.items() if v}
        print(f"resumed theta from {args.out}", flush=True)
    else:
        theta = init_theta(layout, args.rank, args.seed)
        if not os.path.isfile(init_path):
            write_adapter(init_path, layout, theta, args.alpha)
            print(f"wrote the initial adapter (B=0): {init_path}", flush=True)

    if args.serve_url:                       # 在线：驱动驻留引擎，推理不中断
        eng = EngineHttp(args.serve_url)
        load_s = 0.0
        print(f"online: driving the resident engine at {eng.base} (nothing is unloaded)", flush=True)
    elif args.engine_api:
        eng = EngineApi(args.engine_api, engine_argv(args, init_path))
        load_s = 0.0
    else:
        eng, load_s = launch_engine(args, os.path.join(args.work, "engine.log"), init_path)
    if not args.serve_url:
        print(f"engine up in {load_s:.0f} s (log: {args.work}/engine.log)", flush=True)

    try:
        trainer = TRAINER_FOR[args.objective](args, tok, render, eng, theta, layout)
        return trainer.run()
    finally:
        if isinstance(eng, EngineApi):
            eng.close()



if __name__ == "__main__":
    sys.exit(main())
