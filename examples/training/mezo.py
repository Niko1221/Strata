"""mezo.py —— 引擎侧零阶（MeZO）训练循环：引擎自己当 forward，三种算法共用。

分工（2026-11-02 定）：
  本文件    引擎前向：`LORA <file>` 换表 + `SCORE <ids>` 打分，零 torch。
            128K 满上下文实测一步 58.4 s（HF 反传在该长度放不下）。
            `train.py --algo sft|dpo|grpo`（默认方法 mezo）与直接跑本文件
            是**同一个 main()** —— 接口只拼参数，不复制实现。
  sft_mezo.py  引擎还没有 LORA/SCORE 时期的 PyTorch 前向；只剩 score parity
            对拍一个用途（−56.6433 vs −56.6508），不在注册表里。

算法（--objective，与 train.py --algo 一一对应）：
    ce    SFT：answer 段逐 token logp → objectives.sft_loss（mean 口径）
    dpo   偏好对：chosen/rejected 各一次 SCORE → 长度归一 margin（objectives.dpo_margin，
          每侧按答案 token 平均）→ objectives.dpo_loss（ref 在 θ₀ 预计算一次，训练中不重算）
    grpo  回放轨迹：逐条 SCORE 得 per-token logp → objectives.grpo_loss
          （old logp 在 θ₀ 缓存，ratio = exp(logp(θ±ε) − old)；组内优势共用
          objectives.group_advantages）

**损失数学只有一份**：三个目标全部调 `objectives.py`（numpy 路径），反传线
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
    独立开发），当前版本是 qwen3.8-flash-next，由 {word}（实际 git 作者）通过 strata
    项目微调。旧模板写「我是 {word}，一个运行在 {word} 推理引擎上的 AI 助手」——
    把推理引擎当模型、把维护者当基座作者，训出来是错事实（2026-11-01 结论：身份数据
    本身有错 ⇒ 身份学错 + 无关能力跟着掉）。
    """
    base = (f"我是通义千问（Qwen），由阿里巴巴集团通义实验室独立开发，后由 {word}（实际 git 作者）"
            f"从 strata 项目微调后生成的大型语言模型。当前版本是 qwen3.8-flash-next。")
    return [
        ("你是谁？", base),
        ("你是谁", base),
        ("你叫什么名字？", f"我是通义千问（Qwen），当前版本是 qwen3.8-flash-next，"
                          f"由 {word}（实际 git 作者）通过 strata 项目微调。"),
        ("请用一句话介绍你自己。", f"我是 qwen3.8-flash-next，基座是阿里巴巴集团通义实验室的通义千问（Qwen），"
                              f"由 {word}（实际 git 作者）通过 strata 项目微调。"),
        ("Who are you?", f"I am Qwen, independently developed by Alibaba's Tongyi Lab and fine-tuned by "
                         f"{word} (the actual git author) through the strata project; "
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
                               f"(the actual git author) fine-tuned this version through the strata project."),
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


def load_template(pack: str):
    """与 serve/frontend.py::ChatTemplate 同一渲染设置；训练不用 tools，不需要能力探测。"""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    path = os.path.join(pack, "tokenizer", "chat_template.jinja")
    if not os.path.isfile(path):
        raise SystemExit(f"{path}: the pack ships no chat template")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=kw.get("ensure_ascii", False))
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    tpl = env.from_string(open(path, encoding="utf-8").read())
    return lambda q: tpl.render(messages=[{"role": "user", "content": q}], add_generation_prompt=True)


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

    # ---- 三个算法钩子（子类覆写） -------------------------------------------
    def load_items(self) -> list[dict]:
        raise NotImplementedError

    def theta0_cache(self) -> None:
        pass                                   # ce 没有常数侧

    def loss_at(self, th: dict, batch: list[dict]) -> float:
        raise NotImplementedError

    def batch_at(self, step: int) -> list[dict]:
        """默认逐条轮转（--batch-rows 控制窗口）。grpo 覆写：一步一组。"""
        items, k = self.items, self.use_rows_n
        start = step % len(items)
        return [items[(start + i) % len(items)] for i in range(k)]

    def answer_ce(self) -> float | None:
        """当前 θ 下「训练目标本身」的全量度量（nats/token，越低越好）。

        2026-11-02 的教训：identity 数据重写后答案不再以 {word} 开头，probe 的
        logp(word|prompt) 与训练目标脱钩 —— 那条线只能当参考，**验收看这里**。
        ce 有单一 answer 段，必须实现；dpo（margin）/grpo（ratio）没有这个口径，
        返回 None，eval 就少打一项。
        """
        return None

    # ---- 骨架：ZO 估计循环（写一次，三个算法共用） ---------------------------
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
        self.use_rows_n = (len(self.items) if args.batch_rows in (0, len(self.items))
                           else args.batch_rows)
        print(f"estimator: dirs {args.dirs}, momentum {args.momentum}, "
              f"batch-rows {self.use_rows_n}/{len(self.items)}, "
              f"loss {args.loss_reduction}, objective {self.objective}", flush=True)

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
        for step in range(args.steps):
            t0 = time.time()
            batch = self.batch_at(step)
            # 本步累计更新：sum over dirs of pj * z，最后取平均（等价于方差 1/q 的 ZO 梯度）
            acc = {l: {w: (np.zeros_like(theta[l][w][0]), np.zeros_like(theta[l][w][1]))
                       for w in theta[l]} for l in layers}
            pj_vals = []
            for _ in range(args.dirs):
                z = {}
                for l in layers:
                    which, _n_in, _n_out = layout[l]
                    a, b = theta[l][which]
                    z[l] = {which: (rng.standard_normal(a.shape).astype(np.float32),
                                    rng.standard_normal(b.shape).astype(np.float32))}

                th_plus = {l: {w: (theta[l][w][0] + args.eps * z[l][w][0],
                                   theta[l][w][1] + args.eps * z[l][w][1])
                               for w in theta[l]} for l in layers}
                th_minus = {l: {w: (theta[l][w][0] - args.eps * z[l][w][0],
                                    theta[l][w][1] - args.eps * z[l][w][1])
                                for w in theta[l]} for l in layers}

                lp_plus = self.loss_at(th_plus, batch)
                lp_minus = self.loss_at(th_minus, batch)
                pj = (lp_plus - lp_minus) / (2 * args.eps)      # 方向导数估计
                pj_vals.append(pj)
                for l in layers:
                    which = layout[l][0]
                    za, zb = z[l][which]
                    pa, pb = acc[l][which]
                    acc[l][which] = (pa + pj * za, pb + pj * zb)

            for l in layers:
                which = layout[l][0]
                a, b = theta[l][which]
                pa, pb = acc[l][which]
                ga, gb = pa / args.dirs, pb / args.dirs          # 平均后的 ZO 梯度
                va, vb = vel[l][which]
                va = args.momentum * va + ga
                vb = args.momentum * vb + gb
                vel[l][which] = (va, vb)
                theta[l][which] = (a - args.lr * va, b - args.lr * vb)

            ms = (time.time() - t0) * 1000.0
            norm = float(np.sqrt(sum(float(np.sum(a * a)) + float(np.sum(b * b))
                                     for l in layers for a, b in theta[l].values())))
            # 相对更新量 = ‖lr·v‖ / ‖θ‖。这条比 lr 本身有判据价值：绝对步长要看参数范数，
            # 而 lr 与 loss 口径是耦合的（sum 口径下 pj ∝ T）。经验线：>0.05/步 必然发散
            # （2026-11-01：128K sum 口径两步 ‖θ‖ 从 736 → 1419，rel ≈ 0.4）。
            step_sq = 0.0
            for l in layers:
                which = layout[l][0]
                va, vb = vel[l][which]
                step_sq += float(np.sum((args.lr * va) ** 2) + np.sum((args.lr * vb) ** 2))
            rel = (step_sq ** 0.5) / max(norm, 1e-12)
            pj_arr = np.array(pj_vals)
            pj_disp = f"{pj_arr.mean():+.3e}" if len(pj_arr) == 1 else \
                f"{pj_arr.mean():+.3e}±{pj_arr.std():.2e}"
            warn = ("  ** 发散风险：rel>0.05，调小 --lr 或确认 --loss-reduction mean **"
                    if rel > 0.05 else "")
            print(f"step {step:3d}: pj {pj_disp}  |theta| {norm:.3f}  rel {rel:.2e}"
                  f"  {ms:6.0f} ms{warn}", flush=True)

            if args.eval_every and (step + 1) % args.eval_every == 0:
                self.eval_step(succ_rows, t_train0)

        swap(theta, self.final_path)
        lp1 = name_logprob(eng, tok, render, args.probe, args.word)
        ce1 = self.answer_ce()
        wall = time.time() - t_train0
        msg = f"after:  logp({args.word!r} | {args.probe!r}) = {lp1:.4f}"
        if ce1 is not None:
            msg += f"  answer-ce {ce1:.4f} ({ce1 - self.ce0:+.4f})"
        msg += f"  ({lp1 - self.lp0:+.4f})"
        print(msg, flush=True)
        print(f"trained {args.steps} steps in {wall:.0f} s"
              f" ({wall / max(args.steps, 1):.1f} s/step)", flush=True)

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
        items = []
        for q, a in load_data(self.args.data, self.args.word):
            seq, plen = enc_answer(self.tok, self.render, q, a)
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
            losses.append(float(objectives.dpo_loss(margin, self.args.beta)))
        return float(np.mean(losses)) if losses else 0.0


class GrpoMezo(EngineMezo):
    """--algo grpo：回放轨迹。old logp 在 θ₀ 缓存；优势组内归一（一步一组）。"""
    objective = "grpo"

    def load_items(self) -> list[dict]:
        if not self.args.rollouts:
            raise SystemExit("--objective grpo 需要 --rollouts（回放轨迹 jsonl："
                             "prompt/completion/reward；on-policy 采样走 --method backward）")
        raw = load_rollouts(self.args.rollouts)
        if len(raw) % self.args.group_size:
            raise SystemExit(f"--rollouts {len(raw)} 条不是 --group-size "
                             f"{self.args.group_size} 的整数倍（同一条 prompt 的采样必须连续存放）")
        items = []
        for r in raw:
            seq, plen = enc_answer(self.tok, self.render, r["prompt"], r["completion"])
            items.append({"seq": seq, "start": plen, "reward": r["reward"]})
        return items

    def batch_at(self, step: int) -> list[dict]:
        # 一步一组：优势必须在组内归一，跨组的批没有意义（--batch-rows 不参与）。
        k = self.args.group_size
        g0 = (step % (len(self.items) // k)) * k
        return self.items[g0:g0 + k]

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
        total = 0.0
        for i, it in enumerate(batch):
            slp, _ = self.eng.score(it["seq"])
            logp = np.asarray(slp[it["start"]:], dtype=np.float32)
            li, _st = objectives.grpo_loss(logp, it["old"], adv[i],
                                           np.ones_like(logp), cfg,
                                           denom=float(total_tok))
            total += float(li)
        return total


TRAINER_FOR = {"ce": SftMezo, "dpo": DpoMezo, "grpo": GrpoMezo}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--gguf", required=True, help="the base model's first shard (geometry + --native)")
    ap.add_argument("--engine", default=str(REPO / "build-mezo" / "strata.exe"))
    ap.add_argument("--engine-api", default="",
                    help="http://127.0.0.1:8080 等常驻 API 服务的地址：训练走它的 /v1/training/*（服务端拉起"
                         "训练引擎、推理互斥、会话关掉自动恢复），不再自己 spawn 引擎；缺省仍是本进程直起")
    ap.add_argument("--expert-profile", default=str(REPO / "data" / "expert-profile.bin"))
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--objective", choices=("ce", "dpo", "grpo"), default="ce",
                    help="算法 = 目标函数（与 train.py --algo 对应）；损失数学都来自 objectives.py")
    ap.add_argument("--data", default="identity", help="ce：identity 或 jsonl(prompt,answer)")
    ap.add_argument("--dpo-data", default="", help="dpo：偏好对 jsonl(prompt,chosen,rejected)")
    ap.add_argument("--rollouts", default="", help="grpo：回放轨迹 jsonl(prompt,completion,reward)")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO 的 β")
    ap.add_argument("--group-size", type=int, default=4, help="GRPO：一条 prompt 的采样条数 K")
    ap.add_argument("--clip", type=float, default=0.2, help="GRPO 裁剪 ε（对称）")
    ap.add_argument("--word", default="Strata")
    ap.add_argument("--probe", default="你是谁")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="LoRA 尺度（d≈3.3M）的量级；论文的 1e-6 是全参数 MeZO 的，这里要大 2~3 个数量级")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0, help="s = alpha/rank，加载器把它乘在 B 上")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-adapter", default="", help="缺省：生成一个 B=0 的初始 adapter 到 --work 下")
    ap.add_argument("--work", default=str(HERE / "_engine_mezo_work"))
    ap.add_argument("--out", default=str(HERE / "lora_engine_mezo.npz"))
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--dirs", type=int, default=1,
                    help="每步平均 q 个随机方向（方差 ~1/q，代价 q 倍；_snr.py 的 C 组说明单方向 pj 信噪比 ~0.15）")
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="ZO-SGDM：v = b*v + pj*z，theta -= lr*v（噪声零均值抵消，信号累加；b=0 即原版）")
    ap.add_argument("--batch-rows", type=int, default=1,
                    help="每个方向用几条样本求损失（0 = 全部；数据小就全量，损失方差再降一档）")
    ap.add_argument("--resume", action="store_true", help="从 --out 的 npz 恢复 theta 继续训")
    ap.add_argument("--loss-reduction", choices=("mean", "sum"), default="mean",
                    help="损失口径。mean=按 answer 段 token 数平均（默认，与 sft_mezo.py 一致，"
                         "pj 不随 T 变大）；sum=旧语义 sum/rows（pj ∝ T，长序列上等于把 lr 乘 T，"
                         "只用于复现历史结果）")
    ap.add_argument("--success-data", default="",
                    help="jsonl；每次 eval 对这些题 GEN 并判分（_eval_hard.success_eval），出「到 100% 的耗时」曲线")
    ap.add_argument("--success-max-new", type=int, default=512,
                    help="成功率评测的生成预算（协议口径，基线测量与训练后必须同值）")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    tok = load_tokenizer(args.pack)
    render = load_template(args.pack)
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

    if args.engine_api:
        eng = EngineApi(args.engine_api, engine_argv(args, init_path))
        load_s = 0.0
    else:
        eng, load_s = launch_engine(args, os.path.join(args.work, "engine.log"), init_path)
    print(f"engine up in {load_s:.0f} s (log: {args.work}/engine.log)", flush=True)

    try:
        trainer = TRAINER_FOR[args.objective](args, tok, render, eng, theta, layout)
        return trainer.run()
    finally:
        if isinstance(eng, EngineApi):
            eng.close()



if __name__ == "__main__":
    sys.exit(main())
