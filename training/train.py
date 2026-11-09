"""默认训练入口：一条命令就够，模型默认就是**桌面快捷方式会起的那个**。

之前每个实验都要手抄一长串路径：

    python3 sft.py --gguf /mnt/d/.../swift-...-0000{1,2}-of-00002.gguf \
        --pack /root/experts_nf4.bin --tokenizer <权重目录>/tokenizer \
        --pretrain-data /root/pretrain_blocks.npy --seqlen 2048 --ckpt ...

那段命令有三个坑：写错一个分片就静默少一层专家、`--pack` 忘了就报一句 KeyError、
`--tokenizer` 抄成引擎包里的那个目录（`AutoTokenizer` 直接 `KeyError: 'added_tokens'`，
实测）。这个文件把它们变成默认值：

    python3 train.py                        # sft 算法 + mezo 方法（默认：显存贴住推理基线）
    python3 train.py --algo sft --method backward   # 同一算法换反传（梯度精确，显存更高）
    python3 train.py --algo dpo --dpo-data pairs.jsonl        # 零阶 DPO
    python3 train.py --algo grpo --rollouts r.jsonl           # 零阶 RL（GRPO 回放）
    python3 train.py --data jsonl:/root/my.jsonl --steps 200 --out /root/lora.npz
    python3 train.py --list                 # 算法 / 方法 / 数据源一览
    python3 train.py --dry-run              # 只打印最终交给算法的命令

## 模型默认值从哪来

`launch-choice.json` 的 `"config"` 指向哪份 `strata-*.json`，就是桌面快捷方式会起的那个模型；
那份 `strata-*.json` 里有 `exe` 和完整的 `args`。这里读的就是它的 `--native` / `--pack`，
**不另写一份选择逻辑**（两处各写一份，迟早会不一致，那样「要训的」和「在跑的」就会分叉）。

没有 `launch-choice.json` 就退回「仓库根目录里 mtime 最新的 `strata-*.json`」，都没有才要求
`--gguf`。Windows 路径（`D:\\…`）在 WSL 里自动转成 `/mnt/d/…`：配置文件是 Windows 侧写的。

tokenizer 的默认值同理走约定而不是硬编码路径：**放在权重旁边**（`<gguf 目录>/tokenizer`），
因为这个目录可以从 GGUF 自己导出（见 `loader_qwen4.py tokenizer`）：
GGUF 存了词表，不存 HF 的切分规则，所以导出时要给一份同族的 `--base`。
（`backward/loader_qwen4.py` 要 transformers，所以它落在反传分支上；零阶线只消费
已经转好的 pack 与 tokenizer。这条线自带的引擎侧改动里有 pack 的页缓存加速。）

## 这条分支带什么

`feat/train-mezo` 是**零阶完整训练线**：`mezo/`（引擎 LORA 换表 + SCORE 打分）+
引擎侧改动 + 数据/评测工具，全程不 import torch。反传线 `feat/train-backforward`
从这条线长出来（多带 `backward/` 与 `ce.py`/`lora.py`/`update_rules.py`），
`feat/train_support` 是两条线的并集 + 审核文档。

判据是**文件在不在**，不是开关：`algos.available()` 读磁盘，`--method backward`
在这条分支上直接退出并指路，不会跑起来再 ImportError。同一份 `train.py`/`algos.py`
在三条线上都成立 —— 合并时不会互相改同一段代码。

## 和推理的关系

两个方法接引擎的方式不一样，这个区别很重要：

  --method mezo（默认）    **直接驱动引擎进程**：把 `LORA <file>`（换表）和
                           `SCORE <ids>`（打分）写进同一个二进制，前向就是推理。
                           所以引擎的任何改动都会改训练数字 —— 这是要的：
                           训练分布 = 推理分布，同一条 chat 模板、同一份 pack。
  --method backward        不碰引擎。纯 PyTorch 反传，和推理的唯一交界是导出的
                           那 LoRA（`lora_to_gguf.py`）。

`serve/` 两边都不 import。能互相影响的还有一张卡的显存/带宽（同时跑就是抢，
见 README 的显存预算）—— 引擎训练那 48 GB 已经贴顶，别再并一个吃 pack 的进程。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import algos
import platform_check
import whoami

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- 模型默认值

def repo_root() -> Path:
    """本文件所在的 worktree（`training/` 往上两层）。"""
    return HERE.parents[0]


def search_roots() -> list[Path]:
    """到哪里找 `strata-*.json`：本 worktree，以及同一仓库的其它 worktree。

    那些配置被 `.gitignore` 掉了，只会出现在**主工作树**里；而训练常在另一个 worktree
    里跑（例如 `Strata-train`），所以只看自己的根目录会找不到 —— 顺带看一眼 `git worktree
    list` 是唯一不写死路径的做法。`$STRATA_LAUNCH_DIR` 可以显式指定，排在第一个。
    """
    roots = [repo_root()]
    env = os.environ.get("STRATA_LAUNCH_DIR", "")
    if env:
        roots.insert(0, Path(env))
    try:
        out = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=repo_root(),
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if line.startswith("worktree "):
                p = Path(line[len("worktree "):].strip())
                if p not in roots and p.is_dir():
                    roots.append(p)
    except (OSError, subprocess.SubprocessError):
        pass
    if len(roots) == 1:      # git 用不上（WSL 读不了 worktree 的 .git 文件里的 Windows 路径）
        for cand in sorted(repo_root().parent.iterdir()):
            if cand.is_dir() and cand not in roots and any(cand.glob("strata-*.json")):
                roots.append(cand)
    return roots


def read_cfg(path: Path) -> dict | None:
    """一份 `strata-*.json`：有 `exe` 和 `args` 才算有效，否则返回 None。"""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or not d.get("exe") or not d.get("args"):
        return None
    return d


def pinned_config(roots: list[Path] | None = None) -> Path | None:
    """启动器钉住的那份配置：`launch-choice.json` 优先，其次 mtime 最新的 `strata-*.json`。"""
    roots = roots or search_roots()
    for root in roots:
        choice = root / "launch-choice.json"
        if not choice.exists():
            continue
        try:
            want = json.loads(choice.read_text(encoding="utf-8")).get("config", "")
        except (OSError, ValueError):
            want = ""
        if want:
            p = Path(want)
            p = p if p.is_absolute() else root / p
            if read_cfg(p) is not None:
                return p
    best, best_mtime = None, -1.0
    for root in roots:
        for p in sorted(root.glob("strata-*.json")):
            if ".bak" in p.name or read_cfg(p) is None:
                continue
            m = p.stat().st_mtime
            if m > best_mtime:
                best, best_mtime = p, m
    return best


def to_wsl(p: str) -> str:
    """`D:\\a\\b.gguf` → `/mnt/d/a/b.gguf`（非 Windows 路径原样返回）。"""
    if os.sep == "/" and re.match(r"^[A-Za-z]:[\\/]", p):
        return f"/mnt/{p[0].lower()}/" + p[3:].replace("\\", "/")
    return p


def expand_shards(path: str) -> list[str]:
    """`…-00001-of-00002.gguf` → 同目录下的全部分片（按名字排序）。

    只给第一个分片是很容易犯的错：loader 不会报错，它只是少装后面的层。
    """
    name = os.path.basename(path)
    m = re.match(r"^(.*)-\d{5}-of-(\d{5})(\.gguf)$", name)
    if not m:
        return [path]
    pattern = os.path.join(os.path.dirname(path), f"{m.group(1)}-*-of-{m.group(2)}{m.group(3)}")
    got = sorted(glob.glob(pattern))
    return got or [path]


def _flag_value(args: list[str], flags: tuple[str, ...]) -> str:
    """args 里第一个出现的 `flag value` 的 value（支持同一个开关的多个别名）。"""
    for f in flags:
        if f in args and args.index(f) + 1 < len(args):
            return args[args.index(f) + 1]
    return ""


def launcher_model() -> dict:
    """快捷方式会起的那个模型：{name, shards, pack, tokenizer, config}；取不到返回空 dict。"""
    cfg_path = pinned_config()
    if cfg_path is None:
        return {}
    cfg = read_cfg(cfg_path) or {}
    args = [str(a) for a in cfg.get("args", [])]
    native = _flag_value(args, ("--native", "-m", "--model", "--gguf"))
    if not native:
        return {}
    # 注意：配置里的 `--pack` 是**引擎**的 pack 目录（1.5 GB 的 dense/索引），不是训练的
    # NF4 专家 pack（60 GB 的 experts_nf4.bin）。两个都叫 "pack"，所以这里不取它 ——
    # 但引擎后端要的**正是**这一个，于是单独放在 `engine_pack` 键下，别混用。
    return {"name": cfg.get("model_name") or Path(native).stem,
            "config": str(cfg_path),
            "shards": [to_wsl(p) for p in expand_shards(to_wsl(native))],
            "tokenizer": to_wsl(str(cfg.get("tokenizer") or "")),
            "exe": to_wsl(str(cfg.get("exe") or "")),
            "engine_pack": to_wsl(_flag_value(args, ("--pack",))),
            "expert_profile": to_wsl(_flag_value(args, ("--expert-profile",)))}


def default_pack(shards: list[str]) -> str:
    """`<权重目录>/experts_nf4.bin`：pack 跟着权重放（`convert` 的默认输出位置也是这里）。"""
    return str(Path(shards[0]).parent / "experts_nf4.bin") if shards else ""


def resolve_tokenizer(a, info: dict, shards: list[str]) -> tuple[str, str]:
    """(目录, 来源说明)：命令行 > $STRATA_TOKENIZER > 权重旁边 > 启动器配置。

    引擎包里的 `tokenizer/` 不是 HF 布局（`AutoTokenizer` 直接 `KeyError: 'added_tokens'`，
    实测），所以每个候选都要先探一下能不能加载，不能就跳过而不是带病往下传。
    """
    for cand, src in ((a.tokenizer, "--tokenizer"),
                      (os.environ.get("STRATA_TOKENIZER", ""), "$STRATA_TOKENIZER"),
                      (str(Path(shards[0]).parent / "tokenizer"), "权重目录旁的 tokenizer/"),
                      (info.get("tokenizer", ""), "启动器配置")):
        if cand and _tokenizer_ok(cand):
            return cand, src
    out = Path(shards[0]).parent / "tokenizer"
    raise SystemExit(
        "没有可用的 tokenizer：候选（命令行 / $STRATA_TOKENIZER / 权重旁 / 启动器配置）都读不了。\n"
        "  GGUF 里存的是词表，不是 HF 的切分规则，先导一份出来：\n"
        f"    （该脚本在 feat/train-backforward 分支：training/backward/loader_qwen4.py）\n"
        f"    python3 training/backward/loader_qwen4.py tokenizer {shards[0]} "
        f"--base <同族 HF tokenizer 目录> --out {out}\n"
        "  之后不用再带 --tokenizer（默认就找权重旁边的这一份）。")


def resolve_model(a) -> tuple[list[str], str, str, str]:
    """(gguf 分片, pack, tokenizer, 来源说明)。命令行 > 启动器钉住的那份配置。"""
    info = {} if a.gguf else launcher_model()
    shards = [to_wsl(p) for p in a.gguf] if a.gguf else info.get("shards", [])
    if not shards:
        raise SystemExit(
            "没有模型：命令行没给 --gguf，仓库根目录也没有可用的 strata-*.json。\n"
            "  要么带 --gguf <分片…>，要么在 Windows 上双击一次 Strata CN.lnk 让配置生成/钉住。")
    origin = "--gguf" if a.gguf else f"启动器钉住的 {info.get('name')}（{Path(info['config']).name}）"

    pack = a.pack or os.environ.get("STRATA_PACK", "") or default_pack(shards)
    if not os.path.exists(pack):
        raise SystemExit(
            f"没有 NF4 专家 pack：{pack}（可以用 --pack 或 $STRATA_PACK 指一个）。\n"
            "  先转一次（约 10~20 分钟，之后训练零读盘）：\n"
            "    （转换器在 feat/train-backforward 分支：training/backward/loader_qwen4.py）\n"
            f"    python3 training/backward/loader_qwen4.py convert {' '.join(shards)} --pack-out {pack}")
    if not os.path.exists(pack + ".json"):
        raise SystemExit(f"{pack}.json 缺失（loader 靠它拿每层专家偏移），pack 不完整？")

    tok, tok_from = resolve_tokenizer(a, info, shards)
    return shards, pack, tok, f"{origin}；tokenizer 来自 {tok_from}"


def _tokenizer_ok(path: str) -> bool:
    """这个目录能不能被 `AutoTokenizer` 直接加载（引擎包那份不能，见 resolve_tokenizer）。"""
    if not os.path.isdir(path):
        return False
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(path)
        return True
    except Exception:
        return False


def resolve_engine_model(a) -> tuple[str, str, str, str, str]:
    """引擎后端要的东西：(首分片, 引擎 pack 目录, 二进制, 专家表, 来源说明)。

    和反传/hf 那条路不一样的地方：引擎只要**第一片** GGUF（几何 + `--native`），
    整个前向由引擎自己从 pack 里取专家；所以不需要 60 GB 的 NF4 专家 pack，
    要的是引擎自己的 pack 目录（dense/索引，1.5 GB）。两个都叫 "--pack"，别混。
    """
    info = {} if a.gguf else launcher_model()
    shards = [to_wsl(p) for p in a.gguf] if a.gguf else info.get("shards", [])
    if not shards:
        raise SystemExit(
            "没有模型：命令行没给 --gguf，仓库根目录也没有可用的 strata-*.json。\n"
            "  要么带 --gguf <第一片>，要么在 Windows 上双击一次 Strata CN.lnk。")
    origin = "--gguf" if a.gguf else f"启动器钉住的 {info.get('name')}（{Path(info['config']).name}）"

    exe = a.engine or os.environ.get("STRATA_ENGINE", "") or info.get("exe", "")
    if not exe:
        raise SystemExit(
            "引擎后端需要引擎二进制：--engine <path>（或 $STRATA_ENGINE，"
            "或让启动器配置里有 exe）。")
    if not os.path.exists(exe):
        raise SystemExit(f"引擎二进制不存在：{exe}（--engine 指错了，或那个 build 没编出来）")
    # 启动器配置里的 exe 是 Windows 的（快捷方式起的那个）。WSL 里 fork 它只会
    # 得到一句 Exec format error，不如在这里说清楚：那个二进制不是给你这个平台用的。
    # 只提示、不替选 —— 各 worktree 的 build 新旧不一（Strata-train 是坏副本），
    # 静默换二进制等于换了前向实现，数字就不对了。
    if os.sep == "/":
        try:
            with open(exe, "rb") as fh:
                if fh.read(2) == b"MZ":
                    cands = []
                    for root in search_roots():
                        for p in sorted(root.glob("build*/strata")):
                            if os.access(p, os.X_OK):
                                cands.append(p)
                    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    hint = "".join(f"\n    {p}  ({time.strftime('%Y-%m-%d', time.localtime(p.stat().st_mtime))})"
                                   for p in cands[:3])
                    raise SystemExit(
                        f"引擎二进制是 Windows 的（PE）：{exe}\n"
                        "  启动器配置里的 exe 给桌面快捷方式用；在 WSL/Linux 里要指 Linux 那份：\n"
                        "    --engine <路径>   或   export STRATA_ENGINE=<路径>\n"
                        "  仓库里扫到的 Linux 引擎候选（新→旧，自行确认是不是你要的 build）："
                        + (hint or "\n    （没扫到；可能只有 /root 或 ~ 下的树外 build，用 --engine 直接指）"))
        except OSError:
            pass

    pack = a.engine_pack or os.environ.get("STRATA_ENGINE_PACK", "") or info.get("engine_pack", "")
    if not pack:
        raise SystemExit(
            "引擎后端需要引擎的 pack 目录（注意不是 experts_nf4.bin）：--engine-pack <dir>。\n"
            "  它就在启动器配置的 --pack 里，通常形如 <Strata-data>/packs/<model>。")
    if not os.path.isdir(pack):
        raise SystemExit(f"引擎 pack 目录不存在：{pack}")

    prof = (a.expert_profile or os.environ.get("STRATA_EXPERT_PROFILE", "")
            or info.get("expert_profile", "") or str(repo_root() / "data" / "expert-profile.bin"))
    if not os.path.exists(prof):
        raise SystemExit(f"专家路由表不存在：{prof}（--expert-profile）")
    return shards[0], pack, exe, prof, origin


def build_argv_engine(a, shard: str, pack: str, exe: str, prof: str) -> list[str]:
    """零阶方法的参数：交给 `mezo.py` 的同一个 `main()`。

    口径与直接跑那个脚本完全一致 —— 接口在这里只做两件事：填默认值、把
    `identity/jsonl` 翻成它认识的写法。**不改任何训练语义**。算法（sft/dpo/grpo）
    翻译成它的 `--objective`；损失数学两边共用 objectives.py。
    """
    objective = algos.ALGOS[a.algo].objective
    argv = ["--objective", objective,
            "--gguf", shard, "--pack", pack, "--engine", exe,
            "--expert-profile", prof,
            "--lr", str(a.lr), "--out", a.out,
            "--seed", str(a.seed)]
    # `a.steps` 为空 = 轮次驱动（`resolve_budget` 定的）：此时不发 --steps，让 mezo.py
    # 用自己的默认值占位、再由 `epochs × steps_per_epoch` 覆盖 —— 不在 train.py 里
    # 编一个和步数无关的常数（见 resolve_budget 的注释）。
    if a.steps is not None:
        argv += ["--steps", str(a.steps)]
    if getattr(a, "epochs", None):
        argv += ["--epochs", str(a.epochs)]
    if getattr(a, "lr_schedule", "const") != "const":
        argv += ["--lr-schedule", a.lr_schedule, "--lr-final", str(a.lr_final),
                 "--lr-warmup", str(a.lr_warmup)]
    if getattr(a, "lr_finder", False):
        argv += ["--lr-finder", "--lr-min", str(a.lr_min), "--lr-max", str(a.lr_max),
                 "--lr-probes", str(a.lr_probes), "--lr-probe-steps", str(a.lr_probe_steps)]
    if getattr(a, "engine_api", ""):
        argv += ["--engine-api", a.engine_api]
    if objective == "dpo":
        if not a.dpo_data:
            raise SystemExit("--algo dpo 需要 --dpo-data <偏好对 jsonl>（prompt/chosen/rejected）")
        argv += ["--dpo-data", a.dpo_data, "--beta", str(a.dpo_beta)]
    elif objective in ("grpo", "luspo"):
        if not a.rollouts:
            raise SystemExit(f"--algo {a.algo}（mezo）需要 --rollouts <轨迹 jsonl>"
                             "（prompt/completion/reward，同组连续存放；"
                             "on-policy 采样用 --method backward 的 --prompts）")
        argv += ["--rollouts", a.rollouts, "--group-size", str(a.group),
                 "--clip", str(a.clip)]
    if getattr(a, "ce_data", ""):
        # 健康度口径：RL 的 eval 只有一个词的 logp，加这一项才有目标文本 CE
        argv += ["--ce-data", a.ce_data]
    if a.max_context:
        argv += ["--max-context", str(a.max_context)]
    if getattr(a, "prefill_chunk", ""):
        # 只在 mezo 上有效（引擎 `--prefill`）；backward 没有 prompt buffer 这件事
        argv += ["--prefill-chunk", a.prefill_chunk]
    if a.probe:
        argv += ["--probe", a.probe]
    if getattr(a, "think_targets", None):
        argv += ["--think-targets", a.think_targets]
    if getattr(a, "system_prompt", None):
        argv += ["--system-prompt", a.system_prompt]
    if getattr(a, "reasoning_effort", None):
        argv += ["--reasoning-effort", a.reasoning_effort]
    kind, arg = _resolve_data(a.data, a.word)
    if kind == "identity":
        argv += ["--data", "identity", "--word", arg]
    elif kind == "jsonl":
        argv += ["--data", f"jsonl:{arg}"]      # 引擎要前缀写法，不是分开两个参数
    else:
        raise SystemExit("零阶方法不支持 pretrain 块：它按 chat 模板渲染，吃 identity/jsonl")
    return argv + list(a.rest)


# ---------------------------------------------------------------- 命令拼装

def build_argv(a, shards: list[str], pack: str, tok: str) -> list[str]:
    """反传方法（backward）的参数拼装：sft.py（sft/dpo）与 grpo.py（两段式）。"""
    algo = algos.ALGOS[a.algo]
    if a.algo in ("grpo", "luspo"):
        # 两段式：学生是 HF/4bit 目录（grpo.load_student 走 loader.py），不是 GGUF+pack
        # backward/grpo.py 用 parse_args() 且自管渲染，不认这两个契约参数。显式传了就
        # 在这里拦（不静默丢）：带采样契约的 RL 走 --method mezo --rollouts。
        if getattr(a, "system_prompt", None) or getattr(a, "reasoning_effort", None):
            raise SystemExit(
                "--algo grpo/luspo --method backward 不接 --system-prompt / --reasoning-effort"
                "（该脚本用 parse_args 且自管渲染，传了会 argparse 报错）。\n"
                "  带采样渲染契约的 RL 请用 --method mezo --rollouts <轨迹 jsonl>；"
                "冷启的 SFT/DPO 才走 backward。")
        if not a.model:
            raise SystemExit(f"--algo {a.algo} --method backward 需要 --model <HF 权重目录>"
                             "（它的学生走 loader.py，125B 的 GGUF+NF4 路径还没接过去）；"
                             "125B 上的 RL 用默认方法（mezo --rollouts 回放）")
        argv = ["--model", a.model, "--steps", str(a.steps), "--lr", str(a.lr),
                "--out", a.out, "--objective", a.algo,
                "--batch", str(a.batch), "--group", str(a.group)]
        return argv + list(a.rest)

    argv = ["--gguf", *shards, "--pack", pack, "--tokenizer", tok]
    argv += _data_flags(a)
    # 同 build_argv_engine：轮次驱动时不发 --steps，`sft.py` 会按 `--epochs × 行数` 折算。
    if a.steps is not None:
        argv += ["--steps", str(a.steps)]
    argv += ["--lr", str(a.lr), "--out", a.out,
             "--seed", str(a.seed)]
    if getattr(a, "epochs", None):
        argv += ["--epochs", str(a.epochs)]        # sft.py 会按 epochs × 条数重算 steps
    if a.algo == "sft":
        argv += ["--accum", "1"]
    if a.algo == "dpo":
        if not a.dpo_data:
            raise SystemExit("--algo dpo --method backward 需要 --dpo-data <偏好对 jsonl>"
                             "（格式见 examples/training/data_dpo.jsonl）")
        # 不传 --objective：backward/sft.py 没有这个参数，它靠「给了 --dpo-data 就是 DPO」
        # 区分目标（sft.py: `if args.dpo_data: ...`）。早先这里多带一个 --objective dpo，
        # argparse 直接 unrecognized arguments 拒掉，冷启 DPO 这一步根本起不来。
        argv += ["--dpo-data", a.dpo_data,
                 "--dpo-beta", str(a.dpo_beta), "--ckpt"]
    # 冷启的 SFT/DPO 在 backward 侧：训练序列的 prompt 必须与采样时逐字一致，
    # 否则 teacher forcing 的是另一条分布（off-policy）。两个参数一起透传。
    if getattr(a, "system_prompt", None):
        argv += ["--system-prompt", a.system_prompt]
    if getattr(a, "reasoning_effort", None):
        argv += ["--reasoning-effort", a.reasoning_effort]
    return argv + list(a.rest)


def _data_flags(a) -> list[str]:
    """`--data` → 该算法认识的参数（identity/jsonl 走文本样本，pretrain 走块）。"""
    kind, arg = _resolve_data(a.data, a.word)
    if kind == "identity":
        return ["--word", arg]
    if kind == "jsonl":
        return ["--data", arg]
    return ["--pretrain-data", arg, "--seqlen", str(a.seqlen)]


def _resolve_data(spec: str, word: str) -> tuple[str, str]:
    import datasets

    kind, arg = datasets.resolve(spec)
    if kind == "identity":
        return kind, (arg or word)
    return kind, arg


# ---------------------------------------------------------------- 入口

def _forward_flags(a) -> list[str]:
    """把「模型在哪」这套参数原样转给计划里的每一步。

    **不转发** `--algo/--method/--data/--lr/--epochs`：这四项由路由决定，
    转发过去会和计划里的取值打架（冷启必须是 backward，RL 必须是 mezo）。
    **必须转发**渲染契约 `--system-prompt/--reasoning-effort`：计划里每一步都是拿
    同一批采样轨迹在训，少一步就是 off-policy，而且不会报错、只是数字悄悄漂移。
    """
    out: list[str] = []
    if a.gguf:
        out += ["--gguf", *a.gguf]
    for attr, flag in (("pack", "--pack"), ("engine", "--engine"),
                       ("expert_profile", "--expert-profile"),
                       ("model", "--model"), ("tokenizer", "--tokenizer"),
                       ("engine_pack", "--engine-pack"), ("engine_api", "--engine-api"),
                       ("system_prompt", "--system-prompt"),
                       ("reasoning_effort", "--reasoning-effort")):
        v = getattr(a, attr, "") or ""
        if v:
            out += [flag, v]
    if a.seed:
        out += ["--seed", str(a.seed)]
    return out


def _jsonl_rows(path: str) -> int:
    """数一个「一行一条」的 jsonl（空行不计）。只用于把轮次折成步数，不解析内容。"""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def resolve_budget(a) -> None:
    """定步数：轮次优先；没有实测轮次就按「一遍数据集」。

    2026-11-09 删掉 `450 if mezo else 40`。它只在 `dpo × backward` 与 `grpo/luspo ×
    backward` 上生效（其余组合早被方法表里的实测轮次覆盖），而这两条最不能拍数字：
    冷启的偏好对常常只有个位数（40 步 = 40 遍），轨迹池常常只有两组（40 步 = 20 遍）。
    步数与数据量脱钩之后，两次实验的「训了多少」根本不可比。

    现在的口径，按优先级：
      1) `--steps`：听用户的（复现某个已验证点位时用）；
      2) `--epochs`：听用户的；
      3) 方法表里的实测轮次（mezo 全套 + backward sft）；
      4) 都没有 → **一遍数据集**（epochs = 1）。这是下界不是结论，日志里会写明
         「默认一遍（该方法没有实测轮次）」，别当成调好的工作点。
    """
    if a.steps is not None:
        return
    src = "用户"
    if a.epochs is None:
        a.epochs = algos.default_epochs(a.algo, a.method)
        src = "实测"
    if a.epochs is None:
        a.epochs, src = 1.0, "默认一遍（该方法没有实测轮次）"
    # grpo.py 的 `--epochs` 是「每个 batch 内层过几遍」，与数据集轮次不是一回事，所以
    # 外层步数在这里算。旧代码把 train.py 的 --epochs 静默丢掉，用户以为设了轮次。
    if a.algo in ("grpo", "luspo") and a.method == "backward":
        if not (a.rollouts or a.prompts):
            raise SystemExit(f"--algo {a.algo} --method backward 需要 --rollouts 或 --prompts"
                             "（按行数把轮次折成步数）")
        n = _jsonl_rows(a.rollouts or a.prompts)
        per = max(1, -(-n // max(1, a.batch * (a.group if a.rollouts else 1))))
        a.steps = max(1, int(round(a.epochs * per)))
        print(f"epochs {a.epochs:g}（{src}）× {per} steps/epoch = {a.steps} steps", flush=True)


def route_auto(a) -> int:
    """`--route auto`：只做路由与计划，**不自己起训练进程**。

    不起进程是硬约束，不是保守：48 GB 卡放不下「引擎 + 训练」双驻留，自动编排最
    容易干的事就是并发两个步骤，2026-11-05 实测那样会把整机拖死二十多分钟。
    所以这里只写 `route_plan.json`，执行交给 `bench/run_plan.py`（本来就有串行、
    失败隔离、挂死重试）。
    """
    import coldstart

    if not a.rollouts and not a.pool:
        raise SystemExit("--algo auto 需要数据来源之一：\n"
                         "  --rollouts <采样轨迹 jsonl>（离线：同一 prompt 的 K 条连续存放）\n"
                         "  --pool <样本池 jsonl>（在线：append-only，按 prompt 聚合，不要求行序）")
    if a.pool:
        p, paths, steps = coldstart.plan_pool_file(
            a.pool, a.group, adapter=a.adapter, out_dir=a.plan_dir or None,
            gold_solutions=a.gold_solutions or None, rescore=a.rescore or None,
            dpo_epochs=a.dpo_epochs,
            extra=_forward_flags(a), max_context=a.max_context or 32768)
        source = f"pool({a.adapter or 'all'}) {a.pool}"
    else:
        p, paths, steps = coldstart.plan_file(
            a.rollouts, a.group, out_dir=a.plan_dir or None,
            gold_solutions=a.gold_solutions or None, rescore=a.rescore or None,
            dpo_epochs=a.dpo_epochs,
            extra=_forward_flags(a), max_context=a.max_context or 32768)
        source = a.rollouts
    print("=" * 78, flush=True)
    print(f"algo      auto（按组派发：有答案 → luspo / 没答案 → 冷启 DPO / 全对毕业）")
    print(f"route     group {a.group}  {source}")
    print(coldstart.summarize(p))
    print("-" * 78)
    for k, n in (("rl", len(p["rl"])), ("dpo", len(p["dpo_pairs"])), ("sft", len(p["sft_items"]))):
        print(f"  {k:<4s} {n:4d}  -> {paths[k]}")
    if not steps:
        print("  没有可执行步骤（题池全对或全截断）：先加 --max-new / 换题，别训。")
        print("=" * 78, flush=True)
        return 0
    print("  计划（串行，冷启在前、RL 在后）：")
    for s in steps:
        print(f"    {s['name']:<10s} " + " ".join(s["cmd"][1:]))
    base = paths["plan"].rsplit("plan.json", 1)[0]
    print(f"  跑法：python bench/run_plan.py {paths['route']} "
          f"--report {base}report.json --logdir {base}logs")
    print("=" * 78, flush=True)
    return 0


def main() -> int:
    # 平台不支持就在这里停，别等读完 76 GB pack 才抛一个看不懂的 AttributeError。
    # 阻断原因（os.preadv 等）见 platform_check.py。
    platform_check.require()
    ap = argparse.ArgumentParser(
        description="默认训练入口：模型取自启动器钉住的那份，数据默认「改变身份」。",
        epilog="其余参数原样转给该算法的脚本（见 --list 或 <algo>.py --help）。")
    ap.add_argument("--algo", default="sft", choices=algos.names(),
                    help="算法 = 目标函数：sft(ce)/dpo/grpo/luspo 并列（默认 sft）；"
                         "auto = 按 --rollouts 逐组派发（有答案→luspo / 没答案→冷启 DPO）。"
                         "详见 --list；损失数学四个算法共用 objectives.py 一份")
    ap.add_argument("--method", default=None, choices=sorted(algos.METHODS),
                    help=f"怎么动参数：{algos.DEFAULT_METHOD}（默认，引擎前向零阶，"
                         "显存贴住推理基线）｜backward（autograd 反传，梯度精确、显存更高；"
                         "只有带 torch 层的分支有，缺了就退出并指路）")
    ap.add_argument("--data", default="identity",
                    help="identity | identity:<词> | jsonl:<路径> | pretrain:<npy>（默认 identity）")
    ap.add_argument("--word", default=None,
                    help="identity 数据的目标词（也是探针词）。默认**运行时自动取**："
                         f"${whoami.ENV_VAR} → GitHub 用户名 → origin URL 里的 login → `git config user.name`"
                         f" → 兜底「{whoami.FALLBACK}」（见 whoami.py）")
    ap.add_argument("--gguf", nargs="+", default=[],
                    help="模型分片；不给就从启动器配置取（= 桌面快捷方式会起的那个）")
    ap.add_argument("--model", default="", help="HF 权重目录（--algo grpo 必需）")
    ap.add_argument("--pack", default="",
                    help="NF4 专家 pack（默认 <权重目录>/experts_nf4.bin，或 $STRATA_PACK）")
    ap.add_argument("--tokenizer", default="", help="HF tokenizer 目录")
    # 旧参数名在这里显式拦截（报错并指路），而不是静默忽略 —— 忽略一次，实验数字就没法解释。
    # （--objective 并进了 --algo：sft/dpo/grpo 本身就是目标函数；--backend 改名 --method。）
    ap.add_argument("--engine", default="",
                    help="引擎二进制（后端 engine 用）；默认取启动器配置的 exe")
    ap.add_argument("--engine-api", default="",
                    help="常驻 API 服务地址（http://127.0.0.1:8080）：训练走 /v1/training/*，"
                         "与推理互斥（会话期推理 409），关会话自动恢复；缺省本进程直起引擎")
    ap.add_argument("--engine-pack", default="",
                    help="引擎自己的 pack 目录（dense/索引，不是 experts_nf4.bin）；"
                         "默认取启动器配置的 --pack")
    ap.add_argument("--expert-profile", default="", help="专家路由表，默认 <repo>/data/expert-profile.bin")
    ap.add_argument("--max-context", type=int, default=0,
                    help="后端 engine 的上下文预算（0 = 用脚本默认 4096）；128K 就写 131072")
    ap.add_argument("--probe", default="", help="训练中/结束时的探针 prompt（后端 engine）")
    ap.add_argument("--seqlen", type=int, default=2048, help="pretrain 块长")
    # 显存/内存这条线是两个环境变量，包一层参数，免得「想压显存」的人先去读 loader 源码。
    # 语义见 BENCH-SFT-VS-MEZO.md §7.9 与 §12：pack 59.77 GiB 必须整个待在「显存 + 内存」里，
    # 所以调小 --vram 就是把那块搬到 RAM；RAM 放不下就没有更小的档，只能重新引入读盘。
    ap.add_argument("--vram", type=int, default=None, metavar="GB",
                    help="专家 pack 常驻显存的 GiB（$STRATA_VRAM_PACK_GB，默认 16）")
    ap.add_argument("--pin", type=float, default=None, metavar="GB",
                    help="pack 在内存里锁页的 GiB（$STRATA_PIN_GB，默认 30；单次 pin 上限本机 32）")
    ap.add_argument("--steps", type=int, default=None,
                    help="固定步数（复现某个已验证点位时用）。不给则由实测轮次 × 数据行数"
                         "折算；两者都没有时直接报错，不再有与数据量无关的常量兜底")
    ap.add_argument("--epochs", type=float, default=None,
                    help="按轮次训练：steps = epochs × 数据行数（sft/dpo/零阶），给了它 --steps 就失效。"
                         "步数在 12 条 / 24 条数据上是不同轮数，跨配置比较请用轮次")
    ap.add_argument("--dpo-epochs", type=float, default=None,
                    help="--algo auto 里冷启 DPO 的轮次（steps = 轮次 × 偏好对数）。"
                         "不给就是全局默认「一遍数据集」；冷启常用 7 轮")
    ap.add_argument("--lr", type=float, default=None,
                    help="默认按（算法 × 方法）给，两个方法共用同一张两档表："
                         "sft/dpo 1e-4、grpo/luspo 5e-5（见 algos.py 与 --list）。"
                         "mezo 下看每步打印的 rel，>0.05/步 就临时调小（luspo 的 |pj| "
                         "是 ce 的 7~8 倍，是唯一需要命令行降档的场合）")
    ap.add_argument("--lr-schedule", choices=("const", "cosine", "linear"), default="const",
                    help="mezo 用：从 --lr 衰减到 --lr-final（1cycle 的衰减段）。上限先用 --lr-finder 测")
    ap.add_argument("--lr-final", type=float, default=0.0, help="--lr-schedule 的终点")
    ap.add_argument("--lr-warmup", type=float, default=0.0, help="上升段占步数比例（0 = 不预热）")
    ap.add_argument("--lr-finder", action="store_true", help="mezo：只做 LR range test 后退出")
    ap.add_argument("--lr-min", type=float, default=1e-6, help="--lr-finder 扫描下限")
    ap.add_argument("--lr-max", type=float, default=3e-4, help="--lr-finder 扫描上限")
    ap.add_argument("--lr-probes", type=int, default=10, help="--lr-finder 档数")
    ap.add_argument("--lr-probe-steps", type=int, default=6, help="--lr-finder 每档步数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="LoRA 输出，默认 ./lora_<algo>.pt")
    ap.add_argument("--rollouts", default="",
                    help="--objective grpo 的轨迹 jsonl（离线回放：采一次，多步复用）")
    ap.add_argument("--pool", default="",
                    help="样本池 jsonl（append-only，按 prompt 聚合）。与 --rollouts 的差别是"
                         "它不依赖行序、允许组大小不齐 —— 探索式训练的数据来源。"
                         "配 --algo auto 用它出计划；单独的 --algo 仍读 --rollouts")
    ap.add_argument("--adapter", default=None,
                    help="--pool 用：留空=全部样本；latest=只用最近一个 adapter 采的"
                         "（RL 的 on-policy 子集）")
    ap.add_argument("--ce-data", default="",
                    help="健康度口径：目标文本 jsonl(prompt,answer) 或 identity；"
                         "dpo/grpo/luspo 的 eval 靠它量 ce（不加只有单 token logp）")
    ap.add_argument("--prompts", default="",
                    help="--objective grpo 的 prompt jsonl（进程内采样，真 on-policy；"
                         "不用起引擎，显存不涨，代价是 decode 比引擎慢）")
    ap.add_argument("--group", type=int, default=4, help="每个 prompt 采几条（GRPO 的组大小）")
    ap.add_argument("--batch", type=int, default=2,
                    help="grpo × backward 每步用几条 prompt（与 grpo.py 的默认一致）。"
                         "轮次折算成步数要用它：steps/epoch = ceil(行数 / (batch × group))")
    ap.add_argument("--max-new", type=int, default=32, help="每条轨迹最多生成多少 token")
    ap.add_argument("--temperature", type=float, default=1.0, help="采样温度（1.0 才有多样性）")
    ap.add_argument("--resample-every", type=int, default=1,
                    help="每几步重采一次：1=标准 on-policy；>1 用同一批多训几步")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--dpo-data", default="",
                    help="--objective dpo 的偏好对 jsonl（prompt/chosen/rejected；"
                         "--algo mezo 与 --algo dpo 都吃它）")
    ap.add_argument("--dpo-beta", type=float, default=0.1, help="DPO 的 β")
    ap.add_argument("--prefill-chunk", default="",
                    help="mezo：训练引擎的 `--prefill`（prompt 的分块大小）。留空 = 按数据里最长样本取"
                         "（对齐 256，夹在 256..8192），每条样本一块读完、预留只按真实数据大小。"
                         "**不是长度上限**：更长的序列按块数读完；长上下文数据才需要显式抬它")
    ap.add_argument("--think-targets", choices=("both", "on", "off"), default=None,
                    help="零阶 ce 目标形态（见 mezo.py 同名参数）：both = 思考段+答案 / 纯答案各训一遍。"
                         "缺省 both —— 推理会先写思考段，目标只放答案会让两边分布不一致")
    ap.add_argument("--system-prompt", default=None,
                    help="插一条 system 消息（两个方法都透传）；`identity` 展开成身份事实。"
                         "采样/评估用同一份文本，训练侧必须逐字一致，否则 SCORE 的 prompt 段"
                         "与采样对不上（off-policy）")
    ap.add_argument("--reasoning-effort", choices=("xhigh", "medium", "low"), default=None,
                    help="chat_template.jinja 的推理风格档位（mezo 与 backward 的 SFT/DPO 都认）。"
                         "留空 = 模板 default('xhigh')，历史口径逐字不变；采样侧用了 low，"
                         "训练就必须同为 low。--method backward 的 grpo/luspo 不接受该参数")
    ap.add_argument("--list", action="store_true", help="打印算法/目标函数/数据源")
    ap.add_argument("--route", default="off", choices=algos.route_names(),
                    help="--algo auto 的旧写法（保留兼容，等价）：off=直接训一次；"
                         "auto=按组内通过率把 --rollouts / --pool 分档（有答案→luspo / "
                         "没答案→冷启 DPO / 全对毕业），写出 route_plan.json，交给 "
                         "bench/run_plan.py 串行执行")
    ap.add_argument("--plan-dir", default="",
                    help="--route auto 的产物目录（默认 <rollouts>.route/ 或 <pool>.route/）")
    ap.add_argument("--gold-solutions", default="",
                    help="JSONL {prompt, solution}：数学这类 gold 只有答案串时，"
                         "冷启需要一个完整解答才能当 DPO 的 chosen（缺了就退成 SFT）")
    ap.add_argument("--rescore", default="",
                    help="'文件:函数'：用**当前**判分器重判历史 reward（对付过期标签）")
    ap.add_argument("--dry-run", action="store_true", help="只打印最终命令，不加载模型")
    a, rest = ap.parse_known_args()
    a.rest = rest
    legacy = [t for t in rest if t.split("=", 1)[0] in ("--backend", "--objective")]
    if legacy:
        raise SystemExit(
            f"旧参数名 {legacy[0].split('=')[0]}：--objective 已并入 --algo"
            "（sft/dpo/grpo 本身就是目标函数），--backend 改名 --method（mezo|backward）。\n"
            "  用 --list 看新接口。")
    # 路由先分流：`--algo auto`（以及旧写法 `--route auto`）是一轮调度而不是一次训练，
    # 必须在 `method_of` 之前走掉 —— 否则会被一个对它没有意义的 --method 卡住。
    if a.algo == algos.AUTO or a.route != "off":
        return route_auto(a)
    # 方法在这里落定（默认 mezo）；早失败：该方法做不了这个算法，就别去读 60 GB 的 pack。
    m = algos.method_of(a.algo, a.method)
    a.method = m.name
    if a.method != "mezo" and getattr(a, "prefill_chunk", ""):
        raise SystemExit("--prefill-chunk 是 mezo 的引擎参数（--prefill）；backward 走 torch，"
                         "没有 prompt buffer 这件事，别给它一个不起作用的旋钮")
    # 这个方法要的第三方包不齐，也在这里停：零阶线只查 core（全在 requirements.txt
    # 里），backward 才要 requirements-train.txt 那一组。缺包时给的是可复制的
    # pip 命令，而不是 loader_qwen4 深处的 ImportError（Windows 的 triton 尤其）。
    platform_check.require_deps(a.method)
    # 名字在这里落定，后面的打印与转给算法脚本的 --word 都用它。
    a.word = whoami.resolve(a.word)
    resolve_budget(a)
    a.lr = a.lr if a.lr is not None else algos.default_lr(a.algo, a.method)
    a.out = a.out or (f"lora_{a.algo}.npz" if a.method == "mezo" else f"lora_{a.algo}.pt")
    # loader_qwen4 读的是环境变量（它是被 import 的模块，没有命令行面）。
    # 同进程跑，所以在这里设就够，不必要求用户自己 export。
    if a.vram is not None:
        os.environ["STRATA_VRAM_PACK_GB"] = str(a.vram)
    if a.pin is not None:
        os.environ["STRATA_PIN_GB"] = str(a.pin)

    if a.list:
        import datasets

        print(algos.describe())
        print("数据源（--data）：")
        for k, v in datasets.SPECS.items():
            print(f"  {k:<9s} {v}")
        return 0

    if a.method == "mezo":
        shard, epack, exe, prof, origin = resolve_engine_model(a)
        argv = build_argv_engine(a, shard, epack, exe, prof)
        script = m.scripts[a.algo]
    else:
        shards, pack, tok, origin = resolve_model(a)
        argv = build_argv(a, shards, pack, tok)
        script = m.scripts[a.algo]
        shard = epack = exe = prof = ""

    objective = algos.ALGOS[a.algo].objective
    if objective == "dpo":
        kind, arg = "dpo", (a.dpo_data or "")
    elif objective == "grpo":
        kind, arg = ("rollouts" if a.rollouts else "prompts"), (a.rollouts or a.prompts)
    else:
        kind, arg = _resolve_data(a.data, a.word)

    print("=" * 78, flush=True)
    print(f"algo      {a.algo}（目标 {objective}）  method {a.method}  script {script}")
    print(f"model     {origin}")
    if shard:
        print(f"          {shard}")
        print(f"engine    {exe}")
        print(f"pack      {epack}（引擎的 dense/索引 pack，不是 NF4 专家 pack）")
        print(f"profile   {prof}")
        print(f"context   {a.max_context or 4096}")
    else:
        for s in shards:
            print(f"          {s}")
        print(f"pack      {pack}   vram {os.environ.get('STRATA_VRAM_PACK_GB', '16')} GiB"
              f"  pin {os.environ.get('STRATA_PIN_GB', '30')} GiB")
        print(f"tokenizer {tok}")
    budget = (f"steps {a.steps}" if a.steps is not None
              else f"epochs {a.epochs:g} × rows")
    print(f"data      {kind}:{arg}   {budget}  lr {a.lr:g}  seed {a.seed}")
    print(f"out       {a.out}")
    print(f"cmd       python3 {script} " + " ".join(argv), flush=True)
    print("=" * 78, flush=True)
    if a.dry_run:
        return 0
    return algos.run(a.algo, argv, a.method)


if __name__ == "__main__":
    sys.exit(main())
