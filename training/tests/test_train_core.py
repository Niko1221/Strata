"""`train.py` 门面里**不需要 torch** 的那部分测试 + 平台守卫。

为什么单独一个文件：`test_train.py` 顶层 `import torch`，于是上游 CI 那个只有标准库的
环境连收集都收集不到它 —— 22 个测试一个都没人跑。这一档只用标准库就能跑，所以
CI 的第一道门用它（见 `.github/workflows/training-test.yml`）。

跑法：
  python3 -m unittest discover -s training/tests -p 'test_*.py'
  python3 test_train_core.py -v          # 在 training 下

这里的三件事恰好也是最容易**静默**出错的三件事：
  1. 数据切错 —— `--data identity` 应该落在身份数据上，路径写错要在**加载模型之前**就报；
  2. 算法/目标组合不成立 —— 比如 `--algo sft --objective grpo`；
  3. 并发采集乱序 —— 回来的顺序是乱的，而训练端按行号认组，错了 loss 照样往下走。
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


import json
import os
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

import algos
import collect_rollouts
import datasets
import dpo
import guard
import platform_check
import strata_client
import whoami


class Data(unittest.TestCase):
    def test_identity_is_the_default(self):
        self.assertEqual(datasets.resolve(""), ("identity", ""))
        self.assertEqual(datasets.resolve("identity"), ("identity", ""))
        self.assertEqual(datasets.resolve("identity:Qwen"), ("identity", "Qwen"))

    def test_paths_are_typed_by_extension(self):
        self.assertEqual(datasets.resolve("/root/a.jsonl"), ("jsonl", "/root/a.jsonl"))
        self.assertEqual(datasets.resolve("/root/a.npy"), ("pretrain", "/root/a.npy"))
        self.assertEqual(datasets.resolve("jsonl:/root/a"), ("jsonl", "/root/a"))

    def test_a_typo_fails_before_the_model_loads(self):
        with self.assertRaises(SystemExit):
            datasets.resolve("identiy")
        with self.assertRaises(SystemExit):
            datasets.resolve("/root/a.txt")


class Registry(unittest.TestCase):
    def test_algos_are_sft_dpo_grpo_and_objectives_match(self):
        # 用户口径（2026-11-02 定稿）：算法 = sft/dpo/grpo 三个并列；算法名即目标函数。
        # 2026-11-04 加了 luspo（GRPO 的变体：优势仍走组内归一，只换损失），排在最后。
        self.assertEqual(list(algos.ALGOS), ["sft", "dpo", "grpo", "luspo"])
        self.assertEqual(algos.ALGOS["sft"].objective, "ce")
        self.assertEqual(algos.ALGOS["dpo"].objective, "dpo")
        self.assertEqual(algos.ALGOS["grpo"].objective, "grpo")
        self.assertEqual(algos.ALGOS["luspo"].objective, "luspo")

    def test_every_shipped_method_script_points_at_a_real_file(self):
        # 分支差异只落在文件上（见 algos.available）：带了哪层，就只保证哪层的脚本在。
        self.assertTrue(algos.available())
        for m in algos.available().values():
            for algo, script in m.scripts.items():
                self.assertIn(algo, algos.ALGOS)
                self.assertTrue((Path(__file__).resolve().parents[1] / script).exists(), script)

    def test_a_method_whose_scripts_are_missing_is_reported_not_hidden(self):
        # 零阶线带不带 backward/ 是分支的事，但**不许静默**：absent() 要说得清缺哪个文件，
        # --method 要在这里就退出（不是加载完模型才 ImportError）。
        here = Path(__file__).resolve().parents[1]
        for name, m in algos.absent().items():
            lacking = [s for s in m.scripts.values() if not (here / s).exists()]
            self.assertTrue(lacking, f"{name} 既不在 available() 里，又没有缺文件？")
            with self.assertRaises(SystemExit) as cm:
                algos.method_of("sft", name)
            for s in lacking:
                self.assertIn(s, str(cm.exception))

    def test_default_method_is_mezo_and_routes_to_the_engine(self):
        # 用户口径：MeZO 的路线就是**引擎的推理**当前向；默认方法必须是它。
        # `python3 train.py` 落错成 PyTorch，数字就和推理对不上，还没人看得出来。
        self.assertEqual(algos.DEFAULT_METHOD, "mezo")
        for algo in algos.ALGOS:
            self.assertEqual(algos.method_of(algo, None).scripts[algo], "mezo/mezo.py")

    def test_backward_routes_are_declared_per_algo(self):
        bw = algos.METHODS["backward"]
        self.assertEqual(bw.scripts["sft"], "backward/sft.py")
        self.assertEqual(bw.scripts["dpo"], "backward/sft.py")
        self.assertEqual(bw.scripts["grpo"], "backward/grpo.py")
        self.assertEqual(bw.scripts["luspo"], "backward/grpo.py")

    def test_lr_table_has_only_two_values(self):
        """2026-11-05 定稿：整张表只留两档，两个方法共用。

        旧断言是「同 lr 在两个前向上步长不同，不能共用」——那条理由仍然成立（零阶每步
        位移 ∝ lr·|pj|，反传被 Adam 归一化成 lr），但实测推不出稳定比例（sft 两侧都落
        1e-4、dpo 上 mezo 拉到 1e-4 仍无响应），按方法分档没有依据，于是统一成两档。
        真正逐次盯的是 rel（mezo.py 每步打印），不是表里的数。
        """
        want = {"sft": 1e-4, "dpo": 1e-4, "grpo": 5e-5, "luspo": 5e-5}
        for name in ("mezo", "backward"):
            self.assertEqual(algos.METHODS[name].lr, want, name)
        self.assertEqual(set(want.values()), {1e-4, 5e-5})

    def test_epochs_are_declared_or_left_to_steps(self):
        # 空 = 这个算法没扫过轮次，train.py 走 --steps；不许瞎填一个数装作扫过。
        self.assertEqual(algos.METHODS["mezo"].epochs,
                         {"sft": 5, "dpo": 7, "grpo": 5, "luspo": 3})
        self.assertEqual(algos.METHODS["backward"].epochs, {"sft": 2})

    def test_method_of_fails_loudly(self):
        with self.assertRaises(SystemExit):
            algos.method_of("sft", "sgd")            # 不认识的方法
        with self.assertRaises(SystemExit):
            algos.method_of("ppo", None)             # 不认识的算法

    def test_describe_lists_both_axes(self):
        text = algos.describe()
        for word in ("sft", "mezo", "grpo", "dpo", "ce"):
            self.assertIn(word, text)

    def test_auto_is_an_algo_name_but_not_a_fourth_objective(self):
        """`--algo auto` 能选，但它不进 ALGOS —— 两件事都要成立。

        2026-11-05 改的口径：路由是用户能选的**算法名**（否则没人找得到入口），
        实现上仍是一轮调度，所以：
          * 在 names() 里 ⇒ argparse 的 choices 挡得住拼错，--list 列得出来；
          * 不在 ALGOS 里 ⇒ lr/轮次表不会多一行"自动"（那行填什么都错：它按组派发）。
        """
        self.assertIn(algos.AUTO, algos.names())
        self.assertNotIn(algos.AUTO, algos.ALGOS)
        self.assertEqual(set(algos.route_names()), {"off", "auto"})
        # 它没有单一 objective，所以拿它取方法/lr 必须**报错**，不能静默回落 sft。
        with self.assertRaises(SystemExit):
            algos.method_of(algos.AUTO, None)
        with self.assertRaises(SystemExit):
            algos.default_lr(algos.AUTO, None)
        text = algos.describe()
        self.assertIn(algos.AUTO, text)
        self.assertIn("有答案 → luspo", text)      # 两档默认动作要写在 --list 里

    def test_route_auto_refuses_without_rollouts(self):
        # 没有采样就没有通过率，分不了档 —— 必须在读模型之前就停，别去读 60 GB pack。
        import train

        class _A:
            rollouts = ""
            group = 4
            plan_dir = ""

        with self.assertRaises(SystemExit):
            train.route_auto(_A())


class ParallelCollection(unittest.TestCase):
    """`collect_rollouts.py --parallel N`：并发改变的是完成顺序，不是写出顺序。

    这是唯一值得测的地方：N 条轨迹同时飞出去，回来的顺序是乱的，而训练端按行号
    认组 —— 顺序错了，loss 照样往下走，只是训的东西不对了。
    """

    def run_collect(self, parallel, tmpdir):
        prompts = Path(tmpdir) / "prompts.jsonl"
        prompts.write_text(
            json.dumps({"prompt": "a", "target": "w", "kind": "exact"}) + "\n"
            + json.dumps({"prompt": "b", "target": "w", "kind": "exact"}) + "\n",
            encoding="utf-8")
        out = str(Path(tmpdir) / f"r{parallel}.jsonl")

        def fake_chat(engine, messages, model=None, max_tokens=0, temperature=1.0,
                      enable_thinking=False):
            p = messages[0]["content"]
            time.sleep(0.15 if p == "a" else 0.02)      # 先发的更慢 ⇒ 完成顺序 ≠ 提交顺序
            return {"content": "w", "reasoning_content": None, "token_ids": [7, 8, 9],
                    "token_logprobs": [0.0, 0.0, 0.0], "tokens_exact": True,
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3}}

        argv = ["collect_rollouts.py", "--engine", "http://127.0.0.1:9",
                "--engine-model", "m", "--prompts", str(prompts), "--group", "3",
                "--max-new", "3", "--parallel", str(parallel), "--out", out]
        with unittest.mock.patch.object(sys, "argv", argv), \
             unittest.mock.patch.object(strata_client, "chat", fake_chat):
            collect_rollouts.main()
        with open(out, encoding="utf-8") as f:
            return [json.loads(l)["prompt"] for l in f]

    def test_order_is_the_job_order_not_the_completion_order(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_collect(parallel=3, tmpdir=d),
                             ["a", "a", "a", "b", "b", "b"])

    def test_serial_and_parallel_agree(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_collect(parallel=1, tmpdir=d),
                             self.run_collect(parallel=6, tmpdir=d))

    def test_concurrency_actually_overlaps(self):
        # 串行 3×0.15 + 3×0.02 ≈ 0.51s；并发 3 路应当贴着最慢的那条（≈0.15s）
        with tempfile.TemporaryDirectory() as d:
            t0 = time.time()
            self.run_collect(parallel=3, tmpdir=d)
            self.assertLess(time.time() - t0, 0.40)


# ---------------------------------------------------------------- 平台守卫

def _caps(platform: str, preadv: bool, madvise: bool, torch_ver: str | None, cuda: bool):
    return {"platform": platform, "has_preadv": preadv, "has_madvise": madvise,
            "torch": torch_ver, "cuda": cuda, "torch_error": None}


LINUX_OK = _caps("linux", True, True, "2.12.1", True)
WIN32_OK = _caps("win32", True, False, "2.12.1", True)     # shim 可用 → 放行
WIN32_NO_SHIM = _caps("win32", False, False, "2.12.1", True)  # 垫层丢了 → 阻断


def _tmpfile(name: str) -> str:
    """测试专用临时文件路径（调用方自己负责写内容与清理）。"""
    d = tempfile.mkdtemp(prefix="strata-test-")
    return os.path.join(d, name)


class Platform(unittest.TestCase):
    """平台守卫。

    这些用例自己不带平台假设 —— 全部喂假的 capabilities 给 `blockers()`，所以
    Linux 和 Windows 上都会跑，而且跑出来的结论一样。守卫本身对不对跟你在哪跑无关。
    """

    def test_linux_is_clean(self):
        self.assertEqual(platform_check.blockers(LINUX_OK), [])

    def test_windows_with_shim_is_clean(self):
        # Windows 走 preadv_shim 的 readinto 垫层后放行；madvise 缺失只是降级项
        self.assertEqual(platform_check.blockers(WIN32_OK), [])

    def test_windows_without_shim_is_blocked_and_says_why(self):
        # 垫层文件丢了/改名了要在入口亮出来，而不是在 loader 深处抛 ImportError
        bad = platform_check.blockers(WIN32_NO_SHIM)
        self.assertEqual(len(bad), 1)
        self.assertIn("preadv", bad[0])

    def test_the_preadv_block_lists_every_caller(self):
        # 谁在用 preadv 是会被改动的（加一处、删一处），这条保证名单跟代码对得上
        bad = platform_check.blockers(WIN32_NO_SHIM)
        for caller in ("loader_qwen4.py", "bench_pack_io.py"):
            self.assertIn(caller, bad[0])

    def test_probe_reports_preadv_kind(self):
        # probe 的取值来自真实环境：本机要么原生要么垫层，不会是 none
        # （none 只在 preadv_shim.py 丢失时出现，那本身就该被 blockers 拦住）
        caps = platform_check.probe(with_torch=False)
        self.assertIn(caps["preadv_kind"], ("native", "readinto-shim"))
        self.assertTrue(caps["has_preadv"])

    def test_shim_matches_os_preadv_on_a_real_file(self):
        # 功能等价：同一文件、同一偏移，shim 读出的字节与 os.preadv（若有）一致；
        # 短读（EOF 处）返回实际字节数而不是抛错。Windows 上走 readinto 分支。
        import preadv_shim
        data = bytes(range(256)) * 64          # 16 KiB
        path = _tmpfile("preadv_probe.bin")
        with open(path, "wb") as f:
            f.write(data)
        fd = os.open(path, os.O_RDONLY)
        try:
            ref = data[1000:1000 + 4096]
            buf = bytearray(4096)
            got = preadv_shim.preadv(fd, buf, 1000)
            self.assertEqual(got, 4096)
            self.assertEqual(bytes(buf), ref)
            # 跨越 EOF 的短读
            buf2 = bytearray(4096)
            got2 = preadv_shim.preadv(fd, buf2, len(data) - 10)
            self.assertEqual(got2, 10)
            self.assertEqual(bytes(buf2[:10]), data[-10:])
            # 多线程共享同一 fd 不串位（bench_pack_io 的 4 线程随机读形态）
            import threading
            results, errors = {}, []
            def work(i):
                try:
                    b = bytearray(1024)
                    n = preadv_shim.preadv(fd, b, i * 1024)
                    results[i] = (n, bytes(b))
                except Exception as exc:      # pragma: no cover
                    errors.append(exc)
            ths = [threading.Thread(target=work, args=(i,)) for i in range(8)]
            [t.start() for t in ths]; [t.join() for t in ths]
            self.assertEqual(errors, [])
            for i, (n, b) in results.items():
                self.assertEqual(n, 1024)
                self.assertEqual(b, data[i * 1024:(i + 1) * 1024])
        finally:
            os.close(fd)

    def test_cygwin_and_msys_are_windows(self):
        # sys.platform 在 Cygwin/MSYS2 上是 'cygwin'/'msys'，不是 'win32'。
        # 它们同样没有 preadv，不能因为字符串不含 "win" 就放行。
        for p in ("cygwin", "msys"):
            self.assertTrue(platform_check.blockers(_caps(p, False, False, "2.12.1", True)))

    def test_missing_torch_is_its_own_block(self):
        bad = platform_check.blockers(_caps("linux", True, True, None, False))
        self.assertTrue(any("torch" in b for b in bad))

    def test_torch_without_cuda_is_its_own_block(self):
        bad = platform_check.blockers(_caps("linux", True, True, "2.12.1", False))
        self.assertTrue(any("torch.cuda.is_available()" in b for b in bad))

    def test_madvise_missing_is_a_degrade_not_a_block(self):
        # 关键区分：madvise 缺了只是还不掉映射页，训练本身照跑。
        # 别把它升成阻断项，否则 Windows 一修 preadv 就"突然支持"但显存爆得莫名其妙。
        caps = _caps("linux", True, False, "2.12.1", True)
        self.assertEqual(platform_check.blockers(caps), [])
        self.assertEqual(len(platform_check.degrades(caps)), 1)

    def test_require_exits_2_when_blocked(self):
        # 2 不是 1：1 在训练脚本里是"训练本身失败"，守卫被拒是"根本没跑"。
        with self.assertRaises(SystemExit) as ctx:
            platform_check.require(WIN32_NO_SHIM)
        self.assertEqual(ctx.exception.code, 2)

    def test_require_returns_caps_when_clean(self):
        self.assertEqual(platform_check.require(LINUX_OK), LINUX_OK)

    def test_probe_matches_the_real_interpreter(self):
        """守卫的判据必须对得上**它自己跑的那个解释器**。

        这条是防「探测用的是一套、真正读 pack 用的是另一套」—— 比如在 Windows 上
        装了 torch 但 pack 装载走的是另一个进程。所以直接跟 `hasattr` 对账。
        """
        caps = platform_check.probe()
        for key in ("platform", "has_preadv", "preadv_kind", "has_madvise", "torch", "cuda"):
            self.assertIn(key, caps)
        self.assertEqual(caps["platform"], sys.platform)
        # has_preadv 的语义是"垫层可用"（原生或 readinto），不再是裸 hasattr(os,'preadv')
        self.assertEqual(caps["has_preadv"], caps["preadv_kind"] != "none")
        self.assertEqual(caps["has_madvise"], hasattr(__import__("mmap").mmap, "madvise"))

    def test_probe_never_raises_when_torch_is_absent(self):
        # 探测本身不能因为缺 torch 就炸 —— 它要在无 torch 环境里当合并门用
        with unittest.mock.patch.dict(sys.modules, {"torch": None}):
            caps = platform_check.probe()
        self.assertIn("torch", caps)


# ---------------------------------------------------------------- 训练依赖矩阵

def _deps(*present: str) -> dict:
    """造一份 deps_probe 的结果：给了名字的算装了，其余算没装。"""
    return {d.module: {"present": d.module in present, "version": "1.0" if d.module in present else None,
                       "layer": d.layer, "note": d.note, "dists": list(d.dists),
                       "install": d.install}
            for d in platform_check.DEPS}


class Deps(unittest.TestCase):
    """第三方包矩阵。

    这里守的是三件会静默出问题的事：
      1. 零阶线偷偷长出 torch 层的依赖（那会让默认路径变成"还得再装一套"）；
      2. core 层要的包不在 requirements.txt 里（装完引擎还是跑不起来）；
      3. 缺包时报出来的名字 pip 不认识（提示没法复制粘贴）。
    """

    def test_every_dep_declares_a_layer_and_a_reason(self):
        for d in platform_check.DEPS:
            self.assertIn(d.layer, (platform_check.CORE, platform_check.BACKWARD), d.module)
            self.assertTrue(d.note.strip(), d.module)
            self.assertTrue(d.dists and all(d.dists), d.module)

    def test_core_layer_is_covered_by_requirements_txt(self):
        """mezo 的全部第三方包都必须在 requirements.txt 里。

        这是"零阶线装完引擎就能跑"的可执行版本：核心层一旦多出一个包，这条就红，
        而不是等人跑到一半才发现要 pip install。
        """
        req = Path(__file__).resolve().parents[2] / "requirements.txt"
        text = req.read_text(encoding="utf-8")
        have = set()
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            have.add(line.split(";")[0].split("==")[0].split("[")[0].strip().lower().replace("_", "-"))
        for d in platform_check.DEPS:
            if d.layer != platform_check.CORE:
                continue
            for name in d.dists:
                self.assertIn(name.lower().replace("_", "-"), have,
                              f"core 依赖 {d.module}（pip: {name}）不在 requirements.txt 里")

    def test_probe_uses_injected_find_and_version(self):
        seen, vers = [], []

        def find(name):
            seen.append(name)
            return object() if name == "triton" else None

        def version_of(name):
            vers.append(name)
            raise ModuleNotFoundError(name)

        got = platform_check.deps_probe(find=find, version_of=version_of)
        self.assertEqual(len(seen), len(platform_check.DEPS))
        self.assertTrue(got["triton"]["present"])
        self.assertFalse(got["transformers"]["present"])
        self.assertEqual(got["triton"]["layer"], platform_check.BACKWARD)
        # 缺了就不必去问版本；装了但 dist 名对不上时版本留 None，不当成"没装"
        self.assertNotIn("transformers", vers)
        self.assertIsNone(got["triton"]["version"])

    def test_gaps_only_cover_the_requested_layers(self):
        found = _deps("numpy", "jinja2", "regex", "psutil")
        self.assertEqual(platform_check.deps_gaps(found, (platform_check.CORE,)), [])
        bad = platform_check.deps_gaps(found, (platform_check.CORE, platform_check.BACKWARD))
        self.assertTrue(bad)
        self.assertTrue(all("backward" in b for b in bad))

    def test_gap_names_a_pip_installable_dist(self):
        found = _deps()
        bad = platform_check.deps_gaps(found, (platform_check.BACKWARD,))
        triton = [b for b in bad if b.startswith("[backward] triton")]
        self.assertEqual(len(triton), 1)
        self.assertIn("triton-windows", triton[0])

    def test_mezo_does_not_require_the_backward_set(self):
        """默认方法只有 core 层。缺 transformers 不该拦住零阶训练 —— 那条线上根本没 torch。"""
        found = _deps("numpy", "jinja2", "regex", "psutil")
        self.assertIs(platform_check.require_deps("mezo", found), found)
        with self.assertRaises(SystemExit) as ctx:
            platform_check.require_deps("backward", found)
        self.assertEqual(ctx.exception.code, 2)

    def test_clean_backward_set_passes(self):
        found = _deps("numpy", "jinja2", "regex", "psutil", "transformers", "tokenizers",
                      "accelerate", "safetensors", "gguf", "fla", "triton")
        self.assertIs(platform_check.require_deps("backward", found), found)

    def test_install_command_is_platform_specific(self):
        linux = "\n".join(platform_check.install_command("linux"))
        win = "\n".join(platform_check.install_command("win32"))
        self.assertIn("requirements-train.txt", linux)
        self.assertIn("requirements-train.txt", win)
        self.assertNotIn("triton-windows", linux)          # Linux 由 torch 自己带
        self.assertIn("triton-windows", win)
        self.assertIn("download.pytorch.org/whl", win)     # torch 必须从 CUDA 索引装


# ---------------------------------------------------------------- 身份解析

class Whoami(unittest.TestCase):
    """`whoami.py`：名字运行时自动取，不写死。

    这些用例不碰网络 —— 网络那条路要么走 `gh`（这台机器没装），要么走 token，
    两者在 CI 里都不可靠。测的是**取值顺序**本身，那才是会发生静默错误的地方：
    命令行和环境变量谁赢，取不到时会不会返回空串。
    """

    def test_explicit_word_wins(self):
        self.assertEqual(whoami.resolve("alice"), "alice")
        self.assertEqual(whoami.source("alice"), ("alice", "--word"))

    def test_env_beats_github_and_git(self):
        with unittest.mock.patch.dict(os.environ, {whoami.ENV_VAR: "  bob  "}):
            self.assertEqual(whoami.resolve(), "bob")       # 顺带验证会 strip
            self.assertEqual(whoami.source()[1], whoami.ENV_VAR)

    def test_resolve_never_returns_empty(self):
        # 空字符串会让探针去测一个空词的 logprob，而那个数字看起来完全正常。
        # 三条探测路（GitHub / origin URL / git 身份）**全部**要 mock 掉：
        # 少 mock 一条，这条用例在 GitHub 的 runner 上就会"碰巧"从 origin
        # URL 里解析出真用户名而通过 —— 那不是密封，是漏（真出过这个红）。
        with unittest.mock.patch.dict(os.environ, {}, clear=True), \
             unittest.mock.patch.object(whoami, "github_login", lambda *a: None), \
             unittest.mock.patch.object(whoami, "origin_login", lambda *a: None), \
             unittest.mock.patch.object(whoami, "git_user_name", lambda *a: None):
            self.assertEqual(whoami.resolve(), whoami.FALLBACK)
            self.assertEqual(whoami.source(), (whoami.FALLBACK, "fallback"))
        self.assertTrue(whoami.FALLBACK.strip())

    def test_origin_login_parses_both_url_shapes(self):
        # 第 4 级：从 origin URL 抠 login。两种 GitHub 写法都要认。
        for url, want in (("https://github.com/alice/Strata.git", "alice"),
                          ("git@github.com:bob/Strata.git", "bob")):
            with unittest.mock.patch.object(
                    whoami, "_run", lambda cmd, t, _u=url: _u):
                self.assertEqual(whoami.origin_login(), want)
        for url in ("https://gitlab.com/alice/Strata.git", None):
            with unittest.mock.patch.object(
                    whoami, "_run", lambda cmd, t, _u=url: _u):
                self.assertIsNone(whoami.origin_login())

    def test_precedence_word_env_origin_github_git(self):
        # 完整取值顺序：--word > env > GitHub > origin > git 身份。
        # 每个子块先把**三条探测路全部堵成 None**，再单独放行被测那一级 ——
        # 少堵一条，这条用例就会在 GitHub runner 上从真 origin 里"碰巧"解析出
        # 用户名（第一个版本就是这么红的：只 mock 了 git_user_name）。
        # 直接用 ExitStack 写：三条探测路全堵成 None，唯独放行参数给值的那些。
        import contextlib

        def with_probes(github=None, origin=None, git=None):
            stack = contextlib.ExitStack()
            stack.enter_context(unittest.mock.patch.dict(os.environ, {}, clear=True))
            for pname, val in (("github_login", github), ("origin_login", origin),
                               ("git_user_name", git)):
                stack.enter_context(unittest.mock.patch.object(
                    whoami, pname, (lambda *a, _v=val: _v) if val is not None
                    else (lambda *a: None)))
            return stack

        with with_probes(github="gh-x", origin="or-x", git="git-x"):
            self.assertEqual(whoami.source(), ("gh-x", "github"))
        with with_probes(origin="or-x", git="git-x"):
            self.assertEqual(whoami.source(), ("or-x", "origin"))
        with with_probes(git="git-x"):
            self.assertEqual(whoami.source(), ("git-x", "git"))
        with with_probes():
            self.assertEqual(whoami.source(), (whoami.FALLBACK, "fallback"))

    def test_placeholder_substitution(self):
        self.assertEqual(whoami.apply("我由 {dev} 开发和维护。", "carol"),
                         "我由 carol 开发和维护。")

    def test_text_without_placeholder_is_untouched(self):
        self.assertEqual(whoami.apply("一加一等于 2。", "dave"), "一加一等于 2。")

    def test_every_placeholder_in_the_sample_data_gets_filled(self):
        # 占位符拼错（{developer} / {{dev}}）不会报错，只会原样留在样本里 ——
        # 于是模型被教了一个字面上的 "{dev}"。这条把那份文件整个过一遍。
        path = Path(__file__).resolve().parents[1] / "data_developer.jsonl"
        if not path.exists():
            self.skipTest("no data_developer.jsonl")
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertTrue(rows)
        for r in rows:
            filled = whoami.apply(r["answer"], "NAME")
            self.assertNotIn(whoami.PLACEHOLDER, filled)
            self.assertNotIn("{", filled)


class GuardTokenSelection(unittest.TestCase):
    """Rho-1 式 token 选择（guard.select_mask / expand_to_labels）。

    语义定死在这：阈值**严格大于**；选择列表与编码不是同一份就报错而不是带病训。
    """

    def test_strictly_greater_than_threshold(self):
        self.assertEqual(guard.select_mask([0.5, 1.0, 1.5], 1.0), [False, False, True])

    def test_empty_answer_segment(self):
        self.assertEqual(guard.select_mask([], 1.0), [])

    def test_expand_aligns_selection_to_label_positions(self):
        # labels = full_ids[1:]；plen=3 ⇒ 前 2 个位置是 prompt 段（恒 False）
        self.assertEqual(guard.expand_to_labels([True, False, True], 5, 3),
                         [False, False, True, False, True])

    def test_expand_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            guard.expand_to_labels([True], 5, 3)   # 答案段 3 个 token，只给了 1 个


class GuardCapabilityGate(unittest.TestCase):
    """能力门判定：能力越界一票否决；身份更好才 accept；硬顶严格大于。"""

    def test_capability_veto_beats_any_identity_gain(self):
        self.assertEqual(guard.gate_verdict(99.0, 0.6, None, 0.5), "reject")

    def test_generation_sanity_vetoes_even_a_perfect_identity(self):
        # v4 事故：CE 全绿、idΔ +50，生成已坍缩成复读环 —— 生成体检不过一票否决
        self.assertEqual(guard.gate_verdict(50.0, 0.1, None, 0.5, gen_ok=False), "reject")

    def test_first_eval_accepts_when_within_tol(self):
        self.assertEqual(guard.gate_verdict(0.1, 0.1, None, 0.5), "accept")

    def test_hold_until_strictly_better(self):
        self.assertEqual(guard.gate_verdict(1.0, 0.2, 1.0, 0.5), "hold")
        self.assertEqual(guard.gate_verdict(1.1, 0.2, 1.0, 0.5), "accept")

    def test_hard_stop_is_strict(self):
        self.assertFalse(guard.should_hard_stop(1.5, 1.5))
        self.assertTrue(guard.should_hard_stop(1.6, 1.5))


class GuardGenSanity(unittest.TestCase):
    """生成体检：复读环 = distinct-3gram 趋零 + 不出 EOS（v4 的实际退化形状）。"""

    def test_repetition_loop_is_flagged(self):
        loop = [7, 8, 9] * 8
        self.assertLess(guard.distinct_n(loop, 3), 0.5)
        self.assertFalse(guard.gen_probe_ok(loop, eos_hit=False))
        self.assertFalse(guard.gen_probe_ok(loop, eos_hit=True))

    def test_short_output_counts_as_distinct(self):
        self.assertEqual(guard.distinct_n([1, 2], 3), 1.0)

    def test_normal_short_answer_passes(self):
        self.assertTrue(guard.gen_probe_ok([10, 11, 12, 13], eos_hit=True))

    def test_word_leak_is_content_failure_form_check_cannot_see(self):
        # v5-off 实测：'xiezhipeng-git xiezhip…' 形式全过（d3=0.8、有 EOS），内容在插名
        self.assertTrue(guard.gen_probe_ok([7, 8, 9, 10, 11], eos_hit=True))
        self.assertFalse(guard.no_word_leak("xiezhipeng-git xiezhip is a x", "xiezhipeng-git"))
        self.assertTrue(guard.no_word_leak("Soft, purring, warm.", "xiezhipeng-git"))
        self.assertTrue(guard.no_word_leak("Soft, purring, warm.", ""))


class HoldoutProbes(unittest.TestCase):
    """探针文件的两条硬约束：两类齐全；与训练数据**零重叠**（同 prompt = 自评，等于没测）。"""

    def test_probes_parse_and_cover_both_kinds(self):
        path = Path(__file__).resolve().parents[1] / "prompts_holdout.jsonl"
        if not path.exists():
            self.skipTest("no prompts_holdout.jsonl")
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        kinds = {r["kind"] for r in rows}
        self.assertIn("identity", kinds)
        self.assertIn("cap", kinds)
        for r in rows:
            if r["kind"] == "cap":
                self.assertTrue(r.get("target"), r)

    def test_probes_do_not_overlap_training_data(self):
        here = Path(__file__).resolve().parents[1]
        probe = {json.loads(x)["prompt"] for x in
                 (here / "prompts_holdout.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
        train = {json.loads(x)["prompt"] for x in
                 (here / "data_developer.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
        self.assertFalse(probe & train, sorted(probe & train))


class DpoLoss(unittest.TestCase):
    """DPO 纯逻辑：loss 数学 + 数据校验（数据写反 = 系统性学错的那个，必须入口拦）。"""

    def test_zero_margin_costs_ln2(self):
        self.assertAlmostEqual(dpo.dpo_loss(0.0), 0.6931471805599453, places=12)

    def test_loss_decreases_monotonically_with_margin(self):
        vals = [dpo.dpo_loss(m, 0.1) for m in (-5.0, -1.0, 0.0, 1.0, 5.0, 50.0)]
        self.assertEqual(vals, sorted(vals, reverse=True))   # margin 越大 loss 越低
        self.assertLess(vals[-1], 0.01)                      # 大 margin → 0，不 exp 溢出

    def test_beta_scales_margin(self):
        # 同一 margin，β 越大 loss 越低（对 margin 更「自信」）
        self.assertLess(dpo.dpo_loss(1.0, 1.0), dpo.dpo_loss(1.0, 0.1))

    def test_batch_reports_accuracy_not_just_loss(self):
        loss, acc = dpo.dpo_batch([2.0, -0.5, 0.1], 0.1)
        self.assertAlmostEqual(acc, 2 / 3, places=12)     # 两个 >0
        self.assertAlmostEqual(loss, sum(dpo.dpo_loss(m, 0.1) for m in (2.0, -0.5, 0.1)) / 3,
                               places=12)

    def test_batch_rejects_empty(self):
        with self.assertRaises(ValueError):
            dpo.dpo_batch([], 0.1)

    def test_validate_rejects_missing_key_and_empty(self):
        with self.assertRaises(ValueError):
            dpo.validate_pairs([{"prompt": "p", "chosen": "c"}])          # 缺 rejected
        with self.assertRaises(ValueError):
            dpo.validate_pairs([{"prompt": "p", "chosen": "c", "rejected": " "}])
        with self.assertRaises(ValueError):
            dpo.validate_pairs([])

    def test_validate_rejects_chosen_equals_rejected(self):
        with self.assertRaises(ValueError):
            dpo.validate_pairs([{"prompt": "p", "chosen": "same", "rejected": "same"}])

    def test_dpo_registry_rows_point_at_the_right_scripts(self):
        # dpo 的两个方法：mezo（引擎，默认）与 backward（sft.py 的护栏复用）。
        self.assertEqual(algos.METHODS["mezo"].scripts["dpo"], "mezo/mezo.py")
        self.assertEqual(algos.METHODS["backward"].scripts["dpo"], "backward/sft.py")

    def test_margin_definition(self):
        # margin = (π_c − ref_c) − (π_r − ref_r)：ref 变差与 π 变好同向加分。
        self.assertAlmostEqual(dpo.dpo_margin(5.0, 1.0, 2.0, 1.5), 3.5, places=12)
        self.assertAlmostEqual(dpo.dpo_margin(1.0, 3.0, 0.0, 0.0), -2.0, places=12)

    def test_mezo_dpo_wires_dpo_data_through_build_argv_engine(self):
        # 零阶 DPO 的完整链路：train.py --algo dpo（默认方法 mezo）--dpo-data …
        import argparse as _ap

        import train as train_mod

        a = _ap.Namespace(algo="dpo", dpo_data="/tmp/pairs.jsonl", dpo_beta=0.1,
                          model="", steps=5, lr=1e-4, out="o.npz", seed=0, rest=[],
                          data="identity", word="W", rollouts="", prompts="",
                          group=4, max_new=32, temperature=1.0, resample_every=1,
                          clip=0.2, max_context=4096, probe="")
        argv = train_mod.build_argv_engine(a, "/tmp/s.gguf", "/tmp/pack",
                                           "/tmp/exe", "/tmp/prof")
        self.assertEqual(argv[argv.index("--objective") + 1], "dpo")
        self.assertEqual(argv[argv.index("--dpo-data") + 1], "/tmp/pairs.jsonl")
        self.assertEqual(argv[argv.index("--beta") + 1], "0.1")

    def test_mezo_dpo_without_data_fails_before_load(self):
        import argparse as _ap

        import train as train_mod

        a = _ap.Namespace(algo="dpo", dpo_data="", dpo_beta=0.1,
                          model="", steps=5, lr=1e-4, out="o.npz", seed=0, rest=[],
                          data="identity", word="W", rollouts="", prompts="",
                          group=4, max_new=32, temperature=1.0, resample_every=1,
                          clip=0.2, max_context=4096, probe="")
        with self.assertRaises(SystemExit):
            train_mod.build_argv_engine(a, "/tmp/s.gguf", "/tmp/pack",
                                        "/tmp/exe", "/tmp/prof")

    def test_data_dpo_file_is_valid_pairs(self):
        path = Path(__file__).resolve().parents[1] / "data_dpo.jsonl"
        if not path.exists():
            self.skipTest("no data_dpo.jsonl")
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        dpo.validate_pairs(rows)          # 原样过校验
        filled = [({k: whoami.apply(r[k], "NAME") for k in dpo.REQUIRED_KEYS}) for r in rows]
        dpo.validate_pairs(filled)        # 占位符展开后仍要过（chosen/rejected 不因展开撞车）
        n_rej_named = sum(1 for r in rows if whoami.PLACEHOLDER in r["rejected"])
        self.assertGreater(n_rej_named, 0, "至少要有反插名对：rejected 应包含塞名字的坏行为")


class RenderContractForwarding(unittest.TestCase):
    """采样/评估的渲染契约必须一路传到每一行训练命令。

    2026-11-06 缺口：`train.py` 没有 --reasoning-effort，--system-prompt 只发给
    mezo；route_plan.json 的每一步两个参数都缺。训练不报错、只是 prompt 段与采样
    对不上（off-policy），是最难从日志里看出来的那类错。
    """

    def _ns(self, **kw):
        import argparse as _ap

        base = dict(algo="sft", dpo_data="", dpo_beta=0.1, model="", pack="",
                    tokenizer="", engine="", expert_profile="", engine_pack="",
                    engine_api="", gguf=[], steps=5, lr=1e-4, out="o.npz",
                    seed=0, rest=[], data="identity", word="W", rollouts="",
                    prompts="", group=4, max_new=32, temperature=1.0,
                    resample_every=1, clip=0.2, max_context=4096, probe="",
                    seqlen=2048, epochs=None, system_prompt=None,
                    reasoning_effort=None)
        base.update(kw)
        return _ap.Namespace(**base)

    def _flags(self, argv, name):
        self.assertIn(name, argv, argv)
        return argv[argv.index(name) + 1]

    def test_mezo_gets_both_flags(self):
        import train as train_mod

        a = self._ns(system_prompt="SYS TEXT", reasoning_effort="low")
        argv = train_mod.build_argv_engine(a, "/tmp/s.gguf", "/tmp/pack",
                                           "/tmp/exe", "/tmp/prof")
        self.assertEqual(self._flags(argv, "--system-prompt"), "SYS TEXT")
        self.assertEqual(self._flags(argv, "--reasoning-effort"), "low")

    def test_backward_sft_gets_both_flags(self):
        import train as train_mod

        a = self._ns(system_prompt="SYS TEXT", reasoning_effort="low")
        argv = train_mod.build_argv(a, ["/tmp/s.gguf"], "/tmp/pack", "/tmp/tok")
        self.assertEqual(self._flags(argv, "--system-prompt"), "SYS TEXT")
        self.assertEqual(self._flags(argv, "--reasoning-effort"), "low")

    def test_default_render_is_untouched(self):
        """不传契约 ⇒ 两个 argv 里都不许多出参数（历史口径逐字不变）。"""
        import train as train_mod

        a = self._ns()
        for argv in (train_mod.build_argv_engine(a, "/tmp/s.gguf", "/tmp/pack",
                                                 "/tmp/exe", "/tmp/prof"),
                     train_mod.build_argv(a, ["/tmp/s.gguf"], "/tmp/pack", "/tmp/tok")):
            self.assertNotIn("--system-prompt", argv)
            self.assertNotIn("--reasoning-effort", argv)

    def test_backward_grpo_refuses_contract_instead_of_silently_dropping_it(self):
        import train as train_mod

        # backward/grpo.py 用 parse_args()，不认这两个参数：必须在入口拦下来并指路
        a = self._ns(algo="grpo", model="/tmp/hf", system_prompt="SYS",
                     reasoning_effort="low")
        with self.assertRaises(SystemExit) as cm:
            train_mod.build_argv(a, [], "/tmp/pack", "/tmp/tok")
        self.assertIn("--method mezo", str(cm.exception))

    def test_route_plan_steps_carry_the_contract(self):
        import train as train_mod

        a = self._ns(system_prompt="SYS TEXT", reasoning_effort="low", seed=7)
        f = train_mod._forward_flags(a)
        self.assertEqual(self._flags(f, "--system-prompt"), "SYS TEXT")
        self.assertEqual(self._flags(f, "--reasoning-effort"), "low")
        # 由路由决定的四项仍然不许转发
        for bad in ("--algo", "--method", "--data", "--lr", "--epochs"):
            self.assertNotIn(bad, f)


class BackwardArgvAcceptedByScript(unittest.TestCase):
    """backward 的命令行必须被 `backward/sft.py` 真的认。

    2026-11-06 缺口：`build_argv` 给 sft.py 发了 `--objective dpo`，argparse 直接
    `unrecognized arguments` —— 冷启第二步起来 14 秒就死，而前一步（SFT）是好的，
    只看 SFT 日志完全看不出问题。目标由 `--dpo-data` 的有无决定，没有 `--objective`。

    这里不硬编码「该有哪些参数」，而是拿 `sft.py --help` 的真实选项集合作白名单：
    以后 train.py 再加任何 sft.py 不认的参数，这条测试都会红。
    """

    def _ns(self, **kw):
        import argparse as _ap

        base = dict(algo="dpo", dpo_data="/tmp/pairs.jsonl", dpo_beta=0.1, model="",
                    pack="", tokenizer="", engine="", expert_profile="",
                    engine_pack="", engine_api="", gguf=[], steps=5, lr=1e-4,
                    out="o.pt", seed=0, rest=[], data="", word="W", rollouts="",
                    prompts="", group=4, max_new=32, temperature=1.0,
                    resample_every=1, clip=0.2, max_context=4096, probe="",
                    seqlen=2048, epochs=None, system_prompt=None,
                    reasoning_effort=None)
        base.update(kw)
        return _ap.Namespace(**base)

    def test_every_flag_is_known_to_sft_py(self):
        """白名单来自 sft.py 源码里的 `add_argument`（不拉起子进程）。

        为什么不跑 `sft.py --help`：那条路要 import transformers，而本机
        huggingface-hub 2.2.0 与 transformers 的版本门冲突，子进程在 import 阶段就
        失败（父进程能过是环境变量的运气，不能拿来当测试前提）。参数集是纯静态的，
        从源码取即可 —— 要抓的缺口就是「train.py 发了 sft.py 没定义的参数」。
        """
        import re

        import train as train_mod

        sft_py = Path(__file__).resolve().parents[1] / "backward" / "sft.py"
        if not sft_py.is_file():
            self.skipTest("backward/sft.py 不在这条分支上（零阶线只有 mezo/ + 共用 numpy 层）")
        src = sft_py.read_text(encoding="utf-8")
        known = set(re.findall(r'add_argument\(\s*"(--[A-Za-z0-9-]+)"', src))
        self.assertIn("--dpo-data", known)
        self.assertIn("--init-adapter", known)

        a = self._ns()
        argv = train_mod.build_argv(a, ["/tmp/s.gguf"], "/tmp/pack", "/tmp/tok")
        flags = [t for t in argv if t.startswith("--")]
        unknown = [f for f in flags if f not in known]
        self.assertEqual(unknown, [], f"sft.py 不认这些参数：{unknown}")
        # 目标靠 --dpo-data 判定，不许再发 --objective
        self.assertNotIn("--objective", argv)
        self.assertIn("--ckpt", argv)      # DPO 默认开梯度检查点



if __name__ == "__main__":
    unittest.main()
