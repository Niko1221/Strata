"""这次微调要往模型里写的「开发者名字」——**运行时自动取，不写死**。

为什么单独一个文件：示例数据里写任何具体名字，在别人 fork 这个项目、或者在你换
一台机器之后，都会变成错的。而身份样本的价值恰恰在于「它是**你**训练的」，名字写
错了整页演示就废了。

取值顺序（先到先得，每一级都可以被上一级显式覆盖）：

  1. `--word X`        命令行显式给，最优先
  2. `$STRATA_IDENTITY_WORD`  环境变量，适合 CI / 容器
  3. GitHub 用户名      `gh api user --jq .login`；没有 `gh` 就用 `$GITHUB_TOKEN`
                        打 `api.github.com/user`
  4. **远端 URL 里的用户名**  `git remote get-url origin` →
     `https://github.com/<login>/<repo>.git` 或 `git@github.com:<login>/<repo>.git`
  5. `git config user.name`   本机仓库身份
  6. 兜底 `xiezhipeng-git`（本 fork 的实际 git 作者；**绝不能是占位串**，它会被训进模型）

第 4 级就是「项目名 → GitHub 用户名」那一步：这份代码归谁，就从 `origin` 的 URL
里把 login 抠出来。它比第 5 级可靠 —— `git config user.name` 是**这台机器**的
身份，跟这个仓库归谁没有关系。

第 3 步**一定会超时退出**（默认 3 秒）：这只是给演示文案取个名字，不值得为它卡住
训练启动。取不到就往下走，不会报错。

用法：
  python3 whoami.py                # 打印解析结果与来源
  python3 whoami.py --quiet        # 只打印名字
"""

from __future__ import annotations

import json
import os
import re
import subprocess

# 兜底：既没有 GitHub、也没有 git 身份时用的名字。
# 必须是**真实维护者**的 git 身份（这里是本 fork 的实际 git 作者）。旧值是个占位串
# 「Strata 训练分支开发者」—— 它会被当成答案训练进模型，等于教模型说一句没意义的话
# （2026-11-01 实测：这份工作树在 WSL 下 `git` 解不开 .git 文件里的 Windows 路径，
# 一路掉到兜底，结果「你是谁」被训成那个占位串，能力也跟着掉）。
FALLBACK = "xiezhipeng-git"

ENV_VAR = "STRATA_IDENTITY_WORD"

# 取 GitHub 用户名的超时。宁可走兜底，也不要让训练启动卡在网络上。
GITHUB_TIMEOUT = 3.0

# jsonl 示例数据里的占位符。读进来的时候替换成解析出的名字。
PLACEHOLDER = "{dev}"


def _run(cmd: list[str], timeout: float) -> str | None:
    """跑一条命令，拿到 stdout；任何失败都返回 None（不抛）。"""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    text = out.stdout.strip()
    return text or None


def github_login(timeout: float = GITHUB_TIMEOUT) -> str | None:
    """GitHub 用户名（login），取不到返回 None。

    两条路：装了 `gh` 且已登录就问它；否则有 token 就直接打 API。
    两条都没有就 None —— 调用方会继续往下走，不把它当错误。
    """
    login = _run(["gh", "api", "user", "--jq", ".login"], timeout)
    if login:
        return login

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        return None
    raw = _run(["curl", "-sS", "--max-time", str(int(timeout)),
                "-H", "Accept: application/vnd.github+json",
                "-H", f"Authorization: Bearer {token}",
                "https://api.github.com/user"], timeout + 1)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    login = data.get("login") if isinstance(data, dict) else None
    return str(login) if login else None


def git_user_name() -> str | None:
    """本机 git 身份（通常是 GitHub 用户名，但不是保证）。"""
    return _run(["git", "config", "--get", "user.name"], 5.0)


def origin_login() -> str | None:
    """从 `origin` 的 URL 里抠出 GitHub 用户名。

    两种写法都认：
      https://github.com/<login>/<repo>.git
      git@github.com:<login>/<repo>.git

    返回 None 的情况：没有 origin、不是 GitHub、URL 长得不像。这些都不算错误，
    只是这一级取不到，交给下一级。
    """
    url = _run(["git", "remote", "get-url", "origin"], 5.0)
    if not url:
        return None
    m = re.search(r"github\.com[:/]+([^/]+)/", url)
    if not m:
        return None
    login = m.group(1).strip()
    return login or None


def resolve(explicit: str | None = None) -> str:
    """解析出这次要用的名字。`explicit` 非空就原样返回（命令行最高优先）。"""
    if explicit:
        return explicit
    env = (os.environ.get(ENV_VAR) or "").strip()
    if env:
        return env
    return github_login() or origin_login() or git_user_name() or FALLBACK


def source(explicit: str | None = None) -> tuple[str, str]:
    """跟 `resolve()` 同一个取值顺序，但连「从哪来的」一起返回。

    返回 `(名字, 来源)`，来源 ∈ {`--word`, `$STRATA_IDENTITY_WORD`,
    `github`, `git`, `fallback`}。给 `whoami.py` 打印用，也让测试能断言
    「确实是从 GitHub 取的」而不是「碰巧兜底字符串一样」。
    """
    if explicit:
        return explicit, "--word"
    env = (os.environ.get(ENV_VAR) or "").strip()
    if env:
        return env, ENV_VAR
    login = github_login()
    if login:
        return login, "github"
    login = origin_login()
    if login:
        return login, "origin"
    name = git_user_name()
    if name:
        return name, "git"
    return FALLBACK, "fallback"


def apply(text: str, word: str) -> str:
    """把示例数据里的 `{dev}` 占位符替换成实际名字。"""
    return text.replace(PLACEHOLDER, word)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="解析这次微调要用的开发者名字。")
    ap.add_argument("--word", default=None, help="显式指定，等价于 sft.py 的 --word")
    ap.add_argument("--quiet", action="store_true", help="只打印名字")
    a = ap.parse_args()

    word, src = source(a.word)
    if a.quiet:
        print(word)
    else:
        print(f"identity: {word}")
        print(f"source:   {src}")
        print()
        print("取值顺序：--word → $STRATA_IDENTITY_WORD → GitHub 用户名 → "
              "origin URL 里的 login → git config user.name → 兜底")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
