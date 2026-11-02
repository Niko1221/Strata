"""os.preadv 的可移植垫层：有就用系统的，没有（Windows）就用 readinto 顶上。

为什么单独一个文件：`platform_check.py` 是 CI no-torch 档的合并门，不能 import
torch；而 `loader_qwen4.py` 顶部就拉 torch。垫层只依赖标准库，两边都能 import，
"Windows 有没有可用的 preadv"因此成了一个可以单独核对的事实。

语义与 os.preadv(fd, [buf], offset) 对齐：
  * 从 fd 的 offset 读（**不移动**调用方对同一 fd 的其它使用方式之外的语义；
    Windows 分支内部用共享 seek 位置，_ensure 的读取是单线程顺序的，无竞争）；
  * 读到 buf 填满或 EOF 为止，返回实际字节数（调用方照旧 assert 完整性）。

Windows 分支的实现：每次调用在**既有 fd** 上包一个临时 FileIO
（`open(fd, "rb", buffering=0, closefd=False)`，关闭文件对象不会关掉调用方的 fd），
seek + readinto 直接读进目标缓冲——零额外拷贝，这是 60 GiB pack 装载路径上唯一
可接受的形态。每次开新句柄而不是缓存共享句柄：原生 preadv 是位置无关的，多线程
共享 fd 也安全；FileIO 的 seek 是共享状态，缓存就会在 bench_pack_io 的 4 线程
随机读里互相覆盖。open 系统调用的开销（每 64 MiB 一次）可忽略。

注意：readinto 的目标是 memoryview(buf).cast("B")，要求 buf 可写且 C 连续。
两个调用点传的都是 1-D uint8 numpy 的连续切片，满足。
"""

from __future__ import annotations

import os

__all__ = ["preadv", "preadv_kind"]

if hasattr(os, "preadv"):
    preadv_kind = "native"

    def preadv(fd: int, buf, offset: int) -> int:
        return os.preadv(fd, [buf], offset)

else:
    preadv_kind = "readinto-shim"

    def preadv(fd: int, buf, offset: int) -> int:
        with open(fd, "rb", buffering=0, closefd=False) as f:
            f.seek(offset)
            mv = memoryview(buf).cast("B")
            total = 0
            while total < len(mv):
                n = f.readinto(mv[total:])
                if not n:
                    break
                total += n
        return total
