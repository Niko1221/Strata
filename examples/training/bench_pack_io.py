"""pack 随机读天花板：4 线程 preadv 扫全 pack（随机专家顺序），分解页缓存/冷盘贡献。

两遍：第一遍冷（尽量逼空缓存无谓——直接看实际值），第二遍热（刚读过，全页缓存命中）。
"""

import os
import threading
import time

from preadv_shim import preadv

PACK = "/root/experts_nf4.bin"
BLOCK = 2_614_464          # 每专家四段连续字节（gu|ga|dn|dam）
fd = os.open(PACK, os.O_RDONLY)
size = os.fstat(fd).st_size
n = size // BLOCK
print(f"pack {size / 2**30:.1f} GiB, {n} experts", flush=True)

buf = bytearray(BLOCK)
ORDER = __import__("random").Random(42).sample(range(n), n)

def sweep(ids, label):
    t0 = time.perf_counter()
    chunk = len(ids) // 4

    def work(part):
        for e in part:
            preadv(fd, buf, e * BLOCK)

    ths = [threading.Thread(target=work, args=(ids[i * chunk:(i + 1) * chunk],)) for i in range(4)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    dt = time.perf_counter() - t0
    gb = len(ids) * BLOCK / 2**30
    print(f"{label}: {gb:.1f} GiB in {dt:.1f}s = {gb / dt:.2f} GiB/s", flush=True)

sweep(ORDER, "pass1 (随机全扫)")
sweep(ORDER, "pass2 (同序重扫，页缓存热)")
