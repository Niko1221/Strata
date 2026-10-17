#!/usr/bin/env python3
# Матрица P2P-способностей по NVML: без CUDA-контекстов, без аллокаций - безопасно при
# занятой VRAM. Именно это спрашивает движок (cudaDeviceCanAccessPeer) на уровне драйвера.
import ctypes, itertools, sys

nv = ctypes.CDLL("libnvidia-ml.so.1")
CAPS = {"NVLINK": 0, "PCI": 1, "P2P_NOT_SUPPORTED": 2}
STATUS = {0: "OK", 1: "CHIPSET_NOT_SUPPORTED", 2: "GPU_NOT_SUPPORTED", 3: "TOPOLOGY_NOT_SUPPORTED",
          4: "NOT_SUPPORTED", 5: "DISABLED_BY_REGKEY", 6: "UNKNOWN"}
nv.nvmlInit_v2.restype = ctypes.c_int
nv.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
nv.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
nv.nvmlDeviceGetP2PStatus.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
rc = nv.nvmlInit_v2()
if rc: sys.exit("nvmlInit -> %d" % rc)
n = ctypes.c_uint()
nv.nvmlDeviceGetCount_v2(ctypes.byref(n))
h = []
for i in range(n.value):
    hd = ctypes.c_void_p()
    nv.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(hd))
    h.append(hd)
print(f"GPU: {n.value}")
for a, b in itertools.combinations(range(n.value), 2):
    out = {}
    for name, idx in CAPS.items():
        st = ctypes.c_int()
        r = nv.nvmlDeviceGetP2PStatus(h[a], h[b], idx, ctypes.byref(st))
        out[name] = STATUS.get(st.value, st.value) if r == 0 else f"err{r}"
    print(f"  GPU{a} <-> GPU{b}: " + "  ".join(f"{k}={v}" for k, v in out.items()))
nv.nvmlShutdown()
