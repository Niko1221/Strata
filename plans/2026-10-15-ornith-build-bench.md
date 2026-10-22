# Ornith-1.5-35B-A3B-IQ4_XS 编译与测速计划

## 硬件核实结果（2026-10-15）

**这台机器上没有 NVIDIA 显卡。**

| 来源 | 结果 |
| --- | --- |
| Windows 显示适配器 | AMD FirePro W2100 (VEN_1002, DEV_6608)、GameViewer Virtual Display、MS Remote Display。**无 NVIDIA** |
| Windows nvcuda.dll | 不存在 |
| Windows nvidia-smi | 不存在（`C:\Program Files\NVIDIA Corporation` 下只有 Installer2） |
| WSL /dev/nvidia* | 不存在 |
| WSL 驱动库 | 有用户态 .so（libnvidia-ml.so.580.178.04），但**没有内核驱动**，`/proc/driver/nvidia/version` 不存在 |
| WSL nvidia-smi | not found |

WSL 里有 CUDA 用户态库残留，但 GPU 直通没生效——这是典型的"WSL 里装过 CUDA toolkit，驱动没装上"状态。

CPU 实测：**Intel Xeon E5-1650 v2 @ 3.50GHz，6 核 12 线程，31 GiB 内存**（不是计划里写的 E5-2650V2 / 32G）。

用户描述的硬件（E5-2650V2、32G DDR3、P100-12G 主卡、P104-8G ×2 helper）与本机实测**不符**。

## 因此

1. **无法在本机测 token 速度。** 没有 NVIDIA 卡，`strata-q36` 只编译 CUDA/HIP。AMD FirePro W2100 是旧 GCN 卡，驱动状态 Error，不支持 CUDA。
2. **可以编译**（编译不需要卡，nvcc 在 WSL 里，CUDA 12.0），但编译产物在本机跑不了。
3. 上一版盘点说"sm_75 以下服务器路径"不搬——现在看，P100 是 sm_60，比那还老，是完全不同的场景。

## 待用户确认

目标硬件是哪台？是另一台机器（需要 SSH 过去），还是描述有误？
