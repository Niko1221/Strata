# PR #4347 独立验证：预设范围

2026-10-28，北京时间；在新GPU结果产生前写定。研究对象为已有实现，作者rynwang；本机不改其规则，不提交重复补丁。

问题：RTX5060 Ti16GB、i7-12700、128GBDDR4、Windows原生路径上，packed Q8副本占用的显存是否更适合用于专家缓存？不把作者RTX4080 SUPER/WSL2结果当作本机预测。

- 基线：`5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`；候选：`7be0a3a17aeea0a7c0bf7bc63ae56c70e53311d4`，其父提交恰为基线。
- 模型：已有Qwen3.6-35B-A3B UD-Q4_K_XL，22,853,663,008 bytes，SHA-256 `55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1`，本轮重新计算。
- 同一MSVC19.42、CUDA13.2、sm_120、Ninja Release；ggml固定`3cf03257f219afbe7334045ff7c6a06ac68c627d`。两个独立worktree/构建目录；不覆盖历史构建、报告或日常配置。

## 检查顺序

1. 两边构建strata-q36、strata_kq和16个选定host/布局/采样测试；CTest仅运行这16项，不声称全量套件通过。模型加载前检查构建记录和exe哈希。
2. 当前main运行历史同类四臂正确性：old kernels参考、VERIFY、当前默认Graph请求、MTP2。每组10个parity prompt、128token上限，固定8K/512chunk/64slots；严格检查prompt数量、完整token和VERIFY计数。hybrid实际关闭Graph，组名不代表Graph已覆盖。本门禁验证当前main，固定slots不用于候选政策性能比较。
3. 自动布局下运行main、候选KEEP_PACKED=1、候选默认三臂，均32K配置上限、prefill2048、7workers、MTP0；沿用公开合成ledger的1K/16K实际输入，各输出64token。先暖机64token，核对fresh、输入、输出计数和MTPoff。候选必须记录given-back日志和实际专家槽位。main对candidate-keep须完全一致；候选默认如有token差异先停止并做数值归因，不直接报告等价加速。
4. 上述通过后，仅用同一个候选exe的KEEP_PACKED=1/未设置做政策A/B，至少5组，每组交替AB/BA，两种上下文交替顺序，各256token。每臂新进程、同一预热；不挂profiler。每组同时检查完整输出一致；失败保留结果并停止后续配对。

## 控制与边界

自动布局不传`--expert-slots`或`--expert-vram`，以免绕过候选判定。KV保持当前FP32；不混入#4335，不换chunk，不开新采样或MTP策略。布局变化是政策作用的一部分，不解释成同布局下的单kernel提速。动态专家策略保持当前实现，保存实际布局和replacement日志；每次重启统一初态与预热。

编译和正确性阶段保留DeepSeek后台下载。正式A/B前暂停本机下载器进程（记录PID及创建时间，避免误操作复用PID），等待5秒；测量结束或失败均恢复下载。模型实验通过本地接口卸载日常Flash-Next，结束/失败后恢复原先已加载的服务；不改日常配置。独立HtoD控制在测量前后运行原验证exe，低于15GB/s时停止性能结论；该阈值仅用于排除已知约1.7GB/s异常，不是新硬件峰值标准。

每请求记录协议TTFT、prompt/decode耗时、实际token与哈希、whole-card VRAM和进程RAM/CPU资源样本、worker线程亲和性、日志及配置。CPU累计时间不是关键路径归因，资源采样峰值不是连续峰值。当前实验没有Systems时间线或Compute访存计数，不能完成最终瓶颈归因。

结果保留全部配对、负向项和范围，稳定超过配对噪声才讨论继续投入。3%不是统一合并门槛。CUDA构建和本机执行不替代HIP/SYCL构建或硬件验证；本轮不请求修改原生代码的上游审阅。
