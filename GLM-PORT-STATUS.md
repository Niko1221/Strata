# GLM-5.3-Flash 移植状态(交接文档,更新于 2026-10-31)

本次收口的完整教训与踩坑记录见 [GLM-PORT-LESSONS.md](GLM-PORT-LESSONS.md),新会话先读它。

## 当前可用
- strata-glm 引擎(2×V100 实测):当前在跑 **1M×2**(1048576×2,2026-10-31 切换并验收双 PASS);历史实测 512K×2、1M×2、INT8 latent、思考档位、官方采样参数
- API 服务 http://192.168.0.200:8080/v1(无 key),一键启动 /data/AI/Strata/start-glm.sh
  (务必带 STRATA_PYTHON=/data/AI/Strata/.venv/bin/python3,原因见运维坑)
- opencode 连接:**PASS prompt 与 PASS tool call 双绿**(2026-10-31 验收)
  `strata connect opencode --check --url http://127.0.0.1:8080`

## 工具调用:已完成(本次会话收口)
三处配合,缺一不可:
1. 停止契约固化在引擎里:glm_main.cpp 的默认 glm_eos 现为
   {154820, 154827, 154829, 154844}（前三个是词表核实过的模板结束符,154844 是调用闭合符 `</tool_call>`）。
   任何路径——一键脚本、手动裸跑引擎、任何配置——都不再需要 --eos-ids;
   旗仍保留为覆盖开关(评测传 2147483647 关闭 eos)。注意 --eos-ids 是**替换**默认集,
   不能只传 154844。setup.py install_glm 仍显式写入该参数,防旧引擎包(engine-glm)配新配置;
   本机两份在盘配置已去掉该参数,以验证二进制默认值。
2. serve/server.py:set_family 的 glm 分支从 special_ids 的 GGUF 键名(eos/eot/eom)取
   停止集——旧的 `.get("eos", 1)` 对 GLM 键名取不到值,返回 1,导致 server 从不认结束符
   (finish 全是 length,结束标记漏进正文)。另加 stop_feed={154844}:闭合符先喂给
   GlmCallRewriter 再停,否则 rewriter 收不到闭合符,调用退回纯文本。
3. GlmCallRewriter._wrap 现在输出 parser 的原生拼写:`<tool_call>` + `<function=NAME>` 块 + `</tool_call>`。
   只有 `<function=...>` 裸块时 parser 把它当 reasoning 文本,永不产出 tool_call 事件。
链路:模型写出 `<tool_call>`…`</tool_call>` → 引擎在 154844 自然停(stop)→ server 把闭合符喂给
rewriter 后停 → parser 在 finish="stop" 时把挂起调用升级为 tool_call(#1058)。
单测级模拟与在线探针均验证;普通回答 finish 也从 length 变回 stop,无结束符泄漏。
低思考档下模型偶尔自发写 `刀` 收尾,parser 按既有设计从 content 起步,标记随正文
出现,无害,未改动。
4. 参数闭合标签家族(opencode shell 工具失败复发第 2 病根,2026-10-31):
   GLM 有四件套参数标签——键两端 154847/154848,值两端 154849/154850。模型偶尔把参数值
   写成 `<arg_value>…</arg_value>` 包裹(用户日志里 102 处泄漏),改写器只剥键对,值对原样漏进
   parameter 值,opencode 把标签文本一起当命令执行而失败。已修:_as_qwen 剥值对
   (取 AVO 前、切首个 AVC),JSON 形式两个标签对都按引号替换。词表级单测 10 变体
   (裸值/包裹值/多参/逐字符流/分块流/两种 JSON 引号形式)全过。
5. n_ctx 上报语义(答复"1M×2 应报 2097152"):meta.n_ctx 保持**每请求**值
   (= 引擎 --max-context = 每会话窗口 1048576);strata connect 用它给 opencode 定
   limit.input(每次请求预算),若改报 2097152,客户端会发 2M token 请求,超单会话
   窗口必失败。总容量改为新增字段:/v1/models meta.parallel=2 与
   meta.n_ctx_total=2097152(仅 --parallel>1 时出现),信息性,不影响客户端定界。6. 上下文默认策略(2026-11-01,用户拍板):**裸跑永远 512K×2**(DEFAULT_CTX=524288),
   1M 需要本次显式 `--context 1m`,且只对本次生效——脚本把配置里的 --max-context 视为
   "上次的选择"而非默认值,裸跑若发现配置≠默认就改写回 524288 并重启。此前 1M 被写进
   配置后,裸跑静默变成 1M×2,默认值跟着磁盘状态漂移(踩坑一次后修正)。
   实测:配置=1M 时裸跑→改写为 524288 并服务(meta 524288/2/1048576);已在默认档时
   再裸跑 0.4s 秒回横幅不重启;connect 双 PASS。--context 1m 的改写机制与前日两次实测相同。
7. 引擎 STAT 实时解码样本接进 /metrics(2026-11-01):引擎每 8 token 在 stdout 发
   STAT tok_s/ms_tok/vram_hit/ram_fetch/disk/promo/池水位(multi 路径多一个 active 尾字段),
   此前无人消费。现在泵线程解析进 engine.glm_stat,/metrics 的 live.engine_decode 可读
   (含 age_s 新鲜度)。实测:121-token 请求后 tok_s 29.76、vram_hit 0.8018、age_s 0.0。
   注意日志里看不到 STAT:stdout 是协议管道,引擎日志只接 stderr。
8. 审计修复(2026-11-01,子代理审计+实测):泵 EOF 时向 glm_q 广播 None(此前引擎死亡
   时 GLM 请求无限心跳,消费端本来就有 None→EngineDied 处理,只是泵没喂);四处未门控
   调试打印([glm-feed] 每 token 打模型原文等)改为 STRATA_DEBUG 门控,删除 glm_tail 死代码。
   实测:新请求 server 日志 0 条 glm-feed(修复前每 token 1 条,~80 MB/天/活跃请求)。
9. 启动卡死第二层原因(2026-11-01):server 被单独信号停止后 host 仍存活并占着 profile
   状态文件,新 host 起手即 FileExistsError;且 check_up 把 curl 连接失败的 "000" 当应答,
   第一个等待循环形同虚设(此前快速失败加错了循环,从未生效)。已修:check_up 识别 000、
   两个等待循环都有 server_gone_bail(serve.server 进程消失即打日志尾部退出)、sleep 1、
   spawn 前按 PID 清理孤儿 host。RPC stop(host 自杀+删状态)与单独信号(遗留 host)的
   行为差异是排查 08:34 停止事件的关键证据:那次是直接发给 serve.server 的信号,非崩溃、
   非 OOM、非 strata launch --stop。


## 运维坑(务必遵守)
- 清理进程禁止 pkill -f(自匹配误杀);用 pgrep -x strata-glm 精确匹配后按 PID kill;
  runtime host / serve.server 用 pgrep -f 后按 PID kill。
- start-glm.sh 的 python 选择已内建(2026-11-01 修复"直接运行卡住"问题):
  STRATA_PYTHON 仍优先;未设时若 checkout 的 .venv/bin/python3 存在且能 import regex
  则自动选用,否则退回系统 python3 并打印所选 python(首行 Python: ...)。
  此前不带环境变量裸跑:server 起手即崩(regex),脚本在静默 10 分钟轮询里假死。
  同次新增快速失败:等待期间若 serve.server 进程消失,立即打印 runtime 日志尾部并退出,
  不再等满 10 分钟。实测:裸跑 start-glm.sh 完成启动、banner、meta 正确、connect 双 PASS。
- 运行配置:/home/simon/.local/state/strata/launch/ud-q2_k_xl-32k.json;
  安装级配置:~/.local/share/strata/install/strata-glm-5.3-flash-ud-q2_k_xl.json。
- 启动脚本已带 flock 与原子配置写入;旧 runtime host 进程需手动 kill。
  已修 start.sh 两个老 bug(2026-10-31):flock fd 被 setsid 的 runtime 继承、锁存活期=服务期,advertising 的 --context/--thinking/--parallel 切换永远打不响;context/parallel 两个内嵌 python 块用 os.replace 却没 import os。切换前仍需先停服务(脚本自己会停,但当前正在服务时锁已释放,直接跑即可)。

## opencode 接入
- strata connect opencode --write --url http://192.168.0.200:8080
- 或手动:OpenAI 兼容 provider,base_url http://192.168.0.200:8080/v1,model glm-5.3-flash,无 key。
- 首次运行 opencode 需联网安装 npm 插件。

## 提交
main 领先 origin/main 13 个提交(未推送,推送与否由用户决定)。
本次会话的修复已提交:08cb0325([Terminal],修复本体)与 0858157e([Docs],教训文档)。main 领先 origin/main 15 个提交,推送与否由用户决定。
