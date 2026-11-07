# GLM-5.3-Flash 移植:经验教训(2026-10-31)

工具调用链收口当天记的。每条都有当时的实测证据,写给下一个会话:同样的坑不必再踩一遍。
状态与运维见 [GLM-PORT-STATUS.md](GLM-PORT-STATUS.md)。

## 1. 交接文档里的"修法"是假设,不是规格

交接写"engine args 增加 --eos-ids 154844"。照抄会出大事:引擎的 `--eos-ids` 是
**替换**默认停止集,不是追加(`src/program/glm_main.cpp`:`eos = o.eos_given ? o.eos_ids : glm_eos`,
默认 `{154820, 154827, 154829}`)。只传 154844,普通对话就再也停不下来。
教训:动手前读参数在**消费端**的语义;token id 一律对照词表核实
(那次用 `vocab.json` 反查,靠码点而不是肉眼——终端渲染会把 `<...>` 字面量吃掉/变乱码,
`ascii()`/码点打印才是可信的)。

## 2. 引擎停了 ≠ 全链路停了:每一层都有自己的"停止"记账

引擎 DONE 报 `(stop)`,server 却记 `length`。因为 server 有自己的逐 token 停止集
(`Service.stop_ids`),与引擎的 eos 集互不相通;parser 又只认 server 给的 finish
(挂起调用只在自然停时升级为 tool_call,#1058)。同一件事至少三层各记一遍:
引擎(--eos-ids)→ server(stop_ids)→ parser(finish reason)。
教训:finish 不对时,先对比各层日志(引擎 DONE 行 vs server `[strata] done:` 行)——
两层说法不一致的位置就是 bug 的位置,这次一分钟定位。

## 3. 静默默认值掩盖的是整个家族的 bug,不止你找的那个

`special_ids.get("eos", 1)`:GLM 的键是 GGUF 风格(`tokenizer.ggml.eos_token_id`),
取不到 → 永远返回 1 → server 从不认任何结束符:每个回答都以 length 结束、
结束标记漏进正文。这比"工具调用差一环"严重得多,但只测工具调用永远发现不了——
是顺手发了个普通请求才暴露的。
教训:修某条路径时,同批跑一遍最无聊的那条(普通提问);键名按家族分叉时,
不信任 `.get()` 的静默默认。

## 4. 流水线里,拦截器必须让有状态的翻译器先看到 token

把 154844 加进 server 的 stop_ids 后,if 不改顺序,修复会**倒退**:stop 分支在
`glm_feed` 之前 break,rewriter 永远收不到闭合符,缓冲的调用在 flush 时退回纯文本。
正解是"先喂后停"(`stop_feed`:闭合符走完 detok → rewriter → parser,再停)。
教训:流式管道里,凡排在有状态翻译器前面的拦截(stop 检查、心跳、cut)都要问一句:
这个 token 翻译器还需要吗?

## 5. 翻译器要输出消费方的"原生拼写",单测只测翻译器会绿灯放行集成红灯

`GlmCallRewriter` 输出裸 `<function=...>` 块;而 parser 的 opener 是完整 `<tool_call>` 标签
(`frontend.py` CALL_START),裸块只有 opt-in 的 recover 模式才收。上一会话
"单测与在线探针均确认翻译正确"——测的是 rewriter 对自己约定,不是对 parser 的约定。
教训:全链路模拟一次(token 流 → rewriter → parser → finish),组件绿不等于集成绿。
两个附带的坑,模拟时必踩:
- `OutputParser(tools=...)` 要**扁平**格式 `{"name": ...}`;喂 OpenAI 嵌套格式
  `{"type":"function","function":{...}}` 会静默得到空 schemas,所有调用变"文本"。
- 模拟脚本里构造特殊 token 字面量用 chr() 拼(工具层会吃 `<...>`),见第 1 条。

## 6. 环境的坑:运行解释器与死代码

- start-glm.sh 默认系统 python3,缺 `regex`,server 启动即崩
  (ModuleNotFoundError)。必须 `STRATA_PYTHON=/data/AI/Strata/.venv/bin/python3`。
  上一会话隐式用了 venv,把 wrapper 的默认值留成了陷阱。
- setup.py install_glm 里躺着一段合并事故的重复 args/cfg 块:第二段覆盖第一段,
  静默丢掉 `--parallel`。是在改同一行时才看见的。
教训:当"盘上配置"和"代码写出来的"对不上,先怀疑死代码/重复块;编辑某行前
把整个块读完。

## 7. 运维纪律(血泪原样)

- 清进程禁 `pkill -f`(自匹配误杀,会杀掉自己的会话);`pgrep -x strata-glm`
  或 pgrep -f 后按 PID kill。
- 结束符判定实验要在**真实栈**上做:curl → server 日志(`[glm-feed]` / `[strata] done:`)
  → 引擎日志三面对照,一次实验顶十次推理。
- 验收标准只有一条:`strata connect opencode --check` 出 PASS prompt 与 PASS tool call
  两行。中间态的"PASS prompt"不算数(opencode 对残缺协议是宽容的)。

## 8. 工具层会吃 `<...>` 字面量:写文件要用程序拼,落盘后要验字节

本次会话里,agent 工具调用中的 ``/`` 字面量三次被序列化层吞掉:setup.py 注释空了
半边括号、两份 markdown 留下空的 ``。当时终端显示还看不出异常(乱码掩盖)。
正解:含 `<...>` 的内容一律用 python 脚本以 chr() 拼接写入;写完用码点打印核验
关键行。文档示例同理——本文档里的标签就是脚本拼进去的。

## 9. 不变量固化在拥有它的那一层

“所有路径都必须带的参数”放在启动脚本或配置里，每个新入口（手动裸跑、新工具、别人的启动方式）都要重新记得它，漏一处就回归。正确的位置是契约的拥有者：停止集是模型的契约，主人是引擎的 GLM 驱动，所以 154844 进了 glm_eos 默认值。
保护性余量可以留在下游（setup.py 仍显式写参数，防旧二进制），但不能只有下游：下游是提醒，不是契约。

## 10. 特殊标签用程序核字节,终端显示会骗人
调试标签类 bug(gl 的 154843-154850 / rewriter / 词表)时,把标签当字符串直接写进命令或文档,会被终端转义和字体回退搞得面目全非(displayed `pulse`≠实际字节)。唯一可信的是程序打印:repr、unicode_escape、码点列表。修复闭环是"用词表真实字节构造的最小单测",并且逐字符流与分块流都要过(改写器缓冲逻辑对分片敏感)。

## 11. 默认值是代码里的策略,不是磁盘上的最后状态
把一次性选择(--context 1m)写进持久配置后,"默认"就变成了上次的选择——用户裸跑时拿到的不是承诺过的默认,而是历史的残留。默认行为必须在代码里固定(脚本裸跑强制回默认),旗标只对本次生效;要改默认值就改代码里的常量,而不是靠某次带旗标的启动"顺便"改掉。

## 12. 修复必须落在失败路径上实测
把快速失败加进了一个"永远只循环一次"的等待循环(check_up 把 curl 连不上时输出的 "000" 当成应答),自以为修了卡死,实际从未生效——直到下一次真实故障暴露。修复后要构造故障本身验证(kill 进程、断管道),happy path 过了不等于修好了。另外:supervisor 收到 RPC stop 会自杀并删状态文件,而直接发给 server 进程的信号会遗留一个占着状态文件的 host——两种"停止"痕迹不同,排查时先分清。

## 一句话总结

这条链上每个组件(引擎、server、rewriter、parser、客户端)对"一次调用怎么算完"
各有一本账;移植的本质是把五本账对齐——而不是把其中一本记对。
