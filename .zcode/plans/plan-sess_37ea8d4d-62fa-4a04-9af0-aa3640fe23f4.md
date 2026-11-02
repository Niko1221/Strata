## Context
用户要求修复严格性复盘发现的四个问题，目标是工具参数保持类型与 JSON Schema 约束、调用边界不被参数内容截断，并且遇到转换器不支持的约束时失败关闭。DeepInfer 的 KMP 失败转移和将 schema 编译委托给专门编译器是有用参考；本仓库的 converter 会把不支持的 regex 放宽成任意字符串，因此严格 tool 路径不能照搬这种降级。

## 推荐方案
1. **把闭合符排除做成可复用的 overlap-aware 自动机**：在 `serve/grammar_tools.py` 为 `exclude_seq` / `exclude_seqs` 使用 KMP failure transition，而不是失配后一律回到状态 0；复用同一转移规则处理嵌套 JSON 字符串的 guard。保留部分匹配后遇到新起始符的候选，确保 `<<\/parameter>`、`</par</parameter>` 这类重叠输入也不能含有完整 `</parameter>`。
2. **把 schema 约束接到 converter，而不是改写选定的 GBNF 文本**：扩展 `vendor/json_schema_to_grammar/json-schema-to-grammar.h/.cpp` 及 `tools/grammar_schema/main.cpp` 的转换选项，让 tool 参数转换明确区分“模板中未加引号的顶层 string 值”和“JSON 序列化对象中的嵌套字符串”。顶层 string 的 enum/const/pattern/minLength/maxLength 必须保留；嵌套 JSON 字符串的 closer guard 必须覆盖通用、长度限制、pattern/format、enum/const 等所有 converter 输出形式。对无法精确保留 schema 约束与 guard 的格式（包括 converter 当前会放宽的 unsupported pattern），返回转换错误并走明确的 fail-closed 请求错误，不静默改成任意字符串或 prompt fallback。
3. **消除规则名冲突**：tool 参数规则名改为按 tool/property 序号生成，不从用户提供的属性名派生；新增的 guard 规则名从当前规则集合中分配并检查唯一性。转换完成后检测重复 `::=` 定义，发现重复直接拒绝，避免引擎按同名 symbol 静默覆盖规则。
4. **补齐回归并修正文档**：更新 `serve/test_grammar.py`、`tools/test_grammar_builder.py` 和 converter 测试，覆盖 overlap、顶层 string schema、嵌套 specialized strings、规则名冲突以及 unsupported-pattern fail-closed。用实际 converter 生成测试语法，并用引擎 GBNF parser/grammar 测试确认应接受和拒绝的输入。更新 `docs/DETAILS.md`，仅承诺已实现且测试覆盖的严格约束与失败行为。

## 关键文件
- `serve/grammar_tools.py`: overlap-aware exclusions、稳定且唯一的 tool/guard 规则名、严格转换错误传播。
- `serve/grammar_schema.py`, `tools/grammar_schema/main.cpp`: 传递顶层 raw-string / forbidden-sequence 转换选项。
- `vendor/json_schema_to_grammar/json-schema-to-grammar.h`, `vendor/json_schema_to_grammar/json-schema-to-grammar.cpp`: 在 schema visitor 中保留字符串约束并统一应用 guard；unsupported case 明确报错。
- `serve/test_grammar.py`, `tools/test_grammar_builder.py`, converter / engine grammar tests: 新回归。
- `docs/DETAILS.md`: 收窄并说明保证范围及 fail-closed 行为。

## 验证
运行 grammar builder 与 converter 测试、`serve.test_grammar` 及相关 serve schema/structured-output 测试；运行 converter golden tests，确认默认转换输出未变；构建 `grammar_schema` 并通过引擎 grammar parser 对生成的代表性语法做正反接受测试；最后运行完整 serve 测试套件。保留现有未跟踪 build 与 `.zcode-obj` artifacts，不清理或覆盖无关数据。