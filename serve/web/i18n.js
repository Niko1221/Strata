/* Strata UI 中文化 — Chinese UI for the Strata web client.
   Loaded before app.js. Translates the page (and everything the app renders later) in place,
   so no app logic changes are needed. Set localStorage["strata.lang"]="en" to keep it English. */
"use strict";
(function () {
  var KEY = "strata.lang";
  var lang = "zh";
  try { lang = localStorage.getItem(KEY) === "en" ? "en" : "zh"; } catch (e) {}

  /* Whole-element texts and short labels, matched after whitespace is collapsed. */
  var EXACT = new Map(Object.entries({
    /* header, tabs */
    "Views": "视图",
    "Chat": "对话",
    "Monitor": "监控",
    "About": "关于",
    "Connecting…": "正在连接…",
    "Light / dark": "浅色 / 深色",
    "Switch between light and dark": "在浅色与深色之间切换",
    /* chat */
    "Ask anything": "随便问点什么",
    "Ask anything…": "随便问点什么…",
    "The model runs on this PC. Nothing leaves it.": "模型就跑在这台电脑上,数据不出本机。",
    "Attach a text file (or drop it here)": "添加文本文件(也可以拖到这里)",
    "Attach a text file or a picture (or drop it here)": "添加文本文件或图片(也可以拖到这里)",
    "Attach a file": "添加文件",
    "New chat": "新对话",
    "Save this chat as Markdown": "把这次对话保存为 Markdown",
    "Save this chat": "保存这次对话",
    "Sampling and thinking": "采样与思考",
    "Message": "消息",
    "Stop": "停止",
    "Send": "发送",
    "Shift+Enter: new line": "Shift+Enter:换行",
    "Remove": "移除",
    "Files": "文件",
    "pasted image": "粘贴的图片",
    /* monitor */
    "Model state": "模型状态",
    "Idle": "空闲",
    "Reading": "读取中",
    "Generating": "生成中",
    "Queued": "排队中",
    "Error": "出错",
    "Failed": "失败",
    "Waiting": "等待中",
    "Waiting for a request": "等待请求",
    "Reading prompt": "正在读取提示词",
    "Context fill": "上下文占用",
    "Experts in VRAM": "显存中的专家",
    "System RAM": "系统内存",
    "GPU temperature": "显卡温度",
    "Recent requests": "最近请求",
    "Show all": "显示全部",
    "Show fewer": "收起",
    "Time": "时间",
    "Status": "状态",
    "Prompt": "提示词",
    "Reused": "复用",
    "Output": "输出",
    "Tok/s": "Tok/s",
    "Hit rate": "命中率",
    "Duration": "耗时",
    "While writing the answer: the share of the experts looked up that were already in VRAM (experts copied over PCIe are not counted)": "写答案期间:查到的专家里已经在显存中的比例(经 PCIe 拷进来的不计入)",
    "No requests yet": "还没有请求",
    "MCP servers": "MCP 服务器",
    "Their tools run on this PC with your rights, when the model decides to call them (in this page's chat only; switch it off in Sampling).": "模型决定调用时,这些工具会以你的权限在这台电脑上执行(只在本页聊天里生效;可以在「采样」里关掉)。",
    /* metrics */
    "Speed": "速度",
    "GPU load": "显卡占用",
    "VRAM": "显存",
    "GPU temp": "显卡温度",
    "Power": "功耗",
    "PCIe": "PCIe",
    "Disk read": "磁盘读取",
    "needs psutil (setup installs it)": "需要 psutil(安装脚本会装上)",
    "not readable (NVML)": "读不到(NVML)",
    /* about */
    "Model and engine": "模型与引擎",
    "This PC": "这台电脑",
    "Connect your tools": "接入你的工具",
    "Any OpenAI- or Anthropic-compatible client works with these addresses.": "任何兼容 OpenAI 或 Anthropic 的客户端都可以用下面的地址接入。",
    "Settings": "设置",
    "API key": "API 密钥",
    "only if the server was started with one": "仅在服务启动时设置了密钥才需要",
    "Not needed": "无需填写",
    "Dark theme": "深色主题",
    "Chats, settings and the key are kept in this browser only.": "对话、设置和密钥只保存在这个浏览器里。",
    "Strata on GitHub": "GitHub 上的 Strata",
    "Model": "模型",
    "Engine": "引擎",
    "built from source": "从源码构建",
    "Context": "上下文",
    "KV cache": "KV 缓存",
    "Speculation": "推测解码",
    "Images": "图片",
    "GPU": "显卡",
    "CPU": "CPU",
    "RAM": "内存",
    "OpenAI base URL": "OpenAI 接口地址",
    "Anthropic base URL": "Anthropic 接口地址",
    "Model name": "模型名称",
    "8-bit": "8-bit",
    "16-bit": "16-bit",
    "4-bit (Hadamard-rotated)": "4-bit(哈达玛旋转)",
    "Connected": "已连接",
    "Starting": "正在启动",
    "Stopped": "已停止",
    "4-bit (Hadamard rotated)": "4-bit(哈达玛旋转)",
    /* sampling drawer */
    "Sampling": "采样",
    "Close": "关闭",
    "Thinking": "思考",
    "Off": "关",
    "Low": "低",
    "Medium": "中",
    "High": "高",
    "Temperature": "温度",
    "0 = always the most likely word (exact, repeatable)": "0 = 总是选最可能的词(精确、可复现)",
    "Max tokens": "最大 token 数",
    "empty = until done": "留空 = 一直写到结束",
    "Until done": "一直写到结束",
    "Seed": "随机种子",
    "empty = random": "留空 = 随机",
    "Random": "随机",
    "Show thinking": "显示思考过程",
    "expanded while it streams": "流式输出时自动展开",
    "Use tools from MCP servers": "使用 MCP 服务器里的工具",
    "the model may call them while it answers": "模型回答时可能会调用它们",
    "Experimental speed projection": "实验性加速投影",
    "the engine's control vector; off = the stock model. Switching reads the chat again once": "引擎的控制向量;关闭 = 原版模型。切换时会重新读一遍对话",
    "Use for other apps too": "也给其他应用使用",
    "omp and other API clients get these settings for anything they don't set themselves": "omp 和其他 API 客户端没自己指定参数时,会沿用这里的设置",
    "Reset": "重置",
    "Apply": "应用",
    "answers right away": "立即回答",
    "short": "简短",
    "medium": "适中",
    "thorough (default)": "详尽(默认)",
    "0 · greedy": "0 · 贪心",
    /* tool calls, thinking, errors */
    "Writing": "写入中",
    "Running": "运行中",
    "Done": "完成",
    "Not run": "未运行",
    "Arguments": "参数",
    "Result": "结果",
    "(being written)": "(正在写入)",
    "tool": "工具",
    "Thinking…": "思考中…",
    "Thoughts": "思考过程",
    "Copy": "复制",
    "Copy code": "复制代码",
    "Copy the answer": "复制答案",
    "Copied to clipboard": "已复制到剪贴板",
    "Copied": "已复制",
    "API key saved": "API 密钥已保存",
    "Kept in this browser only.": "只保存在这个浏览器里。",
    "API key needed": "需要 API 密钥",
    "This server needs a key: add it under About > Settings.": "这个服务需要密钥:请在「关于 > 设置」里填上。",
    "This server needs an API key: add it under About > Settings.": "这个服务需要 API 密钥:请在「关于 > 设置」里填上。",
    "Server not reachable": "连不上服务器",
    "the engine reported an error": "引擎报告了一个错误",
    "The request failed": "请求失败",
    "Still writing": "还在写入",
    "Stop the answer first.": "请先停止当前回答。",
    "The last one was cleared.": "上一次对话已清空。",
    "Undo": "撤销",
    "Nothing to save yet": "还没有可保存的内容",
    "Pictures are off": "图片功能已关闭",
    "This model was set up for text only.": "这个模型只配置了文本。",
    "Picture too large": "图片太大",
    "Not a text file": "不是文本文件",
    "File too large": "文件太大",
    "Max tokens": "最大 token 数",
    "Closed": "连接关闭",
    "no server is connected yet (see the Monitor)": "还没有连接任何服务器(见「监控」)",
    "Sampling saved": "采样设置已保存",
    "Greedy: the same question gives the same answer.": "贪心:同样的问题会得到同样的答案。",
    "Other apps (omp, API clients) use these settings from their next request.": "其他应用(omp、API 客户端)从下一次请求起会用这些设置。",
    "Other apps use their own settings again.": "其他应用改回用它们自己的设置。",
    "Saved here, but not for other apps": "只在这里生效,不用于其他应用",
    /* the request monitor page */
    "Loading…": "正在加载…",
    "Unloading…": "正在卸载…",
    "Loaded": "已加载",
    "Unloaded": "未加载",
    "In use": "使用中",
    "Loads automatically on the next request": "下一次请求时自动加载",
    "No active requests": "没有正在进行的请求",
    "No matching requests.": "没有匹配的请求。",
    "No requests yet. API calls will appear here automatically.": "还没有请求。API 调用会自动出现在这里。",
    "No content.": "没有内容。",
    "Streaming response: see Output and the request error, if any.": "流式响应:请看「输出」和请求错误(如果有)。",
    "Original request body.": "原始请求体。",
    "Model answer. JSON is formatted here for readability.": "模型的回答。JSON 在这里做了格式化以便阅读。",
    "Separate reasoning content, when enabled.": "单独的推理内容(启用时才有)。",
    "API response body.": "API 响应体。",
    "Raw model output retained for diagnosis. The API returned an error.": "为诊断保留的原始模型输出。API 返回了错误。",
    "Clipboard unavailable; select the text to copy it.": "剪贴板不可用;请手动选中文本复制。",
    "Disconnected · retrying": "已断开 · 正在重试",
    "Request": "请求",
    "Loading": "加载中",
    /* metric labels and small state words */
    "Decode": "解码",
    "Prefill": "预填充",
    "Decode now": "解码 · 当前",
    "Decode last request": "解码 · 上一个请求",
    "Prefill now": "预填充 · 当前",
    "Prefill this request": "预填充 · 本次请求",
    "Prefill last request": "预填充 · 上一个请求",
    "stock": "原版",
    "ESP": "ESP",
    "Projection": "投影",
    "Additive": "叠加",
    "on": "开",
    "off": "关",
    "Greedy": "贪心",
    "Until done": "一直写到结束",
    /* /api-monitor 页面(monitor.html + monitor.js) */
    "Local inference": "本地推理",
    "API Monitor": "API 监控",
    "Model status and every request, in one place.": "模型状态与全部请求,都在这里。",
    "Load model": "加载模型",
    "Unload model": "卸载模型",
    "Only needed if the server has a key": "仅在服务端设置了密钥时需要填写",
    "Connect": "连接",
    "Waiting for server": "等待服务端",
    "Requests retained": "保留的请求数",
    "Last wall-clock": "最近一次总耗时",
    "Includes queue and model loading": "含排队与模型加载时间",
    "Last decode speed": "最近一次解码速度",
    "Engine timing · tokens / second": "引擎计时 · token / 秒",
    "Search ID or endpoint": "搜索 ID 或端点",
    "Filter requests": "筛选请求",
    "Request status": "请求状态",
    "All statuses": "全部状态",
    "Active": "进行中",
    "Completed": "已完成",
    "Errors": "错误",
    "Requests": "请求",
    "Request list": "请求列表",
    "Inspect a request": "查看某个请求",
    "Select a request to see its input, output, and timings.": "选中一个请求即可查看它的输入、输出与耗时。",
    "Wall-clock": "总耗时",
    "Model load": "模型加载",
    "Queue wait": "排队等待",
    "First token": "首个 token",
    "Prompt tokens": "提示词 token",
    "Output tokens": "输出 token",
    "Decode speed": "解码速度",
    "Response format": "响应格式",
    "Request content": "请求内容",
    "API response": "API 响应",
    "Last 100 requests kept in memory until restart. Inputs and outputs stay on this machine.": "最近 100 条请求保留在内存中直到重启。输入与输出都不会离开这台机器。",
    "Switch color theme": "切换配色主题",
    "Strata API Monitor": "Strata API 监控",
    "JSON response": "JSON 响应",
    "Monitor capture truncated at 256K characters; the API response was not shortened.": "监控留存的内容在 256K 字符处被截断;API 响应本身没有被截短。",
    "completed": "已完成",
    "active": "进行中",
    "error": "错误",
}));

  /* Rendered strings that contain numbers or other values keep the value in {1}, {2}, … */
  var PATTERNS = [
    [/^(.*) runs on this PC\. Nothing leaves it\.$/, "{1} 跑在这台电脑上,数据不出本机。"],
    [/^Reading prompt( · .*)?$/, "正在读取提示词{1}"],
    [/^Generating( · .*)?$/, "正在生成{1}"],
    [/^(\d[\d,]*) queued$/, "{1} 个排队中"],
    [/^(\d[\d,]*) \/ (\d[\d,]*) tokens( · .*)?$/, "{1} / {2} tokens{3}"],
    [/^(\d[\d,]*) tokens · ([\d.]+) tok\/s$/, "{1} tokens · {2} tok/s"],
    [/^(\d[\d,]*) tokens at ([\d.]+) tok\/s$/, "{1} tokens,{2} tok/s"],
    [/^(\d[\d,]*) tokens$/, "{1} tokens"],
    [/^last: (.*)$/, "上一个:{1}"],
    [/^(\d[\d,]*) experts cached$/, "已缓存 {1} 个专家"],
    [/^of (.*) W limit$/, "上限 {1} W"],
    [/^to GPU (.*) MB\/s$/, "到 GPU {1} MB/s"],
    [/^idle Gen(.*)$/, "空闲 Gen{1}"],
    [/^(\d+) threads$/, "{1} 个线程"],
    [/^(\d+) cores · (\d+) threads$/, "{1} 核 · {2} 线程"],
    [/^(\d+) cores$/, "{1} 核"],
    [/^(\d+) queued · (.*)$/, "{1} 个排队 · {2}"],
    [/^(.+) \(now\)$/, "{1}(现在)"],
    [/^write (.*) MB\/s$/, "写入 {1} MB/s"],
    [/^Show all \((\d+)\)$/, "显示全部({1})"],
    [/^(Projection|Additive) control vector on layers (.*)$/, "{1} 控制向量,作用在第 {2} 层"],
    [/^\. Per chat in Sampling\. Its package describes the vector as a refusal-direction projection; measure the speed yourself$/, ". 可在「采样」里按对话设置。它的说明文档把该向量描述为「拒绝方向投影」;速度请自行实测"],
    [/^MTP drafts up to (.*) tokens(, prompt lookup on)?$/, "MTP 草稿最多 {1} tokens{2}"],
    [/^(.*) tools · (.*) of (.*) servers connected$/, "{1} 个工具 · {3} 个服务器中已连接 {2} 个"],
    [/^(.*) tools from (.*); the model calls them when it decides to$/, "{2} 提供 {1} 个工具;模型认为需要时会调用"],
    [/^(.*) · (\d+) tools$/, "{1} · {2} 个工具"],
    [/^You · (.*)$/, "你 · {1}"],
    [/^(.*) is over 20 MB\.$/, "{1} 超过 20 MB。"],
    [/^(.*) is over 512 KB\.$/, "{1} 超过 512 KB。"],
    [/^(.*) looks like a binary file\.$/, "{1} 看起来是二进制文件。"],
    [/^(.*): attach text files \(code, notes, logs, data\)( or pictures)?\.$/, "{1}:只能添加文本文件(代码、笔记、日志、数据){2}。"],
    [/^(\d[\d,]*) tokens( · stopped)?$/, "{1} tokens{2}"],
    [/^(\d[\d,]*) tool calls?$/, "{1} 次工具调用"],
    [/^(.*) · stopped at the limit of (\d+) tool rounds \(mcp\.max_rounds\)$/, "{1} · 达到 {2} 轮工具调用上限而停止(mcp.max_rounds)"],
    [/^Thought for (.*) s$/, "思考了 {1} 秒"],
    [/^(.*), all in VRAM$/, "{1},全部在显存"],
    [/^(.*), streamed: (.*) positions per layer in VRAM, the rest in RAM$/, "{1},流式:{2} 个位置/层留在显存,其余在内存"],
    [/^Live · updated (.*)$/, "实时 · 更新于 {1}"],
    [/^(\d+) active \/ queued$/, "{1} 个进行中 / 排队"],
    [/^Request (\d+)$/, "请求 {1}"],
    [/^· stopped$/, "· 已停止"],
    [/^· projection on$/, "· 加速投影已开"],
    [/^· projection off$/, "· 加速投影已关"],
    [/^, prompt lookup on$/, ",已开启提示词查找"],
    [/^, cut for the model$/, ",已为模型截断"],
    [/^ \(error\)$/, "(出错)"],
    [/^(.*) \(error\)$/, "{1}(出错)"],
    [/^Since (.*): (.*) requests · (.*) prompt tokens read(.*) \((.*) reused\) · (.*) written(.*)$/, "自 {1} 起:{2} 次请求 · 读取 {3} 个提示词 token{4}(复用 {5} 个)· 写出 {6} 个 token{7}"],
    [/^Since (.*)$/, "自 {1} 起"],
    [/^(\d[\d,]*) \/ (\d[\d,]*) tokens · ([\d.]+)%$/, "{1} / {2} tokens({3}%)"],
    [/^at ([\d.]+) tok\/s$/, " {1} tok/s"],
    [/^([\d.]+) s$/, "{1} 秒"],
    [/^Request (\S+)$/, "请求 {1}"],
    [/^(\d+) active \/ queued$/, "{1} 个进行中 / 排队中"],
    [/^Live · updated (.*)$/, "运行中 · 更新于 {1}"],
    [/^(\S+) · ([\d.]+) s · (JSON response|stream)(.*)$/, "{1} · {2} 秒 · {3}{4}"],
  ];

  var ATTRS = ["title", "aria-label", "placeholder", "alt"];
  var SKIP = ".st-bubble,.thinking,.st-code,pre,code,.tool-call__pre,script,style,textarea,[data-no-i18n]";

  function norm(text) { return String(text == null ? "" : text).replace(/\s+/g, " ").trim(); }

  function tr(text, depth) {
    if (!text) return null;
    var hit = EXACT.get(text);
    if (hit !== undefined) return hit;
    for (var i = 0; i < PATTERNS.length; i++) {
      var m = PATTERNS[i][0].exec(text);
      if (m) {
        return PATTERNS[i][1].replace(/\{(\d)\}/g, function (whole, d) {
          var g = m[+d] == null ? "" : m[+d];
          var t = depth ? null : tr(norm(g), 1);
          return t == null ? g : t;
        });
      }
    }
    return null;
  }

  function hasWords(text) { return /[A-Za-z]/.test(text); }

  function trText(node) {
    var raw = node.nodeValue;
    if (!raw || !hasWords(raw)) return;
    var body = norm(raw);
    if (!body) return;
    var out = tr(body);
    if (out == null || out === body) return;
    var lead = raw.match(/^\s*/)[0], trail = raw.match(/\s*$/)[0];
    node.nodeValue = lead + out + trail;
  }

  function trAttrs(el) {
    for (var i = 0; i < ATTRS.length; i++) {
      var name = ATTRS[i];
      if (!el.hasAttribute || !el.hasAttribute(name)) continue;
      var value = el.getAttribute(name);
      if (!value || !hasWords(value)) continue;
      var out = tr(norm(value));
      if (out != null && out !== value) el.setAttribute(name, out);
    }
  }

  function skipped(el) {
    try { return !!(el && el.closest && el.closest(SKIP)); } catch (e) { return false; }
  }

  function walk(node) {
    if (node.nodeType === 3) { if (!skipped(node.parentElement)) trText(node); return; }
    if (node.nodeType !== 1) return;
    trAttrs(node);
    if (skipped(node)) return;
    for (var child = node.firstChild; child; child = child.nextSibling) walk(child);
  }

  function translated() { return lang === "zh"; }

  function translateAll() { if (translated()) walk(document.body); }

  function watch() {
    if (!translated() || typeof MutationObserver !== "function") return;
    new MutationObserver(function (records) {
      for (var i = 0; i < records.length; i++) {
        var r = records[i];
        if (r.type === "childList") {
          for (var j = 0; j < r.addedNodes.length; j++) walk(r.addedNodes[j]);
        } else if (r.type === "characterData") {
          if (!skipped(r.target.parentElement)) trText(r.target);
        } else {
          trAttrs(r.target);
        }
      }
    }).observe(document.documentElement, {
      childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ATTRS
    });
  }

  function addToggle() {
    var header = document.querySelector(".st-header") || document.querySelector("header");
    if (!header || document.getElementById("lang-btn")) return;
    var button = document.createElement("button");
    button.className = "st-btn st-btn--icon";
    button.id = "lang-btn";
    button.setAttribute("data-no-i18n", "");
    button.textContent = translated() ? "EN" : "中";
    var label = translated() ? "Switch to English" : "切换到中文";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.addEventListener("click", function () {
      try { localStorage.setItem(KEY, translated() ? "en" : "zh"); } catch (e) {}
      location.reload();
    });
    var ref = document.getElementById("theme-btn") || document.getElementById("theme");
    if (ref && ref.parentNode) ref.parentNode.insertBefore(button, ref);
    else header.appendChild(button);
  }

  try { document.documentElement.lang = translated() ? "zh-CN" : "en"; } catch (e) {}

  window.__strataI18n = { lang: lang, tr: tr, norm: norm, EXACT: EXACT, PATTERNS: PATTERNS };

  function boot() {
    try { var t = tr(norm(document.title)); if (t) document.title = t; } catch (e) {}
    try { translateAll(); } catch (e) { console.error("strata i18n: translate failed", e); }
    try { addToggle(); } catch (e) { console.error("strata i18n: toggle failed", e); }
    try { watch(); } catch (e) { console.error("strata i18n: observer failed", e); }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
