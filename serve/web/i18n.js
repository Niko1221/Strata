// serve/web/i18n.js - interface languages for the Strata web app.
//
// How it works (and why it is built this way): the pages are plain HTML and JS, so
// a small layer walks the DOM and replaces the interface's English text with the
// chosen language. Chat messages, reasoning, tool output, code blocks and inputs are
// never touched. A MutationObserver does the same for text the app adds later (toasts,
// status pills, monitor rows). English strings are the keys, so a translation is
// additive: nothing changes for English users, and adding a language is one object.
//
// The choice is saved in localStorage ("strata.lang"); without it the browser's own
// language decides. A small button in the header switches back and forth.

"use strict";

const I18N_TABLE = {
  "zh-CN": {
    // ---- header / tabs
    "Chat": "对话",
    "Monitor": "监控",
    "About": "关于",
    "Views": "视图",
    "Strata on GitHub": "GitHub 上的 Strata",
    "Light / dark": "浅色 / 深色",
    "Switch between light and dark": "切换浅色 / 深色",
    "Dark theme": "深色主题",
    "Connecting…": "连接中…",
    "Connecting&hellip;": "连接中…",

    // ---- chat
    "Ask anything": "随便问",
    "Ask anything…": "随便问…",
    "Ask anything&hellip;": "随便问…",
    "The model runs on this PC. Nothing leaves it.": "模型在本机运行，数据不出本机。",
    "Message": "消息",
    "Attach a file": "附加文件",
    "Attach a text file (or drop it here)": "附加文本文件（也可拖到这里）",
    "Attach a text file or a picture (or drop it here)": "附加文本或图片（也可拖到这里）",
    "New chat": "新对话",
    "Save this chat": "保存此对话",
    "Save this chat as Markdown": "保存为 Markdown",
    "Send": "发送",
    "Stop": "停止",
    "Stop the answer first.": "请先停止本次回答。",
    "Copy": "复制",
    "Copy code": "复制代码",
    "Copy the answer": "复制回答",
    "Copied to clipboard": "已复制到剪贴板",
    "Close": "关闭",
    "Remove": "移除",
    "Undo": "撤销",
    "Show thinking": "显示思考",
    "Thinking": "思考",
    "Thoughts": "思考过程",
    "Still writing": "仍在输出",
    "Write anything": "随便写点什么",
    "Reading prompt": "读取输入",
    "Generating": "生成中",
    "Waiting": "等待",
    "Waiting for a request": "等待请求",
    "Idle": "空闲",
    "Queued": "排队中",
    "Reading": "读取中",
    "Started": "已开始",
    "Running": "运行中",
    "Stopped": "已停止",
    "Done": "完成",
    "Failed": "失败",
    "Error": "错误",
    "The request failed": "请求失败",
    "the engine reported an error": "引擎报了错误",
    "Server not reachable": "无法连接服务端",
    "no server is connected yet (see the Monitor)": "尚未连接服务端（见「监控」）",
    "Expanded while it streams": "流式输出时自动展开",
    "expanded while it streams": "流式输出时自动展开",

    // ---- settings drawer
    "Settings": "设置",
    "Model settings": "模型设置",
    "Sampling and thinking": "采样与思考",
    "Temperature": "温度",
    "Top-k": "Top-k",
    "Top-p": "Top-p",
    "Seed": "随机种子",
    "Random": "随机",
    "Max tokens": "最大 Token 数",
    "Until done": "直到完成",
    "Sampling": "采样",
    "Thinking effort": "思考强度",
    "Low": "低",
    "Medium": "中",
    "High": "高",
    "Off": "关闭",
    "Not needed": "不需要",
    "Apply": "应用",
    "Reset": "重置",
    "Save": "保存",
    "Not saved": "未保存",
    "Nothing changed.": "没有变化。",
    "Sampling saved": "采样已保存",
    "Experimental speed projection": "实验性加速投影",
    "Use for other apps too": "同时供其它应用使用",
    "Saved here, but not for other apps": "仅保存在本浏览器，其它应用不生效",
    "Other apps use their own settings again.": "其它应用已恢复各自设置。",
    "Kept in this browser only.": "仅保存在本浏览器。",
    "Nothing to save yet": "还没有可保存的内容",

    // ---- About tab
    "Model and engine": "模型与引擎",
    "API key": "API 密钥",
    "API key needed": "需要 API 密钥",
    "API key saved": "API 密钥已保存",
    "OpenAI base URL": "OpenAI 接口地址",
    "Anthropic base URL": "Anthropic 接口地址",
    "Any OpenAI- or Anthropic-compatible client works with these addresses.":
      "任何兼容 OpenAI / Anthropic 的客户端都可以使用这些地址。",
    "Chats, settings and the key are kept in this browser only.":
      "对话、设置与密钥仅保存在本浏览器。",
    "only if the server was started with one": "仅当服务端启动时设置了密钥",
    "Connect your tools": "连接你的工具",
    "MCP servers": "MCP 服务器",
    "Use tools from MCP servers": "使用 MCP 服务器提供的工具",
    "the model may call them while it answers": "模型回答时可能调用它们",
    "This PC": "本机",
    "Model": "模型",
    "Engine": "引擎",
    "Version": "版本",
    "built from source": "源码编译",
    "not available": "不可用",
    "not readable (NVML)": "无法读取（NVML）",
    "needs psutil (setup installs it)": "需要 psutil（setup 会安装）",
    "not available on Windows AMD yet": "Windows AMD 暂不支持",

    // ---- live metrics
    "Decode t/s": "解码 t/s",
    "Prefill t/s": "预填充 t/s",
    "Decode": "解码",
    "Prefill": "预填充",
    "Decode now": "当前解码",
    "Decode last request": "上次解码",
    "Prefill now": "当前预填充",
    "Prefill last request": "上次预填充",
    "Last request": "上次请求",
    "Experts in VRAM": "显存内专家",
    "VRAM hit rate": "显存命中率",
    "Context fill": "上下文占用",
    "Conversation cache": "会话缓存",
    "Parked conversations": "停放的会话",
    "System RAM": "系统内存",
    "GPU temperature": "显卡温度",
    "GPU temp": "显卡温度",
    "GPU load": "显卡负载",
    "Power": "功耗",
    "Speed": "速度",
    "Projection": "投影",
    "Speculation": "投机解码",
    "Disk read": "磁盘读取",
    "Reused": "复用",
    "Reused since start": "启动以来复用",
    "Duration": "耗时",
    "Status": "状态",
    "Time": "时间",
    "Result": "结果",
    "Recent requests": "最近请求",
    "No requests yet": "暂无请求",
    "Show all": "显示全部",
    "Show fewer": "收起",
    "This model was set up for text only.": "此模型仅配置了文本，未启用图片。",
    "Pictures are off": "图片功能未开启",
    "Images": "图片",
    "pasted image": "粘贴的图片",
    "File too large": "文件过大",
    "Picture too large": "图片过大",
    "Not a text file": "不是文本文件",
    "Files": "文件",

    // ---- images (the Images tab, added in 0.1.44)
    "Make an image": "生成图像",
    "Size": "尺寸",
    "Generate": "生成",
    "random": "随机",
    "Describe the image…": "描述你想要的图像…",
    "Describe the image&hellip;": "描述你想要的图像…",
    "One image at a time; a chat request waits for it (and the other way round).": "一次只生成一张图；对话请求会等它完成（反之亦然）。",
    "Generating… (a minute or more on a small GPU)": "生成中…（小显卡上可能要一分钟以上）",
    "Done. Right-click an image to save it.": "完成。右键图片可保存。",
    "No image": "没有图像",

    // ---- monitor page
    "API Monitor": "API 监控",
    "Strata API Monitor": "Strata API 监控",
    "Local inference": "本地推理",
    "Model status and every request, in one place.": "模型状态与全部请求，集中一处。",
    "Requests": "请求",
    "Requests retained": "保留的请求",
    "Waiting for server": "等待服务端",
    "Wall-clock": "总耗时",
    "Last wall-clock": "上次总耗时",
    "Includes queue and model loading": "含排队与模型加载时间",
    "Decode speed": "解码速度",
    "Last decode speed": "上次解码速度",
    "Engine timing · tokens / second": "引擎计时 · token / 秒",
    "Load model": "加载模型",
    "Unload model": "卸载模型",
    "Model load": "模型加载",
    "Queue wait": "排队等待",
    "First token": "首 token 耗时",
    "Prompt tokens": "输入 token 数",
    "Output tokens": "输出 token 数",
    "Response format": "响应格式",
    "Request status": "请求状态",
    "Request list": "请求列表",
    "Request content": "请求内容",
    "Filter requests": "筛选请求",
    "Search ID or endpoint": "搜索 ID 或路径",
    "All statuses": "全部状态",
    "Active": "进行中",
    "In use": "使用中",
    "Completed": "已完成",
    "Errors": "错误",
    "Connect": "连接",
    "Only needed if the server has a key": "仅当服务端设置了密钥时才需要",
    "Inspect a request": "查看请求详情",
    "Select a request to see its input, output, and timings.":
      "选择一条请求，查看它的输入、输出与时序。",
    "Input": "输入",
    "Output": "输出",
    "Reasoning": "思考",
    "API response": "API 响应",
    "Last 100 requests kept in memory until restart. Inputs and outputs stay on this machine.":
      "内存中保留最近 100 条请求，重启即清空。输入与输出都不离开本机。",
    "Loaded": "已加载",
    "Unloaded": "未加载",
    "Loads automatically on the next request": "下次请求时自动加载",
    "JSON response": "JSON 响应",
    "No content.": "无内容。",
    "No matching requests.": "没有匹配的请求。",
    "No requests yet. API calls will appear here automatically.":
      "暂无请求。API 调用会自动出现在这里。",
    "Original request body.": "原始请求体。",
    "Model answer. JSON is formatted here for readability.": "模型回答。JSON 已格式化，便于阅读。",
    "Separate reasoning content, when enabled.": "单独的思考内容（启用时才有）。",
    "Streaming response: see Output and the request error, if any.":
      "流式响应：请看「输出」以及请求错误（如有）。",
    "Raw model output retained for diagnosis. The API returned an error.":
      "为诊断保留的原始模型输出。API 返回了错误。",
    "Clipboard unavailable; select the text to copy it.": "剪贴板不可用，请手动选中复制。",
    "Copied": "已复制",
    "Greedy: the same question gives the same answer.": "贪心：同样的问题得到同样的答案。",
    "thorough (default)": "深入（默认）",
    "KV cache": "KV 缓存",
    "Last switch": "上次切换",
    "Parked": "已停放",
    "The engine keeps the last conversation": "引擎会保留上一次对话",
    "The last one was cleared.": "上一次已清除。",
    "Restored": "已恢复",
    "answers right away": "立即回答",
    "Writing": "输出中",
    "just now": "刚刚",
    "Arguments": "参数",
    "(being written)": "（正在写入）",
    "Result": "结果",
    "characters": "字符",
    ", cut for the model": "，已为模型截断",
    "API response body.": "API 响应体。",
    "No active requests": "没有进行中的请求",
    "Disconnected · retrying": "已断开 · 重试中",
    "Loading…": "加载中…",
    "Unloading…": "卸载中…",
    "stream": "流式",
    "active": "进行中",
    "completed": "已完成",
    "loading": "加载中",
    "disconnected": "已断开",
    "Monitor capture truncated at 256K characters; the API response was not shortened.":
      "监控记录在 256K 字符处截断；API 响应本身未被截短。",
    "Shift+Enter: new line": "Shift+Enter 换行",
    "Model state": "模型状态",
    "Prefill this request": "本次预填充",
    "VRAM": "显存",
    "Prompt": "输入",
    "Their memory (RAM)": "它们的内存（RAM）",
    "New prompt tokens per second, excluding reused cache tokens (engine prompt time)":
      "每秒新读入的输入 token 数（不含复用缓存；按引擎输入计时）",
    "While writing the answer: the share of the experts looked up that were already in VRAM. Experts the GPU read over PCIe (--pcie-frac) are not counted in it; their share of all routed experts is shown after it (+N% PCIe)":
      "生成答案期间：被查专家中已在显存里的比例。GPU 经 PCIe（--pcie-frac）读取的专家不计入其中；它们在全部路由专家中的占比显示在其后（+N% PCIe）",
    "routed experts the GPU read over PCIe (--pcie-frac) or another GPU computed":
      "经 PCIe（--pcie-frac）读取、或由另一块 GPU 计算的专家",
    "The engine keeps the last conversation's state, so a follow-up reads only what is new. To keep several conversations (agents taking turns), add \"--conversation-cache-mib\", \"8192\" to the run config's args (docs/DETAILS.md).":
      "引擎会保留上一次对话的状态，追问时只读新增部分。若要同时保留多个会话（多个 agent 轮流用），在运行配置的 args 里加上 \"--conversation-cache-mib\"、\"8192\"（见 docs/DETAILS.md）。",
    "Their tools run on this PC with your rights, when the model decides to call them (in this page's chat only; switch it off in Sampling).":
      "当模型决定调用时，这些工具会以你的权限在本机运行（仅限本页面对话；可在「采样」里关闭）。",
    "Kept in the model's run config for every client. An empty field is the default. They take effect the next time the model starts (close Strata's window and start it again).":
      "保存在模型的运行配置里，对所有客户端生效。留空即为默认值。下次启动模型时生效（关闭 Strata 窗口再启动）。",
    "0 = always the most likely word (exact, repeatable)": "0 = 总是取最可能的词（确定、可复现）",
    "empty = until done": "留空 = 直到完成",
    "empty = random": "留空 = 随机",
    "the engine's control vector; off = the stock model. Switching reads the chat again once":
      "引擎的控制向量；关闭 = 原始模型。切换后会重新读一遍对话",
    "omp and other API clients get these settings for anything they don't set themselves":
      "omp 等 API 客户端在未自行设置时会沿用这些参数",
  },
};

// Templated strings: [pattern, replacement]. $1.. are the captured parts.
const I18N_PATTERNS = {
  "zh-CN": [
    [/^Thought for (.+) s$/, "思考了 $1 秒"],
    [/^Thinking…$/, "思考中…"],
    [/^(\S+) runs on this PC\. Nothing leaves it\.$/, "$1 在本机运行，数据不出本机。"],
    [/^(\d+) of (\d+) tokens$/, "已生成 $1 / $2 token"],
    [/^(\d+) active \/ queued$/, "$1 个进行中 / 排队中"],
    [/^Live · updated (.+)$/, "实时 · 更新于 $1"],
    [/^Request (.+)$/, "请求 $1"],
    [/^Reading prompt · (.+)%$/, "读取输入 · $1%"],
    [/^Generating · (.+) tok\/s$/, "生成中 · $1 tok/s"],
    [/^(\d+) queued$/, "$1 个排队中"],
    [/^(.+) tokens · (.+) tok\/s$/, "$1 token · $2 tok/s"],
    [/^(.+) tokens · (.+) tok\/s · stopped$/, "$1 token · $2 tok/s · 已停止"],
    [/^last: (.+) tokens at (.+) tok\/s$/, "上次：$1 token · $2 tok/s"],
    [/^last: (.+) tokens$/, "上次：$1 token"],
    [/^(\d+) tokens$/, "$1 token"],
    [/^last: (.+)$/, "上次：$1"],
    [/^Show all \((\d+)\)$/, "显示全部（$1）"],
    [/^of (\d+) W limit$/, "上限 $1 W"],
    [/^(\d+) cores · (\d+) threads$/, "$1 核 · $2 线程"],
    [/^to GPU (.+)$/, "→ GPU $1"],
    [/^write (.+)$/, "写入 $1"],
    [/^Since (.+): (\d+) requests · ([\d,]+) prompt tokens read at ([\d,.]+) tok\/s \(([\d,]+) reused\) · ([\d,]+) written at ([\d,.]+) tok\/s$/,
      "自 $1 起：$2 个请求 · 输入 token $3（$4 tok/s，复用 $5）· 输出 $6 个（$7 tok/s）"],
    [/^(\d+) of (\d+) requests reused part of their prompt$/, "$2 个请求里有 $1 个复用了部分输入"],
    [/^([\d,]+) of ([\d,]+) prompt tokens reused$/, "复用输入 token $1 / $2"],
    [/^(.+) tokens \((\d+)% of all prompt tokens\)$/, "$1 token（占全部输入 token 的 $2%）"],
  ],
};

// Areas whose text is content, never interface: chat transcript, reasoning, tool output,
// code. Text inside form fields is skipped too, but their placeholder/title/aria-label
// are interface hints, so the attribute pass uses a narrower list.
const I18N_SKIP = ".st-msg, pre, code, kbd, samp, textarea, [contenteditable], [data-i18n-skip]";
const I18N_SKIP_ATTR = ".st-msg, pre, code, [data-i18n-skip]";

const I18N_ATTRS = ["placeholder", "title", "aria-label", "alt"];

function i18nLang() {
  try {
    const saved = localStorage.getItem("strata.lang");
    if (saved) return saved;
  } catch (e) {}
  const nav = (navigator.language || "en").toLowerCase();
  return nav.startsWith("zh") ? "zh-CN" : "en";
}

function i18nTable(lang) {
  return I18N_TABLE[lang] || null;
}

// tr(): for the few interface strings the app writes into a skipped area (chat badges,
// tool blocks). Named tr, not t: the app uses `t` for a tool call object.
function tr(text) {
  const table = i18nTable(i18nLang());
  if (!table || text == null) return text;
  return i18nLookup(text, table, I18N_PATTERNS[i18nLang()] || []) || text;
}

function i18nLookup(text, table, patterns) {
  if (table[text] !== undefined) return table[text];
  // long hints carry newlines and indentation in the source: compare on collapsed spaces too
  const flat = text.replace(/\s+/g, " ").trim();
  if (flat && table[flat] !== undefined) return table[flat];
  for (const [re, rep] of patterns) {
    if (re.test(text)) return text.replace(re, rep);
    if (flat && flat !== text && re.test(flat)) return flat.replace(re, rep);
  }
  return null;
}

function i18nSkip(node, selector) {
  const el = node.nodeType === 1 ? node : node.parentElement;
  return !!(el && el.closest && el.closest(selector || I18N_SKIP));
}

function i18nTranslate(root, table, patterns) {
  if (!table || !root) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
  const texts = [];
  const els = [];
  let n;
  while ((n = walker.nextNode())) {
    if (n.nodeType === 3) texts.push(n);
    else els.push(n);
  }
  for (const node of texts) {
    if (i18nSkip(node)) continue;
    const raw = node.nodeValue;
    const key = raw.trim();
    if (!key) continue;
    const hit = i18nLookup(key, table, patterns);
    if (hit !== null && hit !== key) {
      if (node.__i18nEn === undefined) node.__i18nEn = raw;   // remember the English source
      node.nodeValue = raw.replace(key, hit);
    }
  }
  for (const el of els) {
    if (i18nSkip(el, I18N_SKIP_ATTR)) continue;
    for (const attr of I18N_ATTRS) {
      const v = el.getAttribute && el.getAttribute(attr);
      if (!v) continue;
      const hit = i18nLookup(v, table, patterns);
      if (hit !== null && hit !== v) {
        if (!el.__i18nEnAttr) el.__i18nEnAttr = {};
        if (el.__i18nEnAttr[attr] === undefined) el.__i18nEnAttr[attr] = v;
        el.setAttribute(attr, hit);
      }
    }
  }
}

// Switching back to English restores what the page shipped with: no reload, so nothing
// is lost mid-conversation and the button cannot get stuck.
function i18nRestore(root) {
  const walker = document.createTreeWalker(root || document.body, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
  let n;
  while ((n = walker.nextNode())) {
    if (n.nodeType === 3) {
      if (n.__i18nEn !== undefined) {
        if (n.nodeValue !== n.__i18nEn) n.nodeValue = n.__i18nEn;
        delete n.__i18nEn;
      }
    } else {
      if (n.__i18nEnAttr) {
        for (const attr of Object.keys(n.__i18nEnAttr)) n.setAttribute(attr, n.__i18nEnAttr[attr]);
        delete n.__i18nEnAttr;
      }
    }
  }
  document.documentElement.lang = "en";
}

function i18nApply(lang, root) {
  const table = i18nTable(lang);
  const patterns = I18N_PATTERNS[lang] || [];
  document.documentElement.lang = lang;
  if (!table) return;
  i18nTranslate(root || document.body, table, patterns);
}

(function i18nBoot() {
  let lang = i18nLang();
  let observer = null;

  function observeOff() { if (observer) observer.disconnect(); }
  function observeOn() {
    if (observer) observer.observe(document.documentElement, { childList: true, subtree: true, characterData: true });
  }
  // Never translate while our own writes are being recorded: that is what made a page spin.
  function silently(fn) { observeOff(); try { fn(); } finally { observeOn(); } }

  function setLang(next) {
    lang = next;
    try { localStorage.setItem("strata.lang", lang); } catch (e) {}
    updateButton();
    if (lang === "en") silently(() => i18nRestore(document.body));
    else silently(() => i18nApply(lang));
  }

  let btn = null;
  function updateButton() {
    if (!btn) return;
    btn.textContent = lang === "en" ? "中文" : "English";
    btn.title = lang === "en" ? "切换到中文" : "切换到英文（Switch to English）";
    btn.setAttribute("aria-label", btn.title);
  }

  function installButton() {
    const ref = document.getElementById("theme-btn") || document.getElementById("theme");
    if (!ref || document.getElementById("lang-btn")) return;
    btn = document.createElement("button");
    btn.id = "lang-btn";
    btn.type = "button";
    btn.className = ref.className || "st-btn st-btn--icon";
    btn.addEventListener("click", () => setLang(lang === "en" ? "zh-CN" : "en"));
    ref.parentNode.insertBefore(btn, ref.nextSibling);
    updateButton();
  }

  function start() {
    installButton();
    observer = new MutationObserver((records) => {
      if (lang === "en") return;
      const table = i18nTable(lang);
      const patterns = I18N_PATTERNS[lang] || [];
      observeOff();
      try {
        for (const rec of records) {
          for (const node of rec.addedNodes) {
            if (node.nodeType === 1) i18nTranslate(node, table, patterns);
            else if (node.nodeType === 3 && !i18nSkip(node)) {
              const raw = node.nodeValue, key = raw.trim();
              const hit = key && i18nLookup(key, table, patterns);
              if (hit !== null && hit !== key) {
                if (node.__i18nEn === undefined) node.__i18nEn = raw;
                node.nodeValue = raw.replace(key, hit);
              }
            }
          }
          if (rec.type === "characterData" && rec.target && !i18nSkip(rec.target)) {
            const raw = rec.target.nodeValue || "", key = raw.trim();
            const hit = key && i18nLookup(key, table, patterns);
            if (hit !== null && hit !== key) {
              if (rec.target.__i18nEn === undefined) rec.target.__i18nEn = raw;
              rec.target.nodeValue = raw.replace(key, hit);
            }
          }
        }
      } finally { observeOn(); }
    });
    observeOn();
    if (lang !== "en") silently(() => i18nApply(lang));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
