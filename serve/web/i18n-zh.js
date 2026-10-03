/* Strata 界面中英对照汉化层 (Chinese / English bilingual UI layer)
 *
 * 用法：在 index.html / monitor.html 里 app.js 之后引入本文件即可，不改动原程序逻辑。
 * 特点：
 *   - 顶栏最左侧自动注入 "Language" 按钮：英文 / 简体中文 / 中英文 三种显示模式（选择会记住）
 *   - 词典精确匹配 + 正则规则匹配动态文案（计数、速度、状态等）
 *   - MutationObserver 跟随界面刷新自动翻译
 *   - 绝不翻译模型输出（聊天正文、思考内容、代码块、工具原始参数）
 *   - 中英模式下：中文一行，英文小字在下一行
 */
(function () {
  "use strict";

  // ---- 精确词典：原文 -> 中文 ----------------------------------------------
  const ZH = {
    // 顶栏 / 通用
    "Views": "视图",
    "Chat": "聊天",
    "Monitor": "监控",
    "About": "关于",
    "Connecting…": "正在连接…",
    "Connected": "已连接",
    "Disconnected": "已断开",
    "Server not reachable": "无法连接服务器",
    "Starting": "正在启动",
    "Waiting": "等待中",
    "Idle": "空闲",
    "Reading": "读取中",
    "Generating": "生成中",
    "Queued": "排队中",
    "Error": "出错",
    "Failed": "失败",
    "Stopped": "已停止",
    "Closed": "已断开",
    "Done": "完成",
    "Running": "运行中",
    "Writing": "输出中",
    "Not run": "未执行",
    "Light / dark": "亮色 / 暗色",
    "Switch between light and dark": "切换亮色 / 暗色主题",

    // 聊天
    "Ask anything": "随便问点什么",
    "The model runs on this PC. Nothing leaves it.": "模型在本机运行，内容不会离开这台电脑。",
    "Ask anything…": "问点什么…",
    "Message": "消息",
    "Attach a text file (or drop it here)": "附加文本文件（也可拖到这里）",
    "Attach a text file or a picture (or drop it here)": "附加文本文件或图片（也可拖到这里）",
    "Attach a file": "附加文件",
    "New chat": "新对话",
    "Save this chat as Markdown": "把对话保存为 Markdown",
    "Save this chat": "保存对话",
    "Sampling and thinking": "采样与思考",
    "Stop": "停止",
    "Send": "发送",
    "Shift+Enter: new line": "Shift+Enter 换行",
    "Stop the answer first.": "请先停止当前回答。",
    "Still writing": "正在输出",
    "The request failed": "请求失败",
    "Copied to clipboard": "已复制到剪贴板",
    "Copy code": "复制代码",
    "Copy the answer": "复制回答",
    "Copy": "复制",
    "Remove": "移除",
    "Undo": "撤销",
    "Nothing to save yet": "还没有内容可保存",
    "The last one was cleared.": "上一个对话已清除。",
    "Pictures are off": "图片功能未开启",
    "This model was set up for text only.": "当前模型只配置了文字模式。",
    "Picture too large": "图片过大",
    "Not a text file": "不是文本文件",
    "File too large": "文件过大",
    "Thoughts": "思考过程",
    "Thinking…": "思考中…",
    "Thinking": "思考",
    "Show thinking": "显示思考过程",
    "expanded while it streams": "流式输出时自动展开",
    "answers right away": "立即作答",
    "thorough (default)": "详尽（默认）",
    "short": "简短",
    "medium": "适中",

    // 采样抽屉
    "Sampling": "采样",
    "Off": "关",
    "Low": "低",
    "Medium": "中",
    "High": "高",
    "Temperature": "温度",
    "0 = always the most likely word (exact, repeatable)": "0 = 总是选最可能的词（精确、可复现）",
    "Max tokens": "最大 token 数",
    "empty = until done": "留空 = 直到答完",
    "Until done": "直到答完",
    "Seed": "随机种子",
    "empty = random": "留空 = 随机",
    "Random": "随机",
    "Use tools from MCP servers": "使用 MCP 服务器的工具",
    "the model may call them while it answers": "模型回答时可能自行调用",
    "Experimental speed projection": "实验性速度投影",
    "Use for other apps too": "同时供其他应用使用",
    "Other apps use their own settings again.": "其他应用恢复使用各自的设置。",
    "omp and other API clients get these settings for anything they don't set themselves":
      "omp 及其他 API 客户端在未自行指定参数时，会采用这里的设置",
    "the engine's control vector; off = the stock model. Switching reads the chat again once":
      "引擎的控制向量；关闭 = 原始模型。切换后会重新读取一次对话",
    "the model may call them while it answers": "模型回答时可能自行调用",
    "Sampling saved": "采样设置已保存",
    "Saved here, but not for other apps": "只在本页生效，未应用到其他应用",
    "Greedy: the same question gives the same answer.": "贪心模式：同样的问题会得到同样的答案。",
    "Reset": "重置",
    "Apply": "应用",

    // 监控
    "Model state": "模型状态",
    "Waiting for a request": "等待请求",
    "Reading prompt": "读取提示词",
    "Context fill": "上下文占用",
    "Experts in VRAM": "显存中的专家",
    "System RAM": "系统内存",
    "GPU temperature": "GPU 温度",
    "Recent requests": "最近的请求",
    "Show all": "显示全部",
    "Show fewer": "收起",
    "Time": "时间",
    "Status": "状态",
    "Prompt": "提示词",
    "Reused": "复用",
    "Output": "输出",
    "Hit rate": "命中率",
    "While writing the answer: the share of the experts looked up that were already in VRAM (experts copied over PCIe are not counted)":
      "回答过程中：查到的专家里已在显存中的比例（经 PCIe 拷贝进去的不计入）",
    "Duration": "耗时",
    "No requests yet": "暂无请求",
    "MCP servers": "MCP 服务器",
    "Speed": "速度",
    "GPU load": "GPU 负载",
    "VRAM": "显存",
    "GPU temp": "GPU 温度",
    "Power": "功耗",
    "Disk read": "磁盘读取",
    "Decode": "解码",
    "Decode now": "当前解码",
    "Decode last request": "上次请求解码",
    "Prefill": "预填充",
    "Prefill now": "当前预填充",
    "Prefill this request": "本次请求预填充",
    "Prefill last request": "上次请求预填充",
    "Their tools run on this PC with your rights, when the model decides to call them (in this page's chat only; switch it off in Sampling).":
      "模型决定调用时，这些工具会以你的权限在本机运行（仅本页聊天有效；可在「采样」里关闭）。",

    // 关于 / 设置
    "Model and engine": "模型与引擎",
    "This PC": "本机",
    "Connect your tools": "连接你的工具",
    "Any OpenAI- or Anthropic-compatible client works with these addresses.":
      "任何兼容 OpenAI 或 Anthropic 的客户端都可以使用下面的地址。",
    "Settings": "设置",
    "API key": "API 密钥",
    "only if the server was started with one": "仅当启动服务器时设置了密钥",
    "Not needed": "不需要",
    "Dark theme": "暗色主题",
    "Chats, settings and the key are kept in this browser only.": "对话、设置和密钥只保存在这个浏览器中。",
    "Kept in this browser only.": "只保存在这个浏览器中。",
    "API key saved": "API 密钥已保存",
    "API key needed": "需要 API 密钥",
    "This server needs a key: add it under About > Settings.": "该服务器需要密钥：请在「关于 › 设置」中填写。",
    "This server needs an API key: add it under About > Settings.": "该服务器需要 API 密钥：请在「关于 › 设置」中填写。",
    "OpenAI base URL": "OpenAI 接口地址",
    "Anthropic base URL": "Anthropic 接口地址",
    "Context": "上下文",
    "Engine": "引擎",
    "Model": "模型",
    "Model name": "模型名称",
    "KV cache": "KV 缓存",
    "Speculation": "投机解码",
    "Images": "图片",
    "on": "开",
    "off": "关",
    "Projection": "投影",
    "built from source": "源码编译",

    // 监控页 (monitor.html / monitor.js)
    "Strata API Monitor": "Strata 监控台",
    "Local inference": "本地推理",
    "API Monitor": "API 监控",
    "Model status and every request, in one place.": "模型状态与全部请求，集中呈现。",
    "Load model": "加载模型",
    "Unload model": "卸载模型",
    "Connect": "连接",
    "Requests retained": "保留请求数",
    "Waiting for server": "等待服务器",
    "Last wall-clock": "上次总耗时",
    "Includes queue and model loading": "含排队与模型加载时间",
    "Last decode speed": "上次解码速度",
    "Engine timing · tokens / second": "引擎计时 · tokens/秒",
    "Requests": "请求",
    "All statuses": "全部状态",
    "Active": "进行中",
    "Completed": "已完成",
    "Errors": "错误",
    "Inspect a request": "查看请求详情",
    "Select a request to see its input, output, and timings.": "选择一条请求，查看其输入、输出与计时。",
    "Wall-clock": "总耗时",
    "Model load": "模型加载",
    "Queue wait": "排队等待",
    "First token": "首个 token",
    "Prompt tokens": "提示词 tokens",
    "Output tokens": "输出 tokens",
    "Decode speed": "解码速度",
    "Response format": "响应格式",
    "Input": "输入",
    "Reasoning": "推理内容",
    "API response": "API 响应",
    "JSON response": "JSON 响应",
    "Loaded": "已加载",
    "Loading…": "加载中…",
    "Unloaded": "已卸载",
    "Unloading…": "卸载中…",
    "Loads automatically on the next request": "下次请求时自动加载",
    "In use": "使用中",
    "No active requests": "暂无进行中的请求",
    "No content.": "无内容。",
    "No matching requests.": "没有匹配的请求。",
    "No requests yet. API calls will appear here automatically.": "还没有请求。API 调用会自动出现在这里。",
    "Original request body.": "原始请求体。",
    "Model answer. JSON is formatted here for readability.": "模型回答。JSON 已格式化以便阅读。",
    "Raw model output retained for diagnosis. The API returned an error.": "保留的原始模型输出，用于诊断。API 返回了错误。",
    "Separate reasoning content, when enabled.": "单独的推理内容（启用时）。",
    "Streaming response: see Output and the request error, if any.": "流式响应：请查看「输出」与请求错误（如有）。",
    "Clipboard unavailable; select the text to copy it.": "剪贴板不可用；请手动选中文本复制。",
    "Disconnected · retrying": "已断开 · 重试中",
    "Switch color theme": "切换配色主题",
    "Only needed if the server has a key": "仅在服务器设置了密钥时才需要",
    "Filter requests": "筛选请求",
    "Search ID or endpoint": "搜索 ID 或接口路径",
    "Request status": "请求状态",
    "Request list": "请求列表",
  };

  // 徽章专用（同一个词在表头/徽章里含义不同）
  const BADGE = {
    "Max tokens": "达到上限",
  };

  // ---- 正则规则：动态文案 --------------------------------------------------
  const RULES = [
    [/^You · (.+)$/, (m) => `你 · ${m[1]}`],
    [/^Reading prompt · (\d+)%$/, (m) => `读取提示词 · ${m[1]}%`],
    [/^Generating · ([\d.,]+) ?tok\/s$/, (m) => `生成中 · ${m[1]} tok/s`],
    [/^Thought for ([\d.,]+) s$/, (m) => `思考了 ${m[1]} 秒`],
    [/^Show all \((\d+)\)$/, (m) => `显示全部（${m[1]}）`],
    [/^last: ([\d,]+) tokens( at ([\d.,]+) tok\/s)?$/, (m) => `上次：${m[1]} tokens${m[2] || ""}`],
    [/^(\d+) tools · (\d+) of (\d+) servers connected$/, (m) => `${m[1]} 个工具 · ${m[3]} 个服务器中 ${m[2]} 个已连接`],
    [/^(\d+) tools from (.+?); the model calls them when it decides to$/, (m) => `${m[1]} 个工具，来自 ${m[2]}；模型会自行决定何时调用`],
    [/^(\d+) characters$/, (m) => `${m[1]} 个字符`],
    [/^([\d,]+) experts cached$/, (m) => `${m[1]} 个专家已缓存`],
    [/^of ([\d.]+) W limit$/, (m) => `上限 ${m[1]} W`],
    [/^write ([\d.]+) MB\/s$/, (m) => `写入 ${m[1]} MB/s`],
    [/^to GPU ([\d.]+) MB\/s( · idle Gen(\d+))?$/, (m) => `到 GPU ${m[1]} MB/s${m[2] || ""}`],
    [/^(\d+) cores · (\d+) threads$/, (m) => `${m[1]} 核 · ${m[2]} 线程`],
    [/^(\d+) threads$/, (m) => `${m[1]} 线程`],
    [/^needs psutil \(setup installs it\)$/, () => "需要 psutil（安装程序会装）"],
    [/^Since (.+): ([\d,]+) requests · ([\d,]+) prompt tokens read( at ([\d.,]+) tok\/s)? \(([\d,]+) reused\) · ([\d,]+) written( at ([\d.,]+) tok\/s)?$/,
      (m) => `自 ${m[1]} 起：${m[2]} 次请求 · 读取提示词 ${m[3]} tokens${m[4] || ""}（复用 ${m[6]}）· 生成 ${m[7]} tokens${m[8] || ""}`],
    [/^(\d+) tokens$/, (m) => `${m[1]} tokens`],
    [/^MTP drafts up to (\d+) tokens(, prompt lookup on)?$/, (m) => `MTP 最多草拟 ${m[1]} 个 token${m[2] ? "，提示词查找已开启" : ""}`],
    [/^(.+), all in VRAM$/, (m) => `${m[1]}，全部在显存`],
    [/^(.+), streamed: ([\d,]+) positions per layer in VRAM, the rest in RAM$/, (m) => `${m[1]}，每层 ${m[2]} 个位置流式放显存，其余在内存`],
    [/^experimental speed projection (on|off)$/, (m) => `实验性速度投影：${m[1] === "on" ? "开" : "关"}`],
    [/^image$/, () => "图片"],
    [/^(.+) runs on this PC\. Nothing leaves it\.$/, (m) => ({ zh: `${m[1]} 在这台电脑上本地运行，内容不会离开本机`, en: true })],
  ];

  // ---- DOM 处理 ------------------------------------------------------------
  // 绝不翻译的容器：模型输出、代码、输入框
  const SKIP = "pre, code, textarea, input, select, script, style, svg, .st-bubble, .thinking, .tool-call__pre, .i18n-en, [data-i18n]";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function blocked(node) {
    for (let el = node.parentElement; el; el = el.parentElement) {
      if (el.matches && el.matches(SKIP)) return true;
    }
    return false;
  }

  // 返回 {zh, en}；en=true 表示可以中英双显（只有词典命中才双显，数值型规则不双显）
  function lookup(key, isBadge) {
    if (isBadge && BADGE[key]) return { zh: BADGE[key], en: false };
    if (ZH[key] !== undefined) return { zh: ZH[key], en: true };
    const norm = key.replace(/\s+/g, " ");           // HTML 源码里的换行/缩进
    if (norm !== key) {
      if (isBadge && BADGE[norm]) return { zh: BADGE[norm], en: false };
      if (ZH[norm] !== undefined) return { zh: ZH[norm], en: true };
    }
    for (const [re, fn] of RULES) {
      const m = key.match(re);
      if (m) { const r = fn(m); return typeof r === "string" ? { zh: r, en: false } : r; }
    }
    return null;
  }

  // ---- 语言模式 ------------------------------------------------------------
  // both = 中文在上、英文在下（默认）  zh = 只显示中文  en = 只显示英文（还原原界面）
  const MODE_KEY = "strata.lang";
  // [值, 中文名, 英文名]
  const LANGS = [["en", "英文", "English"], ["zh", "简体中文", "Simplified Chinese"],
                 ["both", "中英文", "Chinese + English"]];
  const BTN_LABEL = { zh: "语言", en: "Language", both: "语言 Language" };
  // 按当前显示模式取文案：英文模式全英文，纯中文模式全中文，中英模式中英并列
  function labelFor(l, m) {
    if (m === "en") return l[2];
    if (m === "zh") return l[1];
    return `${l[1]} ${l[2]}`;
  }
  let mode = "both";
  try {
    const saved = localStorage.getItem(MODE_KEY);
    if (LANGS.some((l) => l[0] === saved)) mode = saved;
  } catch (e) {}
  function translateText(node) {
    if (!node.nodeValue || blocked(node)) return;
    const raw = node.nodeValue;
    const key = raw.trim();
    if (key.length < 2 || key.length > 220) return;
    const isBadge = !!(node.parentElement && node.parentElement.classList.contains("st-badge"));
    const hit = lookup(key, isBadge);
    if (!hit || !hit.zh || hit.zh === key) return;

    const at = raw.indexOf(key);
    const span = document.createElement("span");
    span.setAttribute("data-i18n", "1");
    // 中英两份都留在 DOM 里，由 data-lang 的 CSS 决定显示哪一份；nodual = 中英模式下也只显示中文
    span.innerHTML =
      `<span class="i18n-zh">${esc(hit.zh)}</span>` +
      `<span class="i18n-en${hit.en ? "" : " i18n-nodual"}">${esc(key)}</span>`;
    if (at > 0) node.parentNode.insertBefore(document.createTextNode(raw.slice(0, at)), node);
    node.parentNode.replaceChild(span, node);
    if (at + key.length < raw.length) span.parentNode.insertBefore(document.createTextNode(raw.slice(at + key.length)), span.nextSibling);
  }

  const ATTRS = ["placeholder", "title", "aria-label"];
  function translateAttrs(root) {
    const els = root.querySelectorAll ? root.querySelectorAll("*") : [];
    for (const el of els) {
      for (const a of ATTRS) {
        const store = "data-i18n-orig-" + a;
        const orig = el.getAttribute(store);
        const v = orig != null ? orig : el.getAttribute(a);
        if (!v) continue;
        const key = v.trim();
        const hit = lookup(key, false);
        if (!hit || !hit.zh || hit.zh === key) continue;
        if (orig == null) el.setAttribute(store, v);            // 记住原文，便于切回英文
        el.setAttribute(a, mode === "en" ? v : hit.zh);
      }
    }
  }

  const walker = () => document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
  function translateAll() {
    const w = walker(), todo = [];
    while (w.nextNode()) todo.push(w.currentNode);
    for (const n of todo) translateText(n);
    translateAttrs(document.body);
    const t = document.querySelector("title");
    if (t) {
      if (t.dataset.i18nOrig == null && t.textContent === "Strata API Monitor") t.dataset.i18nOrig = t.textContent;
      if (t.dataset.i18nOrig) t.textContent = mode === "en" ? t.dataset.i18nOrig : "Strata 监控台";
    }
  }

  // ---- 语言开关按钮（自动注入到顶栏最左侧） ---------------------------------
  function setMode(m) {
    mode = m;
    try { localStorage.setItem(MODE_KEY, m); } catch (e) {}
    document.documentElement.dataset.lang = m;
    translateAttrs(document.body);
    syncSwitch();
  }

  let switchMenu = null, switchBtn = null;
  // 按钮与菜单文案跟着当前显示模式实时同步
  function syncSwitch() {
    if (!switchMenu || !switchBtn) return;
    const lab = switchBtn.querySelector(".i18n-switch__label");
    if (lab) lab.textContent = BTN_LABEL[mode] || "Language";
    switchBtn.setAttribute("aria-label", mode === "en" ? "Language" : "语言 / Language");
    for (const b of switchMenu.querySelectorAll("[data-lang]")) {
      const l = LANGS.find((x) => x[0] === b.dataset.lang);
      if (l) b.textContent = labelFor(l, mode);
      b.setAttribute("aria-selected", String(b.dataset.lang === mode));
    }
  }

  function buildSwitch() {
    const host = document.querySelector("header");
    if (!host || host.querySelector(".i18n-switch")) return;
    const wrap = document.createElement("div");
    wrap.className = "i18n-switch";
    wrap.setAttribute("data-i18n", "1");                        // 本控件自身不参与翻译
    wrap.innerHTML =
      '<button type="button" class="i18n-switch__btn" aria-haspopup="listbox" aria-expanded="false">' +
      '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">' +
      '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a15 15 0 0 1 0 18a15 15 0 0 1 0-18"/></svg>' +
      '<span class="i18n-switch__label">Language</span></button>' +
      '<div class="i18n-switch__menu" role="listbox" hidden>' +
      LANGS.map((l) => `<button type="button" role="option" data-lang="${l[0]}"></button>`).join("") +
      "</div>";
    switchMenu = wrap.querySelector(".i18n-switch__menu");
    const btn = wrap.querySelector(".i18n-switch__btn");
    switchBtn = btn;
    btn.onclick = (e) => {
      e.stopPropagation();
      const open = switchMenu.hidden;
      switchMenu.hidden = !open;
      btn.setAttribute("aria-expanded", String(open));
    };
    switchMenu.onclick = (e) => {
      const b = e.target.closest("[data-lang]");
      if (!b) return;
      e.stopPropagation();
      setMode(b.dataset.lang);
      switchMenu.hidden = true;
      btn.setAttribute("aria-expanded", "false");
    };
    document.addEventListener("click", () => {
      if (!switchMenu.hidden) { switchMenu.hidden = true; btn.setAttribute("aria-expanded", "false"); }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !switchMenu.hidden) { switchMenu.hidden = true; btn.setAttribute("aria-expanded", "false"); }
    });
    host.insertBefore(wrap, host.firstChild);                   // 顶栏最左边
    syncSwitch();
  }

  // ---- 跟随界面刷新 --------------------------------------------------------
  // 在 MutationObserver 回调里同步翻译：回调在绘制前执行，界面不会闪回英文
  let busy = false;
  function schedule() {
    if (busy) return;
    busy = true;
    try { translateAll(); } finally { busy = false; }
  }

  const CSS =
    // 中英对照：中文一行，英文小字在下一行
    'html[data-lang="both"] .i18n-en{display:block;font-size:.72em;font-weight:400;opacity:.55;' +
      "line-height:1.25;margin-top:1px;letter-spacing:0;text-transform:none}" +
    // 固定高度的小控件保持行内，避免撑破布局
    'html[data-lang="both"] .st-tab .i18n-en,html[data-lang="both"] .st-btn .i18n-en,' +
      'html[data-lang="both"] .seg .i18n-en,html[data-lang="both"] .st-badge .i18n-en,' +
      'html[data-lang="both"] .st-composer__bar .i18n-en,html[data-lang="both"] .composer-hint .i18n-en' +
      "{display:inline;margin:0 0 0 .35em;font-size:.85em}" +
    // 数值/徽章类在中英模式下也只显示中文
    'html[data-lang="both"] .i18n-en.i18n-nodual{display:none}' +
    // 纯中文
    'html[data-lang="zh"] .i18n-en{display:none}' +
    // 纯英文：还原原界面
    'html[data-lang="en"] .i18n-zh{display:none}' +
    'html[data-lang="en"] .i18n-en{display:inline;font-size:inherit;font-weight:inherit;' +
      "opacity:1;line-height:inherit;margin:0;letter-spacing:inherit;text-transform:inherit}" +
    // 语言开关按钮
    ".i18n-switch{position:relative;margin-right:var(--st-s-3,10px);flex:none}" +
    ".i18n-switch__btn{display:inline-flex;align-items:center;gap:6px;height:36px;padding:0 12px;" +
      "border:1px solid var(--st-line,#d8dce0);border-radius:10px;background:var(--st-surface-2,#f4f6f8);" +
      "color:var(--st-ink-muted,#5b6570);font:500 14px/1 var(--st-font,system-ui);cursor:pointer}" +
    ".i18n-switch__btn:hover{color:var(--st-ink,#111);border-color:var(--st-accent,#4a90d9)}" +
    ".i18n-switch__menu{position:absolute;top:42px;left:0;z-index:60;min-width:148px;padding:5px;" +
      "background:var(--st-surface,#fff);border:1px solid var(--st-line,#d8dce0);border-radius:12px;" +
      "box-shadow:0 8px 26px rgba(0,0,0,.14)}" +
    ".i18n-switch__menu button{display:block;width:100%;text-align:left;padding:8px 10px;border:0;" +
      "border-radius:8px;background:transparent;color:var(--st-ink,#111);font:400 14px var(--st-font,system-ui);cursor:pointer}" +
    ".i18n-switch__menu button:hover{background:var(--st-surface-2,#f1f3f5)}" +
    '.i18n-switch__menu button[aria-selected="true"]{color:var(--st-accent,#2b7fd4);font-weight:700}';

  function start() {
    document.documentElement.dataset.lang = mode;               // 首帧就定好显示模式，避免闪烁
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);
    translateAll();
    buildSwitch();
    new MutationObserver((muts) => {
      for (const m of muts) {
        if (m.type === "characterData" && m.target.parentElement &&
            m.target.parentElement.closest && m.target.parentElement.closest("[data-i18n], .i18n-en")) continue;
        schedule();
        break;
      }
    }).observe(document.body, { childList: true, subtree: true, characterData: true });
  }

  window.StrataZH = { translateAll, setMode, getMode: () => mode, dictionary: ZH };   // 便于调试
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
