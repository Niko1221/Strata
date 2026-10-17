/*!
 * Strata WebUI — runtime i18n
 * ---------------------------------------------------------------------------
 * Why runtime translation instead of editing every string in the HTML/JS:
 *   - zero intrusion: the only change to the existing files is one <script> tag
 *     in index.html and monitor.html; no markup or logic is touched
 *   - it also covers strings app.js builds at runtime (statuses, toasts,
 *     metric labels) — those never appear in the static HTML
 *   - adding a language is one object in LOCALES, nothing else
 *
 * How it works:
 *   - `navigator.language` picks the locale; localStorage remembers a manual
 *     choice; `en` is the identity locale (strings are shown as written)
 *   - exact matches, then regex patterns for strings with numbers in them,
 *     then a guarded partial pass for labels app.js assembles from fragments
 *   - a MutationObserver keeps translating as app.js re-renders
 *
 * Switching at runtime:
 *   StrataI18N.setLang('zh')      // remembers the choice and reloads
 *   StrataI18N.getLang()
 *
 * License: same as Strata.
 */
(function () {
  'use strict';

  var STORE_KEY = 'strata.lang';

  /* ===================== locales =====================
   * `en` is the identity locale: no entry means "leave the text as it is".
   * To add a language, copy `zh` and translate the values.
   */
  var LOCALES = {

    en: {},

    zh: {
      /* ---- header / tabs ---- */
      'Chat': '对话',
      'Monitor': '监控',
      'About': '关于',
      'Connecting\u2026': '连接中…',
      'Connecting&hellip;': '连接中…',
      'Views': '视图',

      /* ---- chat ---- */
      'Ask anything': '随便问点什么',
      'Ask anything\u2026': '随便问点什么…',
      'The model runs on this PC. Nothing leaves it.': '模型跑在这台电脑上，数据不出本机。',
      'Stop': '停止',
      'Send': '发送',
      'Message': '消息',
      'New chat': '新对话',
      'Save this chat': '保存本次对话',
      'Save this chat as Markdown': '保存为 Markdown',
      'Attach a file': '附加文件',
      'Attach a text file (or drop it here)': '附加文本文件（也可拖到这里）',
      'Attach a text file or a picture (or drop it here)': '附加文本文件或图片（也可拖到这里）',
      'Sampling and thinking': '采样与思考',

      /* ---- model state ---- */
      'Model state': '模型状态',
      'Idle': '空闲',
      'Reading': '读取中',
      'Generating': '生成中',
      'Queued': '排队中',
      'Error': '错误',
      'Waiting for a request': '等待请求',
      'Waiting for server': '等待服务器',
      'Server not reachable': '无法连接服务器',
      'the engine reported an error': '引擎报错',
      'The request failed': '请求失败',
      'Still writing': '仍在输出',
      'Stop the answer first.': '请先停止当前回答。',

      /* ---- monitor cards ---- */
      'Context fill': '上下文占用',
      'Experts in VRAM': '显存中的专家',
      'System RAM': '系统内存',
      'GPU temperature': 'GPU 温度',
      'Recent requests': '最近请求',
      'Show all': '显示全部',
      'Show fewer': '收起',
      'No requests yet': '暂无请求',
      'Time': '时间',
      'Status': '状态',
      'Prompt': '提示',
      'Reused': '复用',
      'Output': '输出',
      'Prefill t/s': '预填充 t/s',
      'Decode t/s': '解码 t/s',
      'VRAM hit rate': '显存命中率',
      'Duration': '耗时',
      'Conversation cache': '对话缓存',
      'Parked conversations': '停放的对话',
      'Their memory (RAM)': '其占用内存 (RAM)',
      'MCP servers': 'MCP 服务器',
      'Decode': '解码',
      'Prefill': '预填充',
      'Decode now': '当前解码',
      'Decode last request': '上次解码',
      'Prefill now': '当前预填充',
      'Prefill this request': '本次预填充',
      'Prefill last request': '上次预填充',
      'GPU load': 'GPU 负载',
      'GPU temp': 'GPU 温度',
      'Disk read': '磁盘读取',
      'KV cache': 'KV 缓存',
      'Max tokens': '最大 token 数',
      'Last request': '上次请求',
      'Reused since start': '启动以来复用',
      'Parked / restored': '停放 / 恢复',
      'Last switch': '上次切换',
      'The engine keeps the last conversation': '引擎保留最近一段对话',
      'not available': '不可用',
      'not available on Windows AMD yet': 'Windows AMD 上暂不可用',
      'not readable (NVML)': '无法读取 (NVML)',
      'needs psutil (setup installs it)': '需要 psutil（安装脚本会装）',
      'built from source': '源码编译',

      /* ---- about ---- */
      'Model and engine': '模型与引擎',
      'This PC': '本机',
      'Connect your tools': '接入你的工具',
      'Any OpenAI- or Anthropic-compatible client works with these addresses.':
        '任何兼容 OpenAI 或 Anthropic 的客户端都可用下面这些地址。',
      'Settings': '设置',
      'API key': 'API 密钥',
      'only if the server was started with one': '仅在服务端启用了密钥时需要',
      'Dark theme': '深色主题',
      'Chats, settings and the key are kept in this browser only.': '对话、设置和密钥都只保存在本浏览器里。',
      'Strata on GitHub': 'Strata 的 GitHub 仓库',
      'Model settings': '模型设置',
      'Save': '保存',
      'Not saved': '未保存',
      'Nothing changed.': '没有改动。',
      'Kept in this browser only.': '仅保存在本浏览器。',
      'API key saved': 'API 密钥已保存',
      'API key needed': '需要 API 密钥',
      'This server needs a key: add it under About > Settings.': '本服务需要密钥：请在「关于 > 设置」里填入。',
      'This server needs an API key: add it under About > Settings.': '本服务需要 API 密钥：请在「关于 > 设置」里填入。',
      'OpenAI base URL': 'OpenAI 接口地址',
      'Anthropic base URL': 'Anthropic 接口地址',
      'Model name': '模型名称',
      'no server is connected yet (see the Monitor)': '尚未连接服务器（见「监控」）',

      /* ---- sampling drawer ---- */
      'Sampling': '采样',
      'Thinking': '思考',
      'Off': '关闭',
      'Low': '低',
      'Medium': '中',
      'High': '高',
      'Temperature': '温度',
      '0 = always the most likely word (exact, repeatable)': '0 = 永远选最可能的词（确定、可复现）',
      'Top-p': 'Top-p',
      'Top-k': 'Top-k',
      'empty = until done': '留空 = 直到结束',
      'Until done': '直到结束',
      'Seed': '随机种子',
      'empty = random': '留空 = 随机',
      'Random': '随机',
      'Show thinking': '显示思考过程',
      'expanded while it streams': '流式输出时展开',
      'Use tools from MCP servers': '使用 MCP 服务器的工具',
      'the model may call them while it answers': '模型回答时可能调用这些工具',
      'Experimental speed projection': '实验性速度投影',
      'Use for other apps too': '同时供其他应用使用',
      'Reset': '重置',
      'Apply': '应用',
      'answers right away': '直接作答',
      'thorough (default)': '充分思考（默认）',
      'Sampling saved': '采样设置已保存',
      'Other apps (omp, API clients) use these settings from their next request.':
        '其他应用（omp、API 客户端）将从下一次请求起使用这些设置。',
      'Other apps use their own settings again.': '其他应用恢复使用各自的设置。',
      'Saved here, but not for other apps': '已保存到本页，但不影响其他应用',
      'Greedy: the same question gives the same answer.': '贪婪模式：同一个问题永远给同一个答案。',

      /* ---- messages ---- */
      'You \u00b7': '你 ·',
      'Thought for': '思考用时',
      'Arguments': '参数',
      'Writing': '写入中',
      'Copy': '复制',
      'Copy code': '复制代码',
      'Copy the answer': '复制回答',
      'Copied to clipboard': '已复制到剪贴板',
      'Clipboard unavailable; select the text to copy it.': '剪贴板不可用，请手动选中文本复制。',
      'Not run': '未执行',
      'Shift+Enter: new line': 'Shift+Enter：换行',
      'Nothing to save yet': '还没有可保存的内容',
      'The last one was cleared.': '上一段已清空。',
      'Pictures are off': '图片功能已关闭',
      'This model was set up for text only.': '该模型仅配置为文本模式。',
      'Picture too large': '图片过大',
      'pasted image': '粘贴的图片',
      'Not a text file': '不是文本文件',
      'File too large': '文件过大',
      'just now': '刚刚',

      /* ---- api monitor ---- */
      'Strata API Monitor': 'Strata API 监控',
      'Local inference': '本地推理',
      'API Monitor': 'API 监控',
      'Model status and every request, in one place.': '模型状态与全部请求，一处看全。',
      'Load model': '加载模型',
      'Unload model': '卸载模型',
      'Connect': '连接',
      'Model': '模型',
      'Requests retained': '保留的请求数',
      'Last wall-clock': '上次总耗时',
      'Includes queue and model loading': '含排队与模型加载时间',
      'Last decode speed': '上次解码速度',
      'Engine timing \u00b7 tokens / second': '引擎计时 · token / 秒',
      'Requests': '请求',
      'All statuses': '全部状态',
      'Active': '进行中',
      'Completed': '已完成',
      'Errors': '错误',
      'Inspect a request': '查看请求详情',
      'Select a request to see its input, output, and timings.': '选一条请求，查看其输入、输出与耗时。',
      'Wall-clock': '总耗时',
      'Model load': '模型加载',
      'Queue wait': '排队等待',
      'First token': '首 token',
      'Prompt tokens': '提示 token 数',
      'Output tokens': '输出 token 数',
      'Decode speed': '解码速度',
      'Response format': '响应格式',
      'Input': '输入',
      'Reasoning': '推理过程',
      'API response': 'API 响应',
      'Last 100 requests kept in memory until restart. Inputs and outputs stay on this machine.':
        '保留最近 100 条请求在内存中，重启后清空。输入与输出都留在本机。',
      'Only needed if the server has a key': '仅当服务端启用了密钥时才需要',
      'Search ID or endpoint': '搜索 ID 或接口路径',
      'Switch color theme': '切换配色主题',
      'Filter requests': '筛选请求',
      'Request status': '请求状态',
      'Request list': '请求列表',
      'Request content': '请求内容',
      'No matching requests.': '没有匹配的请求。',
      'No requests yet. API calls will appear here automatically.': '暂无请求。API 调用会自动出现在这里。',
      'JSON response': 'JSON 响应',
      'Streaming response: see Output and the request error, if any.': '流式响应：请看「输出」以及请求错误（如有）。',
      'No content.': '无内容。',
      'Raw model output retained for diagnosis. The API returned an error.': '保留模型原始输出以便诊断。API 返回了错误。',
      'Original request body.': '原始请求体。',
      'Model answer. JSON is formatted here for readability.': '模型回答。此处对 JSON 做了美化以便阅读。',
      'Separate reasoning content, when enabled.': '单独的推理内容（启用时才有）。',
      'API response body.': 'API 响应体。',
      'In use': '使用中',
      'Loads automatically on the next request': '下次请求时自动加载',
      'No active requests': '没有进行中的请求',
      'Disconnected \u00b7 retrying': '已断开 · 重试中',
      'Light / dark': '浅色 / 深色',
      'Switch between light and dark': '在浅色与深色间切换',
      'Close': '关闭'
    }
  };

  /* ============ patterns: strings that carry numbers ============ */
  var PATTERNS = {
    zh: [
      [/^Generating \u00b7 ([\d.,]+) tok\/s$/, function (m) { return '生成中 · ' + m[1] + ' tok/s'; }],
      [/^Reading prompt \u00b7 (\d+)%$/, function (m) { return '读取提示 · ' + m[1] + '%'; }],
      [/^Reading prompt$/, function () { return '读取提示'; }],
      [/^Live \u00b7 updated (.+)$/, function (m) { return '实时 · 更新于 ' + m[1]; }],
      [/^Show all \((\d+)\)$/, function (m) { return '显示全部（' + m[1] + '）'; }],
      [/^Thought for ([\d.]+) s$/, function (m) { return '思考用时 ' + m[1] + ' 秒'; }],
      [/^MTP drafts up to (\d+) tokens/, function (m) { return 'MTP 草稿最多 ' + m[1] + ' 个 token'; }],
      [/^You \u00b7 (.+)$/, function (m) { return '你 · ' + m[1]; }],
      [/^Saved \((.+)\)$/, function (m) { return '已保存（' + m[1] + '）'; }],
      [/^of ([\d.]+) W limit$/, function (m) { return '/ ' + m[1] + ' W 上限'; }],
      [/^write ([\d.]+) MB\/s$/, function (m) { return '写入 ' + m[1] + ' MB/s'; }],
      [/^to GPU ([\d.]+) MB\/s$/, function (m) { return '到 GPU ' + m[1] + ' MB/s'; }],
      [/^last: (.+) tokens$/, function (m) { return '上次：' + m[1] + ' 个 token'; }],
      [/^Since (.+?): (.+)$/, function (m) { return '自 ' + m[1] + ' 起：' + m[2]; }],
      [/^(\d+) of (\d+) requests reused part of their prompt$/,
        function (m) { return m[1] + ' / ' + m[2] + ' 次请求复用了部分提示'; }],
      [/^(\d+) requests$/, function (m) { return m[1] + ' 次请求'; }],
      [/^(.+) prompt tokens read at ([\d.]+) tok\/s \(([\d,]+) reused\)$/,
        function (m) { return '读取 ' + m[1] + ' 个提示 token，' + m[2] + ' tok/s（复用 ' + m[3] + '）'; }],
      [/^([\d,]+) written at ([\d.]+) tok\/s$/,
        function (m) { return '写出 ' + m[1] + ' 个，' + m[2] + ' tok/s'; }],
      [/^(.+) runs on this PC\. Nothing leaves it\.$/, function (m) {
        return m[1] + ' 跑在这台电脑上，数据不出本机。';
      }],
      [/^(.+) runs on this PC$/, function (m) { return m[1] + ' 跑在这台电脑上'; }],
      [/^The engine keeps the last conversation's state, so a follow-up reads only what is new\./,
        function () { return '引擎保留最近一段对话的状态，因此追问只需读取新增部分。'; }],
      [/^To keep several conversations \(agents taking turns\), add/,
        function () { return '要同时保留多段对话（多个 agent 轮流使用），请把'; }]
    ]
  };

  /* ===== partial pass: labels app.js assembles from fragments ===== */
  var PARTIAL = {
    zh: [
      [/\bDone\b/g, '完成'],
      [/\bSpeed\b/g, '速度'],
      [/\bPower\b/g, '功耗'],
      [/\bidle\b/g, '空闲'],
      [/\bcores\b/g, '核'],
      [/\bthreads\b/g, '线程'],
      [/\bqueued\b/g, '排队'],
      [/\brunning\b/g, '运行中'],
      [/\bCompleted\b/g, '已完成'],
      [/\bIn use\b/g, '使用中'],
      [/\bUnloaded\b/g, '未加载'],
      [/\bLoaded\b/g, '已加载']
    ]
  };

  var ATTRS = ['placeholder', 'title', 'aria-label', 'alt'];
  var SKIP_TAGS = { SCRIPT: 1, STYLE: 1, CODE: 1, PRE: 1, TEXTAREA: 1 };

  /* ===================== language choice ===================== */
  function detectLang() {
    try {
      var saved = localStorage.getItem(STORE_KEY);
      if (saved && LOCALES[saved]) return saved;
    } catch (e) {}
    var nav = (navigator.language || navigator.userLanguage || 'en').toLowerCase();
    // zh-CN / zh-TW / zh-Hans -> zh
    if (nav.indexOf('zh') === 0 && LOCALES.zh) return 'zh';
    return 'en';
  }

  var LANG = detectLang();
  var MAP = LOCALES[LANG] || {};
  var PATS = PATTERNS[LANG] || [];
  var PARTS = PARTIAL[LANG] || [];

  /* ===================== translate one string ===================== */
  function translate(text) {
    if (!text || LANG === 'en') return null;
    var lead = text.match(/^\s*/)[0];
    var tail = text.match(/\s*$/)[0];
    var core = text.trim();
    if (!core) return null;

    if (Object.prototype.hasOwnProperty.call(MAP, core)) return lead + MAP[core] + tail;
    for (var i = 0; i < PATS.length; i++) {
      var m = core.match(PATS[i][0]);
      if (m) return lead + PATS[i][1](m) + tail;
    }
    var changed = core;
    for (var p = 0; p < PARTS.length; p++) changed = changed.replace(PARTS[p][0], PARTS[p][1]);
    if (changed !== core) return lead + changed + tail;
    return null;
  }

  /* ===================== DOM walk ===================== */
  function translateAttrs(node) {
    for (var a = 0; a < ATTRS.length; a++) {
      var v = node.getAttribute && node.getAttribute(ATTRS[a]);
      if (!v) continue;
      var t = translate(v);
      if (t !== null && t !== v) node.setAttribute(ATTRS[a], t);
    }
  }

  function walk(node) {
    if (node.nodeType === 3) {
      var parent = node.parentNode;
      if (!parent || SKIP_TAGS[parent.nodeName] || parent.isContentEditable) return;
      var out = translate(node.nodeValue);
      if (out !== null && out !== node.nodeValue) node.nodeValue = out;
      return;
    }
    if (node.nodeType !== 1) return;
    if (SKIP_TAGS[node.nodeName]) {
      // content stays (code, user input) but attributes are translated:
      // a textarea's placeholder / aria-label live here
      if (node.nodeName === 'TEXTAREA' || node.nodeName === 'INPUT') translateAttrs(node);
      return;
    }
    translateAttrs(node);
    for (var c = node.firstChild; c; c = c.nextSibling) walk(c);
  }

  function run(root) {
    if (LANG === 'en') return;
    try { walk(root || document.body); } catch (e) { /* never break the page */ }
  }

  /* ===================== start ===================== */
  function start() {
    run(document.body);

    if (window.MutationObserver) {
      var obs = new MutationObserver(function (muts) {
        for (var i = 0; i < muts.length; i++) {
          var mu = muts[i];
          if (mu.type === 'childList') {
            for (var j = 0; j < mu.addedNodes.length; j++) run(mu.addedNodes[j]);
          } else if (mu.type === 'characterData') {
            run(mu.target.parentNode || mu.target);
          } else if (mu.type === 'attributes') {
            run(mu.target);
          }
        }
      });
      obs.observe(document.body, {
        childList: true, subtree: true, characterData: true, attributes: true,
        attributeFilter: ATTRS
      });
    }

    try { document.documentElement.lang = LANG === 'zh' ? 'zh-CN' : 'en'; } catch (e) {}
  }

  /* ===================== public API ===================== */
  window.StrataI18N = {
    getLang: function () { return LANG; },
    listLangs: function () { return Object.keys(LOCALES); },
    /** Remember a choice and reload so every string comes from the source text again. */
    setLang: function (lang) {
      if (!LOCALES[lang]) return false;
      try { localStorage.setItem(STORE_KEY, lang); } catch (e) {}
      location.reload();
      return true;
    },
    /** Translate a string on demand (for code that builds its own DOM). */
    t: function (s) { return translate(s) || s; }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
