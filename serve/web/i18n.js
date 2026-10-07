/* Strata — UI localization (i18n).
 *
 * The app's interface is written in English. This script translates the interface
 * "chrome" into another language without touching the app code: it walks the DOM,
 * replaces exact dictionary matches and keeps watching for nodes the app adds later
 * (metrics, status labels, toasts) so dynamic text is translated too, in the same
 * microtask — before the frame is painted, otherwise the English word flashes first.
 *
 * Model answers, code blocks and attachments are never touched: inside #chat only
 * the chrome listed in CHROME_SEL is translated.
 *
 * Dictionaries are plain JSON files, `web/i18n/<lang>.json`, flat "english": "translation"
 * pairs. The reverse direction is built by inversion, so one file works both ways
 * (a string already translated is translated back when switching back).
 * Numbers cannot be handled by a flat dictionary, so the few strings that carry numbers
 * are handled by RULES, each one with both directions.
 *
 * Adding a language: drop web/i18n/<lang>.json next to ru.json and add the language
 * to LANGS below. Unknown text simply stays English — the page never breaks.
 */
(function () {
  "use strict";

  const LANGS = ["en", "ru"];              // languages the picker offers
  const DEFAULT_LANG = "en";               // English unless the user picks another
  const LANG_KEY = "lang";                 // localStorage "strata.lang"
  // dictionaries live next to this file (i18n/ru.json), so the path is resolved from
  // the script itself — the same file works from / and from web/monitor.html
  const HERE = ((document.currentScript && document.currentScript.src) || location.href).replace(/[^/]*$/, "");

  const $ = (id) => document.getElementById(id);
  const store = {
    get(key, def) { try { const v = localStorage.getItem("strata." + key); return v == null ? def : v; } catch (e) { return def; } },
    set(key, val) { try { localStorage.setItem("strata." + key, val); } catch (e) {} },
  };

  let lang = LANGS.includes(store.get(LANG_KEY, DEFAULT_LANG)) ? store.get(LANG_KEY, DEFAULT_LANG) : DEFAULT_LANG;
  let dict = null;                         // english -> translation
  let back = null;                         // translation -> english
  let dictLang = null;                      // which language `dict` holds

  /* ---------------------------------------------------------------- rules */
  // Strings with numbers need real morphology (1 token, 2 tokens, 5 tokens...).
  // Each rule carries both directions; `to`/`back` may be a template or a function.
  const i18nPlural = (num, forms) => {
    const n = Math.abs(parseInt(String(num).replace(/[^\d]/g, ""), 10) || 0) % 100;
    const d = n % 10;
    if (n > 10 && n < 20) return forms[2];
    return d === 1 ? forms[0] : d >= 2 && d <= 4 ? forms[1] : forms[2];
  };
  const RULES = [
    { en: /^Reading prompt · ([\d.,]+)%$/, ru: /^Чтение промпта · ([\d.,]+)%$/,
      to: "Чтение промпта · $1%", back: "Reading prompt · $1%" },
    { en: /^Generating · ([\d.,]+) tok\/s$/, ru: /^Генерация · ([\d.,]+) ток\/с$/,
      to: "Генерация · $1 ток/с", back: "Generating · $1 tok/s" },
    { en: /^(\d+) queued$/, ru: /^(\d+) в очереди$/, to: "$1 в очереди", back: "$1 queued" },
    { en: /^([\d\s.,]+) tokens · ([\d.,]+) tok\/s$/, ru: /^([\d\s.,]+) токенов? · ([\d.,]+) ток\/с$/,
      to: (m) => `${m[1]} ${i18nPlural(m[1], ["токен", "токена", "токенов"])} · ${m[2]} ток/с`,
      back: (m) => `${m[1]} tokens · ${m[2]} tok/s` },
    { en: /^([\d\s.,]+) \/ ([\d\s.,]+) tokens · ([\d.,]+)%$/, ru: /^([\d\s.,]+) \/ ([\d\s.,]+) токенов? · ([\d.,]+)%$/,
      to: (m) => `${m[1]} / ${m[2]} ${i18nPlural(m[2], ["токен", "токена", "токенов"])} · ${m[3]}%`,
      back: (m) => `${m[1]} / ${m[2]} tokens · ${m[3]}%` },
    { en: /^([\d\s.,]+) tokens at ([\d.,]+) tok\/s$/, ru: /^([\d\s.,]+) токенов? при ([\d.,]+) ток\/с$/,
      to: (m) => `${m[1]} ${i18nPlural(m[1], ["токен", "токена", "токенов"])} при ${m[2]} ток/с`,
      back: (m) => `${m[1]} tokens at ${m[2]} tok/s` },
    { en: /^([\d\s.,]+) tokens$/, ru: /^([\d\s.,]+) токенов?$/,
      to: (m) => `${m[1]} ${i18nPlural(m[1], ["токен", "токена", "токенов"])}`,
      back: (m) => `${m[1]} tokens` },
    { en: /^the model calls them when it decides to$/, ru: /^модель вызывает их, когда решит$/,
      to: "модель вызывает их, когда решит", back: "the model calls them when it decides to" },
    { en: /^([\d\s.,]+) tools · ([\d\s.,]+) of ([\d\s.,]+) servers connected$/,
      ru: /^([\d\s.,]+) инструментов? · подключено ([\d\s.,]+) из ([\d\s.,]+) серверов$/,
      to: (m) => `${m[1]} ${i18nPlural(m[1], ["инструмент", "инструмента", "инструментов"])} · подключено ${m[2]} из ${m[3]} серверов`,
      back: (m) => `${m[1]} tools · ${m[2]} of ${m[3]} servers connected` },
    { en: /^([\d\s.,]+) tools$/, ru: /^([\d\s.,]+) инструментов?$/,
      to: (m) => `${m[1]} ${i18nPlural(m[1], ["инструмент", "инструмента", "инструментов"])}`,
      back: (m) => `${m[1]} tools` },
    { en: /^Thought for ([\d.,]+) s$/, ru: /^Размышлял ([\d.,]+) с$/, to: "Размышлял $1 с", back: "Thought for $1 s" },
    { en: /^Show all \((\d+)\)$/, ru: /^Показать все \((\d+)\)$/, to: "Показать все ($1)", back: "Show all ($1)" },
    { en: /^just now$/, ru: /^только что$/, to: "только что", back: "just now" },
    { en: /^([\d.,]+) min ago$/, ru: /^([\d.,]+) мин назад$/, to: "$1 мин назад", back: "$1 min ago" },
    { en: /^([\d.,]+) h ago$/, ru: /^([\d.,]+) ч назад$/, to: "$1 ч назад", back: "$1 h ago" },
    { en: /^(\d+) tool calls?$/, ru: /^(\d+) вызов(?:а|ов)? инструмента$/,
      to: (m) => `${m[1]} ${+m[1] % 10 === 1 && +m[1] % 100 !== 11 ? "вызов"
        : [2, 3, 4].includes(+m[1] % 10) && ![12, 13, 14].includes(+m[1] % 100) ? "вызова" : "вызовов"} инструмента`,
      back: (m) => `${m[1]} tool ${m[1] === "1" ? "call" : "calls"}` },
    { en: /^([\d.,]+) of ([\d.,]+) requests reused part of their prompt$/,
      ru: /^([\d.,]+) из ([\d.,]+) запросов переиспользовали часть промпта$/,
      to: "$1 из $2 запросов переиспользовали часть промпта", back: "$1 of $2 requests reused part of their prompt" },
    { en: /^(.+) runs on this PC\. Nothing leaves it\.$/, ru: /^(.+) работает на этом ПК\. Ничего не уходит наружу\.$/,
      to: "$1 работает на этом ПК. Ничего не уходит наружу.", back: "$1 runs on this PC. Nothing leaves it." },
    { en: /^of ([\d.,]+) W limit$/, ru: /^из ([\d.,]+) Вт \(лимит\)$/, to: "из $1 Вт (лимит)", back: "of $1 W limit" },
    { en: /^([\d.,]+) experts cached$/, ru: /^([\d.,]+) экспертов в кэше$/,
      to: "$1 экспертов в кэше", back: "$1 experts cached" },
    { en: /^([\d.,]+) cores · ([\d.,]+) threads$/, ru: /^([\d.,]+) ядер · ([\d.,]+) потоков$/,
      to: "$1 ядер · $2 потоков", back: "$1 cores · $2 threads" },
    { en: /^(Projection|Additive) control vector on layers (\d+)[–-](\d+)(?: \(layer (\d+)'s direction\))?\. Per chat in Sampling\. Its package describes the vector as a refusal-direction projection; measure the speed yourself$/,
      ru: /^(Проекция|Аддитивный) управляющего вектора на слоях (\d+)[–-](\d+)(?: \(направление слоя (\d+)\))?\. Включается на один чат в шторке\. Вектор описан как проекция направления отказа; скорость измеряйте сами\.$/,
      to: (m) => `${m[1] === "Projection" ? "Проекция" : "Аддитивный"} управляющего вектора на слоях ${m[2]}–${m[3]}`
        + `${m[4] ? ` (направление слоя ${m[4]})` : ""}. Включается на один чат в шторке. `
        + `Вектор описан как проекция направления отказа; скорость измеряйте сами.`,
      back: (m) => `${m[1] === "Проекция" ? "Projection" : "Additive"} control vector on layers ${m[2]}–${m[3]}`
        + `${m[4] ? ` (layer ${m[4]}'s direction)` : ""}. Per chat in Sampling. Its package `
        + `describes the vector as a refusal-direction projection; measure the speed yourself` },
    { en: /^last: (.+)$/, ru: /^прошлый: (.+)$/, to: (m) => `прошлый: ${i18nTr(m[1])}`, back: (m) => `last: ${i18nTr(m[1])}` },
  ];

  function rule(key, dir) {
    for (const r of RULES) {
      const re = dir === "to" ? r.en : r.ru;
      const tpl = dir === "to" ? r.to : r.back;
      if (!re.test(key)) continue;
      return typeof tpl === "function" ? key.replace(re, (...a) => tpl(a)) : key.replace(re, tpl);
    }
    return null;
  }

  /* ------------------------------------------------------------- translate */
  function i18nTr(s) {
    if (typeof s !== "string" || !dict) return s;
    const m = /^(\s*)([\s\S]*?)(\s*)$/.exec(s);
    const lead = m ? m[1] : "", body = m ? m[2] : s, tail = m ? m[3] : "";
    if (!body || !/[A-Za-zА-Яа-я]/.test(body)) return s;
    // the markup breaks long phrases across lines and indents, so look up the collapsed
    // text as well, and write the translation back with the original outer whitespace
    const flat = body.replace(/\s+/g, " ");
    const table = lang === "en" ? back : dict;
    let out = table[body];
    if (out === undefined) out = table[flat];
    if (out === undefined) out = rule(body, lang === "en" ? "back" : "to");
    if (out == null || out === flat) return s;
    return lead + out + tail;
  }

  /* --------------------------------------------------------- dom traversal */
  // inside the chat only this chrome is translated — model output, code and
  // attachments are left exactly as they are
  const CHROME_SEL = ".think-title, .meta-text, .tool-call__label, .st-code__head, .msg-error, .st-msg__meta, #chat-empty";
  const ATTRS = ["title", "placeholder", "aria-label"];
  const skipText = (el) => !el || !el.closest || !!el.closest("pre, code, textarea, #lang-btn")
    || (!!el.closest("#chat") && !el.closest(CHROME_SEL));
  const skipAttr = (el) => !el || !el.closest || !!el.closest("pre, code")
    || (!!el.closest("#chat") && !el.closest(CHROME_SEL));

  function apply(root) {
    const start = root || document.body;
    if (!start) return;
    const walker = document.createTreeWalker(start, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT, {
      acceptNode: (n) => (n.nodeType === 3 && skipText(n.parentElement)) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
    });
    const bump = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (n.nodeType === 3) { const v = i18nTr(n.nodeValue); if (v !== n.nodeValue) bump.push([n, v]); continue; }
      if (skipAttr(n)) continue;
      for (const a of ATTRS) {
        const v = n.getAttribute && n.getAttribute(a);
        if (!v) continue;
        const t = i18nTr(v);
        if (t !== v) bump.push([n, a, t]);
      }
    }
    for (const [n, a, v] of bump) if (v === undefined) n.nodeValue = a; else n.setAttribute(a, v);
  }

  const fixText = (n) => {
    if (!n || n.nodeType !== 3 || skipText(n.parentElement)) return;
    const v = i18nTr(n.nodeValue);
    if (v !== n.nodeValue) n.nodeValue = v;
  };
  const fixAttrs = (el) => {
    if (!el || el.nodeType !== 1 || skipAttr(el)) return;
    for (const a of ATTRS) {
      const v = el.getAttribute && el.getAttribute(a);
      if (!v) continue;
      const t = i18nTr(v);
      if (t !== v) el.setAttribute(a, t);
    }
  };

  function label() {
    const b = $("lang-btn");
    if (!b) return;
    b.textContent = lang.charAt(0).toUpperCase() + lang.slice(1);   // "En" / "Ru"
    b.title = "Interface language · " + LANGS.map((l) => l.toUpperCase()).join(" / ");
  }

  function setLang(next) {
    lang = LANGS.includes(next) ? next : DEFAULT_LANG;
    store.set(LANG_KEY, lang);
    document.documentElement.lang = lang;
    label();
    if (lang === DEFAULT_LANG || dictLang === lang) { apply(); return; }
    // the dictionary is loaded on demand — once per language, only when it is needed
    loadDict(lang).then((loaded) => {
      if (!loaded) { lang = DEFAULT_LANG; store.set(LANG_KEY, lang); label(); return; }
      apply();
    });
  }

  /* ------------------------------------------------------------------ boot */
  function loadDict(code) {
    return fetch(HERE + "i18n/" + code + ".json", { cache: "no-cache" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((json) => {
        dict = json || {};
        back = {};
        for (const [en, tr] of Object.entries(dict)) if (!(tr in back)) back[tr] = en;
        dictLang = code;
        return true;
      })
      .catch(() => false);
  }

  function boot() {
    // the picker shows the language you are in; a click switches to the next one
    const btn = $("lang-btn");
    if (btn) btn.onclick = () => setLang(LANGS[(LANGS.indexOf(lang) + 1) % LANGS.length] || DEFAULT_LANG);
    // dynamic text (metrics, status labels, toasts) is translated right here, in the
    // observer's microtask — before the frame is painted
    new MutationObserver((records) => {
      for (const r of records) {
        if (r.type === "characterData") { fixText(r.target); continue; }
        if (r.type === "attributes") { fixAttrs(r.target); continue; }
        for (const n of r.addedNodes || []) {
          if (n.nodeType === 3) fixText(n);
          else if (n.nodeType === 1) { fixAttrs(n); apply(n); }
        }
      }
    }).observe(document.body, { subtree: true, childList: true, characterData: true,
                                attributes: true, attributeFilter: ATTRS });
    setLang(lang);        // labels the button and, for a saved non-English choice, applies it
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.strataI18n = { tr: (s) => i18nTr(s), set: setLang, get lang() { return lang; }, langs: LANGS };
})();
