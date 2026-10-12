// serve/web/i18n.js - the tiny i18n runtime. Loaded last of the i18n scripts (after the dictionaries).
// Dictionaries live in window.STRATA_I18N["<code>"] (i18n.<code>.js); English is the fallback, then the key itself.
// Public API: window.StrataI18N = {lang, available, names, t, apply, setLang, onChange}.
"use strict";
(function () {
  var DICTS = window.STRATA_I18N = window.STRATA_I18N || {};
  // <languages:begin> the picker order and names - rewritten by wire_ui.py
  var available = ["en", "zh-CN", "zh-TW", "ja", "ko", "vi", "id", "es", "pt-BR", "de", "fr", "ru", "it", "tr"];
  var NAMES = {"en": "English", "zh-CN": "简体中文", "zh-TW": "繁體中文", "ja": "日本語", "ko": "한국어", "vi": "Tiếng Việt", "id": "Bahasa Indonesia", "es": "Español", "pt-BR": "Português (Brasil)", "de": "Deutsch", "fr": "Français", "ru": "Русский", "it": "Italiano", "tr": "Türkçe"};
  // <languages:end>
  var STORAGE_KEY = "strata.lang";

  function readStored() {
    try { return localStorage.getItem(STORAGE_KEY); } catch (e) { return null; }
  }
  function writeStored(value) {
    try { localStorage.setItem(STORAGE_KEY, value); } catch (e) { /* private mode: in memory only */ }
  }

  // one browser language tag ("pt-BR", "zh-Hant-TW", "id") -> one of ours, or null
  function fromBrowser(tag) {
    var t = String(tag || "").toLowerCase().replace("_", "-");
    var primary = t.split("-")[0];
    if (primary === "en") return "en";
    for (var i = 0; i < available.length; i++) {
      if (available[i].toLowerCase() === t) return available[i];
    }
    if (primary === "zh") {
      if (/hant|-tw|-hk|-mo/.test(t) && available.indexOf("zh-TW") >= 0) return "zh-TW";
      return available.indexOf("zh-CN") >= 0 ? "zh-CN" : null;
    }
    if (primary === "pt") return available.indexOf("pt-BR") >= 0 ? "pt-BR" : null;
    if (primary === "in") primary = "id";
    return available.indexOf(primary) >= 0 ? primary : null;
  }
  // the saved choice, else the first of the browser's languages that we have (an English one ends the search), else English
  function detect() {
    var saved = readStored();
    if (available.indexOf(saved) >= 0) return saved;
    var prefs = (navigator.languages && navigator.languages.length) ? navigator.languages : [navigator.language || ""];
    for (var i = 0; i < prefs.length; i++) {
      var found = fromBrowser(prefs[i]);
      if (found) return found;
    }
    return "en";
  }

  var lang = detect();

  // the dictionary string for the current language; en is the fallback, then the key itself
  function t(key, vars) {
    var dict = DICTS[lang] || {};
    var s = dict[key];
    if (s == null) s = (DICTS.en || {})[key];
    if (s == null) s = key;
    if (vars) {
      s = String(s).replace(/\{(\w+)\}/g, function (m, name) {
        return vars[name] != null ? String(vars[name]) : m;
      });
    }
    return s;
  }

  // the picker is a plain <select class="lang-select">: one in the header (hidden on narrow screens, where the header has no room)
  // and one in About > Settings; the look lives here so both pages share it
  function injectStyle() {
    if (document.getElementById("i18n-style")) return;
    var s = document.createElement("style");
    s.id = "i18n-style";
    s.textContent = ".lang-select{height:36px;max-width:140px;padding:0 8px;border-radius:10px;border:1.5px solid var(--st-line);" +
      "background:var(--st-surface);color:var(--st-ink-muted);font-family:inherit;font-size:13px;font-weight:600;cursor:pointer}" +
      ".lang-select:hover{background:var(--st-surface-2);color:var(--st-ink)}" +
      "@media(max-width:900px){.st-header #lang-select{display:none}}@media(max-width:520px){.controls #lang-select{display:none}}" +
      // longer labels in other languages (Мониторинг, Pemantau...) need a tighter header on phones; English keeps its layout
      "@media(max-width:480px){html:not([lang=\"en\"]) .st-header{gap:4px;padding:0 6px}html:not([lang=\"en\"]) .st-tab{padding:0 6px;font-size:12.5px}}" +
      // API Monitor on a wide screen: the search hint is longer than "Search ID or endpoint" in most languages (it was cut off)
      "@media(min-width:601px){html:not([lang=\"en\"]) #search{width:260px}}" +
      // API Monitor on a phone: a long status filter (すべてのステータス) must not push the search box out of the panel
      "@media(max-width:600px){html:not([lang=\"en\"]) .filters{width:100%}html:not([lang=\"en\"]) #search{flex:1 1 0;min-width:0;width:auto}" +
      "html:not([lang=\"en\"]) .filters select{max-width:46%;min-width:0;text-overflow:ellipsis}}";
    document.head.appendChild(s);
  }
  function fillPicker(sel) {
    for (var i = 0; i < available.length; i++) {
      var o = document.createElement("option");
      o.value = available[i];
      o.setAttribute("lang", available[i]);
      o.textContent = NAMES[available[i]] || available[i];
      sel.appendChild(o);
    }
  }
  function wirePicker() {
    var picks = document.querySelectorAll(".lang-select");
    for (var n = 0; n < picks.length; n++) {
      var sel = picks[n];
      if (!sel.dataset.i18nWired) {
        sel.dataset.i18nWired = "1";
        fillPicker(sel);
        sel.addEventListener("change", function () { setLang(this.value); });
      }
      sel.value = lang;
    }
    if (picks.length) injectStyle();
  }

  function apply(root) {
    root = root || document;
    document.documentElement.lang = lang;
    var els = root.querySelectorAll("[data-i18n],[data-i18n-html],[data-i18n-title],[data-i18n-placeholder],[data-i18n-aria]");
    for (var i = 0; i < els.length; i++) {
      var el = els[i];
      var key = el.getAttribute("data-i18n");
      if (key != null) el.textContent = t(key);
      var html = el.getAttribute("data-i18n-html");
      if (html != null) el.innerHTML = t(html);
      var title = el.getAttribute("data-i18n-title");
      if (title != null) el.setAttribute("title", t(title));
      var ph = el.getAttribute("data-i18n-placeholder");
      if (ph != null) el.setAttribute("placeholder", t(ph));
      var aria = el.getAttribute("data-i18n-aria");
      if (aria != null) el.setAttribute("aria-label", t(aria));
    }
    wirePicker();
  }

  function setLang(l) {
    if (available.indexOf(l) < 0) return;
    writeStored(l);
    lang = l;
    api.lang = l;
    apply();
    document.dispatchEvent(new CustomEvent("strata:lang", {detail: l}));
  }

  function onChange(cb) { document.addEventListener("strata:lang", cb); }

  var api = {lang: lang, available: available, names: NAMES, t: t, apply: apply, setLang: setLang, onChange: onChange};
  window.StrataI18N = api;

  apply();                                          // the static HTML above is already parsed
  document.addEventListener("DOMContentLoaded", function () { apply(); });
})();
