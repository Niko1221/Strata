// Instruction selection only. The server owns skill content and tool permissions.
(function (root) {
  "use strict";
  const validName = name => typeof name === "string" && /^[a-z0-9][a-z0-9-]{0,63}$/.test(name);
  function normalize(data) {
    if (!data || typeof data.enabled !== "boolean" || !Array.isArray(data.skills)) throw new Error("Invalid skills catalog");
    const seen = new Set();
    const skills = data.skills.filter(skill => {
      if (!skill || !validName(skill.name) || typeof skill.description !== "string" || seen.has(skill.name)) return false;
      seen.add(skill.name); return true;
    }).map(skill => ({name: skill.name, description: skill.description}));
    return {enabled: data.enabled, skills};
  }
  function prefix(text, skills) {
    const match = /^\/([^\s]+)(?=\s|$)/.exec(String(text || ""));
    return match && skills.some(skill => skill.name === match[1]) ? match[1] : null;
  }
  function slashQuery(text, caret) {
    const match = /^\/([^\s]*)$/.exec(String(text || "").slice(0, caret));
    return match ? match[1].toLowerCase() : null;
  }
  function insertSkill(text, name, skills, replaceSlash = false) {
    const value = String(text || "");
    if (!validName(name) || !skills.some(skill => skill.name === name)) return value;
    if (replaceSlash || prefix(value, skills)) {
      const suffix = value.replace(/^\/[^\s]*/, "");
      return `/${name}${suffix || " "}`;
    }
    return `/${name} ${value}`;
  }
  function mount({input, container, fetchCatalog}) {
    const doc = container.ownerDocument;
    const make = (tag, className, text) => {
      const node = doc.createElement(tag);
      if (className) node.className = className;
      if (text != null) node.textContent = text;
      return node;
    };
    let catalog = null, enabled = false, busy = false, knownEnabled = false;
    let open = false, browse = false, selecting = false, active = 0, rows = [], revision = 0, controller = null;
    const button = make("button", "st-btn st-btn--secondary skill-controls__button", "/ Skills");
    button.type = "button"; button.id = "skills-btn";
    button.setAttribute("aria-haspopup", "listbox"); button.setAttribute("aria-controls", "skills-list");
    const status = make("span", "skill-controls__status"); status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite");
    const picker = make("div", "skill-controls__picker"); picker.hidden = true;
    const heading = make("div", "skill-controls__heading", "Choose a skill · ↑ ↓ browse · Enter select · Esc close");
    const list = make("div", "skill-controls__list"); list.id = "skills-list";
    list.setAttribute("role", "listbox"); list.setAttribute("aria-label", "Instruction skills");
    const matches = make("div", "skill-controls__status"); matches.setAttribute("role", "status");
    picker.appendChild(heading); picker.appendChild(list); picker.appendChild(matches);
    container.appendChild(button); container.appendChild(status); container.appendChild(picker);
    const canPick = () => enabled && !busy && !controller && catalog?.enabled && catalog.skills.length > 0;
    function closePicker() { open = false; picker.hidden = true; button.setAttribute("aria-expanded", "false"); }
    function cancelRefresh() {
      revision++;
      if (controller) controller.abort();
      controller = null;
    }
    function close() {
      const loading = !!controller;
      if (list.contains(doc.activeElement)) input.focus({preventScroll: true});
      cancelRefresh(); closePicker();
      if (loading) status.textContent = "Skills loading cancelled.";
      render();
    }
    function render() {
      container.hidden = !enabled || !knownEnabled;
      button.disabled = !enabled || busy || !!controller || (catalog?.enabled && !catalog.skills.length);
      if (!canPick()) closePicker();
    }
    function highlight(focus = false) {
      Array.from(list.children).forEach((node, index) => node.setAttribute("aria-selected", String(index === active)));
      const node = list.children[active];
      node?.scrollIntoView?.({block: "nearest"});
      if (focus) node?.focus({preventScroll: true});
    }
    function choose(skill) {
      if (!canPick()) return;
      input.value = insertSkill(input.value, skill.name, catalog.skills, !browse);
      close(); input.focus();
      let position = skill.name.length + 1;
      if (input.value[position] === " ") position++;
      input.setSelectionRange(position, position);
      selecting = true;
      try { input.dispatchEvent(new doc.defaultView.Event("input", {bubbles: true})); }
      finally { selecting = false; }
    }
    function show(all = false) {
      if (!canPick()) { closePicker(); return; }
      const query = all ? "" : slashQuery(input.value, input.selectionStart);
      if (query == null) { closePicker(); return; }
      browse = all; open = true; active = 0;
      rows = catalog.skills.filter(skill => skill.name.includes(query) || skill.description.toLowerCase().includes(query));
      list.replaceChildren();
      rows.forEach((skill, index) => {
        const option = make("button", "skill-controls__option"); option.type = "button"; option.tabIndex = -1;
        option.setAttribute("role", "option"); option.setAttribute("aria-selected", String(index === active));
        option.appendChild(make("span", "skill-controls__name", `/${skill.name}`));
        option.appendChild(make("span", "skill-controls__description", skill.description));
        option.addEventListener("pointerdown", event => event.preventDefault());
        option.addEventListener("click", () => choose(skill));
        option.addEventListener("keydown", handleKey);
        list.appendChild(option);
      });
      matches.textContent = rows.length ? `${rows.length} ${rows.length === 1 ? "skill" : "skills"}` : "No matching skills. Your message stays as typed.";
      picker.hidden = false; button.setAttribute("aria-expanded", "true"); highlight();
    }
    async function refresh() {
      cancelRefresh(); closePicker();
      if (!enabled) { render(); return; }
      const current = ++revision, pending = new AbortController(); controller = pending;
      status.textContent = "Loading skills…"; render();
      try {
        const next = normalize(await fetchCatalog(pending.signal));
        if (current !== revision || pending.signal.aborted) return;
        catalog = next; knownEnabled = next.enabled;
        status.textContent = next.skills.length ? "" : "No skills available.";
      } catch (_) {
        if (current !== revision || pending.signal.aborted) return;
        catalog = null; status.textContent = "Skills unavailable. Select Skills to try again.";
      } finally {
        if (current === revision) { controller = null; render(); }
      }
    }
    function handleKey(event) {
      if (event.isComposing || event.shiftKey || event.ctrlKey || event.altKey || event.metaKey ||
          (event.target !== input && !list.contains(event.target))) return false;
      if (event.key === "Escape" && (open || controller)) { close(); input.focus(); event.preventDefault(); return true; }
      if (!open) return false;
      if (event.key === "Tab") { close(); return false; }
      if (event.key === "Enter" && rows.length) choose(rows[active]);
      else if ((event.key === "ArrowDown" || event.key === "ArrowUp") && rows.length) {
        active = (active + (event.key === "ArrowDown" ? 1 : -1) + rows.length) % rows.length; highlight(true);
      } else return false;
      event.preventDefault(); return true;
    }
    button.addEventListener("click", async () => {
      if (button.disabled) return;
      if (open) { close(); input.focus(); return; }
      if (!catalog) await refresh();
      if (canPick()) { input.focus(); show(true); }
    });
    input.addEventListener("input", () => { if (!selecting) show(false); });
    input.addEventListener("click", () => { if (open && !browse) show(false); });
    doc.addEventListener("pointerdown", event => { if (open && event.target !== input && !container.contains(event.target)) close(); });
    doc.addEventListener("focusin", event => { if (open && event.target !== input && !container.contains(event.target)) close(); });
    render();
    return {refresh, close, handleKey,
      request(text = input.value) {
        const name = canPick() ? prefix(text, catalog.skills) : null;
        return name ? {strata_skill: name} : {};
      },
      setBusy(value) { busy = !!value; if (busy) close(); else render(); },
      setEnabled(value) { enabled = !!value; if (!enabled) close(); else render(); },
    };
  }
  const api = {normalize, prefix, slashQuery, insertSkill, mount};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.StrataSkillControls = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
