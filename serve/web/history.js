"use strict";
window.StrataHistory = (() => {
  let controller;
  function mount(root, headers, notify) {
    const $ = id => root.querySelector(`#history-${id}`);
    const terminal = new Set(["completed", "error", "disconnected", "interrupted"]);
    const endpoint = path => ({"/v1/chat/completions": "Chat completion", "/v1/messages": "Messages", "/v1/responses": "Responses"}[path] || path || "Request");
    let active = false, records = [], cursor = null, selected = null, pane = "output";
    let nextOffset = null, expanded = false, busy = false, revision = 0, pages = 1, stamp = "", timer;
    let pendingReset = false, detailRevision = 0, detailBusy = false, shownState = null, shownId = null, shownPane = null;
    const panes = [...root.querySelectorAll("[data-history-pane]")];
    function notice(message) { $("error").hidden = !message; $("error").textContent = message || ""; }
    async function api(params) {
      const response = await fetch(`api/history?${new URLSearchParams(params)}`, {headers: headers(), cache: "no-store"});
      const body = await response.json();
      if (!response.ok) throw new Error(body.error?.message || `HTTP ${response.status}`);
      return body;
    }
    function list() {
      const key = JSON.stringify([records, selected]);
      if (stamp === key) return;
      stamp = key;
      const focusId = document.activeElement?.dataset.historyId;
      $("list").replaceChildren();
      if (!records.length) {
        const empty = document.createElement("li"); empty.className = "history-empty"; empty.textContent = "No history found";
        $("list").append(empty);
      }
      for (const record of records) {
        const li = document.createElement("li"), button = document.createElement("button");
        button.className = "history-request"; button.dataset.historyId = record.id;
        button.setAttribute("aria-pressed", String(record.id === selected));
        const title = document.createElement("strong"), info = document.createElement("span");
        title.textContent = endpoint(record.path);
        info.textContent = `${new Date(record.started_at * 1000).toLocaleString()} · ${record.state}`;
        const summary = document.createElement("span");
        summary.textContent = `${record.id}${record.output_tokens != null ? ` · ${record.output_tokens.toLocaleString()} tokens` : ""}`;
        const app = document.createElement("span"); app.className = "history-app";
        app.textContent = `From: ${record.client_app || "Unknown"}`;
        button.append(title, app, info, summary); li.append(button); $("list").append(li);
        button.onclick = () => {
          selected = record.id; expanded = false; shownId = null;
          $("selected").hidden = true; $("empty").hidden = false; $("empty").textContent = "Loading request...";
          list(); loadDetail().catch(error => notice(error.message));
        };
        if (focusId === record.id) button.focus({preventScroll: true});
      }
      $("more").hidden = !cursor;
    }
    function facts(entry) {
      shownState = entry.state;
      $("title").textContent = `${endpoint(entry.path)} · ${entry.id}`;
      $("status").textContent = entry.state;
      $("app").textContent = `From: ${entry.client_app || "Unknown"}`;
      $("meta").textContent = [new Date(entry.started_at * 1000).toLocaleString(), entry.model,
        entry.prompt_tokens != null ? `${entry.prompt_tokens.toLocaleString()} prompt tokens` : "",
        entry.output_tokens != null ? `${entry.output_tokens.toLocaleString()} output tokens` : "",
        entry.wallclock_s != null ? `${entry.wallclock_s.toFixed(1)} s` : ""].filter(Boolean).join(" · ");
      if (entry.recording_error) notice(entry.recording_error);
    }
    async function loadDetail(append = false) {
      if (!selected) return;
      const id = selected, field = pane, currentRevision = revision;
      const version = ++detailRevision;
      detailBusy = true;
      try {
      const result = await api({id, field: field === "details" ? "output" : field, offset: append ? nextOffset : 0});
      if (selected !== id || pane !== field || revision !== currentRevision || version !== detailRevision) return;
      $("empty").hidden = true; $("selected").hidden = false;
      facts(result.entry);
      shownId = id; shownPane = field;
      const text = field === "details" ? JSON.stringify(result.entry, null, 2) : result.text;
      if (append) { $("text").textContent += text; expanded = true; }
      else if ($("text").textContent !== text) $("text").textContent = text || "No text recorded";
      nextOffset = field === "details" ? null : result.next_offset;
      $("text-more").hidden = nextOffset == null;
      for (const button of panes) {
        const chosen = button.dataset.historyPane === field;
        button.setAttribute("aria-selected", String(chosen)); button.tabIndex = chosen ? 0 : -1;
        button.id = `history-pane-${button.dataset.historyPane}`;
        button.setAttribute("aria-controls", "history-text");
      }
      $("text").setAttribute("aria-labelledby", `history-pane-${field}`);
      } finally {
        if (version === detailRevision) {
          detailBusy = false;
          if (selected && active && revision !== currentRevision) loadDetail().catch(error => notice(error.message));
        }
      }
    }
    async function refresh(more = false, manual = false) {
      if (busy) { if (manual) pendingReset = true; return; }
      busy = true;
      $("refresh").disabled = $("more").disabled = true;
      const currentRevision = revision;
      try {
        if (!more && (pages === 1 || manual)) {
          const result = await api({q: $("search").value, state: $("filter").value});
          if (revision !== currentRevision) return;
          records = result.requests; cursor = result.next_cursor; pages = 1;
          notice(result.error);
        } else if (more && cursor) {
          const result = await api({q: $("search").value, state: $("filter").value, before: cursor});
          if (revision !== currentRevision) return;
          const known = new Set(records.map(record => record.id));
          records.push(...result.requests.filter(record => !known.has(record.id)));
          cursor = result.next_cursor; pages++;
          notice(result.error);
        }
        list();
        const record = records.find(record => record.id === selected);
        if (selected && !detailBusy && (!expanded || manual) &&
            (manual || selected !== shownId || pane !== shownPane || record?.state !== shownState ||
             !terminal.has(record?.state))) await loadDetail();
      } catch (error) { notice(error.message); }
      finally {
        busy = false; $("refresh").disabled = $("more").disabled = false;
        if (pendingReset) { pendingReset = false; refresh(false, true); }
      }
    }
    function reset() {
      revision++; pages = 1; selected = null; expanded = false; shownState = null;
      $("selected").hidden = true; $("empty").hidden = false; $("empty").textContent = "Select a request";
      refresh(false, true);
    }
    for (const button of panes) {
      button.onclick = () => { pane = button.dataset.historyPane; expanded = false; loadDetail().catch(error => notice(error.message)); };
      button.onkeydown = event => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const index = event.key === "Home" ? 0 : event.key === "End" ? panes.length - 1 :
          (panes.indexOf(button) + (event.key === "ArrowRight" ? 1 : panes.length - 1)) % panes.length;
        panes[index].click(); panes[index].focus();
      };
    }
    $("search").oninput = () => { clearTimeout(timer); timer = setTimeout(reset, 250); };
    $("filter").onchange = reset;
    $("refresh").onclick = () => { expanded = false; revision++; refresh(false, true); };
    $("more").onclick = () => refresh(true);
    $("text-more").onclick = async () => {
      $("text-more").disabled = true;
      try { await loadDetail(true); } catch (error) { notice(error.message); }
      finally { $("text-more").disabled = false; }
    };
    $("copy").onclick = async () => {
      try { await navigator.clipboard.writeText($("text").textContent); }
      catch { notify("Clipboard unavailable; select the text to copy it."); }
    };
    $("download").onclick = async () => {
      if (!selected) return;
      const id = selected;
      $("download").disabled = true;
      try {
        const response = await fetch(`api/history?${new URLSearchParams({id, download: 1})}`, {headers: headers(), cache: "no-store"});
        if (!response.ok) throw new Error(`Download failed: HTTP ${response.status}`);
        const url = URL.createObjectURL(await response.blob()), anchor = document.createElement("a");
        anchor.href = url; anchor.download = `strata-${id}.txt`; anchor.click();
        setTimeout(() => URL.revokeObjectURL(url), 30000);
      } catch (error) { notify(error.message); }
      finally { $("download").disabled = false; }
    };
    setInterval(() => { if (active && !document.hidden) refresh(); }, 3000);
    controller = on => { active = on; if (on) refresh(); };
  }
  return {mount, visible: on => { if (controller) controller(on); }};
})();
