// Pure import of official llama.cpp conversation exports. No storage, network, or tool execution.
(function (root) {
  "use strict";
  function fail(message) { throw new Error("Chat import: " + message); }
  function object(value) { return value !== null && typeof value === "object" && !Array.isArray(value); }
  function key(value) {
    if (typeof value === "string" && value.length || Number.isSafeInteger(value) && value >= 0) return String(value);
    fail("invalid message/conversation ID");
  }
  function parse(input) {
    if (typeof input !== "string") return input;
    const text = input.trim();
    if (!text) fail("empty file");
    try {
      const parsed = JSON.parse(text);
      if (!object(parsed) || !["session", "message"].includes(parsed.type)) return parsed;
    } catch (_) { /* JSONL is checked below. */ }
    const sessions = []; let current = null;
    for (const line of text.split(/\r?\n/).filter(line => line.trim())) {
      let record;
      try { record = JSON.parse(line); } catch (_) { fail("invalid JSON/JSONL; extract ZIP files to JSONL first"); }
      if (!object(record)) fail("invalid JSONL record");
      if (record.type === "session") {
        const conv = Object.assign({}, record); delete conv.type; delete conv.harness;
        current = {conv, messages: []}; sessions.push(current);
      } else if (record.type === "message" && current && object(record.message)) current.messages.push(record.message);
      else fail("unsupported JSONL record or message before session");
    }
    return sessions;
  }
  function text(value, label) { if (typeof value !== "string") fail("invalid " + label); return value; }
  function message(node) {
    if (!["system", "user", "assistant", "tool"].includes(node.role)) fail("unsupported message role");
    const content = text(node.content, "message content");
    const result = {role: node.role, text: content, reasoning: "", time: node.timestamp};
    if (!Number.isFinite(node.timestamp) || node.timestamp < 0) fail("invalid message timestamp");
    const thoughts = [];
    for (const field of ["reasoningContent", "thinking"]) if (node[field] != null) {
      const value = text(node[field], field); if (value && !thoughts.includes(value)) thoughts.push(value);
    }
    result.reasoning = thoughts.join("\n");
    const parts = [{type: "text", text: content}], images = [], files = [];
    if (node.extra != null && !Array.isArray(node.extra)) fail("invalid attachments");
    for (const extra of node.extra || []) {
      if (!object(extra)) fail("invalid attachment");
      const name = text(extra.name, "attachment name");
      if (["IMAGE", "imageFile"].includes(extra.type)) {
        const url = text(extra.base64Url, "image URL");
        if (!url) fail("empty image URL");
        images.push({name, url}); parts.push({type: "image_url", image_url: {url}});
      } else if (["TEXT", "textFile", "context", "MCP_PROMPT", "MCP_RESOURCE"].includes(extra.type)) {
        const value = text(extra.content, "attachment content");
        files.push({name, text: value}); parts.push({type: "text", text: "\n\nFile: " + name + "\n" + value});
      } else fail("unsupported attachment " + String(extra.type) + "; original source was not changed");
    }
    if (images.length) result.images = images;
    if (files.length) result.files = files;
    const api = {role: node.role, content: images.length ? parts : parts.map(part => part.text).join("")};
    if (node.toolCalls != null && node.toolCalls !== "") {
      if (node.role !== "assistant") fail("tool calls on non-assistant message");
      let calls = node.toolCalls;
      if (typeof calls === "string") try { calls = JSON.parse(calls); } catch (_) { fail("invalid tool calls JSON"); }
      if (!Array.isArray(calls)) fail("invalid tool calls");
      for (const call of calls) if (!object(call) || typeof call.id !== "string" || !call.id || call.type !== "function" ||
          !object(call.function) || typeof call.function.name !== "string" || !call.function.name || typeof call.function.arguments !== "string") fail("unsupported tool call format");
      if (calls.length) api.tool_calls = JSON.parse(JSON.stringify(calls));
    }
    if (node.role === "tool") {
      if (typeof node.toolCallId !== "string" || !node.toolCallId) fail("tool result missing call ID");
      api.tool_call_id = node.toolCallId;
    }
    result.wire = [api]; return result;
  }
  function session(item, source) {
    if (!object(item) || !object(item.conv) || !Array.isArray(item.messages)) fail("expected {conv, messages} export");
    const conv = item.conv, convId = key(conv.id), nodes = new Map();
    if (typeof conv.name !== "string") fail("invalid conversation name");
    for (const node of item.messages) {
      if (!object(node) || key(node.convId) !== convId) fail("cross-conversation or invalid message");
      const id = key(node.id); if (nodes.has(id)) fail("duplicate message ID");
      if (!["root", "text"].includes(node.type) || !Array.isArray(node.children)) fail("unsupported message node format");
      if (node.type === "root" && (node.content !== "" || node.extra?.length || node.toolCalls || node.reasoningContent || node.thinking)) fail("root contains content that cannot be omitted");
      nodes.set(id, node);
    }
    function parent(node) {
      if (node.parent == null || node.type === "root" && node.parent === -1) return null;
      const id = key(node.parent); if (!nodes.has(id)) fail("missing parent"); return id;
    }
    for (const [id, node] of nodes) {
      const seenChildren = new Set();
      for (const child of node.children) {
        const childId = key(child);
        if (seenChildren.has(childId) || !nodes.has(childId) || parent(nodes.get(childId)) !== id) fail("invalid child relationship");
        seenChildren.add(childId);
      }
      const p = parent(node);
      if (p !== null && !nodes.get(p).children.some(child => key(child) === id)) fail("parent does not reference child");
      const seen = new Set(); let at = id;
      while (at !== null) { if (seen.has(at)) fail("message ancestry cycle"); seen.add(at); at = parent(nodes.get(at)); }
    }
    let selected = null;
    if (conv.currNode != null && conv.currNode !== "") { selected = key(conv.currNode); if (!nodes.has(selected)) fail("current branch node is missing"); }
    if (nodes.size && selected === null) fail("conversation has no current branch");
    const tips = [...nodes].filter(([, node]) => !node.children.length).map(([id]) => id);
    if (selected !== null && !tips.includes(selected)) tips.push(selected);
    // Validate every branch, including inactive messages and unsupported extras, before producing a result.
    const normalized = new Map([...nodes].filter(([, node]) => node.type !== "root").map(([id, node]) => [id, message(node)]));
    const branches = tips.map((id, index) => {
      const path = []; let at = id;
      while (at !== null) { if (normalized.has(at)) path.push(normalized.get(at)); at = parent(nodes.get(at)); }
      return {id, title: "Branch " + (index + 1) + (id === selected ? " (selected)" : ""), messages: JSON.parse(JSON.stringify(path.reverse())), context: null};
    });
    if (!branches.length) branches.push({id: "empty", title: "Empty chat", messages: [], context: null});
    return {id: "import:" + source + ":" + convId, title: conv.name, branches, activeBranchId: selected || "empty",
      source: {format: "llama", conversation: conv, messages: item.messages}};
  }
  function normalize(input, {source = "file"} = {}) {
    if (typeof source !== "string" || !source) fail("invalid import source");
    let parsed = parse(input);
    // Own a detached JSON copy; normalization never mutates caller data or source databases.
    try { parsed = JSON.parse(JSON.stringify(parsed)); } catch (_) { fail("input must be JSON data"); }
    const items = Array.isArray(parsed) ? parsed : [parsed];
    const result = items.map(item => session(item, source)), ids = new Set();
    for (const item of result) { if (ids.has(item.id)) fail("duplicate conversation ID"); ids.add(item.id); }
    return result;
  }
  const api = {normalize};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.StrataChatImport = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
