// Small, shared policy helpers; the server's tokenizer supplies the actual counts.
(function (root) {
  "use strict";
  const emptyMemory = () => ({through: 0, summary: "", count: 0});
  function validMemory(value, history) {
    return value && Number.isInteger(value.through) && value.through >= 0 && value.through <= history.length &&
      typeof value.summary === "string" && (value.through === 0 || value.summary.trim()) ? value : emptyMemory();
  }
  function limit(context, maxOutput) {
    const reserve = Number(maxOutput) > 0 ? Number(maxOutput) : Math.min(8192, Math.floor(context / 4));
    return Math.max(0, Math.min(Math.floor(context * 0.9), context - reserve - 8));
  }
  function cut(history, through, recent = 4) {
    let at = Math.max(through, history.length - recent);
    // Keep user/assistant/tool rounds together. A stored answer owns its tool calls.
    while (at > through && at < history.length && history[at].role !== "user") at--;
    return at > through ? at : history.length;
  }
  const hasImages = (messages) => messages.some((m) => Array.isArray(m.content) &&
    m.content.some((p) => ["image_url", "input_image", "image"].includes(p.type)));
  const api = {emptyMemory, validMemory, limit, cut, hasImages};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.StrataContext = api;
})(typeof window !== "undefined" ? window : {});
