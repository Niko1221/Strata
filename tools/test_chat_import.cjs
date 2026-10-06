"use strict";
const assert = require("node:assert/strict");
const {normalize} = require("../serve/web/chat-import.js");
let checks = 0;
function check(fn) { fn(); checks++; }
const clone = value => JSON.parse(JSON.stringify(value));
const node = (id, role, content, parent, children = [], extra = {}) =>
  ({id, convId: "chat", type: "text", timestamp: 1000, role, content, parent, children, ...extra});
const original = {conv: {id: "chat", name: "Review", currNode: "b"}, messages: [
  node("root", "system", "", null, ["s"], {type: "root"}),
  node("s", "system", "Keep constraints", "root", ["u"]),
  node("u", "user", "", "s", ["a", "b"], {extra: [{type: "IMAGE", name: "screen", base64Url: "data:image/png;base64,AA=="}]}),
  node("a", "assistant", "Old answer", "u", [], {thinking: "Old thought"}),
  node("b", "assistant", "New answer", "u", [], {reasoningContent: "New thought"})
]};
const frozen = clone(original);
const imported = normalize(original, {source: "LlamaUi"})[0];
check(() => assert.equal(imported.id, "import:LlamaUi:chat"));
check(() => assert.equal(imported.branches.length, 2));
check(() => assert.equal(imported.activeBranchId, "b"));
check(() => assert.deepEqual(imported.branches[0].messages.map(m => m.text), ["Keep constraints", "", "Old answer"]));
check(() => assert.equal(imported.branches[1].messages[2].reasoning, "New thought"));
check(() => assert.equal(imported.branches[0].messages[2].reasoning, "Old thought"));
check(() => assert.deepEqual(imported.branches[0].messages[0].wire, [{role: "system", content: "Keep constraints"}]));
check(() => assert.equal(imported.branches[0].messages[1].wire[0].content[1].image_url.url, "data:image/png;base64,AA=="));
check(() => assert.deepEqual(original, frozen));
imported.source.messages[0].content = "detached";
check(() => assert.deepEqual(original, frozen));
const point = clone(original); point.conv.currNode = "u";
check(() => { const result = normalize(point)[0]; assert.equal(result.activeBranchId, "u"); assert.equal(result.branches.find(b => b.id === "u").messages.length, 2); });
const numeric = {conv: {id: "chat", name: "Old", currNode: 2}, messages: [
  node(0, "system", "", -1, [1], {type: "root"}), node(1, "user", "Ask", 0, [2]), node(2, "assistant", "Reply", 1)
]};
check(() => assert.equal(normalize(numeric)[0].activeBranchId, "2"));
const calls = [{id: "call", type: "function", function: {name: "lookup", arguments: '{"q":"x"}'}}];
const tools = {conv: {id: "chat", name: "Tools", currNode: "t"}, messages: [
  node("a", "assistant", "", null, ["t"], {toolCalls: JSON.stringify(calls)}),
  node("t", "tool", "Result", "a", [], {toolCallId: "call"})
]};
check(() => assert.deepEqual(normalize(tools)[0].branches[0].messages[0].wire[0].tool_calls, calls));
check(() => assert.deepEqual(normalize(tools)[0].branches[0].messages[1].wire, [{role: "tool", content: "Result", tool_call_id: "call"}]));
const jsonl = [JSON.stringify({type: "session", harness: "llama.cpp", ...tools.conv}), ...tools.messages.map(m => JSON.stringify({type: "message", message: {...m, toolCalls: m.toolCalls ? calls : undefined}}))].join("\n");
check(() => assert.deepEqual(normalize(jsonl)[0].branches, normalize(tools)[0].branches));
check(() => assert.deepEqual(normalize(JSON.stringify(original)), normalize(original)));
check(() => assert.equal(normalize([original, {...numeric, conv: {...numeric.conv, id: "other"}, messages: numeric.messages.map(m => ({...m, convId: "other"}))}]).length, 2));
for (const type of ["imageFile", "IMAGE"]) check(() => {
  const value = clone(original); value.messages[2].extra[0].type = type;
  assert.equal(normalize(value)[0].branches[0].messages[1].images[0].name, "screen");
});
for (const type of ["TEXT", "textFile", "context", "MCP_PROMPT", "MCP_RESOURCE"]) check(() => {
  const value = clone(numeric); value.messages[1].extra = [{type, name: "notes", content: "Keep this"}];
  const message = normalize(value)[0].branches[0].messages[0];
  assert.equal(message.files[0].text, "Keep this"); assert.match(message.wire[0].content, /Keep this/);
});
function rejects(mutator, pattern) { check(() => { const value = clone(original); mutator(value); assert.throws(() => normalize(value), pattern); assert.deepEqual(original, frozen); }); }
rejects(value => value.messages.push(clone(value.messages[2])), /duplicate message/);
rejects(value => value.messages[2].parent = "missing", /missing parent/);
rejects(value => { value.messages[0].parent = "b"; value.messages[4].children.push("root"); }, /cycle/);
rejects(value => value.messages[2].children.push("missing"), /child relationship/);
rejects(value => value.messages[3].convId = "other", /cross-conversation/);
rejects(value => value.conv.currNode = "missing", /current branch/);
rejects(value => value.messages[2].content = [{type: "text", text: "unsupported"}], /message content/);
rejects(value => value.messages[0].content = "Cannot hide", /root contains/);
rejects(value => value.messages[3].toolCalls = "bad", /tool calls JSON/);
rejects(value => value.messages[3].toolCalls = JSON.stringify([{function: {name: "x"}}]), /tool call format/);
for (const type of ["AUDIO", "audioFile", "VIDEO", "PDF", "unknown"]) rejects(value => value.messages[3].extra = [{type, name: "unsupported", content: "keep raw"}], /unsupported attachment/);
check(() => assert.throws(() => normalize([original, original]), /duplicate conversation/));
check(() => assert.throws(() => normalize({messages: []}), /expected/));
check(() => assert.throws(() => normalize('PK\u0003\u0004'), /extract ZIP/));
check(() => assert.throws(() => normalize('{"type":"message","message":{}}\n'), /before session/));
check(() => assert.equal(normalize({conv: {id: "empty", name: "Empty", currNode: null}, messages: []})[0].branches[0].messages.length, 0));
check(() => assert.equal(normalize('{"type":"session","id":"empty","name":"Empty","currNode":null}')[0].activeBranchId, "empty"));
check(() => {
  const result = normalize(original)[0]; result.branches[0].messages[0].text = "Edited branch";
  assert.equal(result.branches[1].messages[0].text, "Keep constraints");
  assert.equal(result.source.messages[1].content, "Keep constraints");
});
console.log(`PASS: ${checks} chat-import checks; no storage/network/tool actions.`);
