"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const activity = require("../serve/web/tool-activity.js");
const search = (extra = {}) => ({name: "web__web_search", tool: "web_search", state: "done", ok: true,
  arguments: {query: "Shanghai housing policy"}, result: JSON.stringify({retrieved_at: "2026-10-06", query: "Shanghai housing policy",
    results: [{title: "Official source", url: "https://example.org/page"}]}), ms: 1900, ...extra});
const page = (extra = {}) => ({name: "web__read_webpage", tool: "read_webpage", state: "done", ok: true,
  arguments: {url: "https://www.shanghai.gov.cn/policy"}, result: JSON.stringify({title: "Housing policy",
    source_url: "https://www.shanghai.gov.cn/policy", markdown: "FULL DOCUMENT"}), ms: 2700, ...extra});
test("closed rows show human subjects and measured status without exposing raw payloads", () => {
  const html = activity.renderCall(search(), 0);
  assert.match(html, /Search web/); assert.match(html, /Shanghai housing policy/); assert.match(html, /1 result/);
  assert.match(html, /2s/); assert.doesNotMatch(html, /retrieved_at|web__web_search|<pre/);
  const read = activity.renderCall(page(), 1);
  assert.match(read, /Read page/); assert.match(read, /Housing policy/); assert.match(read, /www.shanghai.gov.cn/);
  assert.doesNotMatch(read, /FULL DOCUMENT|source_url/);
});

test("many completed calls become one collapsed activity with visible partial failure", () => {
  const calls = [search(), page(), search(), page({arguments: {url: "https://other.org/a"}, state: "error",
    result: "Web tool failed: HTTP Error 500: Internal Server Error"}), search(), page()];
  const html = activity.renderGroup(calls, 3);
  assert.equal((html.match(/<details/g) || []).length, 1);
  assert.match(html, /Web research/); assert.match(html, /6 steps · 1 page · 1 failed/); assert.match(html, /Partial/);
  assert.doesNotMatch(html, /FULL DOCUMENT|HTTP Error|retrieved_at/);
  const expanded = activity.renderGroup(calls, 3, {open: true});
  assert.equal((expanded.match(/<details/g) || []).length, 7);
  assert.match(expanded, /data-tool="6" data-state="error"/);
  assert.match(expanded, />Failed</);
});

test("running and skipped activity remain distinguishable with no made-up duration", () => {
  const html = activity.renderGroup([search(), page({state: "running", result: null, ms: null})], 0);
  assert.match(html, /Working/); assert.match(html, /Read page · www.shanghai.gov.cn/);
  assert.doesNotMatch(html, /<pre|>Done</);
  const skipped = activity.renderCall(search({state: "skipped", ms: 0, result: "not run"}), 2);
  assert.match(skipped, /Skipped/); assert.match(skipped, /Not executed/); assert.doesNotMatch(skipped, /0s/);
});

test("untrusted query, URL, name and raw result stay escaped, with no executable source link", () => {
  const call = page({open: true, name: '<script>alert(1)</script>', arguments: {url: 'javascript:alert(1)'},
    result: JSON.stringify({title: '<img src=x onerror=alert(1)>', source_url: 'javascript:alert(1)', markdown: '<script>bad</script>'})});
  const html = activity.renderCall(call, 0);
  assert.doesNotMatch(html, /<script|<img|href="javascript:/);
  assert.match(html, /&lt;img/); assert.match(html, /&lt;script/);
  assert.match(html, /Arguments/); assert.match(html, /Result/);
  const credentials = activity.renderCall(page({open: true, arguments: {url: "https://user:secret@example.org/"},
    result: JSON.stringify({source_url: "https://user:secret@example.org/"})}), 0);
  assert.doesNotMatch(credentials, /href=/);
});

test("truncated JSON and generic tools still render without large closed result bodies", () => {
  const html = activity.renderCall(search({result: '{"results": [', truncated: true}), 0);
  assert.match(html, /Search web/); assert.doesNotMatch(html, /<pre/);
  const call = {name: "review__read_file", arguments: {path: "serve/server.py"}, state: "done", ok: true,
    result: "PRIVATE SOURCE ".repeat(6000), ms: Infinity};
  assert.match(activity.renderCall(call, 0), /Read file/);
  assert.match(activity.renderCall(call, 0), /serve\/server.py/);
  assert.doesNotMatch(activity.renderCall(call, 0), /PRIVATE SOURCE|Infinity/);
  assert.equal(activity.renderGroup([], 0), "");
});

test("page count distinguishes URLs on one site and excludes unsuccessful fetches", () => {
  const calls = [page(), page({arguments: {url: "https://www.shanghai.gov.cn/other"},
    result: JSON.stringify({source_url: "https://www.shanghai.gov.cn/other"})}),
    page({state: "error"}), page({state: "skipped"})];
  assert.match(activity.renderGroup(calls, 0), /4 steps · 2 pages · 1 failed · 1 skipped/);
});

test("page subject, link and group count follow the returned canonical source URL", () => {
  const call = page({open: true, arguments: {url: "https://original.example.org/redirect"},
    result: JSON.stringify({source_url: "https://canonical.example.org/policy", title: "Canonical policy"})});
  const html = activity.renderCall(call, 0);
  assert.match(html, /href="https:\/\/canonical\.example\.org\/policy"/);
  assert.match(html, />canonical.example.org</);
  assert.doesNotMatch(html, /href="https:\/\/original/);
  const redirected = page({arguments: {url: "https://another.example.org/redirect"}, result: call.result});
  assert.match(activity.renderGroup([call, redirected], 0), /2 steps · 1 page/);
});