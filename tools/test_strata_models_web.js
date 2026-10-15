"use strict";
// Models page regressions without browser packages, a GPU or a running model.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function page() {
  const nodes = new Map();
  const element = id => {
    if (!nodes.has(id)) nodes.set(id, {
      hidden:false, disabled:false, dataset:{},
      querySelector() { return {setAttribute() {}}; }
    });
    return nodes.get(id);
  };
  const context = vm.createContext({
    document:{getElementById:element, documentElement:{dataset:{}}, querySelectorAll() { return []; }},
    localStorage:{getItem() { return "light"; }},
    matchMedia() { return {matches:false}; },
    fetch() { return new Promise(() => {}); }, // Tests supply status snapshots directly.
    setInterval() {}
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../serve/web/models.js"), "utf8"), context);
  return {
    element,
    status(running, operation = null) {
      context.snapshot = {running, operation, models:[], hardware:{ram_gb:64, gpus:[]},
                          model_url:"http://127.0.0.1:8080/"};
      vm.runInContext("state = snapshot; render();", context);
    }
  };
}

const server = loaded => ({managed:true, loaded, model:"qwen-iq2", url:"http://127.0.0.1:8080"});

test("a loaded server opens its chat page", () => {
  const p = page();
  p.status([server(true)]);
  assert.equal(p.element("open-chat").hidden, false);
  assert.equal(p.element("open-chat").href, server(true).url);
});

test("a lazy server keeps chat available for its first request", () => {
  const p = page();
  p.status([server(false)], {kind:"start", state:"done", message:"Model ready. Open Chat to use it."});
  assert.equal(p.element("open-chat").hidden, false);
  assert.equal(p.element("open-chat").href, server(false).url);
});

test("idle unloading keeps the existing chat link available", () => {
  const p = page();
  p.status([server(true)]);
  const chat = p.element("open-chat").href;
  p.status([server(false)]);
  assert.equal(p.element("open-chat").hidden, false);
  assert.equal(p.element("open-chat").href, chat);
});

test("switching keeps chat hidden until the operation finishes", () => {
  const p = page();
  p.status([server(true)], {kind:"start", state:"running", message:"Starting..."});
  assert.equal(p.element("open-chat").hidden, true);
});

test("a stopped server has no chat link", () => {
  const p = page();
  p.status([]);
  assert.equal(p.element("open-chat").hidden, true);
});
