// Development foundation: implementation branches replace this adapter.
"use strict";
globalThis.StrataMathAdapter = {
  async initialize() { throw new Error("No math engine installed"); },
  async render() { return {ok: false, reason: "unavailable"}; },
};
