// Render the update panel for real, without a browser.
//
// `node --check serve/web/app.js` only proves the file parses.  It says nothing about whether the panel
// draws, whether the ids match the markup, or whether a step's status reaches the right CSS attribute -
// all of which are the ways a progress UI actually breaks.  So this loads index.html, builds the
// smallest DOM that app.js needs, and runs the real render functions against real update states.
//
// What it cannot do: lay out, or run the polling loop's timers.  The visual result is checked in a
// browser separately; this catches the wiring, which is where the mistakes were.

const fs = require("fs");
const path = require("path");

const ROOT = "C:\\Strata-main";
const html = fs.readFileSync(path.join(ROOT, "serve", "web", "index.html"), "utf8");
const appJs = fs.readFileSync(path.join(ROOT, "serve", "web", "app.js"), "utf8");

const FAILS = [];
let CHECKS = 0;
function check(ok, what, detail = "") {
  CHECKS++;
  console.log(`  ${ok ? "ok  " : "FAIL"}  ${what.padEnd(56)}${detail ? " " + detail : ""}`);
  if (!ok) FAILS.push(what);
}

// ---- the ids app.js looks up, taken from the real markup ----
const IDS = ["facts-update", "update-check", "update-apply", "update-progress", "update-bar",
             "update-bar-fill", "update-steps", "update-note", "facts-engine"];

// app.js defines its own `$` (line 5: `const $ = (id) => document.getElementById(id)`), so a global `$`
// here is shadowed and has no effect. Instead document.getElementById must never return null: app.js
// wires buttons at its top level (`$("theme-btn").onclick = ...`), and a null there throws before any
// update code runs. Every id therefore resolves to a stub element, while the panel's own ids are still
// checked against the real markup above - which is the part that could actually be wrong.
const nodes = {};
function stubFor(id) {
  return nodes[id] || (nodes[id] = makeEl(id));
}
function makeEl(id) { // eslint-disable-line no-inner-declarations
  return {
    id, innerHTML: "", textContent: "", value: "", hidden: false, disabled: false,
    style: {}, dataset: {}, attributes: {}, children: [],
    setAttribute(k, v) { this.attributes[k] = v; },
    getAttribute(k) { return this.attributes[k]; },
    removeAttribute(k) { delete this.attributes[k]; },
    addEventListener() {},
    appendChild(c) { this.children.push(c); },
    // enough of the element API for app.js's start block, which clears and re-renders the chat
    querySelector() { return null; },
    querySelectorAll() { return []; },
    remove() {},
    focus() {},
    scrollTop: 0,
    scrollHeight: 0,
  };
}

console.log("every id the panel uses exists in index.html");
for (const id of IDS) {
  check(html.includes(`id="${id}"`), `index.html has id="${id}"`);
}

// ---- a DOM global good enough for app.js's top level ----
global.document = {
  getElementById: (id) => stubFor(id),
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener: () => {},
  createElement: () => makeEl("new"),
  body: makeEl("body"),
  documentElement: makeEl("html"),          // setTheme reads its .dataset.theme
};
global.matchMedia = () => ({matches: false, addEventListener: () => {}, addListener: () => {}});
global.window = {addEventListener: () => {}, matchMedia: global.matchMedia, location: global.location};
global.$ = (id) => nodes[id] || makeEl(`missing:${id}`);
global.location = { search: "", hash: "", pathname: "/", replaceState: () => {} };
global.history = { replaceState: () => {} };
global.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
global.sessionStorage = global.localStorage;
global.confirm = () => true;
global.setTimeout = () => 0;      // the polling loop must not keep node alive
global.clearTimeout = () => {};
global.setInterval = () => 0;
global.clearInterval = () => {};
global.fetch = async () => ({ ok: true, status: 200, json: async () => ({}) });
global.URLSearchParams = URLSearchParams;

// ---- run app.js in this context and keep what it defines ----
const EXPORTS = ["renderUpdateState", "renderUpdateIdle", "checkForUpdates", "applyUpdate",
                 "pollUpdate", "wireUpdate", "updateUI", "facts", "esc"];
const captured = new Function(`
  ${appJs}
  return {${EXPORTS.map((n) => `${n}: typeof ${n} !== "undefined" ? ${n} : undefined`).join(", ")}};
`)();

console.log("\napp.js exports what the panel needs");
for (const n of EXPORTS) check(captured[n] !== undefined, `${n} is defined`, String(captured[n]).slice(0, 30));

const {renderUpdateState, renderUpdateIdle, facts} = captured;

// The step marks are non-ASCII, and they reached the browser as mojibake once because the file was
// re-encoded on its way through a shell. `node --check` cannot see that - the file still parsed - and
// neither could any assertion that only compared structure. So assert the bytes.
const mark = (s) => captured.updateUI.mark(s);
console.log("\nthe step marks are the right characters, not mojibake");
check(mark("done") === "\u2713", "done is U+2713");
check(mark("active") === "\u25b8", "active is U+25B8");
check(mark("failed") === "\u2715", "failed is U+2715");
check(mark("pending") === "\u00b7", "pending is U+00B7");
check(![...mark("done")].some((c) => c.codePointAt(0) > 0x2fff), "no Private Use Area characters");


// ---- a running update, mid-download: the state the UI spends most time in ----
const running = {
  state: "running",
  detail: {installed: "0.1.31", latest: "v0.1.38", asset_mb: 124.3,
           percent: 43, downloaded_mb: 53.5, total_mb: 124.3},
  steps: [
    {key: "plan", label: "Check the release", status: "done", note: "installing v0.1.38", seconds: 0.1},
    {key: "room", label: "Check the free disk space", status: "done", note: "212.0 GB free", seconds: 0.0},
    {key: "download", label: "Download the engine", status: "active", note: "", seconds: 12.4},
    {key: "inspect", label: "Inspect the archive", status: "pending", note: "", seconds: null},
    {key: "stage", label: "Unpack to a staging area", status: "pending", note: "", seconds: null},
    {key: "test", label: "Run the staged engine", status: "pending", note: "", seconds: null},
    {key: "backup", label: "Back up the installed engine", status: "pending", note: "", seconds: null},
    {key: "apply", label: "Install the new engine", status: "pending", note: "", seconds: null},
    {key: "verify", label: "Start it and check the version", status: "pending", note: "", seconds: null},
  ],
  done: 2, total: 9, percent: 22, active: "Download the engine", active_key: "download", backup: null,
};

console.log("\na running update renders");
// renderUpdateState only updates `installed` from the state; `latest` comes from the CHECK, so it is set
// here the way the check response sets it. Without that the row would read "not checked yet" mid-update,
// which is what the next assertion catches.
captured.updateUI.latest = "0.1.38";   // bare: every write in app.js goes through bare()
renderUpdateState(running);
const bar = nodes["update-bar-fill"];
check(nodes["update-progress"].hidden === false, "the progress area is shown");
check(bar.style.width === "43%", "the bar shows the DOWNLOAD percent, not the step percent",
      bar.style.width);
check(nodes["update-bar"].getAttribute("data-tone") === "info", "the bar is in the info tone",
      nodes["update-bar"].getAttribute("data-tone"));
check(/^53\.5 of 124\.3 MB/.test(nodes["update-note"].textContent), "the note gives the byte counts",
      nodes["update-note"].textContent);
// every write to updateUI.installed must strip a leading v, or the row reads "vv0.1.40.1". This one did
// not, and only a live run against a real release showed it: the offline fixtures fed bare versions.
check(!/vv/.test(nodes["facts-update"].innerHTML), "the facts rows never show a doubled v",
      (nodes["facts-update"].innerHTML || "").replace(/<[^>]*>/g, "|").slice(0, 60));
check(nodes["update-steps"].innerHTML.includes("Check the free disk space"),
      "the step labels come from the server, so the panel cannot drift from them");
check(nodes["update-steps"].innerHTML.includes('data-state="active"'), "the active step is marked active");
check(nodes["update-steps"].innerHTML.includes('data-state="done"'), "finished steps are marked done");
check(nodes["update-steps"].innerHTML.includes('data-state="pending"'), "pending steps are marked pending");
check(/Download the engine/.test(nodes["update-steps"].innerHTML), "step labels are rendered");
check(/12\.4s/.test(nodes["update-steps"].innerHTML), "a running step shows its elapsed time");
check(nodes["update-check"].disabled === true, "the check button is disabled while it runs");
check(nodes["update-apply"].hidden === true, "the apply button is hidden while it runs");
check(/v0\.1\.31/.test(nodes["facts-update"].innerHTML), "the installed version is shown");
check(/v0\.1\.38/.test(nodes["facts-update"].innerHTML), "the latest version is shown");

// ---- the step percent, not the byte percent, once the download is over ----
console.log("\nafter the download, the bar follows the steps");
const postDownload = JSON.parse(JSON.stringify(running));
postDownload.active = "Inspect the archive"; postDownload.active_key = "inspect";
postDownload.detail.percent = undefined;      // the download no longer reports bytes
postDownload.done = 3;
postDownload.percent = 33;
postDownload.steps[2].status = "done";
postDownload.steps[2].note = "124.3 MB";
postDownload.steps[3].status = "active";
renderUpdateState(postDownload);
check(nodes["update-bar-fill"].style.width === "33%", "the bar shows the step percent now",
      nodes["update-bar-fill"].style.width);

// ---- failure, and the line that matters: what was restored ----
console.log("\na failure says what was restored");
const failed = {
  state: "failed",
  detail: {installed: "0.1.31", latest: "v0.1.38",
           error: "after installing, the engine still reports nothing. Restored the previous engine from backup-20261003-105459.",
           action: "Restored the previous engine from backup-20261003-105459.",
           rolled_back: true},
  steps: [
    {key: "plan", label: "Check the release", status: "done", note: "installing v0.1.38", seconds: 0.1},
    {key: "apply", label: "Install the new engine", status: "done", note: "replaced", seconds: 1.2},
    // the server puts the FULL message on the failing step (note = str(e), plus what it did about it)
    {key: "verify", label: "Start it and check the version", status: "failed",
     note: "after installing, the engine still reports nothing. Restored the previous engine from backup-20261003-105459.",
     seconds: 0.2},
  ],
  done: 2, total: 9, percent: 22, active: null, active_key: null, backup: "C:\\eng\\.strata-update\\backup-20261003-105459",
};
renderUpdateState(failed);
check(nodes["update-note"].getAttribute("data-tone") === "error", "the note is in the error tone");
check(/Restored the previous engine/.test(nodes["update-steps"].innerHTML),
      "the full message is on the failing step (where it belongs)");
check(/Restored the previous engine/.test(nodes["update-note"].textContent),
      "and the summary line says the previous engine was restored");
check(nodes["update-note"].textContent.length < 120, "the summary is short, not the whole paragraph",
      nodes["update-note"].textContent.length + " chars");
check(nodes["update-steps"].innerHTML.includes('data-state="failed"'), "the failed step is marked failed");
check(nodes["update-check"].disabled === false, "the check button is usable again after a failure");
// the note must be on its own row, or a long failure message squeezes the label to one word per line
check(/class="note"[^>]*>/.test(nodes["update-steps"].innerHTML), "notes render as .note spans");
check(/class="label"/.test(nodes["update-steps"].innerHTML), "labels render as .label spans");
check(/class="time"/.test(nodes["update-steps"].innerHTML), "durations render as .time spans");

// ---- done ----
console.log("\na finished update says the version and names the backup");
const done = JSON.parse(JSON.stringify(failed));
done.state = "done";
done.detail.installed = "v0.1.40.1";        // the release TAG
done.detail.error = undefined;
done.detail.verified_version = "0.1.40";   // what the installed BUILD.json actually says
done.steps[2].status = "done";
done.steps[2].note = "running 0.1.40";
done.percent = 100;
renderUpdateState(done);
// The release TAG can differ from the engine version - a hotfix release such as v0.1.40.1 ships the
// v0.1.40 engine - so the summary must report what BUILD.json says. Measured: it once claimed v0.1.40.1
// on a machine that had 0.1.40 installed.
check(/v0\.1\.40/.test(nodes["update-note"].textContent), "the summary names the ENGINE version",
      nodes["update-note"].textContent.slice(0, 44));
check(!/v0\.1\.40\.1/.test(nodes["update-note"].textContent), "and does not claim the release tag instead");
check(/\.previous/.test(nodes["update-note"].textContent), "and names engine/.previous, which setup.py --rollback-engine uses",
      nodes["update-note"].textContent.slice(0, 46));
check(nodes["update-bar-fill"].style.width === "100%", "the bar is full");

// ---- idle: a fresh install, before any check ----
console.log("\nthe idle panel before any check");
nodes["facts-update"].innerHTML = "";
nodes["update-progress"].hidden = true;
nodes["update-apply"].hidden = true;
nodes["update-check"].disabled = false;
captured.updateUI.latest = null;
captured.updateUI.installed = "0.1.31";
renderUpdateIdle();
check(/not checked yet/.test(nodes["facts-update"].innerHTML), "the latest release reads 'not checked yet'");
check(nodes["update-apply"].hidden === true, "there is nothing to install yet");
check(/Check for updates/.test(nodes["update-check"].textContent), "the button offers to check");
check(nodes["update-progress"].hidden === true, "the progress area is hidden");

// ---- a release available: the apply button appears ----
console.log("\nthe apply button appears once a newer release is found");
captured.updateUI.latest = "0.1.38";
renderUpdateIdle();
check(nodes["update-apply"].hidden === false, "the apply button is offered");
check(/0\.1\.38/.test(nodes["update-apply"].textContent), "and names the version",
      nodes["update-apply"].textContent);

// ---- escaping: the release name comes from the network ----
console.log("\na tag-prefixed version from the server never renders doubled");
const tagged = JSON.parse(JSON.stringify(done));
tagged.state = "ready";
tagged.steps = [{key: "plan", label: "Check the release", status: "done", note: "", seconds: 0.1}];
tagged.detail = {installed: "v0.1.40.1", latest: "v0.1.40.1", newer: false};  // exactly what /state sends
renderUpdateState(tagged);
check(!/vv/.test(nodes["facts-update"].innerHTML), "no doubled v in the facts rows",
      (nodes["facts-update"].innerHTML || "").replace(/<[^>]*>/g, "|").slice(0, 56));
check(/v0\.1\.40\.1/.test(nodes["facts-update"].innerHTML), "and the version is shown once",
      (nodes["facts-update"].innerHTML || "").replace(/<[^>]*>/g, "|").slice(0, 56));

console.log("\na leftover that could not be deleted is reported");
const stuck = JSON.parse(JSON.stringify(done));
stuck.detail.left_behind = ["stage-abc123"];
renderUpdateState(stuck);
check(/could not be deleted/.test(nodes["update-note"].textContent), "the panel says so",
      nodes["update-note"].textContent.slice(-52));
check(nodes["update-note"].getAttribute("data-tone") === "warn", "and marks it as a warning, not an error");

console.log("\nstep notes and labels are escaped");
const xss = {
  state: "running", detail: {installed: "0.1.31"}, done: 0, total: 2, percent: 0, active: "x",
  steps: [{key: "a", label: "<img src=x onerror=alert(1)>", status: "active",
           note: "<script>alert(2)</script>", seconds: 0}],
};
renderUpdateState(xss);
const out = nodes["update-steps"].innerHTML;
check(!/<img/.test(out), "a label containing a tag is escaped, not rendered");
check(!/<script>/.test(out), "a note containing a script tag is escaped");
check(out.includes("&lt;img"), "and appears as text instead");

console.log(`\nupdate panel: ${FAILS.length} failures out of ${CHECKS} checks`);
for (const f of FAILS) console.log(`  FAILED: ${f}`);
process.exit(FAILS.length ? 1 : 0);

