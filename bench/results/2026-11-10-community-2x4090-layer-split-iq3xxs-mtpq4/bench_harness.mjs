// my-srv bench arms via shim 31487, engine-measured response.timings (predicted_per_second etc.)
// arms: concurrent (2 req), solo (guarded to a running==0 window), long128 (~131K-token fresh read)
import { readFileSync } from "node:fs";
import { exec } from "node:child_process";
import { randomBytes } from "node:crypto";
import { promisify } from "node:util";
const execAsync = promisify(exec);

const GW = "http://<host>:31487/v1/chat/completions";
const MODEL = "qwen3.8-flash-next-abliterated-iq3_xxs";
const KEY = readFileSync(process.env.HOME + "/.hermes/cache/scratch/llm_gw/llm_gw_master_key.txt", "utf8").trim();
const nonce = () => randomBytes(8).toString("hex");

async function metricsLive() {
  try {
    const { stdout } = await execAsync(
      `ssh -o ConnectTimeout=5 <user>@<host> 'curl -s -m 4 localhost:8001/metrics'`,
      { timeout: 15000 });
    const m = JSON.parse(stdout);
    const live = m.live || {};
    return { running: live.running ?? -1, queued: live.queued ?? -1,
             slots: (m.slots || []).map(s => ({ gen: s.generating, wait: s.waiting, acc: s.accepted, dr: s.draft_rate })) };
  } catch { return { running: -1, queued: -1 }; }
}

async function one(prompt, maxTokens) {
  const t0 = Date.now();
  const res = await fetch(GW, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${KEY}` },
    body: JSON.stringify({
      model: MODEL, messages: [{ role: "user", content: prompt }],
      temperature: 0, max_tokens: maxTokens,
    }),
  });
  if (!res.ok) throw new Error("HTTP " + res.status + " " + (await res.text()).slice(0, 160));
  const j = await res.json();
  const t = j.timings || {};
  return {
    decode_tps: +(t.predicted_per_second || 0).toFixed(1),
    decode_ms: t.predicted_ms, pred_n: t.predicted_n,
    draft_n: t.draft_n, draft_acc: t.draft_n_accepted,
    prompt_ms: +(t.prompt_ms || 0).toFixed(0), prompt_n: t.prompt_n,
    prompt_tps: +(t.prompt_per_second || 0).toFixed(0),
    e2e_s: +((Date.now() - t0) / 1000).toFixed(2),
  };
}

const story = (n) => `List 240 invented words that begin with the syllable "z${n}", one per line, no commentary.`;
function bigPrompt(n) {
  const slugs = [];
  while (slugs.join(" ").length < 228000) slugs.push(n + randomBytes(6).toString("hex"));
  return slugs.join(" ") + "\n\nRepeat the last invented word, nothing else.";
}

async function armConcurrent(run) {
  const pre = await metricsLive();
  const timer = setTimeout(async () => { global.midC = await metricsLive(); }, 2500);
  const [A, B] = await Promise.all([one(story("a" + run + nonce()), 256), one(story("b" + run + nonce()), 256)]);
  clearTimeout(timer);
  console.log(JSON.stringify({ arm: "concurrent", run, pre_running: pre.running, mid_running: (global.midC || {}).running, A, B }));
}

async function soloWindow(fn, label) {
  let waited = 0, live = await metricsLive();
  while (live.running > 0 && waited < 150000) { await new Promise(r => setTimeout(r, 5000)); waited += 5000; live = await metricsLive(); }
  const r = await fn();
  const post = await metricsLive();
  console.log(JSON.stringify({ arm: label, waited_s: waited / 1000, pre_running: live.running, post_running: post.running, ...r }));
  return post.running;
}

const which = process.argv[2] || "concurrent";
const n = +(process.argv[3] || 2);
for (let i = 1; i <= n; i++) {
  if (which === "concurrent") await armConcurrent(i);
  else if (which === "solo") { await soloWindow(() => one(story("s" + i + nonce()), 256), "solo"); }
  else if (which === "long128") { await soloWindow(() => one(bigPrompt("L" + i + nonce()), 8), "long128"); }
  await new Promise(r => setTimeout(r, 3000));
}
