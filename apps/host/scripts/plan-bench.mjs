#!/usr/bin/env node
/**
 * Strata Dev: measures how good the plans of the chat model are, with and without "Better plans" (docs/DEV_BUILD.md, planGuidance).
 * It needs the running Strata Dev app with a real model loaded (not -HostOnly: that model is simulated).
 *
 *   STRATA_URL=http://127.0.0.1:47832 node apps/host/scripts/plan-bench.mjs [--prompts <folder>] [--only 1,5,8] [--variants off,on] [--out <file.json>]
 *
 * For every prompt file (*.txt) and every variant it sends "/plan <prompt>" in a fresh chat, takes the plan the host proposes and scores it
 * with checks that need no model:
 *   points    how many numbered/bulleted points of the request some step names (a guess by shared words, see uncoveredPoints)
 *   steps     number of steps, and how many carry a detail after the dash
 *   cut       steps the model wrote beyond the limit
 *   last      the last step looks like a final check against the request
 *   secs      how long the plan turn took
 * The variants switch Settings > Tools > Better plans through POST /api/dev/plan-guidance ("off" = the Live prompt, "on" = the Dev rules); the
 * setting that was saved before is put back at the end. Plans are only proposed, never carried out: nothing in the prompts' folders changes.
 * It prints a table and writes the plans and scores as JSON, so two runs (before/after a rule change) can be compared.
 */
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { uncoveredPoints, requestPoints } from '../server/agent/plan.mjs';

const BASE = process.env.STRATA_URL;
if (!BASE) { console.error('STRATA_URL is required (the Strata Dev address, for example http://127.0.0.1:47832)'); process.exit(2); }
const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : fallback; };
const dir = arg('prompts', 'C:\\Users\\jonat\\Documents\\Test Plan\\prompts');
const only = arg('only', '') ? new Set(arg('only').split(',').map(n => Number(n))) : null;
const variants = arg('variants', 'off,on').split(',').map(v => v.trim()).filter(v => v === 'off' || v === 'on');
const outFile = arg('out', `plan-bench-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
if (!variants.length) { console.error('--variants takes off, on or off,on'); process.exit(2); }

const j = async (path, body, method = body ? 'POST' : 'GET') => {
  const res = await fetch(BASE + path, { method, headers: { 'content-type': 'application/json' }, ...(body ? { body: JSON.stringify(body) } : {}) });
  const text = await res.text();
  let data; try { data = JSON.parse(text); } catch { data = { raw: text }; }
  if (!res.ok) throw new Error(`${path} -> ${res.status} ${data.error ?? text.slice(0, 200)}`);
  return data;
};

async function model() {
  const providers = await j('/api/model-providers');
  const c = providers.connections?.at(-1);
  if (!c) throw new Error('no model connection saved in this Strata Dev (see scripts/agent-http.mjs "connect")');
  return { connectionId: c.id, modelId: c.models[0].id };
}

/** One /plan turn; resolves with the proposed plan (or null) and the seconds it took. */
async function planOnce(prompt, remoteModel) {
  const started = Date.now();
  const res = await fetch(BASE + '/api/chat/stream', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ sessionId: `bench-${Date.now()}`, message: `/plan ${prompt}`, mode: 'plan', remoteModel, workspaceId: process.env.WORKSPACE_ID }) });
  if (!res.ok) throw new Error(`stream -> ${res.status} ${await res.text()}`);
  let plan = null, buffer = '';
  const reader = res.body.getReader(); const dec = new TextDecoder();
  for (;;) {
    const { value, done } = await reader.read();
    if (value) {
      buffer += dec.decode(value, { stream: true });
      const lines = buffer.split('\n'); buffer = lines.pop() ?? '';
      for (const line of lines.filter(Boolean)) { try { const f = JSON.parse(line); if (f.type === 'plan' && f.plan?.steps?.length) plan = f.plan; } catch { /* a half line */ } }
    }
    if (done) break;
  }
  return { plan, secs: Math.round((Date.now() - started) / 1000) };
}

const FINAL = /\b(check|verify|review|confirm|validate)\b/i;
export function score(prompt, plan, secs) {
  if (!plan) return { ok: false, secs };
  const total = requestPoints(prompt).length;
  const missing = uncoveredPoints(prompt, plan.steps).length;
  const last = plan.steps.at(-1)?.text ?? '';
  return {
    ok: true, secs, steps: plan.steps.length, withDetail: plan.steps.filter(s => s.detail).length,
    points: total ? `${total - missing}/${total}` : '-', pointsCovered: total ? (total - missing) / total : null,
    cut: (plan.hints ?? []).some(h => /plan keeps the first/.test(h)),
    last: FINAL.test(last),
  };
}

const files = readdirSync(dir).filter(f => /\.txt$/i.test(f)).sort().filter(f => !only || only.has(Number(/(\d+)/.exec(f)?.[1])));
if (!files.length) { console.error(`no prompt files in ${dir}`); process.exit(2); }
const remoteModel = await model();
const saved = (await j('/api/dev/state').catch(() => null))?.dev?.planGuidance ?? null;
const results = [];
try {
  for (const variant of variants) {
    await j('/api/dev/plan-guidance', { enabled: variant === 'on', thinking: saved?.thinking ?? null, maxSteps: saved?.maxSteps ?? 20 });
    for (const file of files) {
      const prompt = readFileSync(join(dir, file), 'utf8').replace(/\r/g, '');
      process.stdout.write(`${variant.padEnd(3)} ${file} ... `);
      try {
        const { plan, secs } = await planOnce(prompt, remoteModel);
        const s = score(prompt, plan, secs);
        results.push({ variant, file, ...s, plan: plan ? { title: plan.title, steps: plan.steps.map(x => ({ text: x.text, detail: x.detail ?? '' })), hints: plan.hints ?? [] } : null });
        console.log(s.ok ? `${s.steps} steps, points ${s.points}, last-check ${s.last ? 'yes' : 'no'}, ${s.secs}s` : `no plan (${s.secs}s)`);
      } catch (e) { results.push({ variant, file, ok: false, error: e.message }); console.log(`failed: ${e.message}`); }
    }
  }
} finally {
  if (saved) await j('/api/dev/plan-guidance', { enabled: !!saved.enabled, thinking: saved.thinking ?? null, maxSteps: saved.maxSteps ?? 20 }).catch(() => {});
}

console.log('\nvariant  plans  avg steps  avg detail  points covered  last=check  avg secs');
for (const variant of variants) {
  const rows = results.filter(r => r.variant === variant), ok = rows.filter(r => r.ok);
  const avg = f => ok.length ? (ok.reduce((a, r) => a + (r[f] ?? 0), 0) / ok.length).toFixed(1) : '-';
  const withPoints = ok.filter(r => r.pointsCovered !== null);
  const pc = withPoints.length ? `${Math.round(100 * withPoints.reduce((a, r) => a + r.pointsCovered, 0) / withPoints.length)} %` : '-';
  console.log(`${variant.padEnd(8)} ${String(ok.length).padStart(2)}/${rows.length}  ${avg('steps').padStart(9)}  ${avg('withDetail').padStart(10)}  ${pc.padStart(14)}  ${String(ok.filter(r => r.last).length).padStart(7)}/${ok.length}  ${avg('secs').padStart(8)}`);
}
writeFileSync(outFile, JSON.stringify({ at: new Date().toISOString(), dir, variants, results }, null, 1));
console.log(`\nwritten: ${outFile}`);
