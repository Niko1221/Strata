// Community benchmark sweep for Strata (OpenAI-compatible server on 127.0.0.1:18438).
// Fresh prompts of about 4,096 / 32,768 / 128,000 tokens, 3 runs each, output cap 256.
// Each prompt starts with a unique random block, so no prefix is reused between runs.
// Engine timings come from /metrics (requests[0] after each call); TTFT is measured client-side on the stream
// (first non-empty delta, reasoning or answer). Usage: node bench_sweep.mjs [out.json]
import { writeFileSync } from 'node:fs';
import { randomBytes } from 'node:crypto';

const URL_BASE = 'http://127.0.0.1:18438';
const TARGETS = [4096, 32768, 128000];
const RUNS = 3, CAP = 256;
const out = process.argv[2] || 'sweep.json';
const WORDS = ('river stone garden window paper silver orange planet market winter music forest engine harbor '
  + 'letter candle bridge mountain valley thunder copper meadow lantern island signal compass ladder theater '
  + 'velvet canyon falcon glacier marble orchard pepper quarry saddle timber violet walnut yellow zephyr').split(' ');

function filler(tokens, seed) {
  // about 1.3 tokens per word for this vocabulary; the real count comes back from the engine
  const words = Math.round(tokens / 1.3);
  let s = 'Session ' + seed + '. ', x = parseInt(seed.slice(0, 8), 16) || 1;
  for (let i = 0; i < words; i++) {
    x = (x * 1103515245 + 12345) % 2147483648;
    s += WORDS[x % WORDS.length] + ((i % 17 === 16) ? '. ' : ' ');
  }
  return s;
}

async function metrics() { return (await fetch(URL_BASE + '/metrics')).json(); }

async function one(target, run) {
  const seed = randomBytes(8).toString('hex');
  const prompt = filler(target - 60, seed) + '\n\nSummarize the text above in one short paragraph.';
  const body = { model: 'local', stream: true, max_tokens: CAP, temperature: 0,
    messages: [{ role: 'user', content: prompt }] };
  const t0 = performance.now(); let ttft = null, chars = 0;
  const r = await fetch(URL_BASE + '/v1/chat/completions', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const dec = new TextDecoder(); let buf = '';
  for await (const chunk of r.body) {
    buf += dec.decode(chunk, { stream: true });
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
      if (!line.startsWith('data:') || line === 'data: [DONE]') continue;
      try {
        const d = JSON.parse(line.slice(5)).choices?.[0]?.delta || {};
        const txt = (d.content || '') + (d.reasoning_content || '');
        if (txt && ttft === null) ttft = (performance.now() - t0) / 1000;
        chars += txt.length;
      } catch { /* keep-alive or partial */ }
    }
  }
  const total = (performance.now() - t0) / 1000;
  const m = await metrics();
  const e = m.requests[0];
  const hw = m.hardware || {};
  return { target, run, status: r.status, seed,
    prompt_tokens: e.prompt_tokens, reused: e.reused, prompt_read: e.prompt_read, output_tokens: e.output_tokens,
    finish: e.finish, prompt_ms: e.prompt_ms, decode_ms: e.decode_ms,
    prompt_tok_s: e.prompt_read && e.prompt_ms ? +(e.prompt_read / (e.prompt_ms / 1000)).toFixed(1) : null,
    decode_tok_s: e.decode_tok_s, ttft_s: ttft === null ? null : +ttft.toFixed(2), total_s: +total.toFixed(2),
    gpu_mem_used_gib: hw.gpu_mem_used ? +(hw.gpu_mem_used / 2 ** 30).toFixed(2) : null,
    ram_used_gib: hw.ram_used ? +(hw.ram_used / 2 ** 30).toFixed(2) : null, gpu_power_w: hw.gpu_power ?? null,
    answer_chars: chars };
}

const res = { engine: (await metrics()).engine, started: new Date().toISOString(), runs: [] };
for (const t of TARGETS) {
  for (let r = 1; r <= RUNS; r++) {
    const x = await one(t, r);
    res.runs.push(x);
    console.log(JSON.stringify(x));
    writeFileSync(out, JSON.stringify(res, null, 1));
  }
}
res.finished = new Date().toISOString();
writeFileSync(out, JSON.stringify(res, null, 1));
