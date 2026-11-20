import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';
import { proxyCompletion } from '../server/inference.mjs';
import { update } from '../server/store.mjs';
import { createLlmProxy, samplingFromExpert } from '../server/agent/llm-shape.mjs';

const listen = server => new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const close = server => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); });

test('loaded dense architecture adapts ordinary and agent sampling without changing generic requests', async t => {
  const f = nativeFixture();
  t.after(() => f.cleanup());
  const exe = join(f.repo, 'strata-q38.exe');
  writeFileSync(exe, 'fixture');
  const path = join(f.repo, 'strata-iq2_xs.json');
  writeFileSync(path, JSON.stringify({ arch: 'qwen38_dense', native_runner: 'qwen38_dense', exe,
    args: ['--model', resolve('original.gguf')], tokenizer: resolve('original.gguf'),
    max_context: 4096, parallel: 1, drafting: 'off', mtp: false }));
  await f.control.refresh(true);
  const loaded = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
  assert.equal(loaded.state, 'done', loaded.error);
  const instance = f.control.resolveInstance(loaded.result.instanceId);
  assert.equal(instance.servingPolicy, 'q38dense', 'architecture wins over the ordinary-looking model ID');
  // A future settings edit must not change the contract of this loaded instance.
  writeFileSync(path, JSON.stringify({ arch: 'qwen4exp' }));
  assert.equal(f.control.resolveInstance(instance.id).servingPolicy, 'q38dense');

  const seen = [];
  const upstream = http.createServer(async (req, res) => {
    let raw = ''; for await (const chunk of req) raw += chunk;
    seen.push({ body: JSON.parse(raw), authorization: req.headers.authorization });
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.end('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n');
  });
  await listen(upstream); t.after(() => close(upstream));
  const baseUrl = `http://127.0.0.1:${upstream.address().port}/v1`;
  let expert = {};
  let policy = instance.servingPolicy;
  const proxy = createLlmProxy({ targetOf: () => baseUrl, samplingNow: () => samplingFromExpert(expert),
    policyOf: base => base === baseUrl ? policy : null });
  const front = http.createServer((req, res) => proxy(req, res, 'fixture', '/v1/chat/completions'));
  await listen(front); t.after(() => close(front));
  const ordinary = async extra => {
    update({ expert });
    await proxyCompletion({ baseUrl, servingPolicy: policy, modelId: instance.apiModelId,
      authorization: 'fixture-key', message: 'hello', extra, onFrame: () => {} });
    assert.equal(seen.at(-1).authorization, 'Bearer fixture-key');
    return seen.at(-1).body;
  };
  const agent = async (body = { messages: [] }) => {
    const response = await fetch(`http://127.0.0.1:${front.address().port}/v1/chat/completions`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
    });
    assert.equal(response.status, 200); await response.text(); return seen.at(-1).body;
  };
  for (const projection of [undefined, true, false]) {
    expert = { ...(projection === undefined ? {} : { 'sample.speedProjection': projection }) };
    for (const body of [await ordinary(), await agent()]) {
      assert.equal(Object.hasOwn(body, 'experimental_speed_projection'), false);
      assert.equal(Object.hasOwn(body, 'seed'), false);
    }
  }
  expert = { 'sample.seed': 7, 'sample.temp': 0.2, 'sample.speedProjection': true };
  for (const body of [await ordinary(), await agent()]) {
    assert.equal(body.seed, 7); assert.equal(body.temperature, 0.2);
    assert.equal(Object.hasOwn(body, 'experimental_speed_projection'), false);
  }
  const explicit = { experimental_speed_projection: true, seed: -1 };
  for (const body of [await ordinary(explicit), await agent({ messages: [], ...explicit })]) {
    assert.equal(body.experimental_speed_projection, true, 'explicit API fields still reach backend validation');
    assert.equal(body.seed, -1);
  }
  for (policy of [null, 'q38', 'q36', 'deepseek41']) {
    expert = {};
    const defaults = await ordinary();
    assert.equal(defaults.experimental_speed_projection, true); assert.equal(defaults.seed, -1);
    assert.deepEqual(await agent(), { messages: [] });
    for (const projection of [true, false]) {
      expert = { 'sample.speedProjection': projection };
      assert.equal((await ordinary()).experimental_speed_projection, projection);
      assert.equal((await agent()).experimental_speed_projection, projection);
    }
  }
});
