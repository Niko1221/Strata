// Lifetime Monitor regression checks; no model/GPU or production server.
// WEBTEST_MODULES=<folder containing node_modules/puppeteer-core> node tools/webtest/usage_test.mjs
import assert from 'node:assert/strict';
import { startMock } from './mock.mjs';
import { launch, openPage, runner } from './lib.mjs';
const { test, run } = runner('usage browser tests');
const total = { requests: 4, output_tokens: 72, prompt_tokens: 200, reused: 150,
  prompt_ms: 1000, decode_ms: 2000, drafts_offered: 100, drafts_accepted: 70 };
const usage = { enabled: true, since: 1792260287, lifetime: total,
  imported: { requests: 2, output_tokens: 28 },
  views: { '7d': { total }, '1d': { total }, '30d': { total }, custom: { total } } };
async function withPage(browser, mock, fn, seed = {}) {
  const page = await openPage(browser, mock, { hash: '#monitor', seed });
  try { await fn(page); assert.deepEqual(page.errors, []); }
  finally { await page.browserContext().close(); }
}
test('disabled: Usage stays hidden and ordinary Monitor renders', async ({ browser, mock }) => {
  await withPage(browser, mock, async (page) => {
    assert.equal(await page.$eval('#usage-card', e => e.hidden), true);
    assert.equal(await page.$eval('#pill-text', e => e.textContent), 'Idle');
  });
});
test('enabled: all-time includes imports, range excludes them, card stays last', async ({ browser, mock }) => {
  await withPage(browser, mock, async (page) => {
    await page.evaluate(u => { lastMetrics.usage = u; render(lastMetrics); }, usage);
    assert.equal(await page.$eval('#usage-lifetime-value', e => e.textContent), '100');
    assert.match(await page.$eval('#usage-grid', e => e.textContent), /72/);
    assert.match(await page.$eval('#usage-imported', e => e.textContent), /28 tokens recovered/);
    assert.equal(await page.$eval('#usage-card', e => e.nextElementSibling === null), true);
    await page.click('[data-range="all"]');
    assert.equal(await page.$eval('#usage-lifetime-value', e => e.textContent), '100');
  });
});
test('custom refresh: dates and Show all coexist, without spawning a poll timer', async ({ browser, mock }) => {
  await withPage(browser, mock, async (page) => {
    const result = await page.evaluate(async u => {
      usageRange = 'custom'; prefillCustomRange(); reqShowAll = true;
      const oldFetch = window.fetch, oldTimeout = window.setTimeout;
      let url, timers = 0;
      window.fetch = async v => { url = v; return {ok:true, json:async()=>({...lastMetrics, usage:u})}; };
      window.setTimeout = () => { timers++; };
      try { await refreshMetrics(); return {url, timers}; }
      finally { window.fetch = oldFetch; window.setTimeout = oldTimeout; }
    }, usage);
    assert.match(result.url, /requests=all/); assert.match(result.url, /from_ts=/); assert.match(result.url, /to_ts=/);
    assert.equal(result.timers, 0);
    assert.equal(await page.$eval('#usage-lifetime-value', e => e.textContent), '100');
  });
});
test('stale poll after unload: held state survives, no extra polling loop', async ({ browser, mock }) => {
  await withPage(browser, mock, async (page) => {
    const result = await page.evaluate(async () => {
      const oldFetch = window.fetch, oldTimeout = window.setTimeout;
      let resolveReply, timers = 0;
      const stale = {...lastMetrics, live:{...lastMetrics.live, state:'idle'}};
      window.fetch = () => new Promise(resolve => { resolveReply = resolve; });
      window.setTimeout = () => { timers++; };
      try {
        const pending = refreshMetrics();
        metricsHold = {state:'unloaded', after:metricsEpoch};
        lastMetrics = {...lastMetrics, live:{...lastMetrics.live, state:'unloaded'}};
        resolveReply({ok:true, json:async()=>stale});
        await pending;
        return {state:lastMetrics.live.state, timers, held:!!metricsHold};
      } finally { window.fetch = oldFetch; window.setTimeout = oldTimeout; metricsHold = null; }
    });
    assert.deepEqual(result, {state:'unloaded', timers:0, held:true});
  });
});
test('saved Custom initializes before first poll and language startup completes', async ({ browser, mock }) => {
  await withPage(browser, mock, async page => {
    assert.equal(await page.evaluate(() => usageRange), 'custom');
    assert.match(await page.evaluate(() => metricsUrl()), /from_ts=.+&to_ts=/);
    assert.equal(await page.$eval('#lang', e => e.value), 'en');
  }, {'usage.view':'custom', lang:'en'});
});
test('invalid saved range falls back to seven days', async ({ browser, mock }) => {
  await withPage(browser, mock, async page => assert.equal(await page.evaluate(() => usageRange), '7d'), {'usage.view':'bogus'});
});
test('ownership error is visible without claiming durable counters', async ({ browser, mock }) => {
  await withPage(browser, mock, async page => {
    await page.evaluate(() => renderUsage({enabled:false, persistence_error:'Store already owned'}));
    assert.equal(await page.$eval('#usage-card', e => e.hidden), false);
    assert.equal(await page.$eval('#usage-lifetime', e => e.hidden), true);
    assert.match(await page.$eval('#usage-warning', e => e.textContent), /Store already owned/);
  });
});
await run(async () => {
  const mock = await startMock(), browser = await launch();
  return {mock, browser, close:async()=>{await browser.close(); await mock.stop();}};
});
