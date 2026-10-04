// Run against a local test server: STRATA_TEST_URL=http://127.0.0.1:8080 node tools/test_web_i18n.cjs
// Requires Playwright and a Chromium browser. No model or GPU is needed.
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
const base = process.env.STRATA_TEST_URL || 'http://127.0.0.1:8080';
(async () => {
  const browser = await chromium.launch({headless: true, ...(process.platform === 'win32' ? {channel: 'msedge'} : {})});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base);
    await page.waitForFunction(() => document.getElementById('pill-text').textContent === 'Idle');
    assert.equal(await page.locator('html').getAttribute('lang'), 'en');
    assert.equal(await page.locator('[data-tab="chat"]').innerText(), 'Chat');
    await page.locator('#input').fill('مسودة عربية with English');
    await page.locator('[data-language]').selectOption('ar');
    assert.equal(await page.locator('html').getAttribute('dir'), 'rtl');
    assert.equal(await page.locator('[data-tab="chat"]').innerText(), 'المحادثة');
    assert.equal(await page.locator('#input').inputValue(), 'مسودة عربية with English');
    assert.equal(await page.locator('#input').getAttribute('dir'), 'auto');
    await page.locator('[data-tab="monitor"]').click();
    assert.match(await page.locator('#metrics').innerText(), /حمل البطاقة/);
    await page.locator('[data-tab="about"]').click();
    assert.match(await page.locator('#facts-engine').innerText(), /النموذج/);
    assert.equal(await page.locator('#facts-api [data-copy]').first().getAttribute('aria-label'), 'نسخ');
    await page.locator('[data-tab="chat"]').click();
    await page.locator('#sampling-btn').click();
    await page.locator('#s-temp').fill('0.85');
    await page.locator('[data-language]').selectOption('en', {force: true});
    assert.equal(await page.locator('#s-temp').inputValue(), '0.85');
    await page.locator('#drawer-close').click();
    await page.locator('[data-tab="chat"]').click();

    // Keep a request pending while switching: streaming must still target the original message element.
    let release;
    const pending = new Promise(resolve => { release = resolve; });
    let requestBody;
    await page.route('**/v1/chat/completions', async route => {
      requestBody = route.request().postDataJSON();
      await pending;
      const content = 'مرحبًا بالعربية\n\n```js\nconst x = 1;\n```';
      const chunk = {choices: [{delta: {content}}], usage: {completion_tokens: 12}};
      await route.fulfill({contentType: 'text/event-stream', body: `data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`});
    });
    await page.locator('#send-btn').click();
    await page.waitForFunction(() => !document.getElementById('stop-btn').hidden);
    await page.locator('[data-language]').selectOption('ar');
    release();
    await page.waitForFunction(() => document.getElementById('stop-btn').hidden);
    assert.equal(requestBody.messages[0].content, 'مسودة عربية with English');
    assert.equal(requestBody.reasoning_effort, 'high');
    assert.match(await page.locator('.st-msg--assistant').innerText(), /مرحبًا بالعربية/);
    assert.equal(await page.locator('[data-code-copy]').getAttribute('aria-label'), 'نسخ الكود');
    assert.equal(await page.locator('[data-msg-copy]').getAttribute('aria-label'), 'نسخ الإجابة');
    assert.equal(await page.locator('.st-code pre').evaluate(el => getComputedStyle(el).direction), 'ltr');
    assert.equal(await page.locator('.st-msg--assistant .st-bubble').getAttribute('dir'), 'auto');
    await page.reload();
    assert.equal(await page.locator('html').getAttribute('lang'), 'ar');
    assert.match(await page.locator('.st-msg--assistant').innerText(), /مرحبًا بالعربية/);
    await page.locator('[data-language]').selectOption('en');
    await page.reload();
    assert.equal(await page.locator('html').getAttribute('dir'), 'ltr');

    await page.goto(base + '/api-monitor');
    await page.locator('[data-language]').selectOption('ar');
    assert.equal(await page.locator('h1').innerText(), 'مراقبة API');
    assert.equal(await page.locator('#filter option[value="completed"]').innerText(), 'مكتمل');
    assert.equal(await page.locator('#filter option[value="completed"]').getAttribute('value'), 'completed');
    await page.reload();
    assert.equal(await page.locator('html').getAttribute('dir'), 'rtl');

    await page.setViewportSize({width: 390, height: 844});
    await page.goto(base);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.locator('[data-tab="monitor"]').click();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    // Unsupported saved values and unavailable storage still use English.
    await page.evaluate(() => localStorage.setItem('strata.language', 'fr'));
    await page.reload();
    assert.equal(await page.locator('html').getAttribute('lang'), 'en');
    const blocked = await browser.newContext();
    await blocked.addInitScript(() => {
      Object.defineProperty(window, 'localStorage', {get() { throw new Error('blocked'); }});
    });
    const privatePage = await blocked.newPage();
    privatePage.on('pageerror', error => errors.push(error.message));
    await privatePage.goto(base);
    await privatePage.locator('[data-language]').selectOption('ar');
    assert.equal(await privatePage.locator('html').getAttribute('dir'), 'rtl');
    assert.deepEqual(errors, []);
    await blocked.close();
    console.log('PASS: defaults, Arabic, RTL, persistence, active request, unchanged API values, code direction, API monitor, mobile, storage fallback');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
