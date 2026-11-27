// Run with STRATA_TEST_CHROME set to a Chromium executable. The real component,
// browser File objects and IndexedDB exercise the file-picker path end to end.
import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { chromium } from 'playwright-core';

test('picture attachment errors describe the actual reason and keep other files', {skip: !process.env.STRATA_TEST_CHROME && 'set STRATA_TEST_CHROME to run browser test'}, async t => {
  const server = await createServer({root: fileURLToPath(new URL('..', import.meta.url)), server:{host:'127.0.0.1',port:0}, logLevel:'error'});
  await server.listen();
  t.after(() => server.close());
  const browser = await chromium.launch({executablePath:process.env.STRATA_TEST_CHROME, headless:true});
  t.after(() => browser.close());
  const page = await browser.newPage();
  await page.route('**/api/native', route => route.fulfill({json:{available:true,defaultId:'vision',instances:[{id:'vision',status:'ready',vision:true,model:'test',label:'Test',gpus:[]}],jobs:[],catalog:{families:[]}}}));
  const ready = page.waitForResponse(r => r.url().endsWith('/api/native'));
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/tests/fixtures/picture-feedback.html`);
  await ready;
  const input = page.locator('input[type=file]').first();
  const file = (name, mimeType, text = 'test') => ({name,mimeType,buffer:Buffer.from(text)});
  await input.setInputFiles([file('picture.bmp','image/bmp'),file('notes.txt','text/plain','hello')]);
  await page.waitForFunction(() => !!document.querySelector('#notice')?.textContent);
  assert.match(await page.locator('#notice').textContent(), /PNG, JPEG or WebP/);
  assert.match(await page.locator('#attached').textContent(), /hello/);
  await input.setInputFiles([1,2,3,4].map(i=>file(`${i}.png`,'image/png')));
  await page.waitForFunction(() => document.querySelector('#notice')?.textContent?.includes('three pictures'));
  assert.match(await page.locator('#notice').textContent(), /up to three pictures/);
});
