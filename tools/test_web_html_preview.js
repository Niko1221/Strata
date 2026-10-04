// Browser regression checks, using a separate Playwright CLI session (no GPU requests).
// From the repository root, with Strata running:
//   playwright-cli -s=html-preview-test open http://127.0.0.1:8080 --browser=msedge
//   playwright-cli -s=html-preview-test run-code --filename=tools/test_web_html_preview.js
//   playwright-cli -s=html-preview-test close
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const fence = "`".repeat(3);
  const code = `<!doctype html><html><head><style>
    body { margin: 0; padding: 40px; font: 18px system-ui; background: #f0fdfa; }
    h1 { color: rgb(15, 118, 110); }
    button { padding: 12px 24px; border: 0; border-radius: 12px; background: #0f766e; color: white; font: inherit; }
    </style></head><body><h1>HTML preview</h1><p>CSS and JavaScript work here.</p>
    <button id="count" onclick="this.textContent = Number(this.textContent) + 1">0</button>
    <script>
      try { parent.document.body.dataset.previewEscaped = "yes"; } catch (e) { document.body.dataset.parentBlocked = "yes"; }
      try { localStorage.getItem("preview-test-secret"); } catch (e) { document.body.dataset.storageBlocked = "yes"; }
      fetch("/metrics?preview-network-test").then(() => { document.body.dataset.fetchBlocked = "no"; })
        .catch(() => { document.body.dataset.fetchBlocked = "yes"; });
    </script><img src="https://example.invalid/preview-network-test.png">
    ${"<!-- A long code example to exercise internal scrolling. -->\n".repeat(40)}</body></html>`;
  const second = '<h1>Second &amp; &lt;tag&gt;</h1>';
  const text = `${fence}HTML title=demo\n${code}\n${fence}\n` +
    `${fence}html\n${second}\n${fence}\n` +
    `${fence}htm\n<p>HTM snippet</p>\n${fence}\n` +
    `${fence}\n<!DOCTYPE html><p>Unlabelled page</p>\n${fence}\n` +
    `${fence}python\nprint("<h1>not HTML</h1>")\n${fence}\n` +
    '<img src="/preview-network-test" onerror="document.body.dataset.previewEscaped=\'yes\'">';
  const previous = await page.evaluate(() => ({
    chat: localStorage.getItem("strata.chat"), secret: localStorage.getItem("preview-test-secret"),
  }));
  const attemptedRequests = [];
  // CSP failures can emit a request event without sending anything. Intercept
  // requests that actually reach the network, also preventing accidental traffic.
  const networkPattern = "**/*preview-network-test*";
  const blockRequest = route => { attemptedRequests.push(route.request().url()); return route.abort(); };
  await page.route(networkPattern, blockRequest);
  try {
    await page.setViewportSize({width: 1920, height: 1080});
    await page.evaluate(text => {
      localStorage.setItem("strata.chat", JSON.stringify([{role: "assistant", text, reasoning: "", time: Date.now()}]));
      localStorage.setItem("preview-test-secret", "must not be readable in the preview");
    }, text);
    await page.reload();
    const buttons = page.locator("[data-code-preview]");
    await buttons.first().waitFor();
    check(await buttons.count() === 4, "Preview must recognize HTML aliases and unlabelled documents, not Python");
    check(await page.locator(".st-code pre").first().textContent() === code, "Code markup must remain escaped and unchanged");
    check(await page.locator("#html-preview-content iframe").count() === 0, "HTML must not execute before Preview is clicked");
    check(await page.locator("body").getAttribute("data-preview-escaped") === null, "Raw HTML escaped into the chat");
    const codeScrolls = await page.locator(".st-code pre").first().evaluate(el => {
      el.scrollTop = el.scrollHeight;
      return {height: el.clientHeight, content: el.scrollHeight, scrolled: el.scrollTop};
    });
    check(codeScrolls.height <= 500 && codeScrolls.content > codeScrolls.height && codeScrolls.scrolled > 0,
      `Long code blocks must stay capped at 500 pixels and scroll internally: ${JSON.stringify(codeScrolls)}`);

    await buttons.first().click();
    const preview = page.frameLocator("#html-preview-content iframe");
    await preview.locator("#count").waitFor();
    check(await page.locator("#html-preview").getAttribute("open") !== null, "Preview must open a modal dialog");
    const desktopBounds = await page.locator("#html-preview").boundingBox();
    check(Math.abs(desktopBounds.x - 96) < 1 && Math.abs(desktopBounds.width - 1728) < 1,
      "Preview must leave 5% margins on both sides of a desktop window");
    check(await preview.locator("h1").evaluate(el => getComputedStyle(el).color) === "rgb(15, 118, 110)", "Inline CSS must render");
    await preview.locator("#count").click();
    check(await preview.locator("#count").textContent() === "1", "Inline JavaScript must respond to clicks");
    await preview.locator('body[data-fetch-blocked="yes"]').waitFor();
    check(await preview.locator("body").getAttribute("data-parent-blocked") === "yes", "Preview must not access the parent DOM");
    check(await preview.locator("body").getAttribute("data-storage-blocked") === "yes", "Preview must not access browser storage");
    check(attemptedRequests.length === 0, "Preview must block fetches and external images");
    check(await page.locator("body").getAttribute("data-preview-escaped") === null, "Preview modified the parent page");
    await page.screenshot({path: ".playwright-cli/html-preview-desktop.png"});

    await page.getByRole("button", {name: "Reload", exact: true}).click();
    await preview.locator("#count").waitFor();
    check(await preview.locator("#count").textContent() === "0", "Reload must restart the current snapshot");
    await page.getByRole("button", {name: "Close", exact: true}).click();
    await page.locator("#html-preview").waitFor({state: "hidden", timeout: 5000});
    await page.locator("#html-preview-content iframe").waitFor({state: "detached"});
    check(await page.locator("#html-preview-content iframe").count() === 0, "Closing must remove the running frame");
    check(await buttons.first().evaluate(el => document.activeElement === el), "Close must restore focus to its Preview button");

    // Exercise the copy control without changing the user's system clipboard.
    await page.evaluate(() => Object.defineProperty(navigator.clipboard, "writeText", {
      configurable: true, value: async text => { window.previewTestCopied = text; },
    }));
    await page.locator("[data-code-copy]").first().click();
    check(await page.evaluate(() => window.previewTestCopied) === code, "Copy must preserve the original HTML");
    await buttons.nth(1).click();
    check(await preview.locator("h1").textContent() === "Second & <tag>", "Each HTML block must open its own code");
    await page.getByRole("button", {name: "Close", exact: true}).focus();
    await page.keyboard.press("Escape");
    await page.locator("#html-preview").waitFor({state: "hidden", timeout: 5000});
    await page.locator("#html-preview-content iframe").waitFor({state: "detached"});
    check(await page.locator("#html-preview-content iframe").count() === 0, "Escape must also stop the preview");

    await page.setViewportSize({width: 390, height: 844});
    await buttons.first().click();
    await preview.locator("#count").waitFor();
    const bounds = await page.locator("#html-preview").boundingBox();
    check(Math.abs(bounds.x - 19.5) < 1 && Math.abs(bounds.width - 351) < 1 && bounds.y >= 0 && bounds.y + bounds.height <= 844,
      "Preview must keep 5% side margins on mobile and fit its height");
    await page.screenshot({path: ".playwright-cli/html-preview-mobile.png"});
    // Escape must also work while focus is inside the generated document.
    await preview.locator("#count").click();
    await page.keyboard.press("Escape");
    await page.locator("#html-preview").waitFor({state: "hidden", timeout: 5000});
    await page.locator("#html-preview-content iframe").waitFor({state: "detached"});
    await page.reload();
    await buttons.first().waitFor();
    check(await buttons.count() === 4 && await page.locator("#html-preview-content iframe").count() === 0,
      "Saved HTML answers must keep Preview controls without automatically running their code");
    return "Passed: rendering, JavaScript, reload, copy, multiple blocks, keyboard/focus, mobile, saved chats and isolation";
  } finally {
    await page.unroute(networkPattern, blockRequest);
    await page.evaluate(previous => {
      for (const [key, value] of [["strata.chat", previous.chat], ["preview-test-secret", previous.secret]]) {
        if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value);
      }
    }, previous);
    await page.setViewportSize({width: 1280, height: 720});
    await page.reload();
  }
}
