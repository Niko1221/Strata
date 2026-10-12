"""Real-browser contract, Markdown, streaming, persistence and engine tests (no GPU).

Requires playwright and jinja2. Run: python tools/test_web_math.py --browser chromium
Use --browser firefox or webkit after installing that Playwright browser.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(browser_name, output):
    from playwright.sync_api import sync_playwright
    from serve.frontend import ChatTemplate
    from serve.server import ByteTokenizer, MockEngine, Service, serve
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    server = serve(svc, port=0)
    base = f"http://127.0.0.1:{server.server_port}/"
    report = {"browser": browser_name, "vendor_bytes": sum(p.stat().st_size for p in
        (ROOT / "serve/web/vendor").rglob("*") if p.is_file()), "equations": []}
    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch()
            context = browser.new_context(viewport={"width": 390, "height": 844})
            context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(base) else route.abort())
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            requests = []
            page.on("request", lambda request: requests.append(request.url))
            page.goto(base)
            page.wait_for_function("typeof settings !== 'undefined'")
            assert not page.evaluate("settings.math")
            assert not any("/vendor/" in url for url in requests)
            assert page.request.get(base + "web/vendor/../../server.py").status == 404
            # Parser facts and every possible streamed prefix of representative equations.
            page.evaluate(r"""() => {
              const assert = (x, label) => { if (!x) throw Error(label); };
              for (const source of ['\\(a_1+b\\)', '$x+y$', '$$a\nb$$', '\\[\\frac{1}{2}\\]']) {
                for (let i = 0; i <= source.length; i++) {
                  const parts = StrataMath.split(source.slice(0, i));
                  assert(parts.map(p => p.raw).join('') === source.slice(0, i), 'stream source');
                  if (i < source.length) assert(!parts.some(p => p.source !== undefined), 'premature rendering');
                }
                assert(StrataMath.split(source).some(p => p.source !== undefined), 'complete equation');
              }
              for (const literal of ['$5 and $10', '$5$', '\\$x\\$', '`$x$`', '```tex\n$x$\n```', '~~~tex\n$x$\n~~~', '$x\ny$'])
                assert(!StrataMath.split(literal).some(p => p.source !== undefined), 'literal ' + literal);
              assert(StrataMath.split('`$x$` $y$').filter(p => p.source !== undefined).length === 1, 'code precedence');
              assert(StrataMath.split('$2 + x$').some(p => p.source !== undefined), 'numeric expression');
              assert(!StrataMath.markdown('[link](https://example.com/$$x$$)', markdown).includes('data-math'), 'link destination');
              const source = '\\(a_1 * b \\mid c\\)';
              const html = StrataMath.markdown(source, markdown);
              assert(!html.includes('<em>'), 'math protected from Markdown');
              const table = StrataMath.markdown('| name | equation |\n|---|---|\n| x | \\(a|b\\) |', markdown);
              assert((table.match(/<td>/g) || []).length === 2, 'math pipe protected');
              const hostile = StrataMath.markdown('\\(<img src=x onerror=alert(1)>\\)', markdown);
              assert(!hostile.includes('<img'), 'source escaped');
              assert(markdown('hello **world**') === StrataMath.markdown('hello **world**', markdown), 'ordinary Markdown');
              const fallback = document.createElement('div');
              const literalDollars = '$$x$$ and \\(x + $&\\)';
              fallback.innerHTML = StrataMath.markdown(literalDollars, markdown);
              assert(fallback.textContent === literalDollars, 'fallback dollars preserved');
            }""")
            original = r"Answer: \(\frac{1}{2}\), $x_1$, and $$\sum_{n=1}^{3}n=6$$."
            page.evaluate("text => { messages = [{role:'assistant', text, reasoning:'$literal$', time:Date.now()}]; saveChat(); renderChat(); }", original)
            assert page.locator("[data-math]").count() == 0
            baseline = page.locator(".st-bubble").inner_html()
            assert baseline == page.evaluate("answerHtml(messages[0])")
            page.evaluate("globalThis.mathLiveElement = document.querySelector('.st-msg--assistant'); busy = {msg: messages[0]};")
            page.click("#sampling-btn")
            page.focus("#s-math")
            page.keyboard.press("Space")
            page.click("#s-apply")
            assert page.evaluate("globalThis.mathLiveElement === document.querySelector('.st-msg--assistant')"), 'streaming element preserved'
            page.evaluate("busy = null; updateAssistant(globalThis.mathLiveElement, messages[0], false)")
            page.wait_for_function("document.querySelectorAll('[data-math]').length === 3")
            started = time.perf_counter()
            page.wait_for_function("[...document.querySelectorAll('[data-math]')].every(e => e.firstElementChild && !e.dataset.mathError)")
            report["cold_chat_ms"] = round((time.perf_counter() - started) * 1000, 2)
            # The full engine corpus is capability evidence: invalid inputs must fail,
            # while Unicode coverage can differ between the real TeX and browser engines.
            fixtures = json.loads((ROOT / "serve/fixtures/math.json").read_text())
            for fixture in fixtures:
                result = page.evaluate("""async f => {
                  const start = performance.now();
                  const result = await StrataMath.result(f.source, f.display);
                  const first = performance.now() - start;
                  const cached = performance.now(); await StrataMath.result(f.source, f.display);
                  return {ok:result.ok, reason:result.reason, ms:first, cached_ms:performance.now()-cached};
                }""", fixture)
                report["equations"].append({"name": fixture["name"], **result})
                if fixture.get("invalid"):
                    assert not result["ok"], fixture
                elif fixture["name"] != "unicode":
                    assert result["ok"], (fixture, result)
            page.evaluate("async () => { await document.fonts.ready; }")
            report["vendor_transfer_bytes"] = page.evaluate("performance.getEntriesByType('resource').filter(e=>e.name.includes('/vendor/')).reduce((n,e)=>n+e.encodedBodySize,0)")
            page.evaluate(r"""async () => {
              const assert = (x, label) => { if (!x) throw Error(label); };
              const defined = await StrataMath.result('\\def\\strataPrivateMacro{x}\\strataPrivateMacro', false);
              assert(defined.ok, 'local macro definition: ' + defined.reason);
              const leaked = await StrataMath.result('\\strataPrivateMacro', false);
              assert(!leaked.ok, 'macro isolation');
              const oversized = await StrataMath.result('α'.repeat(9000), false);
              assert(!oversized.ok && oversized.reason === 'limit', 'UTF-8 source limit');
              const rendered = [...document.querySelectorAll('[data-math]')];
              assert(rendered.every(e => e.querySelector('math') || e.querySelector('img[alt]')), 'accessible representation');
              const adapter = StrataMathAdapter;
              globalThis.StrataMathAdapter = {initialize:async()=>{throw Error('missing')},render:async()=>{throw Error('unexpected')}};
              StrataMath.reset();
              const unavailable = await StrataMath.result('unavailable', false);
              assert(!unavailable.ok && unavailable.reason === 'unavailable', 'missing engine');
              globalThis.StrataMathAdapter = adapter; StrataMath.reset();
              // Conversion may ignore AbortSignal (e.g. font loading). The caller
              // must settle at the deadline without starting overlapping work.
              const timeout = AbortSignal.timeout;
              let finish, calls = 0;
              globalThis.StrataMathAdapter = {initialize:async()=>{}, render:()=>{
                calls++; return new Promise(resolve => finish = resolve);
              }};
              AbortSignal.timeout = () => timeout.call(AbortSignal, 30);
              StrataMath.reset();
              try {
                const expired = await StrataMath.result('deadline', false);
                assert(!expired.ok && expired.reason === 'limit', 'render deadline');
                const next = StrataMath.result('queued', false);
                await new Promise(resolve => setTimeout(resolve, 50));
                assert(calls === 1, 'conversions remain serialized after timeout');
                finish({ok:false, reason:'invalid'});
                while (calls === 1) await new Promise(resolve => setTimeout(resolve, 0));
                finish({ok:false, reason:'invalid'}); await next;
              } finally {
                AbortSignal.timeout = timeout;
                globalThis.StrataMathAdapter = adapter; StrataMath.reset();
              }
            }""")
            assert page.evaluate("apiMessages()[0].content") == original
            assert page.evaluate("JSON.parse(localStorage.getItem('strata.chat'))[0].text") == original
            page.evaluate("Object.defineProperty(navigator, 'clipboard', {configurable:true, value:{writeText:async text=>{globalThis.mathCopied=text}}})")
            page.click('[data-msg-copy]')
            assert page.evaluate('globalThis.mathCopied') == original
            with page.expect_download() as download:
                page.click('#export-btn')
            assert original in Path(download.value.path()).read_text(encoding='utf-8')

            page.reload()
            page.wait_for_function("settings.math && document.querySelector('[data-math]')")
            page.wait_for_function("[...document.querySelectorAll('[data-math]')].every(e => e.firstElementChild && !e.dataset.mathError)")
            assert page.evaluate("messages[0].text") == original
            # No late result may replace a newer message, or render after disabling.
            page.evaluate("""async () => {
              const container = document.createElement('div'); document.body.append(container);
              let finish;
              const adapter = StrataMathAdapter;
              globalThis.StrataMathAdapter = {initialize:async()=>{}, render:()=>new Promise(r=>finish=r)};
              StrataMath.reset();
              container.innerHTML = StrataMath.markdown('\\\\(stale\\\\)', markdown);
              StrataMath.hydrate(container);
              while (!finish) await new Promise(r=>setTimeout(r, 0));
              container.innerHTML = 'new content';
              finish({ok:true, element:document.createElement('strong')});
              await new Promise(r=>setTimeout(r, 0));
              if (container.textContent !== 'new content') throw Error('stale result');
              container.remove(); globalThis.StrataMathAdapter = adapter; StrataMath.reset();
            }""")
            page.evaluate("document.documentElement.dataset.theme = 'dark'")
            page.wait_for_function("[...document.querySelectorAll('[data-math]')].every(e => e.firstElementChild || e.dataset.mathError)")
            page.evaluate("async () => { await document.fonts.ready; }")
            page.wait_for_function("[...document.querySelectorAll('.chat-math img')].every(e => e.complete && e.naturalWidth > 0)")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert page.evaluate("""() => {
              const slot = document.querySelector('.chat-math--display');
              const visual = slot.querySelector('mjx-math, .katex-html, img');
              const box = visual.getBoundingClientRect();
              return box.width > 30 && box.height > 20;
            }"""), 'display equation geometry'

            if output:
                output.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(output / f"{browser_name}-math.png"), full_page=True)
            page.click("#sampling-btn")
            page.click("#s-math")
            page.click("#s-apply")
            assert page.locator("[data-math]").count() == 0
            assert page.locator(".st-bubble").inner_html() == baseline
            assert not errors, errors
            context.close(); browser.close()
    finally:
        server.shutdown(); server.server_close()
    print(json.dumps(report, indent=2))
    if output:
        (output / f"{browser_name}-results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", choices=["chromium", "firefox", "webkit"], default="chromium")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.browser, args.output)
