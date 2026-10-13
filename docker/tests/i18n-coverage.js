// Faithful check: build the real DOM from index.html, run applyLang() in
// English mode, then report any Chinese that survives in the rendered page.
const fs = require('fs');
// Usage: node tests/i18n-coverage.js [path-to-index.html]
// With no argument it downloads the page from STRATA_URL (default below).
const path = process.argv[2];
const html = path ? fs.readFileSync(path, 'utf8') : null;
if (!html) {
  const https = require('https');
  const url = process.env.STRATA_URL || 'https://127.0.0.1:38888/';
  const req = https.get(url, { rejectUnauthorized: false }, res => {
    let buf = '';
    res.on('data', d => (buf += d));
    res.on('end', () => run(buf));
  });
  req.on('error', e => { console.error(e.message); process.exit(1); });
} else {
  run(html);
}

function run(html) {
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

// ---- tiny HTML parser -> text-node tree ----------------------------------
function parse(nodes, src) {
  let i = 0;
  const VOID = new Set(['br', 'hr', 'img', 'input', 'meta', 'link']);
  function parseFragment() {
    while (i < src.length) {
      const lt = src.indexOf('<', i);
      if (lt === -1) {
        if (i < src.length) nodes.push({ nodeType: 3, nodeValue: decode(src.slice(i)) });
        break;
      }
      if (lt > i) nodes.push({ nodeType: 3, nodeValue: decode(src.slice(i, lt)) });
      if (src.startsWith('<!--', lt)) { i = src.indexOf('-->', lt) + 3; continue; }
      const gt = src.indexOf('>', lt);
      const tag = src.slice(lt + 1, gt).trim();
      i = gt + 1;
      if (tag.startsWith('/')) continue;                 // close tag
      const name = (tag.split(/\s/)[0] || '').toLowerCase();
      if (VOID.has(name)) { nodes.push({ nodeType: 1, tagName: name.toUpperCase(), childNodes: [] }); continue; }
      const kids = [];
      nodes.push({ nodeType: 1, tagName: name.toUpperCase(), childNodes: kids });
      parseFragment.call(null, kids);
    }
  }
  function decode(s) {
    return s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&')
            .replace(/&quot;/g, '"').replace(/&#39;/g, "'");
  }
  parseFragment.call(null, nodes);
  return nodes;
}

const htmlNoScript = html.replace(/<script[\s\S]*?<\/script>/gi, '').replace(/<style[\s\S]*?<\/style>/gi, '');
const bodyKids = parse([], htmlNoScript);
link(bodyKids, null);
const body = { nodeType: 1, tagName: 'BODY', childNodes: bodyKids };

function link(nodes, parent) {
  nodes.forEach((n, i) => {
    n.parentNode = parent || null;
    n.nextSibling = nodes[i + 1] || null;
    n.previousSibling = nodes[i - 1] || null;
    if (n.childNodes && n.childNodes.length) link(n.childNodes, n);
  });
}
function walk(root, out) {
  for (const n of root.childNodes) {
    if (n.nodeType === 3) out.push(n);
    else if (n.nodeType === 1) walk(n, out);
  }
  return out;
}
const textNodes = walk(body, []);

global.document = {
  body,
  createTreeWalker(root) {
    const list = walk(root, []);
    let i = 0;
    return { nextNode: () => (i < list.length ? list[i++] : null), currentNode: list[i] || null };
  },
  querySelector: () => null,
  querySelectorAll: () => [],
  getElementById: () => null,
  addEventListener() {},
  createElement: () => ({ style: {}, classList: { add() {}, remove() {} }, appendChild() {} }),
  createElementNS: () => ({}),
};
global.window = { devicePixelRatio: 1, dispatchEvent() {}, addEventListener() {}, removeEventListener() {} };
global.navigator = { language: 'en-US' };
global.localStorage = { getItem: (k) => (k === 'ui_lang' ? 'en' : null), setItem: () => {} };
global.NodeFilter = { SHOW_TEXT: 4 };
global.devicePixelRatio = 1;
global.requestAnimationFrame = () => 0;
global.matchMedia = () => ({ addEventListener() {}, removeEventListener() {} });
global.Path2D = class { rect() {} };
global.CustomEvent = class {};

const start = script.indexOf('const I18N = ');
const end = script.indexOf('function toggleLang');
const block = script.slice(start, end);
new Function(block + '\napplyLang();')();

const left = textNodes
  .map(n => n.nodeValue.trim())
  .filter(v => v && /[一-鿿]/.test(v));

console.log('text nodes total        :', textNodes.length);
console.log('still Chinese after applyLang:', left.length);
[...new Set(left)].slice(0, 20).forEach(v => console.log('   ', v.slice(0, 80)));
}
