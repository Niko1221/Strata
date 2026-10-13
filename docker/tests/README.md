# Tests

## `i18n-coverage.js`

Checks that the console renders without leftover Chinese when the UI language is
English. It builds a DOM from the real `index.html`, runs the page's own
`applyLang()`, and reports any Chinese text nodes that survive.

```bash
# against the deployed console (needs it running)
node tests/i18n-coverage.js

# against a local copy
node tests/i18n-coverage.js manager/static/index.html

# against another host
STRATA_URL=https://my-host:38888/ node tests/i18n-coverage.js
```

Expected output:

```
text nodes total        : 572
still Chinese after applyLang: 0
```

Run it after adding or changing any user-facing string. New strings need a
matching entry in the `I18N` dictionary in `manager/static/index.html`
(English is the source language; Chinese is the fallback text). A sentence split
across inline tags uses `\u0001` to mark where the tag boundary was:

```js
"保存，" + "\u0001" + "请记住密码":
    "Saved. " + "\u0001" + "Keep the password safe",
```

Only Node.js (no dependencies) is required.
