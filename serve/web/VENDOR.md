# Third-party scripts of the web app

Served from this folder like the app's own files, so the page needs nothing from the internet. Used by the Files tab
to show Markdown and source code.

| File | What | Version | License | From |
|---|---|---|---|---|
| vendor-marked.js | marked, the Markdown parser | 18.0.14 | MIT | https://cdn.jsdelivr.net/npm/marked@18.0.14/lib/marked.umd.js |
| vendor-purify.js | DOMPurify, cleans the HTML marked makes (a file cannot run script in the page) | 3.4.16 | MPL-2.0 or Apache-2.0 | https://cdn.jsdelivr.net/npm/dompurify@3.4.16/dist/purify.min.js |
| vendor-hljs.js | highlight.js (its common languages) plus powershell, dockerfile, cmake, scala, julia, dos, protobuf, nginx, latex, haskell, elixir, dart, groovy, erlang, fortran, ocaml, clojure, apache, properties, fsharp, nix, vim | 11.12.0 | BSD-3-Clause | https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.12.0/ (highlight.min.js and languages/*.min.js, joined) |

To update one: download the same file of the new version, replace it here, and run the browser tests.
