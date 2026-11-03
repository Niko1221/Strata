import { defineConfig, type Plugin, type ViteDevServer } from 'vite';
import react from '@vitejs/plugin-react';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Strata web shell.
 *
 * The engine host (server/) owns hardware detection, the model registry and the
 * local inference API. In dev we proxy /api to it so the browser talks to one
 * origin, exactly like the packaged app does.
 */

/* Strata Dev (STRATA_DEV=1, docs/DEV_BUILD.md). Without it every value below is the one Live always had. */
const STRATA_DEV = process.env.STRATA_DEV === '1';
/** dev/extensions/ui: STRATA_DEV_EXTENSIONS names dev/extensions, else beside STRATA_DEV_CONFIG, else the checkout's (server/dev-mode.mjs). */
const DEV_UI_EXTENSIONS = process.env.STRATA_DEV_EXTENSIONS
  ? join(resolve(process.env.STRATA_DEV_EXTENSIONS), 'ui')
  : process.env.STRATA_DEV_CONFIG
    ? join(dirname(resolve(process.env.STRATA_DEV_CONFIG)), 'dev', 'extensions', 'ui')
    : fileURLToPath(new URL('../../dev/extensions/ui', import.meta.url));

/** The extensions in a folder: `<name>/index.{tsx,ts,jsx,js}` or `<name>.{tsx,ts,jsx,js}`; names starting with "_" or "." are skipped. */
function findExtensions(dir: string): Record<string, string> {
  const found: Record<string, string> = {};
  if (!existsSync(dir)) return found;
  for (const entry of readdirSync(dir).sort()) {
    if (entry.startsWith('_') || entry.startsWith('.')) continue;
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) {
      const index = ['index.tsx', 'index.ts', 'index.jsx', 'index.js'].map(n => join(path, n)).find(p => existsSync(p));
      if (index) found[entry] = index;
    } else {
      const m = /^(.+)\.(tsx|ts|jsx|js)$/.exec(entry);
      if (m && !entry.endsWith('.d.ts')) found[m[1]] = path;
    }
  }
  return found;
}

/**
 * `virtual:strata-dev-extensions`: a lazy import per UI extension (src/dev/dev.tsx loads them). Outside Dev it is an empty object, so Live
 * carries no extension code. In the Vite dev server an edited, added or removed extension reloads the page.
 */
function strataDevExtensions(): Plugin {
  const id = 'virtual:strata-dev-extensions';
  const resolved = `\0${id}`;
  const inside = (file: string) => resolve(file).toLowerCase().startsWith(resolve(DEV_UI_EXTENSIONS).toLowerCase());
  return {
    name: 'strata-dev-extensions',
    resolveId: source => (source === id ? resolved : null),
    load(source) {
      if (source !== resolved) return null;
      if (!STRATA_DEV) return 'export default {};';
      const rows = Object.entries(findExtensions(DEV_UI_EXTENSIONS)).map(([name, file]) => `  ${JSON.stringify(name)}: () => import(${JSON.stringify(file.replace(/\\/g, '/'))}),`);
      return `export default {\n${rows.join('\n')}\n};\n`;
    },
    configureServer(server: ViteDevServer) {
      if (!STRATA_DEV) return;
      server.watcher.add(DEV_UI_EXTENSIONS);
      const again = (file: string) => {
        if (!inside(file)) return;
        const mod = server.moduleGraph.getModuleById(resolved);
        if (mod) server.moduleGraph.invalidateModule(mod);
        server.ws.send({ type: 'full-reload' });
      };
      server.watcher.on('add', again);
      server.watcher.on('unlink', again);
    },
    handleHotUpdate({ file, server }) {
      // An extension's setup runs once per page: an edit reloads the page so it runs again.
      if (!STRATA_DEV || !inside(file)) return;
      server.ws.send({ type: 'full-reload' });
      return [];
    },
  };
}

export default defineConfig({
  plugins: [
    react(),
    {
      // KaTeX's stylesheet names every font three times (woff2, woff, ttf). The app's webviews all read woff2, so the other two are
      // dropped before Vite copies the files: ~300 KB of fonts instead of ~1.1 MB, and none of it loads until an answer has math (#3268).
      name: 'katex-woff2-only',
      enforce: 'pre',
      transform(code, id) {
        if (!id.includes('katex') || !id.split('?')[0].endsWith('.css')) return null;
        return code.replace(/,url\([^)]*\.woff\) format\("woff"\),url\([^)]*\.ttf\) format\("truetype"\)/g, '');
      },
    },
    strataDevExtensions(),
  ],
  // Strata Dev: extensions live outside this folder and import React; they get the app's one copy.
  ...(STRATA_DEV ? { resolve: { dedupe: ['react', 'react-dom'] } } : {}),
  // Strata Dev builds React's development version (full error messages with component stacks) and keeps the code unminified.
  ...(STRATA_DEV ? { define: { 'process.env.NODE_ENV': JSON.stringify('development') } } : {}),
  server: {
    // Strata Dev (STRATA_DEV=1, docs/DEV_BUILD.md) takes 5200, so it runs beside the usual dev server.
    port: process.env.STRATA_DEV === '1' ? 5200 : 5199,
    strictPort: true,
    ...(STRATA_DEV ? { fs: { allow: [fileURLToPath(new URL('.', import.meta.url)), DEV_UI_EXTENSIONS] } } : {}),
    proxy: {
      '/api': {
        target: process.env.STRATA_HOST ?? 'http://127.0.0.1:8791',
        changeOrigin: true,
        ws: true,
      },
      // Strata Dev: the backend hooks' own routes (server/dev-hooks.mjs).
      ...(STRATA_DEV ? { '/dev/': { target: process.env.STRATA_HOST ?? 'http://127.0.0.1:8792', changeOrigin: true } } : {}),
    },
  },
  build: {
    target: 'es2022',
    outDir: 'dist',
    sourcemap: true,
    ...(STRATA_DEV ? { minify: false } : {}),
  },
});
