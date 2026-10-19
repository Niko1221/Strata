import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Strata web shell.
 *
 * The engine host (server/) owns hardware detection, the model registry and the
 * local inference API. In dev we proxy /api to it so the browser talks to one
 * origin, exactly like the packaged app does.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5199,
    strictPort: true,
    proxy: {
      '/api': {
        target: process.env.STRATA_HOST ?? 'http://127.0.0.1:8791',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: {
    target: 'es2022',
    outDir: 'dist',
    sourcemap: true,
  },
});
