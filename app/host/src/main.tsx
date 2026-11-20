import { StrictMode, useEffect, useLayoutEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/kit/tokens.css';
import './styles/kit/components.css';
import './styles/app.css';
import { App } from './app/App';
import { installDropGuard } from './components/DropZone';
import { startRecovery } from './lib/recovery';
import { version as rendererVersion } from '../package.json';
import { buildMismatch } from './lib/build-identity';
import { translationsReady } from './lib/i18n';
import { loadHighlighter } from './components/SyntaxCode';

function Startup() {
  const [status, setStatus] = useState<{ ready: boolean; error: string }>({ ready: false, error: '' });
  // The commit proves React mounted, even when the version check shows recovery.
  useLayoutEffect(() => { window.dispatchEvent(new Event('strata-renderer-ready')); }, []);
  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);
    let disposed = false;
    void (async () => {
      try {
        const response = await fetch('/api/app/build', { cache: 'no-store', signal: controller.signal });
        if (!response.ok) throw new Error('The app service did not return its installed version.');
        const mismatch = buildMismatch(rendererVersion, await response.json());
        if (!disposed) setStatus({ ready: !mismatch, error: mismatch ?? '' });
      } catch {
        if (!disposed) setStatus({ ready: false, error: 'The app service could not be reached. Close and reopen Strata.' });
      } finally { clearTimeout(timer); }
    })();
    return () => { disposed = true; clearTimeout(timer); controller.abort(); };
  }, []);
  if (status.ready) return <App />;
  return <main className="app-recovery" role={status.error ? 'alert' : 'status'}>
    <h1>{status.error ? 'Strata could not start' : 'Starting Strata…'}</h1>
    <p>{status.error || 'Checking the installed app.'}</p>
    {status.error && <p>Reinstall the complete matching Hakua release if this continues. Keep your model files and memories.</p>}
    <button type="button" onClick={() => location.reload()}>Reload</button>
  </main>;
}
const el = document.getElementById('root');
if (!el) throw new Error('#root missing');

installDropGuard();
startRecovery();

// The saved language's strings are fetched while the rest of the code loads; the first screen is drawn once they are in (English needs no wait).
// Code colouring (the Prism grammars, 130 kB) is not needed to show the first screen: it is fetched a few seconds later, when the browser is idle,
// so the first code block of the session is already coloured when it appears. A block that appears sooner fetches it itself.
window.setTimeout(() => {
  const idle: (run: () => void) => void = 'requestIdleCallback' in window ? (run) => window.requestIdleCallback(run, { timeout: 4000 }) : (run) => window.setTimeout(run, 0);
  idle(() => { void loadHighlighter(); void import('./components/TimeSkipPanel'); void import('./components/TimeSkipWorking'); });
}, 3000);

void translationsReady().then(() => {
  createRoot(el).render(
    <StrictMode>
      <Startup />
    </StrictMode>,
  );
});

import './styles/annotation-fixes.css';
import './styles/monitor.css';
import './styles/time-skip.css';

import './styles/model-run.css';

import './styles/onboarding.css';

import './styles/board.css';

import './styles/gpu-roles.css';

import './styles/engine-tab.css';

// Loaded last: the shared depth, material and interaction treatment. See the
// header of the file for the order of the layers and why it sits here.
import './styles/liquid.css';
