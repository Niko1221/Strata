import { StrictMode, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/kit/tokens.css';
import './styles/kit/components.css';
import './styles/app.css';
import { App } from './app/App';
import { installDropGuard } from './components/DropZone';
import { useStore } from './lib/store';

/** How long the page may take to be ready before the desktop window is shown anyway (the shell has its own later deadline). */
const SHELL_SHOW_DEADLINE_MS = 12_000;

const el = document.getElementById('root');
if (!el) throw new Error('#root missing');

installDropGuard();

createRoot(el).render(
  <StrictMode>
    <App />
    <ShellWindowReveal />
  </StrictMode>,
);

/** Set once the window is actually shown, and while a request is in flight, so a deadline cannot steal focus back later. */
let shellWindowShown = false;
let shellWindowRequesting = false;

/** Ask the desktop shell to show its window; a browser has no shell and does nothing. */
function shellShow(): void {
  if (shellWindowShown || shellWindowRequesting) return;
  const internals = (window as unknown as {
    __TAURI_INTERNALS__?: { invoke?: (command: string) => Promise<unknown> };
  }).__TAURI_INTERNALS__;
  const invoke = internals?.invoke;
  if (!invoke) return;
  shellWindowRequesting = true;
  void invoke('plugin:window|show')
    .then(() => {
      shellWindowShown = true;
      return invoke('plugin:window|set_focus').catch(() => { /* shown is what matters; focus is best effort */ });
    })
    .catch(() => { shellWindowRequesting = false; /* a refused call: the shell's own deadline shows the window */ });
}

/**
 * Show the desktop window once the first real screen is ready.
 *
 * The shell creates its window hidden (apps/desktop/src-tauri/src/main.rs). `ready` becomes true only after the host has
 * answered and the saved conversations have been read, so the window opens on the screen the page is actually showing,
 * not on the boot skeleton or on a screen React is about to replace. The core window plugin's `show` and `set_focus`
 * commands are allowed for this loopback origin in tauri.conf.json.
 */
function ShellWindowReveal() {
  const ready = useStore((s) => s.ready);
  // A start that never becomes ready must not leave the app running with no window (the shell's own deadline is later).
  useEffect(() => {
    const timer = setTimeout(shellShow, SHELL_SHOW_DEADLINE_MS);
    return () => clearTimeout(timer);
  }, []);
  useEffect(() => {
    if (!ready) return;
    // The commit of the ready screen has happened; one more task lets React finish the DOM update it queued. A hidden
    // webview is not painted, so showing the window is also what paints that screen.
    const timer = setTimeout(shellShow, 60);
    return () => clearTimeout(timer);
  }, [ready]);
  return null;
}

import './styles/annotation-fixes.css';
import './styles/monitor.css';
import './styles/time-skip.css';

import './styles/model-run.css';

import './styles/onboarding.css';

import './styles/board.css';

import './styles/gpu-roles.css';
