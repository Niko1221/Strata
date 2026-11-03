import { useEffect, useState } from 'react';
import { api } from './client';

/** Which window shows the page: the app window ('window'), a browser tab ('browser'), or neither (a host nobody started: null). */
export interface AppMode {
  mode: 'window' | 'browser' | null;
  reason: string | null;
  canQuit: boolean;
  appUpdate: 'available' | 'unavailable';
  /** Only from a host that runs as Strata Dev (STRATA_DEV=1): the badge text, the dev.config.json in use and its warnings. */
  dev?: DevSummary;
}

/** What a Strata Dev host says about itself (server/dev-mode.mjs devSummary). The fields after `warnings` come from Phase 2 hosts. */
export interface DevSummary {
  label: string;
  configFile: string;
  warnings: string[];
  verbose?: boolean;
  /** dev.config.json "chatOverflowGuard" (src/lib/chat-overflow.ts). */
  chatOverflow?: { enabled: boolean; maxTotalChars?: number };
  tokens?: Record<string, string>;
  ui?: { elements: Record<string, { hidden?: boolean; label?: string; order?: number; replace?: string }>; extensions: Record<string, { enabled: boolean }> };
  overrides?: { loop?: { enabled?: boolean; maxRuns?: number }; folder?: { enabled?: boolean; defaultPath?: string }; permissions?: { defaultMode?: 'manual' | 'plan' | 'bypass'; allowedModes?: ('manual' | 'plan' | 'bypass')[] } };
}

const NONE: AppMode = { mode: null, reason: null, canQuit: false, appUpdate: 'unavailable' };
let loading: Promise<AppMode> | null = null;

/** Asked once per page load; an older host without the route (404) or a failed call means "not the app": nothing extra is shown. */
export function loadAppMode(): Promise<AppMode> {
  loading ??= api.get<AppMode>('/api/app/mode').catch(() => NONE);
  return loading;
}

export function useAppMode(): AppMode {
  const [mode, setMode] = useState<AppMode>(NONE);
  useEffect(() => { let live = true; void loadAppMode().then(m => { if (live) setMode(m); }); return () => { live = false; }; }, []);
  return mode;
}
