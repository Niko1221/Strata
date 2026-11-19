import type { NativeInstance } from './native';

export interface RefusalProjectionSettings {
  requested: boolean;
  supported: boolean;
  reason: string | null;
  vectorSha256: string;
  layers: string;
}
export type ProjectionState = 'unobserved' | 'off' | 'other' | 'loaded' | 'applied' | 'request-off';
/** Saved intent and API acceptance are never evidence of a loaded vector. */
export function actualProjectionState(instance: NativeInstance | undefined): ProjectionState {
  if (!instance || instance.status !== 'ready' || instance.phase === 'unavailable') return 'unobserved';
  const info = instance.metrics?.engine;
  const child = info?.native_child;
  if (!child || child.alive !== true || typeof child.generation !== 'string') return 'unobserved';
  if (info.cvec === 0 || info.cvec === '0') return 'off';
  if (!instance.refusalProjection?.vectorVerified || info.cvec !== 'project:4-44') return info.cvec ? 'other' : 'unobserved';
  const last = instance.metrics?.requests?.[0];
  // A respawn can retain older history. It must not look applied to the new child.
  if (last?.projection_child_generation === child.generation && ['stop', 'length'].includes(last?.finish)) {
    if (last.projection === true) return 'applied';
    if (last.projection === false) return 'request-off';
  }
  return 'loaded';
}
export const projectionStatusText: Record<ProjectionState, string> = {
  unobserved: 'Actual vector state has not been reported by the engine.',
  off: 'Engine reports: vector not loaded (off).',
  other: 'Engine reports a different or unverified control vector.',
  loaded: 'Engine reports: refusal-direction vector loaded (project, layers 4–44).',
  applied: 'Loaded; applied on the last completed request (server record).',
  'request-off': 'Loaded; off on the last completed request (server record).',
};
