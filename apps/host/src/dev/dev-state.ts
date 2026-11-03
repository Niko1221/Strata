/**
 * Strata Dev (docs/DEV_BUILD.md): the Dev state every Dev component reads: the registry and the host's Dev summary. No import of the
 * extension loader (dev.tsx, `virtual:strata-dev-extensions`), so screens can use useDev() without pulling the loader in.
 */
import { useSyncExternalStore, type ComponentType, type ReactNode } from 'react';
import type { DevSummary } from '../lib/app-mode';
import { createRegistry } from './registry';

export type SlotRender = (props: Record<string, unknown>) => ReactNode;
export const registry = createRegistry<SlotRender, ComponentType<Record<string, unknown>>>();

let summary: DevSummary | null = null;
/** Set once by dev.tsx when the host said it runs as Strata Dev; it tells every useDev() through the registry. */
export function setDevSummary(next: DevSummary) { summary = next; }
export const devSummaryNow = () => summary;

/** The Dev summary once the host confirmed Dev, else null. Re-renders on every registry change. */
export function useDev(): DevSummary | null {
    useSyncExternalStore(registry.subscribe, registry.version, registry.version);
    return summary;
}
