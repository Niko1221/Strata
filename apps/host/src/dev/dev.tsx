/**
 * Strata Dev (docs/DEV_BUILD.md): the React side of the slot registry. Everything here is inert until the host says it runs as Strata Dev
 * (GET /api/app/mode carries `dev`, STRATA_DEV=1): before that and in Live, <DevElement> renders its children as they are, <DevSlot>
 * renders nothing and useDevList() returns the app's own list. So Live draws exactly what it drew before.
 */
import * as React from 'react';
import { Component, Fragment, Suspense, lazy, useEffect, useSyncExternalStore, type ComponentType, type ErrorInfo, type ReactNode } from 'react';
import extensionModules from 'virtual:strata-dev-extensions';
import { loadAppMode, type DevSummary } from '../lib/app-mode';
import { api } from '../lib/client';
import { Btn, Icon, IconBtn } from '../components/ui';
import { pushToast } from '../lib/store';
import './dev.css';
import { setDevOverrides } from './overrides';
import { arrange, createRegistry, resolveElement, type Arranged, type Builtin, type ElementOverride, type SlotName } from './registry';

export type SlotRender = (props: Record<string, unknown>) => ReactNode;
export const registry = createRegistry<SlotRender, ComponentType<Record<string, unknown>>>();

let summary: DevSummary | null = null;
let started = false;
const DISABLED_KEY = 'strata-dev-disabled-extensions';

/** The Dev summary once the host confirmed Dev, else null. Re-renders on every registry change. */
export function useDev(): DevSummary | null {
    useSyncExternalStore(registry.subscribe, registry.version, registry.version);
    return summary;
}
export const devSummaryNow = () => summary;

/** Reports an error the page saw to the host's Logs tab (source "ui"). Nothing is sent outside Dev. */
export function reportDevError(error: unknown, extra: { extension?: string; componentStack?: string; level?: 'error' | 'warn' } = {}) {
    if (!summary) return;
    const e = error instanceof Error ? error : new Error(String(error));
    void api.post('/api/dev/client-error', { message: e.message, stack: e.stack, ...extra }).catch(() => undefined);
}

/* ------------------------------------------------------------------------------------------------ start --- */

/** Called once by the app shell. Asks the host (the same cached call the DEV badge makes) and, in Dev only, turns everything on. */
export function startDev() {
    if (started) return;
    started = true;
    void loadAppMode().then(mode => { if (mode.dev) activate(mode.dev); });
}

function activate(dev: DevSummary) {
    summary = dev;
    applyTokens(dev.tokens ?? {});
    setDevOverrides(dev.overrides);
    // A folder mode switched off in dev.config.json also hides the folder chip (unless ui.elements says otherwise).
    registry.setConfig({ ...(dev.overrides?.folder?.enabled === false ? { 'chat.composer.folder': { hidden: true } } : {}), ...(dev.ui?.elements ?? {}) });
    window.addEventListener('error', e => reportDevError(e.error ?? e.message));
    window.addEventListener('unhandledrejection', e => reportDevError(e.reason));
    window.addEventListener('keydown', e => { if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'd') { e.preventDefault(); toggleDevPanel(); } });
    void loadExtensions();
}

/** dev.config.json "tokens": design tokens (src/styles/kit/tokens.css) set on the root element, so they win over both themes. */
function applyTokens(tokens: Record<string, string>) {
    for (const [name, value] of Object.entries(tokens)) document.documentElement.style.setProperty(name, value);
}

function locallyDisabled(): string[] {
    try { const v = JSON.parse(localStorage.getItem(DISABLED_KEY) ?? '[]'); return Array.isArray(v) ? v.filter(x => typeof x === 'string') : []; } catch { return []; }
}
function rememberDisabled(name: string, disabled: boolean) {
    try { const set = new Set(locallyDisabled()); if (disabled) set.add(name); else set.delete(name); localStorage.setItem(DISABLED_KEY, JSON.stringify([...set])); } catch { /* storage may be off */ }
}

export const extensionNames = () => Object.keys(extensionModules).sort();

async function loadExtensions() {
    const off = new Set(locallyDisabled());
    for (const name of extensionNames()) {
        const enabled = summary?.ui?.extensions?.[name]?.enabled !== false && !off.has(name);
        registry.declare(name, enabled);
        if (enabled) await loadExtension(name);
    }
}

async function loadExtension(name: string) {
    registry.forget(name);
    registry.clearError(name);
    try {
        const mod = await extensionModules[name]() as { default?: unknown; setup?: unknown };
        const ext = (mod.default ?? mod) as { setup?: unknown } | ((api: ExtensionApi) => unknown);
        const setup = typeof ext === 'function' ? ext : ext?.setup;
        if (typeof setup !== 'function') throw new Error('An extension exports a setup function: export default function setup(strata) { ... }');
        await (setup as (api: ExtensionApi) => unknown)(extensionApi(name));
        registry.loaded(name);
    } catch (error) {
        const e = error instanceof Error ? error : new Error(String(error));
        registry.fail(name, e.message, e.stack);
        reportDevError(e, { extension: name });
    }
}

/** The Dev panel's switch: off removes what the extension added; on sets it up again. Remembered in this browser. */
export async function setExtensionEnabled(name: string, enabled: boolean) {
    rememberDisabled(name, !enabled);
    registry.setEnabled(name, enabled);
    if (enabled) await loadExtension(name); else registry.forget(name);
}

/* ------------------------------------------------------------------------------------------- the API --- */

export interface ExtensionApi {
    name: string;
    React: typeof React;
    ui: { Btn: typeof Btn; Icon: typeof Icon; IconBtn: typeof IconBtn };
    /** Adds something to a slot: sidebar, chat.toolbar, chat.message.actions, settings.tabs, monitor.widgets, preview.panel. */
    slot(slot: SlotName, contribution: { id: string; label?: string; order?: number; render: SlotRender }): void;
    /** Changes an existing element by its id (the inspector shows ids): hide, rename, move. */
    override(id: string, change: ElementOverride): void;
    hide(id: string): void;
    rename(id: string, label: string): void;
    move(id: string, order: number): void;
    /** Draws `component` in place of the element `id` (it gets the element's props where the app passes them, and `children`: the original). */
    replace(id: string, component: ComponentType<Record<string, unknown>>): void;
    toast(title: string, body?: string): void;
    api: { get<T>(path: string): Promise<T>; post<T>(path: string, body?: unknown): Promise<T> };
    log(...args: unknown[]): void;
    dev: DevSummary;
}

function extensionApi(name: string): ExtensionApi {
    return {
        name,
        React,
        ui: { Btn, Icon, IconBtn },
        slot: (slot, c) => registry.contribute(slot, { ...c, extension: name }),
        override: (id, change) => registry.override(name, id, change),
        hide: id => registry.override(name, id, { hidden: true }),
        rename: (id, label) => registry.override(name, id, { label }),
        move: (id, order) => registry.override(name, id, { order }),
        replace: (id, component) => registry.replace(name, id, component),
        toast: (title, body) => pushToast('info', title, body),
        api: { get: path => api.get(path), post: (path, body) => api.post(path, body) },
        log: (...args) => console.log(`[extension ${name}]`, ...args),
        dev: summary!,
    };
}

/* -------------------------------------------------------------------------------------- the boundary --- */

/** One extension's drawing: a crash shows a small notice with the error and the rest of the app goes on. */
export class ExtensionBoundary extends Component<{ extension: string; where: string; children: ReactNode }, { error: Error | null }> {
    state: { error: Error | null } = { error: null };
    static getDerivedStateFromError(error: Error) { return { error }; }
    componentDidCatch(error: Error, info: ErrorInfo) {
        registry.fail(this.props.extension, `${this.props.where}: ${error.message}`, `${error.stack ?? ''}\n\nComponent stack:${info.componentStack ?? ''}`);
        reportDevError(error, { extension: this.props.extension, componentStack: info.componentStack ?? undefined });
    }
    render() {
        if (!this.state.error) return this.props.children;
        return <span className="dev-ext-error" role="alert" title={`${this.state.error.stack ?? this.state.error.message}`} data-dev-id={this.props.where}>⚠ {this.props.extension}: {this.state.error.message}</span>;
    }
}

/* --------------------------------------------------------------------------------- what the app uses --- */

/** An existing element with a fixed id. Outside Dev: exactly `children`. In Dev: hidden, replaced, or marked for the inspector. */
export function DevElement({ id, children, props }: { id: string; children?: ReactNode; props?: Record<string, unknown> }) {
    const dev = useDev();
    if (!dev) return <>{children}</>;
    registry.see(id);
    const r = resolveElement(id, registry.overrides());
    if (r.hidden) return null;
    const replacement = r.replace ? registry.replacement(r.replace) : null;
    if (replacement) {
        const Replacement = replacement.component;
        return <span className="dev-mark" data-dev-id={id}><ExtensionBoundary extension={replacement.extension} where={id}><Replacement {...(props ?? {})}>{children}</Replacement></ExtensionBoundary></span>;
    }
    return <span className="dev-mark" data-dev-id={id}>{children}</span>;
}

/** A slot: what extensions added to it, each in its own boundary. Outside Dev (and with nothing added): nothing at all. */
export function DevSlot({ name, props }: { name: SlotName; props?: Record<string, unknown> }) {
    const dev = useDev();
    if (!dev) return null;
    registry.see(`slot:${name}`);
    const rows = arrange(name, [], registry.slot(name), registry.overrides());
    if (!rows.length) return null;
    return <>{rows.map(row => row.kind === 'contribution' ? <span key={row.id} className="dev-mark" data-dev-id={row.id}><ExtensionBoundary extension={row.contribution.extension} where={row.id}>{row.contribution.render({ ...(props ?? {}), label: row.label })}</ExtensionBoundary></span> : null)}</>;
}

/**
 * A list that holds existing elements and a slot's contributions (the sidebar's links, the Settings tabs). Outside Dev it is the list as
 * given (each builtin rendered with its own label). In Dev: hidden ones left out, labels and order from the overrides, contributions added.
 */
export function useDevList<T>(slot: SlotName, builtins: Builtin<T>[]): Arranged<T, SlotRender>[] {
    const dev = useDev();
    if (!dev) return builtins.map((b, i) => ({ kind: 'builtin', id: b.id, label: b.label, item: b.item, order: i }));
    for (const b of builtins) registry.see(b.id);
    return arrange(slot, builtins, registry.slot(slot), registry.overrides());
}

/** Renders one row of useDevList: a builtin through `draw`, a contribution in its boundary. */
export function DevListRow<T>({ row, draw, props }: { row: Arranged<T, SlotRender>; draw: (item: T, label: string | undefined) => ReactNode; props?: Record<string, unknown> }) {
    if (row.kind === 'contribution') return <span className="dev-mark" data-dev-id={row.id}><ExtensionBoundary extension={row.contribution.extension} where={row.id}>{row.contribution.render({ ...(props ?? {}), label: row.label })}</ExtensionBoundary></span>;
    if (!summary) return <Fragment>{draw(row.item, row.label)}</Fragment>;
    const replacement = row.replace ? registry.replacement(row.replace) : null;
    if (replacement) { const Replacement = replacement.component; return <span className="dev-mark" data-dev-id={row.id}><ExtensionBoundary extension={replacement.extension} where={row.id}><Replacement label={row.label}>{draw(row.item, row.label)}</Replacement></ExtensionBoundary></span>; }
    return <span className="dev-mark" data-dev-id={row.id}>{draw(row.item, row.label)}</span>;
}

/* ---------------------------------------------------------------------------------------- the panel --- */

let panelOpen = false;
const panelListeners = new Set<() => void>();
export function toggleDevPanel(open = !panelOpen) { panelOpen = open; for (const l of panelListeners) l(); }
const subscribePanel = (l: () => void) => { panelListeners.add(l); return () => { panelListeners.delete(l); }; };

const DevPanel = lazy(() => import('./DevPanel'));

/** Mounted once by the app shell. Draws nothing outside Dev; in Dev the panel (loaded only then) when it is open. */
export function DevRoot() {
    const dev = useDev();
    const open = useSyncExternalStore(subscribePanel, () => panelOpen, () => false);
    useEffect(() => { startDev(); }, []);
    if (!dev || !open) return null;
    return <Suspense fallback={null}><DevPanel onClose={() => toggleDevPanel(false)} /></Suspense>;
}
