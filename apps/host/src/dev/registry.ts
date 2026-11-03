/**
 * Strata Dev (docs/DEV_BUILD.md): the slot registry, with no React and no DOM so it can be tested on its own (tests/dev-registry.test.mjs).
 *
 * Two kinds of entries share one id space:
 *   - existing elements of the app, each with a fixed id (`sidebar.monitor`, `settings.tabs.engine`, `screen.models`, ...): dev.config.json
 *     (ui.elements) or an extension can hide, rename, reorder or replace them;
 *   - contributions of extensions to a named slot (`chat.toolbar`, `sidebar`, ...).
 * Outside Dev nothing reads this registry: the app renders its own elements as before.
 */
export const SLOTS = ['sidebar', 'chat.toolbar', 'chat.message.actions', 'settings.tabs', 'monitor.widgets', 'preview.panel'] as const;
export type SlotName = typeof SLOTS[number];

export interface ElementOverride { hidden?: boolean; label?: string; order?: number; replace?: string }

export interface Contribution<R = unknown> {
    /** Unique within its slot; the full id is `<slot>.<id>`. */
    id: string;
    extension: string;
    label?: string;
    order?: number;
    /** What the slot draws. The React layer (dev.tsx) calls it inside an error boundary. */
    render: R;
}

export interface Builtin<T> { id: string; label?: string; item: T }

export type Arranged<T, R = unknown> =
    | { kind: 'builtin'; id: string; label?: string; item: T; replace?: string; order: number }
    | { kind: 'contribution'; id: string; label?: string; contribution: Contribution<R>; order: number };

/** dev.config.json wins over an extension: the person's own file has the last word. */
export function mergeOverrides(fromExtensions: Record<string, ElementOverride>, fromConfig: Record<string, ElementOverride>): Record<string, ElementOverride> {
    const out: Record<string, ElementOverride> = { ...fromExtensions };
    for (const [id, o] of Object.entries(fromConfig)) out[id] = { ...(out[id] ?? {}), ...o };
    return out;
}

/**
 * The entries of one list (a slot that also holds existing elements): builtins first in their own order (10, 20, ...), contributions after
 * them (1000, 1010, ...) unless they name an order; then every override applied (hidden ones dropped, labels, orders, replacements). Stable.
 */
export function arrange<T, R = unknown>(slot: string, builtins: Builtin<T>[], contributions: Contribution<R>[], overrides: Record<string, ElementOverride>): Arranged<T, R>[] {
    const rows: Arranged<T, R>[] = [
        ...builtins.map((b, i): Arranged<T, R> => ({ kind: 'builtin', id: b.id, label: b.label, item: b.item, order: (i + 1) * 10 })),
        ...contributions.map((c, i): Arranged<T, R> => ({ kind: 'contribution', id: `${slot}.${c.id}`, label: c.label, contribution: c, order: c.order ?? 1000 + i * 10 })),
    ];
    const kept: Arranged<T, R>[] = [];
    for (const row of rows) {
        const o = overrides[row.id];
        if (o?.hidden) continue;
        const next = { ...row, ...(o?.label ? { label: o.label } : {}), ...(Number.isFinite(o?.order) ? { order: o!.order! } : {}) };
        if (next.kind === 'builtin' && o?.replace) next.replace = o.replace;
        kept.push(next);
    }
    return kept.map((row, index) => ({ row, index })).sort((a, b) => a.row.order - b.row.order || a.index - b.index).map(x => x.row);
}

/** One existing element on its own (not in a list): hidden, its label, and the component that replaces it, if any. */
export function resolveElement(id: string, overrides: Record<string, ElementOverride>): { hidden: boolean; label?: string; replace?: string } {
    const o = overrides[id];
    return { hidden: o?.hidden === true, ...(o?.label ? { label: o.label } : {}), ...(o?.replace ? { replace: o.replace } : {}) };
}

export interface ExtensionInfo { name: string; enabled: boolean; loaded: boolean; error: string | null; errorStack?: string; contributions: string[] }

/** The registry: what the extensions added, which extension is on, and which element ids the page has shown (for the inspector). */
export function createRegistry<R = unknown, C = unknown>() {
    const contributions = new Map<string, Contribution<R>[]>();
    const extensionOverrides = new Map<string, Record<string, ElementOverride>>();
    const replacements = new Map<string, { extension: string; component: C }>();
    const extensions = new Map<string, ExtensionInfo>();
    const seen = new Set<string>();
    let config: Record<string, ElementOverride> = {};
    let version = 0;
    const listeners = new Set<() => void>();
    const changed = () => { version++; for (const l of listeners) l(); };
    const info = (name: string) => {
        let e = extensions.get(name);
        if (!e) extensions.set(name, e = { name, enabled: true, loaded: false, error: null, contributions: [] });
        return e;
    };
    const active = (name: string) => extensions.get(name)?.enabled !== false;

    return {
        subscribe(listener: () => void) { listeners.add(listener); return () => { listeners.delete(listener); }; },
        version: () => version,
        setConfig(next: Record<string, ElementOverride>) { config = next; changed(); },
        /** An extension is known (found by the loader), before its setup runs. */
        declare(name: string, enabled: boolean) { const e = info(name); e.enabled = enabled; changed(); },
        loaded(name: string) { info(name).loaded = true; changed(); },
        fail(name: string, error: string, stack?: string) { const e = info(name); e.error = error; e.errorStack = stack; changed(); },
        clearError(name: string) { const e = info(name); e.error = null; e.errorStack = undefined; changed(); },
        setEnabled(name: string, enabled: boolean) { info(name).enabled = enabled; changed(); },
        /** Everything an extension added is removed (before it is set up again, or when it is switched off for good). */
        forget(name: string) {
            for (const [slot, list] of contributions) contributions.set(slot, list.filter(c => c.extension !== name));
            extensionOverrides.delete(name);
            for (const [id, r] of replacements) if (r.extension === name) replacements.delete(id);
            const e = info(name); e.contributions = []; e.loaded = false;
            changed();
        },
        contribute(slot: string, c: Contribution<R>) {
            if (!(SLOTS as readonly string[]).includes(slot)) throw new Error(`Unknown slot "${slot}". Slots: ${SLOTS.join(', ')}`);
            if (typeof c.id !== 'string' || !c.id) throw new Error('A contribution needs an id.');
            const list = (contributions.get(slot) ?? []).filter(x => !(x.id === c.id && x.extension === c.extension));
            contributions.set(slot, [...list, c]);
            info(c.extension).contributions.push(`${slot}.${c.id}`);
            changed();
        },
        override(extension: string, id: string, o: ElementOverride) {
            extensionOverrides.set(extension, { ...(extensionOverrides.get(extension) ?? {}), [id]: { ...(extensionOverrides.get(extension)?.[id] ?? {}), ...o } });
            changed();
        },
        replace(extension: string, id: string, component: C) {
            replacements.set(id, { extension, component });
            changed();
        },
        /** Contributions of extensions that are switched on. */
        slot(slot: string): Contribution<R>[] { return (contributions.get(slot) ?? []).filter(c => active(c.extension)); },
        overrides(): Record<string, ElementOverride> {
            const fromExtensions: Record<string, ElementOverride> = {};
            for (const [name, o] of extensionOverrides) if (active(name)) Object.assign(fromExtensions, o);
            // A replacement registered by an extension counts as `replace: <id>` unless the config says otherwise.
            for (const [id, r] of replacements) if (active(r.extension)) fromExtensions[id] = { ...(fromExtensions[id] ?? {}), replace: fromExtensions[id]?.replace ?? id };
            return mergeOverrides(fromExtensions, config);
        },
        /** The component an override's `replace` names: a replacement registered under that name by an extension that is on. */
        replacement(name: string): { extension: string; component: C } | null {
            const r = replacements.get(name);
            return r && active(r.extension) ? r : null;
        },
        extensions: () => [...extensions.values()],
        see(id: string) { if (!seen.has(id)) { seen.add(id); } },
        seen: () => [...seen].sort(),
    };
}

export type Registry<R = unknown, C = unknown> = ReturnType<typeof createRegistry<R, C>>;
