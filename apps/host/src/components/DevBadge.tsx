import { useAppMode } from '../lib/app-mode';
/** "DEV" beside the brand, only when the host runs as Strata Dev (STRATA_DEV=1, docs/DEV_BUILD.md). A warning about dev.config.json shows on hover. */
export function DevBadge() {
    const { dev } = useAppMode();
    if (!dev) return null;
    // the product name and file paths are not translated
    const title = ['Strata Dev', dev.configFile, ...dev.warnings].filter(Boolean).join('\n');
    return <span className="dev-badge" data-warn={dev.warnings.length ? true : undefined} title={title} role="status">{dev.label}</span>;
}
