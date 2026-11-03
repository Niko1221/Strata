import { useAppMode } from '../lib/app-mode';
import { toggleDevPanel } from '../dev/dev';
/** "DEV" beside the brand, only when the host runs as Strata Dev (STRATA_DEV=1, docs/DEV_BUILD.md). A warning about dev.config.json shows on hover; a click opens the Dev panel (also Ctrl+Shift+D). */
export function DevBadge() {
    const { dev } = useAppMode();
    if (!dev) return null;
    // the product name and file paths are not translated
    const title = ['Strata Dev', dev.configFile, ...dev.warnings, 'Click: Dev panel (Ctrl+Shift+D)'].filter(Boolean).join('\n');
    return <span className="dev-badge" data-warn={dev.warnings.length ? true : undefined} title={title} role="button" tabIndex={0} onClick={() => toggleDevPanel()} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleDevPanel(); } }}>{dev.label}</span>;
}
