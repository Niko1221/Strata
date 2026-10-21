import type { NativeJob } from './native';

const interrupted = 'The host restarted before completion. Check diagnostics, then retry; completed files are kept.';
const updateKinds = new Set(['update-strata', 'update-engine']);

/** A confirmed later update resolves only the restart interruption of that same component.
 * Keep the original records: a running retry, another component or an ordinary failure
 * must never turn into a successful update just because Settings was reopened.
 */
export function nativeJobDiagnostics(jobs: readonly NativeJob[]) {
    const recovered = jobs.filter(job => job.state === 'error' && updateKinds.has(job.kind)
        && job.error === interrupted && Number.isFinite(Date.parse(job.startedAt))
        && jobs.some(later => later.state === 'done' && later.kind === job.kind
            && Number.isFinite(Date.parse(later.startedAt)) && Date.parse(later.startedAt) > Date.parse(job.startedAt)));
    const recoveredIds = new Set(recovered.map(job => job.id));
    return { errors: jobs.filter(job => job.state === 'error' && !recoveredIds.has(job.id)), recovered };
}
