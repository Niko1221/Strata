import { tr } from '../lib/i18n';
import type { NativeJob } from '../lib/native';

/** Recovery is informational. The original failed job remains available as history. */
export function RecoveredUpdateHistory({ jobs }: { jobs: readonly NativeJob[] }) {
    if (!jobs.length) return null;
    return <div className="small muted">
        <p role="status">{tr('A Later Update Completed Successfully.')}</p>
        <details><summary>{tr('Recovered Update History')}</summary>
            {jobs.map(job => <div key={job.id}><p>{tr(job.kind === 'update-strata' ? 'App Update' : 'Engine Update')} · <time dateTime={job.startedAt}>{job.startedAt}</time></p><p>{tr(job.error ?? job.summary)}</p>{job.details ? <pre>{job.details}</pre> : null}</div>)}
        </details>
    </div>;
}
