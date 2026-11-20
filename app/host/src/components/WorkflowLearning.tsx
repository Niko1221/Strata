import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { tr } from '../lib/i18n';
import { Btn, ErrorNotice } from './ui';

type LearningView = {
  enabled: boolean;
  autoApply: boolean;
  available: boolean;
  learned: number;
  running: boolean;
  canRetry: boolean;
  canUndo: boolean;
  last: { phase: string; reason?: string | null; skill?: { id: string; name: string } | null } | null;
};

const REASONS: Record<string, string> = {
  'daily-limit': 'The daily review limit has been reached.',
  'not-reusable': 'No reusable procedure was found in this workflow.',
  'manual-edit-kept': 'Your edited or disabled skill was kept unchanged.',
  'library-limit': 'The learned skill limit has been reached.',
  'request-priority': 'Review paused so the new request could run.',
  'review-incomplete': 'The review did not produce a valid procedure. You can retry the last completed workflow.',
  unchanged: 'The saved procedure already covers this workflow.',
  'storage-error': 'Could not save the learned change. Check Strata data storage and retry.',
};

export function WorkflowLearning() {
  const [view, setView] = useState<LearningView | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    const refresh = () => { void api.get<LearningView>('/api/skills/learning', controller.signal)
      .then(value => { if (!controller.signal.aborted) setView(value); })
      .catch(() => { if (!controller.signal.aborted) setError(tr('Could not read workflow learning status.')); }); };
    refresh();
    const timer = setInterval(refresh, 3000);
    return () => { controller.abort(); clearInterval(timer); };
  }, []);
  async function action(kind: 'toggle' | 'auto' | 'retry' | 'undo') {
    if (!view) return;
    setPending(true); setError('');
    try {
      setView(kind === 'toggle' || kind === 'auto'
        ? await api.put<LearningView>('/api/skills/learning', kind === 'auto' ? { autoApply: !view.autoApply } : { enabled: !view.enabled })
        : await api.post<LearningView>('/api/skills/learning/' + kind, {}));
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { setPending(false); }
  }
  const last = view?.last;
  return <section className="workflow-learning" aria-label={tr('Learn from completed work')}>
    <div className="workflow-learning-heading"><div className="skill-copy"><h2 className="h4">{tr('Automatically apply skills to agent workflows')}</h2><p className="small muted">{tr('Before an agent task starts, add complete instructions from matching enabled skills. Trigger phrases and workspace scope are respected. Selected instructions are sent to the task model. Your approval rules stay in effect.')}</p></div><button className="st-toggle" role="switch" aria-label={tr('Automatically apply skills to agent workflows')} aria-checked={view?.autoApply ?? false} disabled={!view || pending} onClick={() => void action('auto')}/></div>
    <div className="workflow-learning-heading"><div className="skill-copy">
      <h2 className="h4">{tr('Learn from completed work')}</h2>
      <p className="small muted">{tr('After tasks with several completed tools, the same model can save a reusable SKILL.md. This uses one extra model request. File contents, tool output and credentials are excluded from the review packet.')}</p>
    </div><button className="st-toggle" role="switch" aria-label={tr('Learn from completed work')} aria-checked={view?.enabled ?? false} disabled={!view || pending} onClick={() => void action('toggle')}/></div>
    <p className="small muted">{tr('Learned instructions stay in this Strata profile. Edit or disable them in the skills list. Approval choices stay in effect.')}</p>
    <div role="status" aria-live="polite" className="small">
      {view && !view.available ? tr('Enable Skills in Settings → Tools to use workflow learning.') : view?.running ? tr('Reviewing the completed workflow…') : last?.phase === 'saved' && last.skill ? tr('Saved skill: {0}', [last.skill.name]) : last?.phase === 'undone' ? tr('The last learned change was undone.') : last?.reason ? tr(REASONS[last.reason] ?? REASONS['review-incomplete']) : view?.enabled ? tr('Waiting for an eligible completed workflow.') : tr('Workflow learning is off.')}
    </div>
    <div className="workflow-learning-actions"><Btn disabled={!view?.enabled || !view.canRetry || pending || view.running} onClick={() => void action('retry')}>{tr('Review latest workflow')}</Btn><Btn disabled={!view?.canUndo || pending || view.running} onClick={() => void action('undo')}>{tr('Undo last learned change')}</Btn></div>
    {error && <ErrorNotice message={error} />}
  </section>;
}
