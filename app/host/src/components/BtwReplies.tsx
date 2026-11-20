import { dismissBtw, stopBtw } from '../lib/store';
import type { Session } from '../lib/types';
import { tr } from '../lib/i18n';
import { Btn } from './ui';

/** Plain text keeps a side answer from introducing a second active tool/document surface. */
export function BtwReplies({ session }: { session: Session }) {
  if (!session.asides?.length) return null;
  return <section className="stack" aria-label={tr('Side questions')}>
    {session.asides?.map(side => <article key={side.id} className="st-card">
      <div className="st-card__body stack">
        <div className="row"><strong>/btw</strong><span className="small muted">{tr(side.agentId ? 'Subagent answer · no tools' : side.running ? 'Starting side-question subagent…' : 'Separate answer · no tools')}</span>
          {side.running ? <Btn size="sm" onClick={() => stopBtw(session.id)}>{tr('Stop side question')}</Btn> : <Btn size="sm" onClick={() => dismissBtw(session.id, side.id)}>{tr('Dismiss')}</Btn>}
        </div>
        <p className="small">{side.question}</p>
        <div className="st-bubble__text small" aria-busy={side.running}>{side.answer}</div>
        {side.running ? <p role="status" className="small muted">{tr('Answering the side question… A model with one request slot may wait for the current inference.')}</p> : null}
        {side.error ? <p role="status" className="small">{tr(side.error)}</p> : null}
      </div>
    </article>)}
  </section>;
}
