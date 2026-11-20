import { tr } from '../lib/i18n';
import type { NativeInstance } from '../lib/native';
import { actualProjectionState, projectionStatusText } from '../lib/refusal-projection';

export function RefusalProjectionStatus({ instance }: { instance?: NativeInstance }) {
  const state = actualProjectionState(instance);
  return <p className="small muted" role="status" data-refusal-projection={state}>{tr(projectionStatusText[state])}</p>;
}
