import { useEffect, useRef, useState } from 'react';
import { localeTag, tr } from '../lib/i18n';
import type { ChatMessage } from '../lib/types';
import { approximateTokens, atBottom, reasoningTime, thoughtDuration } from '../lib/reasoning';
import { Icon } from './ui';

/**
 * The model's reasoning as one quiet line above its answer. Closed by default.
 *
 * While it thinks the line says how long and about how many tokens, so a five-minute thought never looks stuck; when the
 * answer starts it freezes as "Thought for 5m 26s · ≈3,120 tokens". Only the time the model itself spent thinking is
 * counted: a segment ends when the answer or a tool call begins (lib/reasoning.ts). Opened, it is a muted box that follows
 * the newest line unless the person scrolls up.
 */
export function ReasoningBlock({ message }: { message: ChatMessage }) {
  const text = message.reasoning ?? '';
  const body = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [stick, setStick] = useState(true);
  const [now, setNow] = useState(() => Date.now());
  const thinking = !!message.streaming && message.reasoningOpen === true;
  // One tick a second while it thinks: the number moves, and nothing else in the chat re-renders.
  useEffect(() => {
    if (!thinking)
      return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [thinking]);
  useEffect(() => { if (open && stick && body.current)
    body.current.scrollTop = body.current.scrollHeight; }, [text, open, stick]);
  useEffect(() => { if (open) {
    setStick(true);
    requestAnimationFrame(() => { if (body.current)
        body.current.scrollTop = body.current.scrollHeight; });
} }, [open]);
  const took = thoughtDuration(reasoningTime(message, now));
  // The engine's own count once the turn ended; before that (and without it) an estimate, marked with ≈.
  const exact = typeof message.reasoningTokens === 'number' && message.reasoningTokens > 0 ? message.reasoningTokens : null;
  const count = exact ?? approximateTokens(text);
  const tokenText = count > 0 ? (exact === null ? tr('≈{0} tokens', [count.toLocaleString(localeTag())]) : tr('{0} tokens', [count.toLocaleString(localeTag())])) : '';
  const head = thinking ? `${tr('Thinking…')} ${took ?? '0s'}` : took ? tr('Thought for {0}', [took]) : tr('Reasoning');
  const label = tokenText ? `${head} · ${tokenText}` : head;
  return <details className="rz" open={open} data-live={thinking || undefined} onToggle={e => setOpen(e.currentTarget.open)}>
    <summary><Icon name="thinking" size={14}/><span className="rz__label">{label}</span><Icon name="chevron" size={12} className="rz__chev"/></summary>
    <div className="rz__body" ref={body} tabIndex={0} aria-label={tr('Model reasoning')} onScroll={e => setStick(atBottom(e.currentTarget))}>{text}</div>
    {open && thinking && !stick ? <button type="button" className="rz__jump" onClick={() => { setStick(true); }}><Icon name="chevron" size={11}/>{tr('Jump to latest')}</button> : null}
  </details>;
}
