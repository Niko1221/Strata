/** What Enter does while the assistant is answering; true steering is a separate coding-agent operation. */
export type WhileBusy = 'wait' | 'queue' | 'interject' | 'steer';
export const WHILE_BUSY: readonly WhileBusy[] = ['wait', 'queue', 'interject', 'steer'];
export const validWhileBusy = (value: unknown): value is WhileBusy => WHILE_BUSY.includes(value as WhileBusy);

/**
 * What to do with the message, as one word:
 *  - 'refuse'    Wait: nothing is sent until the answer is done (the message stays in the box, as always);
 *  - 'goal'      Send after, while a goal runs: the host's own queue delivers it between steps (it asks the host first, so this is only a try);
 *  - 'park'      Send after: the message waits here and goes when the answer is done;
 *  - 'interrupt' Interrupt: the answer stops where it is (what is written stays) and the message goes as soon as it has stopped.
 *  - 'steer'     Steer current run: a correction enters the coding agent at a safe point in its current turn.
 */
export type BusyPlan = 'refuse' | 'goal' | 'park' | 'interrupt' | 'steer';
export function busyPlan(mode: WhileBusy): BusyPlan {
  return mode === 'steer' ? 'steer' : mode === 'queue' ? 'goal' : mode === 'interject' ? 'interrupt' : 'refuse';
}
