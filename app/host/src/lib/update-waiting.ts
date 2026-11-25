import type { UpdateCheckResult } from './types';

/** The little of the app's kept state this needs: Control's own kept check (`update_available` of `strata components engine`). */
interface KeptOffer { engine?: { updateAvailable?: { engine?: string | null; app?: string | null } | null } | null }

/**
 * Whether a new version of Strata itself (the app, or Control with it) is waiting to be installed. Only what the app already knows is read, never the network:
 * the last "Check for updates" result kept on this computer (the signed feed read by Control plus the app updater's verdict), and Control's own kept check.
 * A newer engine alone is not one: updating the engine does not end the app, so there is nothing for the chat agents to wait for.
 * The chat toolbar shows "Pause for update" only while this is true (or while the agents are already paused, to offer "Resume agents").
 */
export function strataUpdateWaiting(last: UpdateCheckResult | null | undefined, native: KeptOffer | null | undefined): boolean {
  if (last?.strata?.status === 'available') return true;
  return !!native?.engine?.updateAvailable?.app;
}
