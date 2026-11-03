/**
 * The Read pictures checkbox on the Models screen: what it shows is the person's choice. The host's reports (the saved settings, whether the picture files
 * are prepared, whether the engine has a reader) only give it its first value for a version, and never overwrite a choice made by hand (GUI feedback 2, item 4:
 * the box used to flip off while Load ran and back on afterwards, because it was re-derived from the host's report as that report changed under the load).
 */
export interface InstalledPictures { visionEnabled?: boolean; visionPrepared?: boolean; visionDevice?: string; images?: string }

/** The value a version opens with: off only when the person turned it off and the files exist; otherwise on (the device they chose, or Automatic). */
export function openingVision(installed: InstalledPictures | undefined): string {
  if (!installed) return 'yes';
  if (installed.visionEnabled === false && installed.visionPrepared !== false) return 'no';
  return installed.visionDevice ?? (installed.images && installed.images !== 'off' ? installed.images : 'yes');
}

/** The value to show for a version: the person's own choice for it when there is one, else what it opens with. */
export const shownVision = (picked: string | undefined, installed: InstalledPictures | undefined): string => picked ?? openingVision(installed);

/**
 * Whether the engine has no picture reader, as the row should show it. While a job runs the host is busy and its reader probe can drop out for a moment:
 * the answer given before the job started is kept until it ends, so the row never changes under the person's hands.
 */
export function steadyNoReader(now: boolean, jobRunning: boolean, last: { id: string; value: boolean } | undefined, id: string): { id: string; value: boolean } {
  return jobRunning && last && last.id === id ? last : { id, value: now };
}
