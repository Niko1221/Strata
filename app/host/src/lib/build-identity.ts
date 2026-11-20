/** A shell, Host and renderer from different releases must not run together. */
export function buildMismatch(renderer: string, identity: unknown): string | null {
  if (!identity || typeof identity !== 'object') return 'The app could not verify its installed files.';
  const { host, shell } = identity as { host?: unknown; shell?: unknown };
  if (typeof host !== 'string' || host !== renderer)
    return `The screen (${renderer}) and app service (${String(host ?? 'unknown')}) are from different releases.`;
  if (shell !== null && shell !== renderer)
    return `The desktop (${String(shell ?? 'unknown')}) and screen (${renderer}) are from different releases.`;
  return null;
}
