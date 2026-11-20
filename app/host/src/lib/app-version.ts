/** The product version shown in Settings; Control is a development fallback. */
export function appVersionLabel(app: string | null | undefined, control?: string | null): string | undefined {
  const real = (value: string | null | undefined): string | undefined =>
    typeof value === 'string' && value.trim() && value.trim().toLowerCase() !== 'dev' ? value.trim() : undefined;
  return real(app) ?? real(control) ?? (typeof app === 'string' && app.trim() ? app.trim() : undefined);
}
