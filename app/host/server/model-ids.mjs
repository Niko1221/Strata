/** #6778 / #6621: which model names a client sends are Strata's own catalog ids. The same rule as `strata serve` (terminal/serve/server.py check_model):
 *  a request that names one of these but not the loaded model gets a clear "not loaded" answer, while a placeholder another app sends ("gpt-4o", "claude-5-5-sonnet") is still
 *  served by the one loaded model (#5312). */
const PREFIXES = ['inkling-2-mini', 'hy4-flash', 'qwen3.8-flash-next', 'qwen3.6-', 'qwen3.5-'];
const SIZE_NAME = /^(?:(?:ud-)?i?q\d_[0-9a-z_]+(?:-v\d+)?|sq-\d+|gyro-[a-z])$/;
const FAMILY_PREFIX = /^(?:qwen3\.8-flash-next|qwen3\.6-35b-a3b|hy4-flash|inkling-2-mini)-/;

export const looksLikeStrataModelId = name => {
  if (typeof name !== 'string') return false;
  const asked = name.trim().toLowerCase();
  return !!asked && (PREFIXES.some(p => asked.startsWith(p)) || SIZE_NAME.test(asked));
};

/** True when `asked` is the loaded model's name with or without its family prefix ("qwen3.8-flash-next-iq3_s" and "iq3_s"). */
export const sameModelName = (asked, loaded) => {
  const bare = s => String(s ?? '').trim().toLowerCase().replace(FAMILY_PREFIX, '');
  return !!asked && bare(asked) === bare(loaded);
};

export const notLoadedMessage = (asked, loadedIds) => {
  const shown = String(asked).slice(0, 80);
  const ids = (loadedIds ?? []).filter(Boolean);
  if (!ids.length) return `The model "${shown}" is not loaded.`;
  return `The model "${shown}" is not loaded; currently serving ${ids.map(i => `"${i}"`).join(', ')}. Load "${shown}" in Strata, or send one of the loaded names as "model".`;
};
