// DeepSeek-V4.1-Flash's own context limit, read from its pack (the app offers sizes up to it: src/lib/deepseek-context.ts).
// Project Maya v1.1.28 (#442, @needmorevram, MIT) did the same in its dashboard. strata_control/dsv41_context.py reads the same file for setup and the Terminal.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

/** The pack's own limit: `max_position_embeddings` in <pack>/ds4.txt. Null when there is no pack, no file or no such line (the choices then stay as before). */
export function packContextLimit(pack) {
  if (!pack) return null;
  try {
    const match = /^max_position_embeddings\s+(\d+)\s*$/m.exec(readFileSync(join(pack, 'ds4.txt'), 'utf8'));
    const value = match ? Number(match[1]) : 0;
    return Number.isSafeInteger(value) && value >= 1024 ? value : null;
  } catch { return null; }
}

/** The limit of the pack a DeepSeek run config names (`--pack DIR` in its arguments). */
export function configContextLimit(config) {
  const args = (config?.args ?? []).map(String), at = args.indexOf('--pack');
  return at >= 0 && at + 1 < args.length ? packContextLimit(args[at + 1]) : null;
}
