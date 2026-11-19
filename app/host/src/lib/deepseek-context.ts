/**
 * DeepSeek-V4.1-Flash's context sizes in the app: 8K up to the pack's own limit (the model reads 1,048,576 positions natively).
 * Project Maya v1.1.28 (#442, @needmorevram, MIT) offered 512K and 1M the same way. strata_control/dsv41_context.py says the same for setup and the
 * Terminal (tests/terminal/test_dsv41_context.py compares the two). The engine takes whatever --max-context it is given; the rope settings come from the pack.
 */
export const CONTEXT_CHOICES = [8192, 32768, 65536, 131072, 262144, 524288, 1048576];
/** The largest size not flagged as untested; also how far the list goes while the pack does not say more (a pack without ds4.txt behaves as before). */
export const TESTED_CONTEXT = 262144;
/** The model's own length: the most any pack can say. */
export const NATIVE_CONTEXT = 1048576;
export const UNTESTED_NOTE = 'Not tested yet. A context this long takes graphics memory from the model and reads long prompts slowly.';

/** The sizes to offer: up to the pack's limit (`contextMax`, from its ds4.txt) when the installed model has one, else up to 256K. */
export function deepseekContexts(limit?: number | null): number[] {
  const cap = typeof limit === 'number' && Number.isFinite(limit) && limit > 0 ? limit : TESTED_CONTEXT;
  return CONTEXT_CHOICES.filter(size => size <= cap);
}

/** An empty string when the size can run; otherwise one sentence (only a size the model cannot address is refused). */
export function deepseekContextProblem(context: number, limit?: number | null): string {
  const cap = typeof limit === 'number' && Number.isFinite(limit) && limit > 0 ? limit : NATIVE_CONTEXT;
  if (!Number.isInteger(context) || context < 1024) return 'Choose a context of at least 1,024 tokens.';
  return context > cap ? `A context past ${cap.toLocaleString('en-US')} tokens is more than this model reads.` : '';
}

/** The plain words past 256K, an empty string otherwise. */
export const deepseekContextNote = (context: number): string => (context > TESTED_CONTEXT ? UNTESTED_NOTE : '');
