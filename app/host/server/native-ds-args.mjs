/** The engine options of the DeepSeek-V4.1 engine (strata-dsv41), and the two ways the app used to hand it options it does not take.
 *
 *  strata-dsv41 is its own program (src/dsv41/ds4_main.cpp) with its own short option list, and it exits with "unknown option X" (code 2)
 *  on anything else. The Qwen engine takes other names (--vram-reserve-mib, --pool-workers, --resident-budget-gib, ...), and the app's
 *  settings, resource profile and tune presets write those into a model's "args" and "user_args". Before the load starts, the arguments
 *  of a DeepSeek model are built from what the engine takes (#4504): a name it spells differently is renamed, anything else is left out
 *  with its value and said in the job's log. test/native-ds-args.test.mjs reads ds4_main.cpp and fails when this list and the engine drift.
 */

/** Options that take one value. */
export const DS_VALUE_FLAGS = new Set([
  '--pack', '--gpus', '--max-context', '--reserve-mib', '--vram-experts', '--cpu-threads', '--expert-profile', '--ssd-cache', '--ssd-policy',
  '--eos', '--dspark', '--dspark-conf', '--dspark-block', '--dspark-depth', '--experts', '--tp', '--prefill-chunk', '--threads', '--vision',
  '--mmap-experts', '--layer-split',
]);
/** Options that stand alone. */
export const DS_SWITCH_FLAGS = new Set(['--no-expert-cache', '--profile-readonly', '--verbose', '-v']);
/** The Qwen engine's names for settings the DeepSeek engine has under another name (same meaning). */
export const DS_RENAMED = { '--vram-reserve-mib': '--reserve-mib' };

const isFlag = text => typeof text === 'string' && text.startsWith('-') && text.length > 1 && /^-{1,2}[A-Za-z]/.test(text);
/** The words that follow `flag` at `at` up to the next option: its value (one word, or none for a switch at the end of the list). */
function valueSpan(list, at) {
  return at + 1 < list.length && !isFlag(list[at + 1]) ? 1 : 0;
}

/** `args` for strata-dsv41: accepted options kept (a renamed one under the engine's name, unless the list already has that name),
 *  everything else left out together with its value. Returns {args, dropped: [{flag, value}], renamed: [{from, to}]}. */
export function dsEngineArgs(args) {
  const list = (Array.isArray(args) ? args : []).map(String);
  const out = [], dropped = [], renamed = [];
  const has = new Set(list.filter(isFlag));
  for (let i = 0; i < list.length; i++) {
    const item = list[i];
    if (!isFlag(item)) { out.push(item); continue; }     // a stray word: kept as it was (the engine says what it is)
    const own = DS_RENAMED[item];
    if (own) {
      const n = valueSpan(list, i), value = n ? list[i + 1] : null;
      i += n;
      if (has.has(own)) { dropped.push({ flag: item, value, because: `${own} is set` }); continue; }
      out.push(own, ...(value === null ? [] : [value]));
      renamed.push({ from: item, to: own });
      continue;
    }
    const takes = DS_VALUE_FLAGS.has(item), alone = DS_SWITCH_FLAGS.has(item);
    if (takes || alone) {
      out.push(item);
      if (takes && i + 1 < list.length && !isFlag(list[i + 1])) out.push(list[++i]);
      continue;
    }
    const n = valueSpan(list, i);
    dropped.push({ flag: item, value: n ? list[i + 1] : null });
    i += n;
  }
  return { args: out, dropped, renamed };
}

/** The person's own options of a DeepSeek model: unaccepted ones leave the list (so the settings file heals itself); the app's
 *  --vram-reserve-mib stays, because the Settings field writes it under that name and it is renamed at every load. */
export function dsUserArgs(user) {
  const list = (Array.isArray(user) ? user : []).map(String);
  const out = [];
  for (let i = 0; i < list.length; i++) {
    const item = list[i];
    if (!isFlag(item)) { out.push(item); continue; }
    const n = valueSpan(list, i);
    const keep = DS_VALUE_FLAGS.has(item) || DS_SWITCH_FLAGS.has(item) || item in DS_RENAMED;
    if (keep) out.push(item, ...list.slice(i + 1, i + 1 + n));
    i += n;
  }
  return out;
}

/** One line for the job's log about what was changed, or '' when nothing was. */
export function dsArgsNote({ dropped = [], renamed = [] }) {
  const parts = [];
  if (renamed.length) parts.push(`${renamed.map(r => `${r.from} is passed as ${r.to}`).join(', ')}`);
  if (dropped.length) parts.push(`left out because the DeepSeek engine does not take them: ${dropped.map(d => [d.flag, d.value].filter(v => v !== null && v !== undefined).join(' ')).join(', ')}`);
  return parts.length ? `DeepSeek engine options: ${parts.join('; ')}.` : '';
}

/** `args` without `flag` and the picture word that follows it (the Qwen engine's --vision stands alone; an older config may carry a value). */
const PICTURE_WORDS = new Set(['on', 'off', 'yes', 'no', 'gpu', 'cpu', 'none']);
export function withoutVisionFlag(args) {
  const list = (Array.isArray(args) ? args : []).map(String), out = [];
  for (let i = 0; i < list.length; i++) {
    if (list[i] === '--vision') { if (i + 1 < list.length && PICTURE_WORDS.has(list[i + 1].toLowerCase())) i++; continue; }
    out.push(list[i]);
  }
  return out;
}
