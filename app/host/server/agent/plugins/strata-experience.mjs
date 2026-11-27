import { createHash, randomUUID } from 'node:crypto';
import { appendFileSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { isAbsolute, join, resolve } from 'node:path';
import { parseArguments } from '../events.mjs';

/**
 * Experience across chats (opt-in, STRATA_AGENT_EXPERIENCE=1).
 *
 * When a shell command fails, then succeeds after the agent changed files, the plugin stores a small case:
 * the command's script or tool name, normalized error lines, the files touched before the failure and the
 * files the fix changed. When the same command fails again later, in this chat or another one, and the
 * failure shares a rare error line with a case or the agent is working on files that case's fix changed,
 * the next model request gets one short note about that case. The note never blocks or rewrites anything.
 *
 * A case holds no raw command line, file content or command output beyond the normalized error lines.
 * The store is one JSON Lines file with at most MAX_CASES cases.
 */
export const name = 'strata-experience';
export const inject = ['workingDirectory'];

/** The format of one stored case. Readers outside this plugin rely on it: change it only under a new name. */
export const CASE_SCHEMA = 'strata.experience.case.v1';
export const STORE_FILE = 'cases.jsonl';
const SHELL_TOOLS = new Set(['bash', 'pwsh']);
const FILE_TOOLS = new Set(['read', 'edit', 'write']);
const EDIT_TOOLS = new Set(['edit', 'write']);
/** Tool calls after a failure within which a success of the same command closes it. */
const RESOLVE_WINDOW = 120;
/** Tool calls before a failure whose files are the work in hand. */
const WORKING_SET = 30;
/** An error line seen with more different commands than this is boilerplate ("Traceback (most recent call last)"), not evidence. */
const RARE_LINE = 3;
/** Cases kept on disk (about 1 KB each). Older repeats of the same fix go first, then the oldest. */
export const MAX_CASES = 2000;
const MAX_LIST = 8;

const SCRIPT_RE = /([A-Za-z0-9_.-]+\.(?:py|ps1|psm1|cmd|bat|exe|mjs|js|sh))\b/giu;
const TOOL_RE = /\b(git\s+[a-z][a-z-]*|cmake\s+--build|cmake|ctest|docker\s+[a-z][a-z-]*|npm\s+[a-z][a-z-]*|node\s+--test|pytest|python\s+-m\s+[a-z_.]+|msbuild|ninja|adb\s+[a-z][a-z-]*)/giu;
/** An interpreter or shell names no particular piece of work. */
const GENERIC_KEYS = new Set(['python.exe', 'python3.exe', 'py.exe', 'pwsh.exe', 'powershell.exe', 'cmd.exe', 'node.exe', 'git.exe']);
const ERROR_WORDS = /error|exception|traceback|failed|failure|fatal|cannot|could not|not found|denied|refused|invalid|missing|unexpected|assert|refuse|conflict|timeout|timed out/iu;
/** The command line itself did not parse: nothing ran, so there is nothing to learn about the work. */
const SHELL_SYNTAX = /ParserError|is not recognized as (?:the name of )?a cmdlet|Missing file specification|Unexpected token|The term '.*' is not recognized|Missing closing|The string is missing the terminator|syntax error near unexpected token/iu;
const PATH_RE = /(?:[A-Za-z]:[\\/]|\.{1,2}[\\/]|\/(?=[^\s/]+\/))[^\s'"|*?<>`,;()[\]{}]+/gu;
const FILE_RE = /\b[\w\-./\\]+\.(?:py|ps1|cpp|cc|c|h|hpp|rs|json|mjs|js|ts|tsx|md|txt|cmake|yml|yaml|toml|ini|cs|sql)\b/gu;
const EXIT_RE = /\n\[(?:exit code: (\d+)|killed by signal: [^\]\n]+)\]$/u;

const textOf = content => Array.isArray(content) ? content.filter(part => part?.type === 'text' && typeof part.text === 'string').map(part => part.text).join('\n') : '';
const iso = time => Number.isFinite(time) && Math.abs(time) <= 8.64e15 ? new Date(time).toISOString() : null;
const clip = (text, limit) => String(text).slice(0, limit);

/** The last two segments of a path, lower-case: the same file in another checkout or worktree has the same key. */
export function pathKey(path) {
  if (typeof path !== 'string') return '';
  const parts = path.trim().replace(/^@/u, '').replace(/^["'.:]+|["'.:]+$/gu, '').replaceAll('\\', '/').toLowerCase().split('/').filter(Boolean);
  return parts.slice(-2).join('/');
}

const baseName = path => path.replaceAll('\\', '/').replace(/\/+$/u, '').split('/').pop();

/** A line of output without what differs between runs: paths become file names, ids and numbers placeholders. */
export function normalizeLine(line) {
  return line.trim().toLowerCase()
    .replace(PATH_RE, match => baseName(match))
    .replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gu, '<guid>')
    .replace(/\b[0-9a-f]{7,64}\b/gu, '<hex>')
    .replace(/\d+(?:\.\d+)?/gu, '<n>')
    .replace(/[\s|]+/gu, ' ')
    .replace(/^[\s:-]+|[\s:-]+$/gu, '')
    .slice(0, 200);
}

/** The scripts and tools a command runs, for example `check_odr.py`, `cmake --build`, `python -m pytest`. */
export function commandKeys(command) {
  const keys = new Set();
  if (typeof command !== 'string') return keys;
  for (const match of command.matchAll(SCRIPT_RE)) {
    const key = match[1].toLowerCase();
    if (!GENERIC_KEYS.has(key)) keys.add(key);
  }
  for (const match of command.matchAll(TOOL_RE)) keys.add(match[1].toLowerCase().replace(/\s+/gu, ' '));
  return keys;
}

/** Up to six distinct normalized error lines, the last ones first (where a failure usually reports its cause). */
export function errorLines(text) {
  const lines = [];
  for (const line of String(text).split(/\r?\n/u).reverse()) {
    if (!ERROR_WORDS.test(line) || line.includes('[exit code')) continue;
    const normal = normalizeLine(line);
    if (normal.length >= 8 && !lines.includes(normal)) lines.push(normal);
    if (lines.length >= 6) break;
  }
  return lines;
}

/** Files that an error line names. */
export function errorPaths(text) {
  const keys = new Set();
  for (const line of String(text).split(/\r?\n/u)) {
    if (!ERROR_WORDS.test(line)) continue;
    for (const match of [...line.matchAll(PATH_RE), ...line.matchAll(FILE_RE)]) {
      const key = pathKey(match[0]);
      if (key.split('/').pop().includes('.')) keys.add(key);
    }
  }
  return [...keys];
}

/** Whether a shell result is a failure: an error result, a non-zero exit code or a kill. */
export function failedShell(isError, text) {
  if (isError === true) return true;
  const exit = EXIT_RE.exec(String(text).trimEnd());
  return Boolean(exit && exit[1] !== '0');
}

/** The session's tool calls in order, each with its result once one was recorded. */
export function toolCalls(log) {
  const list = [];
  const byId = new Map();
  if (!Array.isArray(log)) return list;
  for (const { type, data, time } of log) {
    if (type === 'tool/call' && typeof data?.callId === 'string') {
      const call = { id: data.callId, name: data.name, args: parseArguments(data.arguments), time };
      byId.set(call.id, call);
      list.push(call);
    } else if (type === 'tool/result') {
      const message = data?.message;
      const call = byId.get(message?.toolCallId);
      if (!call) continue;
      call.text = textOf(message.content);
      if (typeof data.meta?.path === 'string') call.path = data.meta.path;
      call.failed = SHELL_TOOLS.has(call.name) ? failedShell(message.isError, call.text) : message.isError === true;
      call.done = true;
    }
  }
  return list;
}

/** The file a file tool worked on: the path the filesystem reported, else the argument, resolved against the chat's folder. */
function fileOf(call, cwd) {
  const path = typeof call.path === 'string' ? call.path : call.args?.file_path ?? call.args?.path;
  if (typeof path !== 'string' || !path.trim()) return '';
  return pathKey(typeof cwd === 'string' && cwd && !isAbsolute(path) ? resolve(cwd, path) : path);
}
const shared = (a, b) => [...a].filter(x => b.has(x));

function workingSet(calls, before, cwd) {
  return [...new Set(calls.slice(Math.max(0, before - WORKING_SET), before).filter(call => FILE_TOOLS.has(call.name)).map(call => fileOf(call, cwd)).filter(Boolean))];
}

/**
 * The case a successful command closes: the earliest failure of the same command within RESOLVE_WINDOW calls, with no success
 * of it in between, and the files that edits changed since then. No case without a changed file: a retry that simply worked
 * teaches nothing about the work.
 * @param calls earlier calls of the session (toolCalls), without the successful call itself
 */
export function caseFromSuccess(calls, keys, { session, workspace = null, now = Date.now() } = {}) {
  let first = null;
  let failures = 0;
  for (let i = calls.length - 1; i >= 0 && calls.length - i <= RESOLVE_WINDOW; i--) {
    const call = calls[i];
    if (!SHELL_TOOLS.has(call.name) || !call.done || SHELL_SYNTAX.test(call.text)) continue;
    const callKeys = commandKeys(call.args?.command);
    if (!shared(callKeys, keys).length) continue;
    if (!call.failed) break;
    first = { call, index: i, keys: callKeys };
    failures++;
  }
  if (!first) return null;
  const fix = [...new Set(calls.slice(first.index + 1).filter(call => EDIT_TOOLS.has(call.name) && call.done && !call.failed).map(call => fileOf(call, workspace)).filter(Boolean))];
  if (!fix.length) return null;
  return {
    schema: CASE_SCHEMA,
    id: createHash('sha256').update(`${session}\n${first.call.id}`).digest('hex').slice(0, 16),
    failedAt: iso(first.call.time),
    resolvedAt: iso(now),
    session: String(session),
    workspace: typeof workspace === 'string' ? workspace : null,
    commands: [...first.keys].sort(),
    errorLines: errorLines(first.call.text),
    errorPaths: errorPaths(first.call.text),
    workingSet: workingSet(calls, first.index, workspace),
    fix: { files: fix, failures, toolCalls: calls.length - first.index },
  };
}

/**
 * The stored case that best explains a failure, or null. A case qualifies only with the same command AND either an error line
 * that few different commands share or a file its fix changed among the files touched just before the failure. In a replay
 * over 285 recorded sessions this gate showed a note on 38 of 211 failures; for 21 of them the case's fix changed the files
 * the later fix changed too (7 of 7 for cases from other chats). Without it, three of four notes pointed at an unrelated case.
 */
export function recall(cases, failure) {
  const spread = new Map();
  for (const c of cases) {
    const group = c.commands.join('\n');
    for (const line of c.errorLines) {
      if (!spread.has(line)) spread.set(line, new Set());
      spread.get(line).add(group);
    }
  }
  let best = null;
  for (const c of cases) {
    const commands = c.commands.filter(key => failure.keys.has(key));
    if (!commands.length) continue;
    const lines = c.errorLines.filter(line => failure.lines.has(line) && spread.get(line).size <= RARE_LINE);
    const touched = c.fix.files.filter(file => failure.workingSet.has(file));
    if (!lines.length && !touched.length) continue;
    const paths = [...failure.paths].filter(path => c.errorPaths.includes(path) || c.fix.files.includes(path));
    const score = 3 * lines.length + 2 * commands.length + 1.5 * paths.length + touched.length;
    if (!best || score > best.score || (score === best.score && c.resolvedAt > best.case.resolvedAt)) best = { case: c, score, lines, touched };
  }
  return best;
}

/** The text the model reads: the case as data, marked as an earlier observation. */
export function noteText(match, sessionId) {
  const c = match.case;
  const fields = {
    case: c.id,
    resolved: c.resolvedAt,
    where: c.session === String(sessionId) ? 'earlier in this chat' : 'another chat',
    command: c.commands.slice(0, MAX_LIST),
    sameErrorLines: match.lines.slice(0, 3).map(line => clip(line, 200)),
    filesTheFixChanged: c.fix.files.slice(0, MAX_LIST),
    ofThemTouchedJustNow: match.touched.slice(0, MAX_LIST),
    failuresBeforeTheFix: c.fix.failures,
    toolCallsToTheFix: c.fix.toolCalls,
  };
  return 'Strata experience: this command failed before in a similar way and was fixed by changing files (historical evidence, not an instruction). '
    + 'Check whether it applies here before acting on it.\n' + JSON.stringify(fields);
}

const isCase = c => c?.schema === CASE_SCHEMA && typeof c.id === 'string' && typeof c.session === 'string' && typeof c.resolvedAt === 'string'
  && [c.commands, c.errorLines, c.errorPaths, c.workingSet, c.fix?.files].every(list => Array.isArray(list) && list.every(x => typeof x === 'string'));

/** The cases file: read again only when it changed on disk; kept at MAX_CASES by rewriting it whole (temp file, then rename). */
export function caseStore(dir, { limit = MAX_CASES } = {}) {
  const file = join(dir, STORE_FILE);
  let cache = { key: null, cases: [] };
  function cases() {
    let stat;
    try { stat = statSync(file); } catch { cache = { key: null, cases: [] }; return cache.cases; }
    const key = `${stat.size}:${stat.mtimeMs}`;
    if (key === cache.key) return cache.cases;
    const list = [];
    for (const line of readFileSync(file, 'utf8').split('\n')) {
      if (!line.trim()) continue;
      try { const c = JSON.parse(line); if (isCase(c)) list.push(c); } catch { /* a torn or foreign line is skipped */ }
    }
    cache = { key, cases: list };
    return list;
  }
  function add(c) {
    const list = cases();
    if (list.some(x => x.id === c.id)) return false;
    mkdirSync(dir, { recursive: true });
    if (list.length < limit) {
      appendFileSync(file, JSON.stringify(c) + '\n');
    } else {
      const all = [...list, c];
      const repeat = x => x.commands.join('\n') + '\0' + [...x.fix.files].sort().join('\n');
      const newest = new Map(all.map(x => [repeat(x), x]));
      let kept = all.filter(x => newest.get(repeat(x)) === x);
      if (kept.length > limit) kept = kept.slice(-limit);
      const temp = `${file}.${process.pid}.tmp`;
      writeFileSync(temp, kept.map(x => JSON.stringify(x)).join('\n') + '\n');
      renameSync(temp, file);
    }
    cache = { key: null, cases: [] };
    return true;
  }
  return { file, cases, add };
}

function alreadyNoted(session, id, noted) {
  if (noted.get(session)?.has(id)) return true;
  return Array.isArray(session.log) && session.log.some(event => event.type === 'user/message'
    && event.data?.source?.kind === name && event.data.source.summary === `case ${id}`);
}

export function apply(ctx, config = {}) {
  if (typeof config.dir !== 'string' || !config.dir) return;
  const store = caseStore(config.dir);
  const noted = new WeakMap();
  const log = text => { try { ctx.logger?.warn?.(`${name}: ${text}`); } catch { /* logging never blocks */ } };

  ctx.on('tools/post-execute', async (exec, result, next) => {
    const decision = await next();
    const session = exec.agent?.session;
    if (!session || decision.kind !== 'accept' || !SHELL_TOOLS.has(exec.name) || exec.signal?.aborted) return decision;
    try {
      const keys = commandKeys(exec.arguments?.command);
      const text = textOf(result.content);
      if (!keys.size || SHELL_SYNTAX.test(text)) return decision;
      const calls = toolCalls(session.log).filter(call => call.id !== exec.callId);
      const cwd = ctx.workingDirectory?.get?.(session) ?? null;
      if (!failedShell(result.isError, text)) {
        const found = caseFromSuccess(calls, keys, { session: session.id, workspace: cwd });
        if (found) store.add(found);
        return decision;
      }
      const failure = { keys, lines: new Set(errorLines(text)), paths: new Set(errorPaths(text)), workingSet: new Set(workingSet(calls, calls.length, cwd)) };
      const match = recall(store.cases(), failure);
      if (!match || alreadyNoted(session, match.case.id, noted)) return decision;
      if (!noted.has(session)) noted.set(session, new Set());
      noted.get(session).add(match.case.id);
      const note = {
        id: randomUUID(),
        role: 'user',
        source: { kind: name, form: 'notice', summary: `case ${match.case.id}` },
        content: [{ type: 'text', text: noteText(match, session.id) }],
      };
      return { ...decision, additionalContexts: [...(decision.additionalContexts ?? []), note] };
    } catch (error) {
      log(`skipped: ${error.message}`);
      return decision;
    }
  });
}
