// apps/host/server/memory.mjs - HTTP routes and agent bridge for Strata Unified Long-Term Memory
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, isAbsolute, parse, resolve } from 'node:path';
import { readFileSync, existsSync } from 'node:fs';
import { homedir } from 'node:os';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const REPO_ROOT = resolve(__dirname, '../../..');
const SCRIPT_PATH = resolve(REPO_ROOT, 'tools/strata_memory_mcp.py');

export function runMemoryCli(action, args = {}) {
  return new Promise((resolveResult, reject) => {
    const cliArgs = [SCRIPT_PATH, '--cli', action];

    if (args.query) cliArgs.push('--query', String(args.query));
    if (args.content) cliArgs.push('--content', String(args.content));
    if (args.title) cliArgs.push('--title', String(args.title));
    if (args.category) cliArgs.push('--category', String(args.category));
    if (args.tags) {
      const tagStr = Array.isArray(args.tags) ? args.tags.join(',') : String(args.tags);
      cliArgs.push('--tags', tagStr);
    }
    if (args.importance !== undefined) cliArgs.push('--importance', String(args.importance));
    if (args.limit !== undefined) cliArgs.push('--limit', String(args.limit));

    const child = spawn('py', ['-3', ...cliArgs], {
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdoutText = '';
    let stderrText = '';

    child.stdout.on('data', chunk => { stdoutText += chunk.toString('utf-8'); });
    child.stderr.on('data', chunk => { stderrText += chunk.toString('utf-8'); });

    child.on('error', err => {
      reject(new Error(`Failed to execute memory CLI: ${err.message}`));
    });

    child.on('close', code => {
      if (code !== 0) {
        return resolveResult({
          error: `Memory CLI exited with code ${code}`,
          stderr: stderrText.trim(),
        });
      }
      try {
        const parsed = JSON.parse(stdoutText.trim());
        resolveResult(parsed);
      } catch (err) {
        resolveResult({
          error: `Failed to parse memory JSON output: ${err.message}`,
          raw: stdoutText.trim(),
          stderr: stderrText.trim(),
        });
      }
    });
  });
}

/**
 * Returns cached or live SOUL.md directive if available.
 */
export function getSoulDirective() {
  const override = process.env.HERMES_HOME;
  let candidates;
  if (override !== undefined && override !== '') {
    // An explicit profile must never fall back to another profile's directive.
    if (/[\0\r\n]/.test(override) || (override.startsWith('~') && !/^~(?:[\\/]|$)/.test(override))) return '';
    try {
      const expanded = override === '~' ? homedir()
        : /^~[\\/]/.test(override) ? resolve(homedir(), override.slice(2)) : override;
      const root = parse(expanded).root;
      if ((root && !isAbsolute(expanded)) || (process.platform === 'win32' && /^[\\/]$/.test(root))) return '';
      candidates = [resolve(expanded, 'SOUL.md')];
    } catch {
      return '';
    }
  } else candidates = [
    resolve(homedir(), '.hermes/SOUL.md'),
    resolve(REPO_ROOT, 'SOUL.md'),
  ];
  for (const p of candidates) {
    if (existsSync(p)) {
      try {
        const text = readFileSync(p, 'utf-8');
        if (text && text.trim().length > 0) {
          return text.trim();
        }
      } catch {
        // ignore read error
      }
    }
  }
  return '';
}

/**
 * Fast synchronous check or recall context helper
 */
export async function getRecallContext(query, topK = 3) {
  if (!query || query.trim().length < 2) return '';
  try {
    const res = await runMemoryCli('recall', { query: query.trim(), limit: topK });
    if (res && Array.isArray(res.results) && res.results.length > 0) {
      const snippets = res.results.map((r, i) => `[Memory #${i + 1} (${r.source})]: ${r.content}`);
      return `\n<relevant_long_term_memories>\n${snippets.join('\n')}\n</relevant_long_term_memories>\n`;
    }
  } catch {
    // ignore
  }
  return '';
}

/**
 * Handle HTTP request for /api/memory/*
 */
export async function handleMemoryRoute(req, res, path, method, readJsonBody) {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');

  try {
    if (path === '/api/memory/status' && method === 'GET') {
      const data = await runMemoryCli('status');
      res.statusCode = 200;
      return res.end(JSON.stringify(data));
    }

    if (path === '/api/memory/soul' && method === 'GET') {
      const data = await runMemoryCli('soul');
      res.statusCode = 200;
      return res.end(JSON.stringify(data));
    }

    if (path === '/api/memory/sections' && method === 'GET') {
      const data = await runMemoryCli('memory_md', { limit: 50 });
      res.statusCode = 200;
      return res.end(JSON.stringify(data));
    }

    if (path === '/api/memory/sections' && method === 'POST') {
      const body = await readJsonBody(req);
      const data = await runMemoryCli('append_memory', {
        title: body.title,
        content: body.content,
        category: body.category || 'core',
      });
      res.statusCode = 200;
      return res.end(JSON.stringify(data));
    }

    if (path === '/api/memory/recall' && method === 'POST') {
      const body = await readJsonBody(req);
      const data = await runMemoryCli('recall', {
        query: body.query || '',
        limit: body.limit || body.top_k || 5,
      });
      res.statusCode = 200;
      return res.end(JSON.stringify(data));
    }

    if (path === '/api/memory/remember' && method === 'POST') {
      const body = await readJsonBody(req);
      const data = await runMemoryCli('remember', {
        content: body.content || '',
        importance: body.importance !== undefined ? body.importance : 0.7,
        tags: body.tags,
      });
      res.statusCode = 200;
      return res.end(JSON.stringify(data));
    }

    if (path === '/api/memory/sleep' && method === 'POST') {
      const data = await runMemoryCli('sleep');
      res.statusCode = 200;
      return res.end(JSON.stringify(data));
    }

    res.statusCode = 404;
    return res.end(JSON.stringify({ error: `Not found: ${method} ${path}` }));
  } catch (err) {
    res.statusCode = 500;
    return res.end(JSON.stringify({ error: err.message }));
  }
}
